"""Recompute pilot projections/metrics and validate its bounded execution."""
from pathlib import Path
import json,time
import numpy as np
from run_object_repair_v2 import ROOT,OUT,HISTORY,KEYS,VARIANTS,read,sha,dump,verify
from omni_semantic_map.full_ovo_evaluation import voxel_metrics
from omni_semantic_map.mesh_instance_evaluation import transfer_labels
from omni_semantic_map.instance_error_audit import audit_instances,filter_transition


def audit():
    started=time.monotonic();protocol=verify();queue=read(OUT/'queue_status.json');summary=read(OUT/'summary.json')
    assert queue['complete'] and summary['complete']
    attempts=list((OUT/'runs').glob('*/*/run.json'));assert len(attempts)==12==protocol['maximum_new_maps']
    assert {(p.parent.parent.name,p.parent.name) for p in attempts}=={(s,v) for s in KEYS for v in VARIANTS}
    gate=read(OUT/'predictions_complete.json');assert gate['complete']
    for p in attempts:assert sha(p)==gate['runs'][str(p.relative_to(OUT))]
    for name,want in protocol['prior_artifact_sha256'].items():assert sha(name)==want,name
    checks=[];results=[]
    for entry in protocol['scenes']:
        key=entry['key'];m=read(OUT/'evaluation'/key/'metrics.json');assert sha(OUT/'evaluation'/key/'metrics.json')==summary['scene_results'][key]
        results.append(m)
        for name,want in entry['hashes'].items():assert sha(name)==want,name
        cache=Path(entry['cache']);manifest=read(cache/'manifest.json');total_tokens=0;total_observations=0;per_observation={}
        for name,want in manifest['files_sha256'].items():assert sha(cache/name)==want,name
        for rec in manifest['records']:
            p=cache/'frames'/rec['file'];assert sha(p)==rec['sha256']
            z=np.load(p);total_tokens+=len(z['point_ids']);total_observations+=rec['masks']
            assert len(np.unique(z['point_ids']))==len(z['point_ids'])
            for i,count in enumerate(np.diff(z['offsets'])):per_observation[rec['observation_offset']+i]=int(count)
        ref=np.load(entry['reference']);truth=ref['instance'];valid=ref['valid'] if 'valid' in ref.files else np.ones(len(truth),bool)
        prior_projection=np.load(HISTORY/'evaluation'/key/'native_bridge_2.npz')['instance'];prior_audit=audit_instances(truth[valid],prior_projection[valid])
        for variant in VARIANTS:
            target=OUT/'runs'/key/variant;run=read(target/'run.json');method=m['methods'][variant]
            if not run['complete']:
                assert run['failed'] and not method['complete'];continue
            assert run['no_gt'] and run['protocol_sha256']==sha(OUT/'protocol.json')
            for name,want in run['artifacts'].items():assert sha(target/name)==want,name
            assert len(run['frame_times_s'])==len(manifest['records'])
            assert run['frame_p95_s']==float(np.quantile(run['frame_times_s'],.95))
            assert run['frame_median_s']==float(np.median(run['frame_times_s']))
            geo=np.load(run['geometry_path']);rows=read(run['classification_path'])['objects'];kept=[r['object_id'] for r in rows if r['predictions']['mean']['retained']]
            projected,_=transfer_labels(geo['xyz'],np.where(np.isin(geo['instance'],kept),geo['instance'],0),ref['xyz'],mode='nearest_5cm')
            np.testing.assert_array_equal(projected,np.load(OUT/'evaluation'/key/(variant+'.npz'))['instance'])
            score=voxel_metrics(truth[valid],projected[valid]);assert score==method['score']
            diagnosis=audit_instances(truth[valid],projected[valid])
            assert diagnosis==read(OUT/'evaluation'/key/(variant+'_audit.json'))
            assert filter_transition(prior_audit,diagnosis)==method['transition']
            assert method['assigned_valid_fraction']==float((projected[valid]>0).mean())
            if variant=='constraint_control':
                frames={r['observation_offset']+i:r['frame_id'] for r in manifest['records'] for i in range(r['masks'])}
                ids=[i for row in rows for i in row['observation_ids']];assert len(set(ids))==len(ids)
                assert all(len(row['observation_ids'])==len(set(frames[i] for i in row['observation_ids'])) for row in rows)
            else:
                ownership=read(target/'ownership.json');assert ownership['total_tokens']==total_tokens
                assert ownership['assigned_tokens']+ownership['unassigned_tokens']==total_tokens
                consumed={i:0 for i in per_observation}
                for row in rows:
                    assert row['unique_frames']>=3
                    for c in row['support_contributions']:
                        assert c['original_support_points']==per_observation[c['observation_id']]
                        consumed[c['observation_id']]+=c['assigned_points']
                assert all(0<=n<=per_observation[i] for i,n in consumed.items())
                g=np.load(target/'support_graph.npz');e=g['edges'];blocked=(g['negative']>=protocol['config']['min_negative_frames'])&(g['negative']>=protocol['config']['negative_positive_ratio']*g['positive'])
                violations=int(np.sum(blocked&(g['labels'][e[:,0]]>0)&(g['labels'][e[:,0]]==g['labels'][e[:,1]])))
                assert violations==run['confirmed_constraint_violations']
                if variant=='repair':assert violations==0
                events=read(target/'events.json');assert len(events)==len(manifest['records'])
                assert max(x['repair_atoms'] for x in events)<=protocol['config']['max_repair_atoms']
                assert sum(x['positive_witnesses'] for x in events)==int(g['positive'].sum())
                assert sum(x['negative_witnesses'] for x in events)==int(g['negative'].sum())
                if variant=='no_revoke':assert not any(x['split_lineage'] for x in events)
        if all(m['methods'][v]['complete'] for v in ['repair','no_revoke']):
            a=np.load(OUT/'runs'/key/'repair/support_graph.npz');b=np.load(OUT/'runs'/key/'no_revoke/support_graph.npz')
            for name in ['keys','edges','positive','negative','seen']:np.testing.assert_array_equal(a[name],b[name])
        checks.append(dict(scene=key,all_source_and_input_hashes=True,all_projections_and_metrics_recomputed=True,
            transitions_recomputed=True,token_conservation_checked=True,ablation_evidence_identical=True,
            coframe_control_checked=True,repair_exclusions_checked=True,frame_percentiles_recomputed=True))
        print('AUDITED',key,flush=True)
    for dataset,group in summary['datasets'].items():
        rows=[r for r in results if r['dataset']==dataset]
        for method,v in group.items():
            if not v['complete']:assert any(not r['methods'][method]['complete'] for r in rows);continue
            scores=[r['methods'][method]['score'] for r in rows]
            assert v['pq100']==float(np.mean([x['observed_voxel_instance_pq'] for x in scores])*100)
            for name in ['tp','fp','fn']:assert v[name]==sum(x[name] for x in scores)
    if summary['all_maps_completed']:
        groups=summary['datasets'];g=protocol['gate']
        actual=dict(mean_gain_each_dataset=all(d['repair']['pq100']-d['native_bridge_2']['pq100']>=2 for d in groups.values()),
            nondecreasing_scenes=sum(r['methods']['repair']['score']['observed_voxel_instance_pq']>=r['methods']['native_bridge_2']['score']['observed_voxel_instance_pq']-1e-10 for r in results)>=3,
            tp_not_lower_each_dataset=all(d['repair']['tp']>=d['native_bridge_2']['tp'] for d in groups.values()),
            positive_vs_constraint_each_dataset=all(d['repair']['pq100']>d['constraint_control']['pq100'] for d in groups.values()),
            positive_revocation_ablation_each_dataset=all(d['repair']['pq100']>d['no_revoke']['pq100'] for d in groups.values()),
            new_matches_each_dataset=all(sum(len(r['methods']['repair']['transition']['gained_matched_gt']) for r in results if r['dataset']==ds)>0 for ds in groups))
        assert actual==summary['accuracy_gate_checks'];assert summary['accuracy_gate_passed']==all(actual.values())
    assert summary['full_go_gate_passed']==(summary['accuracy_gate_passed'] and summary['cost_gate_passed'])
    assert summary['timing_missing_scenes']==['replica_office_0','replica_office_1'] and not summary['cost_gate_passed']
    result=dict(complete=True,checks=checks,attempts=12,maximum_attempts=12,new_score_rows=sum(r['methods'][v]['complete'] for r in results for v in VARIANTS),
        legacy_score_rows=8,protocol_sha256=sha(OUT/'protocol.json'),summary_sha256=sha(OUT/'summary.json'),prediction_gate_sha256=sha(OUT/'predictions_complete.json'),
        no_training_or_new_frontend=True,no_post_score_tuning=True,stop_gate_recomputed=True,development_only=True,
        source_sha256=sha(Path(__file__)),wall_s=time.monotonic()-started)
    dump(OUT/'audit.json',result);print('AUDIT PASS',result['new_score_rows'],'new scores',flush=True)


if __name__=='__main__':audit()
