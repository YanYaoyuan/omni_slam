"""Fixed-domain evaluation and stop decision for the bounded v2 pilot."""
from collections import Counter
from pathlib import Path
import json
import numpy as np
from run_object_repair_v2 import OUT,HISTORY,KEYS,VARIANTS,verify,read,dump,sha
from omni_semantic_map.full_ovo_evaluation import voxel_metrics
from omni_semantic_map.mesh_instance_evaluation import transfer_labels
from omni_semantic_map.instance_error_audit import audit_instances,filter_transition


def evaluate():
    protocol=verify();gate=read(OUT/'predictions_complete.json');assert gate['complete']
    for name,want in gate['runs'].items():assert sha(OUT/name)==want
    results=[]
    for entry in protocol['scenes']:
        key=entry['key'];folder=OUT/'evaluation'/key;folder.mkdir(parents=True,exist_ok=False)
        for p,want in entry['hashes'].items():assert sha(p)==want,p
        reference=np.load(entry['reference']);truth=reference['instance'];vertices=reference['xyz']
        valid=reference['valid'] if 'valid' in reference.files else np.ones(len(truth),bool)
        old=read(HISTORY/'evaluation'/key/'metrics.json');methods={};old_audit={}
        for name in ['baseline','native_bridge_2']:
            path=HISTORY/'evaluation'/key/(name+'.npz');assert sha(path)==protocol['prior_artifact_sha256'][str(path)]
            projected=np.load(path)['instance'];score=voxel_metrics(truth[valid],projected[valid])
            for k in ['observed_voxel_instance_pq','tp','fp','fn','sq','rq']:assert score[k]==old['methods'][name][k],(key,name,k)
            audit=audit_instances(truth[valid],projected[valid]);old_audit[name]=audit
            methods[name]=dict(complete=True,score=score,audit_counts=dict(predicted=audit['predicted_counts'],gt=audit['gt_counts']),
                assigned_valid_fraction=float((projected[valid]>0).mean()),source='immutable v1 cache')
        for variant in VARIANTS:
            target=OUT/'runs'/key/variant;run=read(target/'run.json')
            if not run['complete']:
                methods[variant]=dict(complete=False,error=run['error']);continue
            for name,want in run['artifacts'].items():assert sha(target/name)==want,name
            geometry=np.load(run['geometry_path']);rows=read(run['classification_path'])['objects']
            kept=[r['object_id'] for r in rows if r['predictions']['mean']['retained']]
            labels=np.where(np.isin(geometry['instance'],kept),geometry['instance'],0)
            projected,_=transfer_labels(geometry['xyz'],labels,vertices,mode='nearest_5cm')
            score=voxel_metrics(truth[valid],projected[valid]);audit=audit_instances(truth[valid],projected[valid])
            transition=filter_transition(old_audit['native_bridge_2'],audit)
            np.savez_compressed(folder/(variant+'.npz'),instance=projected)
            dump(folder/(variant+'_audit.json'),audit)
            methods[variant]=dict(complete=True,score=score,audit_counts=dict(predicted=audit['predicted_counts'],gt=audit['gt_counts']),
                transition=transition,assigned_valid_fraction=float((projected[valid]>0).mean()),
                regions=len(rows),retained_regions=len(kept),frame_p95_s=run['frame_p95_s'],frame_median_s=run['frame_median_s'],
                finalization_s=run['finalization_s'],preparation_s=run.get('preparation_s',0),wall_s=run['wall_s'],
                cpu_peak_rss_bytes=run['cpu_peak_rss_bytes'],constraint_violations=run['confirmed_constraint_violations'],
                split_events=run.get('split_events',0))
        # Ablation uses exactly the same edge evidence, not a newly tuned graph.
        if all(methods[v]['complete'] for v in ['repair','no_revoke']):
            a=np.load(OUT/'runs'/key/'repair/support_graph.npz');b=np.load(OUT/'runs'/key/'no_revoke/support_graph.npz')
            for name in ['keys','edges','positive','negative','seen']:np.testing.assert_array_equal(a[name],b[name])
        historical_p95=None
        if entry['dataset']=='SceneNN':
            prior=read(Path(entry['baseline_classification']).with_name('run.json'))
            historical_p95=float(np.quantile(prior['frame_times'],.95))
        result=dict(complete=True,key=key,dataset=entry['dataset'],methods=methods,valid_voxels=int(valid.sum()),
            historical_causal_p95_s=historical_p95,reference_sha256=sha(entry['reference']),
            scope='Already-viewed development data; fixed observed domain; no official leaderboard claim.')
        dump(folder/'metrics.json',result);results.append(result)
        print('EVALUATED',key,{k:round(v['score']['observed_voxel_instance_pq']*100,4) if v['complete'] else 'FAILED' for k,v in methods.items()},flush=True)
    groups={};all_maps=all(r['methods'][v]['complete'] for r in results for v in VARIANTS)
    for dataset in ['Replica','SceneNN']:
        rows=[r for r in results if r['dataset']==dataset];summary={}
        for method in ['baseline','native_bridge_2']+VARIANTS:
            if not all(r['methods'][method]['complete'] for r in rows):summary[method]=dict(complete=False);continue
            scores=[r['methods'][method]['score'] for r in rows]
            summary[method]=dict(complete=True,pq100=float(np.mean([x['observed_voxel_instance_pq'] for x in scores])*100),
                **{k:sum(x[k] for x in scores) for k in ['tp','fp','fn']})
        groups[dataset]=summary
    accuracy=False;cost=False;checks={}
    if all_maps:
        g=protocol['gate'];changes=[r['methods']['repair']['score']['observed_voxel_instance_pq']-r['methods']['native_bridge_2']['score']['observed_voxel_instance_pq'] for r in results]
        checks=dict(mean_gain_each_dataset=all(d['repair']['pq100']-d['native_bridge_2']['pq100']>=g['dataset_mean_delta_pq100_min'] for d in groups.values()),
            nondecreasing_scenes=sum(x>=-1e-10 for x in changes)>=g['nondecreasing_scenes_min'],
            tp_not_lower_each_dataset=all(d['repair']['tp']>=d['native_bridge_2']['tp'] for d in groups.values()),
            positive_vs_constraint_each_dataset=all(d['repair']['pq100']>d['constraint_control']['pq100'] for d in groups.values()),
            positive_revocation_ablation_each_dataset=all(d['repair']['pq100']>d['no_revoke']['pq100'] for d in groups.values()),
            new_matches_each_dataset=all(sum(len(r['methods']['repair']['transition']['gained_matched_gt']) for r in results if r['dataset']==ds)>0 for ds in groups))
        accuracy=all(checks.values())
        cost=all(r['historical_causal_p95_s'] is not None and r['methods']['repair']['frame_p95_s']<=2*r['historical_causal_p95_s'] for r in results)
    result=dict(complete=True,all_maps_completed=all_maps,new_map_attempts=12,maximum_new_maps=12,datasets=groups,accuracy_gate_checks=checks,
        accuracy_gate_passed=accuracy,cost_gate_passed=cost,full_go_gate_passed=accuracy and cost,
        decision='Proceed only to a separately defined v2 main protocol' if accuracy and cost else 'STOP this pilot branch; no automatic sweep, model or scene expansion.',
        timing_missing_scenes=[r['key'] for r in results if r['historical_causal_p95_s'] is None],
        scene_results={r['key']:sha(OUT/'evaluation'/r['key']/'metrics.json') for r in results},
        protocol_sha256=sha(OUT/'protocol.json'),prediction_gate_sha256=sha(OUT/'predictions_complete.json'),no_post_score_tuning=True,development_only=True)
    dump(OUT/'summary.json',result);print('PILOT DECISION',result['decision'],flush=True)


if __name__=='__main__':evaluate()
