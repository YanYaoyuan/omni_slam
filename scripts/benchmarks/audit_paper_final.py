"""Completion audit of the fixed matrix; no new model inference or tuning."""
from pathlib import Path
import json
import time
import numpy as np
from paper_final_common import ROOT,RUN,FROZEN,SCENES,read,sha,dump,verify_execution,run_scenes,scored_scenes,unavailable_scenes
from paper_final_metrics import reference,complete_metrics,graph_diagnostics
from omni_semantic_map.mesh_instance_evaluation import transfer_labels
from omni_semantic_map.shared_observations import observation_table,select_views
from omni_semantic_map.consensus_category import classify


def audit():
    began=time.monotonic();verify_execution();queue=read(RUN/'queue_status.json');assert queue['complete']
    summary=read(RUN/'summary.json');assert summary['complete']
    gate=read(RUN/'predictions_complete.json');assert gate['complete']
    amendment=read(RUN/'data_amendment.json');scenes=scored_scenes();assert summary['scenes']==scenes
    unavailable=unavailable_scenes()
    assert set(scenes)|set(unavailable)==set(SCENES)
    assert set(scenes)&set(unavailable)==set()
    assert summary['unavailable_scenes']==unavailable
    assert summary['sufficient_final_scene_coverage']==(len(scenes)>=4)
    if len(scenes)<4:assert not summary['association_claim_supported'] and not summary['independent_category_claim_supported']
    assert amendment['before_final_test_decoding_inference_and_scores'] and not amendment['replacement_scenes']
    for p,want in amendment['evidence_sha256'].items():assert sha(ROOT/p)==want,p
    for record in read(RUN/'download.json')['files']:assert sha(record['path'])==record['sha256'],record['path']
    checks=[];method_ids={x['id'] for x in read(FROZEN/'protocol.json')['variants']}
    ontology=read(FROZEN/'config/nyu_prediction_mapping.json')
    prediction_checks=[]
    assert set(gate['files'])==set(run_scenes())
    parent=RUN/'execution_revisions/v2_before_annotation_adapter/execution_manifest.json'
    for scene in run_scenes():
        assert sha(RUN/'predictions'/(scene+'.json'))==gate['files'][scene]
        pred=read(RUN/'predictions'/(scene+'.json'));assert pred['complete'] and pred['no_gt']
        assert pred['execution_manifest_sha256']==sha(parent)
        assert set(pred['variants'])==method_ids
        geometries=set()
        for key,v in pred['variants'].items():
            path=Path(v['geometry_path']);assert sha(path)==v['geometry_sha256']
            assert read(path.parent/'run.json')['complete'];geometries.add(str(path))
        assert len(geometries)==5
        cache=RUN/'frontend'/scene/'cache';m=read(cache/'manifest.json')
        assert m['complete'] and m['frames']==400 and m['frames_cached']==200 and m['no_semantic_gt']
        for name,want in m['files_sha256'].items():assert sha(cache/name)==want,name
        for rec in m['records']:assert sha(cache/'frames'/rec['file'])==rec['sha256']
        for phase in ['masks','triplets','features','geometry']:
            phase_run=read(RUN/'frontend'/scene/(phase+'_run.json'));assert phase_run['complete']
            assert phase_run['job_sha256']==sha(RUN/'jobs'/(scene+'.json'))
            for name,want in phase_run['files_sha256'].items():assert sha(RUN/'frontend'/scene/name)==want
        prediction_checks.append(dict(scene=scene,geometry_outputs=5,method_rows=8,
            immutable_prediction_hash=True,all_frontend_hashes=True,geometry_frames=400,semantic_frames=200))
    for scene in scenes:
        assert sha(RUN/'evaluation'/scene/'metrics.json')==summary['scene_results'][scene]
        pred=read(RUN/'predictions'/(scene+'.json'));metrics=read(RUN/'evaluation'/scene/'metrics.json')
        assert set(metrics['methods'])==method_ids
        cache=RUN/'frontend'/scene/'cache'
        ref=reference(scene);saved=np.load(RUN/'evaluation'/scene/'reference.npz')
        for a,b in [('xyz','xyz'),('truth','instance'),('valid','valid'),('semvalid','semvalid')]:np.testing.assert_array_equal(ref[a],saved[b])
        geometry_count=0;geometry_predictions={}
        for key,v in pred['variants'].items():
            path=Path(v['geometry_path']);assert sha(path)==v['geometry_sha256'];base=read(path.parent/'classification.json')['objects']
            assert len(v['objects'])==len(base)
            for a,b in zip(v['objects'],base):
                assert a['object_id']==b['object_id'] and a['observation_ids']==b['observation_ids']
                assert a['base_retained']==b['predictions']['mean']['retained']
                if key=='B1':assert not a['base_retained'] or b['unique_frames']>=3
            if v['geometry'] not in geometry_predictions:
                geometry_count+=1;geom=np.load(path);keep=[x['object_id'] for x in v['objects'] if x['base_retained']]
                projection,_=transfer_labels(geom['xyz'],np.where(np.isin(geom['instance'],keep),geom['instance'],0),ref['xyz'],mode='nearest_5cm')
                stored=np.load(RUN/'evaluation'/scene/(v['geometry']+'_projection.npz'))['instance']
                np.testing.assert_array_equal(stored,projection);geometry_predictions[v['geometry']]=projection
            projection=geometry_predictions[v['geometry']]
            ins=complete_metrics(ref['truth'][ref['valid']],projection[ref['valid']]);assert ins==metrics['methods'][key]['instance']
            classes={x['object_id']:ontology.get(x['prediction']['name'],-1) for x in v['objects']}
            sem=complete_metrics(ref['truth'][ref['semvalid']],projection[ref['semvalid']],ref['categories'],classes)['semantic']
            assert sem==metrics['methods'][key]['semantic']
            assert all(v>=0 for k,v in metrics['cost'][key].items())
        assert geometry_count==5
        f,frames,areas=observation_table(cache);text=np.load(cache/'text_features.npy');text/=np.linalg.norm(text,axis=1,keepdims=True);vocab=read(cache/'vocabulary.json')
        for obj in pred['variants']['M']['objects']:
            assert classify(obj['observation_ids'],f,frames,areas,text,vocab)==obj['prediction']
        diagnostics=graph_diagnostics(cache,RUN/'maps'/scene/'cg',ref['support_truth'])
        assert json.loads(json.dumps(diagnostics,allow_nan=False))==read(RUN/'evaluation'/scene/'graph_diagnostics.json')
        checks.append(dict(scene=scene,geometry_outputs=geometry_count,method_rows=8,geometry_frames=400,semantic_frames=200,
            cache_and_input_hashes=True,reference_exact=True,all_geometry_projections_exact=True,
            all_metrics_reproduced=True,main_readout_exact=True,foreground_and_memberships_fixed=True,mechanism_diagnostics_reproduced=True))
        print('AUDITED',scene,flush=True)
    # Independent recomputation of all aggregate metrics and prespecified contrasts.
    rows=[read(RUN/'evaluation'/s/'metrics.json') for s in scenes]
    for k,aggregate in summary['methods'].items():
        values=[r['methods'][k] for r in rows]
        for name,group,field in [('instance_pq100','instance','observed_voxel_instance_pq'),
            ('category_pq100','semantic','class_aware_pq'),('macro_category_pq100','semantic','macro_pq'),
            ('foreground_miou100','semantic','foreground_miou')]:
            assert aggregate[name]==float(np.mean([v[group][field] for v in values])*100)
        for name,group,field in [('tp','instance','tp'),('fp','instance','fp'),('fn','instance','fn'),
            ('correct_classes','semantic','correct_class_matches'),('category_geometry_matches','semantic','geometry_matches')]:
            assert aggregate[name]==sum(v[group][field] for v in values)
    draws=np.random.default_rng(0).integers(0,len(rows),(10000,len(rows)))
    for baseline,group,field in [('B0','instance','observed_voxel_instance_pq'),('B3','semantic','class_aware_pq'),('A3','semantic','class_aware_pq')]:
        d=np.array([100*(r['methods']['M'][group][field]-r['methods'][baseline][group][field]) for r in rows])
        c=summary['primary_contrasts'][baseline]
        assert c['per_scene_delta']==dict(zip(scenes,d.tolist())) and c['mean_delta']==float(d.mean())
        assert c['paired_scene_bootstrap_ci95']==np.quantile(d[draws].mean(1),[.025,.975]).tolist()
        assert c['passes_frozen_effect_gate']==bool(d.mean()>0 and (d>=-1e-8).sum()>=int(np.ceil(2*len(rows)/3)))
    result=dict(complete=True,selected_scenes=SCENES,scored_scenes=scenes,
        unavailable_scenes=unavailable,missing_data_amendment_sha256=sha(RUN/'data_amendment.json'),
        annotation_adapter_amendment_sha256=sha(RUN/'annotation_adapter_amendment.json'),
        checks=checks,prediction_checks=prediction_checks,
        method_scene_rows=sum(x['method_rows'] for x in checks),prediction_method_rows=sum(x['method_rows'] for x in prediction_checks),
        selected_method_rows_accounted_for=len(scenes)*8+len(unavailable)*8,
        geometry_outputs=sum(x['geometry_outputs'] for x in prediction_checks),aggregate_and_bootstrap_exact=True,
        frozen_paper_manifest_sha256=sha(FROZEN/'manifest.json'),execution_manifest_sha256=sha(RUN/'execution_manifest.json'),
        predictions_gate_sha256=sha(RUN/'predictions_complete.json'),summary_sha256=sha(RUN/'summary.json'),
        no_new_inference=True,no_score_based_retuning=True,no_scene_replacement=True,
        measurement_limits=['CG RSS is a cumulative shared-process high-water mark, not independent per-variant memory.',
            'Timing includes inherited audit readouts and staged process setup; no robot real-time claim.'],
        audit_wall_s=time.monotonic()-began)
    dump(RUN/'completion_audit.json',result);print('COMPLETION AUDIT PASSED',result['method_scene_rows'],'rows',flush=True)


def audit_timing():
    """Extend the completed score audit with the one permitted timing repeat."""
    verify_execution();path=RUN/'completion_audit.json';result=read(path);assert result['complete']
    assert result['summary_sha256']==sha(RUN/'summary.json')
    plan=read(RUN/'timing_replay/plan.json');done=read(RUN/'timing_replay/complete.json')
    assert done['complete'] and done['predictions_gate_unchanged'] and done['summary_unchanged']
    assert plan['successful_repetitions_per_scene']==1 and plan['scenes']==run_scenes()
    assert all(1<=n<=3 for n in plan['attempt_counts'].values())
    script=ROOT/'scripts/benchmarks/measure_paper_final_geometry.py';assert sha(script)==plan['source_sha256']
    assert sha(RUN/'predictions_complete.json')==plan['predictions_gate_sha256']
    checks=[]
    for scene in run_scenes():
        p=RUN/'timing_replay'/(scene+'.json');assert sha(p)==done['files'][scene];t=read(p)
        assert t['source_sha256']==sha(script) and t['frozen_geometry_source_sha256']==sha(ROOT/'scripts/benchmarks/split_strong_frontend.py')
        assert t['job_sha256']==sha(RUN/'jobs'/(scene+'.json'))
        assert t['original_cache_manifest_sha256']==sha(RUN/'frontend'/scene/'cache/manifest.json')
        assert t['prediction_sha256']==sha(RUN/'predictions'/(scene+'.json'))
        assert t['complete'] and t['exact_cached_arrays'] and t['exact_cached_manifest'] and t['original_outputs_unchanged']
        assert t['timing_repetition']==plan['attempt_counts'][scene]
        records=t['records'];frames=read(RUN/'jobs'/(scene+'.json'))['frames']
        assert [(x['frame_id'],x['semantic']) for x in records]==[(x['frame_id'],x['semantic']) for x in frames]
        assert len(records)==400 and sum(x['semantic'] for x in records)==200
        values=[x['wall_s'] for x in records];assert all(x>0 for x in values)
        assert t['frame_median_s']==float(np.median(values)) and t['frame_p95_s']==float(np.quantile(values,.95))
        for phase in ['masks','triplets','features']:
            r=read(RUN/'frontend'/scene/(phase+'_run.json'))['records']
            assert len(r)==200 and [x['frame_id'] for x in r]==[x['frame_id'] for x in frames if x['semantic']]
            assert all(x['wall_s']>0 for x in r)
        checks.append(dict(scene=scene,geometry_timing_frames=400,semantic_timing_frames_per_phase=200,
            source_and_original_cache_hashes=True,exact_outputs=True,percentiles_reproduced=True))
    saved=RUN/'completion_scoring_audit.json';assert not saved.exists();saved.write_bytes(path.read_bytes())
    result.update(timing_replay_checks=checks,timing_replay_manifest_sha256=sha(RUN/'timing_replay/complete.json'),
        scoring_audit_sha256=sha(saved),successful_timing_replays_per_scene=1,timing_attempt_counts=plan['attempt_counts'],
        no_new_detection_descriptor_or_association_inference=True)
    dump(path,result);print('TIMING AUDIT PASSED',len(checks),'scenes',flush=True)


if __name__=='__main__':
    import sys
    if '--timing-only' in sys.argv:audit_timing()
    else:audit()
