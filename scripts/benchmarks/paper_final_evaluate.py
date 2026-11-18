"""Score all sealed predictions, including failures and regressions."""
import argparse
from pathlib import Path
import time
import numpy as np
from paper_final_common import ROOT,RUN,FROZEN,SCENES,read,sha,dump,verify_execution,run_scenes,scored_scenes,unavailable_scenes
from paper_final_metrics import complete_metrics,reference,geometry_signatures,graph_diagnostics
from omni_semantic_map.mesh_instance_evaluation import transfer_labels


def evaluate_scene(scene):
    verify_execution();gate=read(RUN/'predictions_complete.json');assert gate['complete']
    assert scene in scored_scenes()
    for s,want in gate['files'].items():assert sha(RUN/'predictions'/(s+'.json'))==want,s
    state=read(RUN/'predictions'/(scene+'.json'));ref=reference(scene);out=RUN/'evaluation'/scene
    assert not (out/'metrics.json').exists()
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'reference.npz',xyz=ref['xyz'],instance=ref['truth'],valid=ref['valid'],semvalid=ref['semvalid'])
    dump(out/'category_reference.json',dict(classes=ref['categories'],purities=ref['purities']))
    mapping=read(FROZEN/'config/nyu_prediction_mapping.json');projected={};results={}
    costs={};front=RUN/'frontend'/scene;frontend={}
    cache_manifest=read(front/'cache/manifest.json')
    for phase in ['masks','triplets','features','geometry']:
        r=read(front/(phase+'_run.json'));u=read(front/(phase+'_resources.json'))
        frontend[phase]=dict(model_phase_wall_s=r['wall_s'],process_wall_s=u['process_wall_s'],
            cold_import_verification_overhead_s=max(0,u['process_wall_s']-r['wall_s']),
            peak_rss_bytes=u['sampled_peak_rss_bytes'],gpu_peak_allocated_bytes=r['gpu_peak_allocated_bytes'],
            gpu_peak_reserved_bytes=r['gpu_peak_reserved_bytes'])
    fullfrontend=sum(x['process_wall_s'] for x in frontend.values())
    for key,v in state['variants'].items():
        path=Path(v['geometry_path']);assert sha(path)==v['geometry_sha256']
        if v['geometry'] not in projected:
            geometry=np.load(path);keep=[x['object_id'] for x in v['objects'] if x['base_retained']]
            labels=np.where(np.isin(geometry['instance'],keep),geometry['instance'],0)
            pr,_=transfer_labels(geometry['xyz'],labels,ref['xyz'],mode='nearest_5cm');projected[v['geometry']]=pr
            np.savez_compressed(out/(v['geometry']+'_projection.npz'),instance=pr)
        prediction=projected[v['geometry']];rows=v['objects']
        predclasses={r['object_id']:mapping.get(r['prediction']['name'],-1) for r in rows}
        ins=complete_metrics(ref['truth'][ref['valid']],prediction[ref['valid']])
        sem=complete_metrics(ref['truth'][ref['semvalid']],prediction[ref['semvalid']],ref['categories'],predclasses)['semantic']
        signatures=geometry_signatures(ref['truth'][ref['valid']],prediction[ref['valid']])
        results[key]=dict(geometry=v['geometry'],category=v['method'],instance=ins,semantic=sem,signatures=signatures,
            regions=len(rows),retained_objects=sum(r['base_retained'] for r in rows),
            observations=sum(len(r['observation_ids']) for r in rows),
            retained_observations=sum(len(r['observation_ids']) for r in rows if r['base_retained']),
            input_observations=cache_manifest['observations'],
            unretained_or_rejected_observations=cache_manifest['observations']-len({i for r in rows if r['base_retained'] for i in r['observation_ids']}),
            effective_views=[r['effective_views'] for r in rows if r['base_retained']],category_s=v['category_s'])
        runtime=read(path.parent/'run.json')
        if v['geometry']!='ovo_3views':base=read(path.parent.parent/'run.json')
        else:base=runtime
        costs[key]=dict(frontend_s=fullfrontend,causal_mapping_s=runtime['causal_mapping_s'],
            offline_finalization_s=runtime['offline_finalization_s'],category_s=v['category_s'],
            total_including_frontend_s=fullfrontend+runtime['causal_mapping_s']+runtime['offline_finalization_s']+v['category_s'],
            cpu_peak_rss_bytes=max(base['cpu_peak_rss_bytes'],runtime['cpu_peak_rss_bytes'],*[x['peak_rss_bytes'] for x in frontend.values()]),
            gpu_peak_reserved_bytes=max(x['gpu_peak_reserved_bytes'] for x in frontend.values()),
            causal_frame_median_s=base['frame_median_s'],causal_frame_p95_s=base['frame_p95_s'],
            backend_cold_start_s=base['cold_start_s'])
    changes={}
    for baseline in ['B2','B3','A3']:
        b={m['gt_id']:m for m in results[baseline]['semantic']['matches']}
        a=results['M']['semantic']['matches'];assert [(x['gt_id'],x['predicted_id'],x['iou']) for x in a]==[(x['gt_id'],x['predicted_id'],x['iou']) for x in b.values()]
        rows=[dict(before=b[x['gt_id']],after=x) for x in a if x['class_correct']!=b[x['gt_id']]['class_correct']]
        changes[baseline]=dict(gains=sum(x['after']['class_correct'] for x in rows),regressions=sum(x['before']['class_correct'] for x in rows),objects=rows)
    diagnostics=graph_diagnostics(front/'cache',RUN/'maps'/scene/'cg',ref['support_truth'])
    dump(out/'graph_diagnostics.json',diagnostics)
    result=dict(complete=True,scene=scene,domain=ref['domain'],methods=results,paired_category_changes=changes,
        frontend_cost=frontend,cost=costs,cache_disk_bytes=sum(p.stat().st_size for p in front.rglob('*') if p.is_file()),
        prediction_sha256=sha(RUN/'predictions'/(scene+'.json')),execution_manifest_sha256=sha(RUN/'execution_manifest.json'))
    dump(out/'metrics.json',result)
    print('SCORED',scene,{k:round(100*v['semantic']['class_aware_pq'],4) for k,v in results.items()},flush=True)


def summarize():
    scenes=scored_scenes()
    rows=[read(RUN/'evaluation'/s/'metrics.json') for s in scenes];assert all(x['complete'] for x in rows)
    rng=np.random.default_rng(0);indices=rng.integers(0,len(rows),(10000,len(rows)));summary={}
    for method in read(FROZEN/'protocol.json')['variants']:
        key=method['id'];values=[r['methods'][key] for r in rows]
        summary[key]=dict(instance_pq100=float(np.mean([v['instance']['observed_voxel_instance_pq'] for v in values])*100),
            category_pq100=float(np.mean([v['semantic']['class_aware_pq'] for v in values])*100),
            macro_category_pq100=float(np.mean([v['semantic']['macro_pq'] for v in values])*100),
            foreground_miou100=float(np.mean([v['semantic']['foreground_miou'] for v in values])*100),
            tp=sum(v['instance']['tp'] for v in values),fp=sum(v['instance']['fp'] for v in values),fn=sum(v['instance']['fn'] for v in values),
            correct_classes=sum(v['semantic']['correct_class_matches'] for v in values),
            category_geometry_matches=sum(v['semantic']['geometry_matches'] for v in values))
    contrasts={}
    for baseline,field,metric in [('B0','instance','observed_voxel_instance_pq'),('B3','semantic','class_aware_pq'),('A3','semantic','class_aware_pq')]:
        delta=np.array([100*(r['methods']['M'][field][metric]-r['methods'][baseline][field][metric]) for r in rows])
        samples=delta[indices].mean(1)
        contrasts[baseline]=dict(per_scene_delta=dict(zip(scenes,delta.tolist())),mean_delta=float(delta.mean()),
            paired_scene_bootstrap_ci95=np.quantile(samples,[.025,.975]).tolist(),wins=int((delta>1e-8).sum()),
            ties=int((abs(delta)<=1e-8).sum()),losses=int((delta< -1e-8).sum()),
            passes_frozen_effect_gate=bool(delta.mean()>0 and (delta>=-1e-8).sum()>=int(np.ceil(2*len(rows)/3))))
    result=dict(complete=True,scenes=scenes,selected_scenes=SCENES,
        unavailable_scenes=unavailable_scenes(),predicted_scenes=run_scenes(),methods=summary,primary_contrasts=contrasts,
        scene_results={r['scene']:sha(RUN/'evaluation'/r['scene']/'metrics.json') for r in rows},
        sufficient_final_scene_coverage=len(rows)>=4,
        association_claim_supported=len(rows)>=4 and contrasts['B0']['passes_frozen_effect_gate'],
        independent_category_claim_supported=len(rows)>=4 and contrasts['B3']['passes_frozen_effect_gate'] and contrasts['A3']['passes_frozen_effect_gate'],
        protocol_sha256=sha(FROZEN/'protocol.json'),no_post_score_tuning=True)
    dump(RUN/'summary.json',result);print('FINAL SUMMARY',summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene');p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.scene:evaluate_scene(a.scene)
    if a.summarize:summarize()
