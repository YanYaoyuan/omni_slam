"""Recover JSON serialization only; consume sealed maps and cached predictions."""
import numpy as np
from review_maskclustering_control import OUT,SRC,ROOT,SCENES,verify,read,dump,sha
from evaluate_kitti360_development import clean_json
from omni_semantic_map.kitti360_evaluation import evaluate
from run_discovery_query_benchmark import matched_ids
verify();seal=read(OUT/'maps_complete.json');assert not (OUT/'summary.json').exists()
for f,h in seal['files'].items():assert sha(f)==h
cached={str(OUT/'evaluation'/k/'reference_prediction.npz'):sha(OUT/'evaluation'/k/'reference_prediction.npz') for k in SCENES}
dump(OUT/'score_recovery.json',dict(reason='Original score() completed all six evaluations but JSON export failed because matched_ids returns a set. Original frozen script/maps unchanged. Reuse all cached predictions and sort matched IDs for JSON.',original_script_sha256=sha(ROOT/'scripts/benchmarks/review_maskclustering_control.py'),recovery_script_sha256=sha(__file__),cached_predictions=cached,unique_logical_scores=6,total_kernel_calls_including_failed_serialization_attempt=12,scientific_parameters_changed=False))
results={}
for k in SCENES:
    ref=np.load(SRC/'evaluation'/k/'reference.npz');pred=np.load(OUT/'evaluation'/k/'reference_prediction.npz')['instance'];info={r['object_id']:r['predictions']['mean'] for r in read(OUT/'maps'/k/'classification.json')['objects']};row={}
    for mode,gt,uniform in [('primary',ref['train_instance'],False),('raw_targets',ref['raw_target_instance'],False),('uniform_scores',ref['train_instance'],True)]:
        cls={o:dict(v,score=1.) for o,v in info.items()} if uniform else info
        z=evaluate(gt,ref['confidence'],pred,cls,ROOT/'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py');row[mode]=dict(metrics=clean_json(z['averages']),matched_gt_ids=sorted(matched_ids(z['matches'])))
    results[k]=row;print(k,{m:v['metrics']['allAp'] for m,v in row.items()},flush=True)
for f,h in cached.items():assert sha(f)==h
means={m:float(np.mean([x[m]['metrics']['allAp'] for x in results.values()])) for m in ['primary','raw_targets','uniform_scores']}
dump(OUT/'summary.json',dict(complete=True,results=results,mean_ap=means,new_maps=2,new_scores=6,native_full_pipeline=False,independent_scenes=False,protocol_sha256=sha(OUT/'protocol.json'),recovery_sha256=sha(OUT/'score_recovery.json')))
