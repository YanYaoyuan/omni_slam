"""Development-only GT-assisted reference and optimistic support relaxation.

Neither is a method result. No newly evaluated0004/0005 inputs are read, no
models are fitted and no frozen prediction is modified.
"""
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from paper_final_common import ROOT,read,dump,sha
from run_surface_ownership import OUT as PARENT,PRIOR,INPUT,verify,sealed
from evaluate_kitti360_development import transfer,clean_json
from omni_semantic_map.surface_ownership import append_owned
from omni_semantic_map.support_recovery_bound import optimistic_iou
from omni_semantic_map.kitti360_evaluation import evaluate
from run_discovery_query_benchmark import matched_ids

OUT=ROOT/'build/benchmarks/surface_recovery_headroom_v1'


def main():
    p=verify(True);sealed('features_complete');sealed('training_labels_complete');sealed('predictions_complete');assert not OUT.exists()
    sources=[__file__,ROOT/'omni_semantic_map/omni_semantic_map/support_recovery_bound.py']
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),source_hashes={str(f):sha(f) for f in sources},
        scope='Posthoc diagnostics on exposed0000/0002 only. No new inference/training or new-window access. No result is a competitive benchmark method.',
        feasible_reference='Use existing training-only source/anchor labels to append source points with a known positive candidate owner. If several owner IDs map to the same GT, choose nearest original core, then smaller owner ID. Unknown source/anchor groups excluded. This is a GT-assisted feasible map, not the maximum or a strict AP upper bound.',
        relaxation='For each reference point independently, permit retention by any tied old nearest owner, or allocation to any candidate owner whose new point can beat/tie old nearest distance within20cm. Original wrong points are irreducible only if no other owner can steal them. UpperIoU=possibleGTintersection/(GTmass+irreducibleFalseMass). Allow conflicting source allocations across references; not jointly realizable.',
        matching='Same-class bound edges at each IoU threshold; maximum bipartite matching gives optimistic TP count, not AP. Fixed original object IDs/classes. Confidence ranking/FP penalties and joint feasibility are not solved.',
        tests='12 random small scenes: enumerate all81 physical-source assignments, ensure every actual object/GT IoU<=bound; explicit immutable false-core and other-owner removal cases.'))
    results={};baseline=read(PARENT/'summary.json')
    for s in p['scenes']:
        key=s['key']
        if not s['reference']:continue
        started=time.monotonic();d=np.load(PARENT/'features'/key/'candidates.npz');y=np.load(PARENT/'training_labels'/key/'TRAINING_ONLY.npz');xyz=np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz'];core=np.load(PRIOR/'maps'/key/'largest/geometry.npz');rows=read(PRIOR/'maps'/key/'largest/classification.json')['objects'];ref=np.load(s['reference'])
        valid=d['valid'];positive=y['positive']&valid&y['eligible'][:,None];distance=np.where(positive,d['features'][:,:,0],np.inf);best=distance.argmin(1);owner=np.where(positive.any(1),d['owners'][np.arange(len(best)),best],0)
        assisted=append_owned(core['xyz'],core['instance'],xyz,d['source_ids'],owner);folder=OUT/key;folder.mkdir();np.savez_compressed(folder/'GT_ASSISTED_ONLY.npz',**assisted)
        prediction=transfer(assisted['xyz'],assisted['instance'],ref['xyz'],.2);score=evaluate(ref['train_instance'],ref['confidence'],prediction,{r['object_id']:r['predictions']['mean'] for r in rows},ROOT/'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py');dump(folder/'GT_ASSISTED_matches.json',clean_json(score['matches']))
        base=set(baseline['results'][key]['methods']['baseline']['matched_gt_ids']);hits=matched_ids(score['matches']);oracle=dict(metrics=clean_json(score['averages']),added_sources=int((owner>0).sum()),gained=sorted(hits-base),lost=sorted(base-hits))
        group,slot=np.where(valid);bound=optimistic_iou(core['xyz'],core['instance'],xyz[d['source_ids'][group]],d['owners'][group,slot],ref['xyz'],ref['train_instance'],ref['confidence'])
        np.savez_compressed(folder/'optimistic_iou.npz',object_ids=bound['object_ids'],gt_ids=bound['gt_ids'],iou_upper=bound['iou_upper'])
        classes={r['object_id']:r['predictions']['mean']['train_id'] for r in rows};gt_mass=np.bincount(ref['train_instance'],weights=ref['confidence']);counts={};per_gt=[]
        for category,cid in [('car',13),('building',2)]:
            op=np.flatnonzero([classes[int(o)]==cid for o in bound['object_ids']]);gp=np.flatnonzero((bound['gt_ids']//1000==cid)&(gt_mass[bound['gt_ids']]>=100));matrix=bound['iou_upper'][np.ix_(op,gp)];counts[category]={}
            for threshold in np.arange(.5,.91,.05):
                matching=maximum_bipartite_matching(csr_matrix(matrix>threshold),perm_type='column') if len(op) and len(gp) else np.empty(0,int)
                counts[category][str(round(float(threshold),2))]=int((matching>=0).sum())
            for j,g in enumerate(gp):per_gt.append(dict(gt_id=int(bound['gt_ids'][g]),category=category,best_optimistic_iou=float(matrix[:,j].max(initial=0.))))
        result=dict(baseline_metrics=baseline['results'][key]['methods']['baseline']['metrics'],gt_assisted=oracle,optimistic_tp_counts=counts,per_gt=per_gt,seconds=time.monotonic()-started)
        dump(folder/'result.json',result);results[key]=result;print('HEADROOM',key,'GT-assisted',oracle,'optimistic TP',counts,flush=True)
    dump(OUT/'summary.json',dict(complete=True,diagnostic_only=True,not_a_method=True,results=results,protocol_sha256=sha(OUT/'protocol.json')))


if __name__=='__main__':main()
