"""Frozen-input semantic readout experiment, with prediction before evaluation."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

from audit_evidence_retention import ROOT, BASE, CACHE, OUT, REPORT, KEY, dump
from omni_semantic_map.flat_benchmark import sha256
from omni_semantic_map.shared_observations import select_views
from omni_semantic_map.evidence_readout import aggregate, METHODS, ANGULAR_BANDWIDTH_DEG
from omni_semantic_map.full_ovo_evaluation import BACKGROUND, voxel_metrics
from omni_semantic_map.mesh_instance_evaluation import transfer_labels
from omni_semantic_map.instance_error_audit import audit_instances


def freeze():
    target = OUT/'stage2_protocol.json'
    assert not target.exists(), 'Never silently replace a frozen experiment'
    source = ROOT/'omni_semantic_map/omni_semantic_map/evidence_readout.py'
    original = json.loads((BASE/'protocol_backends.json').read_text())
    dump(target, dict(stage=2, scenes=original['scenes'], methods=METHODS,
        method_under_test='area_density', angular_bandwidth_deg=ANGULAR_BANDWIDTH_DEG,
        rule='w_i = area_i / sum_j area_j exp(-(1-dot(view_i,view_j))/(1-cos(30deg)))',
        rationale='Give correlated camera-direction groups comparable influence; area weights within groups. Fixed 30-degree scale, no sweep.',
        controls=dict(mean='existing unique-frame mean', mean_min3='same mean, >=3 distinct frames',
            area='raw-mask area weighted mean', view_fps10='largest-area seed, up to 10 farthest angular views',
            density='inverse unweighted angular density; area-quality ablation'),
        selection='one largest raw mask per object per distinct frame, fixed observation ID tie',
        input='frozen guarded CG geometry/memberships, cached raw SigLIP descriptors and camera poses',
        classification='unchanged vocabulary, text features and background list',
        evaluation=original['evaluation'], baseline_protocol_sha256=sha256(BASE/'protocol_backends.json'),
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [source,Path(__file__)]},
        limitations=['All eight scenes already viewed. This is development, not held-out testing.',
                    'Only semantic aggregation/filtering changes; no geometric completion or identity association claim.',
                    '25 audited lamp losses have all-background per-view predictions; convex weighting may be insufficient.'],
        primary_comparison='area_density versus mean AND area AND view_fps10; report every scene and negative outcome',
        no_training=True, no_sam3=True, no_dino=True))
    print('FROZEN',target,flush=True)


def run(scene, protocol):
    out = OUT/'readout_runs'/scene
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    cache=CACHE/scene
    meta_path=OUT/'observations'/scene/'metadata.npz'
    prov=json.loads(meta_path.with_name('provenance.json').read_text())
    assert prov['no_gt'] and sha256(meta_path)==prov['metadata_sha256']
    a=np.load(meta_path)
    rows=json.loads((BASE/'runs/full'/scene/'cg/guarded_classification.json').read_text())['objects']
    native=np.load(BASE/'runs/full'/scene/'cg/guarded_geometry.npz')
    pts,ids=native['xyz'],native['instance']
    text=np.load(cache/'text_features.npy');text=text/np.linalg.norm(text,axis=1,keepdims=True)
    vocab=json.loads((cache/'vocabulary.json').read_text());class_ids=np.array(vocab['replica_class_ids'])
    predictions={m:[] for m in METHODS}
    for row in rows:
        oid=row['object_id']; selected=np.array(select_views(row['observation_ids'],a['frames'],a['areas'],top_k=0))
        center=pts[ids==oid].mean(0)
        for m in METHODS:
            r=aggregate(a['features'][selected],a['areas'][selected],a['camera_positions'][a['pose_indices'][selected]],center,
                        method=m,observation_ids=selected)
            scores=text@r['descriptor'];order=np.argsort(-scores,kind='stable');best=order[0]; cls=int(class_ids[best])
            predictions[m].append(dict(object_id=oid,replica_class_id=cls,name=vocab['names'][best],
                score=float(scores[best]),margin=float(scores[best]-scores[order[1]]),effective_views=r['effective_views'],
                retained=bool(r['retained_by_count'] and cls not in BACKGROUND),
                selected_observation_ids=selected.tolist(),weights=r['weights'].tolist()))
            if m=='mean': assert cls==row['predictions']['all_views_mean']['replica_class_id'], (scene,oid)
    dump(out/'predictions.json',dict(no_gt=True,methods=predictions))
    prediction_hash=sha256(out/'predictions.json')
    predict_wall=time.monotonic()-started
    # Reference labels are not loaded until all predictions are finalized.
    reference_path=ROOT/'build/benchmarks/expanded_semantic_20261109/conceptgraphs_evaluation/evaluation'/scene/'reference.npz'
    original=json.loads((BASE/'protocol_backends.json').read_text())
    assert sha256(reference_path)==original['reference_sha256'][scene]
    ref=np.load(reference_path);vertices,truth=ref['xyz'],ref['instance']
    baseline=np.load(BASE/'evaluation/full'/scene/(KEY+'.npz'))['instance']
    metrics={}
    for method, classified in predictions.items():
        keep=[r['object_id'] for r in classified if r['retained']]
        labels=np.where(np.isin(ids,keep),ids,0)
        projected,_=transfer_labels(pts,labels,vertices,mode='nearest_5cm')
        if method=='mean':np.testing.assert_array_equal(projected,baseline)
        score=voxel_metrics(truth,projected);audit=audit_instances(truth,projected)
        score.update(predicted_error_counts=audit['predicted_counts'],gt_error_counts=audit['gt_counts'])
        metrics[method]=score
        np.savez_compressed(out/(method+'.npz'),instance=projected)
        dump(out/(method+'_audit.json'),audit)
    result=dict(scene=scene,complete=True,methods=metrics,prediction_sha256=prediction_hash,
        protocol_sha256=sha256(OUT/'stage2_protocol.json'),baseline_labels_exactly_reproduced=True,
        prediction_wall_s=predict_wall,total_wall_s=time.monotonic()-started)
    dump(out/'metrics.json',result)
    print('EVALUATED',scene,{k:round(v['observed_voxel_instance_pq']*100,3) for k,v in metrics.items()},flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    if not a.run:return
    protocol=json.loads((OUT/'stage2_protocol.json').read_text())
    for source,h in protocol['source_sha256'].items():assert sha256(ROOT/source)==h,source
    assert protocol['angular_bandwidth_deg']==ANGULAR_BANDWIDTH_DEG
    results=[run(s,protocol) for s in protocol['scenes']]
    summary=dict(complete=True,scenes=protocol['scenes'],methods={},baseline_reproduced=True)
    for method in METHODS:
        scores=[r['methods'][method] for r in results];pq=[r['observed_voxel_instance_pq'] for r in scores]
        base=[r['methods']['mean']['observed_voxel_instance_pq'] for r in results];delta=np.array(pq)-base
        summary['methods'][method]=dict(macro_pq=float(np.mean(pq)),pq_by_scene=pq,
            delta_vs_mean=float(np.mean(delta)),wins=int((delta>1e-8).sum()),ties=int((abs(delta)<=1e-8).sum()),losses=int((delta < -1e-8).sum()),
            totals={key:sum(r[key] for r in scores) for key in ['tp','fp','fn']})
    dump(REPORT/'stage2/summary.json',summary)
    dump(REPORT/'stage2/protocol.json',protocol)
    print(json.dumps(summary['methods'],indent=2),flush=True)


if __name__=='__main__':main()
