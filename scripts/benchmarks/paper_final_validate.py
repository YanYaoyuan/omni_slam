"""Validate new adapters on synthetic inputs and previously viewed caches only."""
import argparse
from pathlib import Path
import numpy as np
from paper_final_common import ROOT,RUN,OLD,read,dump,sha
from paper_final_mapping import graph_without_veto,categories
from paper_final_metrics import complete_metrics,reference,unique_annotation
from omni_semantic_map.history_graph import build_history_graph


def validate():
    rng=np.random.default_rng(0)
    xyz=np.array([[0.,0.,0.],[0.,0.,0.],[1.,0.,0.],[1.,0.,0.],[2.,0.,0.]])
    labels=np.array([0,14,3,3,7])
    for _ in range(20):
        order=rng.permutation(len(labels));coords,cls,conflicts=unique_annotation(xyz[order],labels[order])
        np.testing.assert_array_equal(coords,xyz[[0,2,4]])
        np.testing.assert_array_equal(cls,[0,3,7]);assert conflicts==1
    rng=np.random.default_rng(0)
    for trial in range(200):
        frames={i:[i] for i in range(12)}
        proposals=[dict(a=int(a),b=int(b),frame=int(t),dino_pass=False) for a,b,t in rng.integers(0,12,(40,3))]
        assert graph_without_veto(frames,proposals)==build_history_graph(frames,proposals,min_votes=2)
    nodes={1:[0],2:[1],3:[0]};edges=[dict(a=a,b=b,frame=f,dino_pass=False) for a,b in [(1,2),(2,3)] for f in [8,9]]
    normal=build_history_graph(nodes,edges,min_votes=2);off=graph_without_veto(nodes,edges)
    assert normal['groups']==[[1,2],[3]] and normal['decisions'][1]['reason']=='coframe_conflict'
    assert off['groups']==[[1,2,3]] and off['decisions'][1]['conflict_frames']==[0]
    one=graph_without_veto(nodes,[edges[0],edges[0]]);assert one['decisions'][0]['reason']=='insufficient_votes'
    truth=np.array([1,1,2,2]);prediction=np.array([10,10,20,20])
    metrics=complete_metrics(truth,prediction,{1:1,2:2},{10:2,20:2})
    assert metrics['instance']['observed_voxel_instance_pq']==1
    assert metrics['semantic']['class_aware_pq']==.5
    assert abs(metrics['semantic']['macro_pq']-1/3)<1e-12
    assert metrics['semantic']['foreground_miou']==.25
    assert complete_metrics(np.array([1,1]),np.array([1,0]))['tp']==0
    assert complete_metrics(np.array([0,0]),np.array([0,0]))['observed_voxel_instance_pq'] is None
    unknown=complete_metrics(truth,prediction,{1:1,2:2},{10:-1,20:2})
    assert unknown['semantic']['class_aware_fp']==1 and unknown['semantic']['unknown_predicted_classes']==1
    checks={'conflicting_coordinate_void_permutation_invariance':20,'random_no_conflict_graphs':200,'transitive_conflict_ablation':True,'distinct_frame_votes':True,
            'hand_calculated_macro_and_pooled_metrics':True,'strict_iou_half':True,'empty_domain':True,'unknown_prediction_counts_wrong':True}
    # New CG adapter preserves the immutable old baseline and history outputs exactly.
    old=OLD/'category_holdout_v1/history/runs/scenenn_025';new=RUN/'validation/025/cg'
    for n,legacy in [('cg','baseline'),('history1','native_bridge_1'),('history2','native_bridge_2')]:
        a=np.load(new/n/'geometry.npz');b=np.load(old/legacy/'geometry.npz')
        for k in a.files:np.testing.assert_array_equal(a[k],b[k])
        assert read(new/n/'classification.json')==read(old/legacy/'classification.json')
        assert read(new/n/'graph.json')==read(old/legacy/'graph.json')
    checks['025_cg_history1_history2_geometry_classes_memberships_graph_exact']=True
    # The OVO adapter changes only final retention (>=3 views), not tracking.
    old=ROOT/'build/benchmarks/shared_frontend_20261110/runs/full/office_0/ovo'
    new=RUN/'validation/office_0/ovo'
    a=np.load(new/'geometry.npz');b=np.load(old/'native_geometry.npz')
    for k in a.files:np.testing.assert_array_equal(a[k],b[k])
    a=read(new/'classification.json')['objects'];b=read(old/'native_classification.json')['objects']
    assert len(a)==len(b)
    for x,y in zip(a,b):
        assert x['object_id']==y['object_id'] and x['observation_ids']==y['observation_ids']
        assert x['predictions']['all_views_mean']==y['predictions']['all_views_mean']
    checks['office0_ovo_native_geometry_membership_mean_exact']=True
    # Reference construction and category metrics remain exactly the historical domain.
    ref=reference('025',OLD/'category_holdout_v1');b=np.load(OLD/'category_holdout_v1/evaluation/025/reference.npz')
    for source,key in [('xyz','xyz'),('truth','instance'),('valid','valid')]:np.testing.assert_array_equal(ref[source],b[key])
    pred=np.load(OLD/'category_holdout_v1/evaluation/025/fixed_prediction.npz')['instance']
    rows=read(OLD/'category_holdout_v1/predictions/025.json')['objects'];mapping=read(OLD/'category_holdout_v1/selection.json')['nyu_prediction_mapping']
    previous=read(OLD/'category_holdout_v1/evaluation/025/metrics.json')
    for method in ['mean','area','consensus_sqrt_area']:
        pc={r['object_id']:mapping.get(r['methods'][method]['name'],-1) for r in rows}
        result=complete_metrics(ref['truth'][ref['semvalid']],pred[ref['semvalid']],ref['categories'],pc)['semantic']
        oldresult=previous['methods'][method]['fixed']
        for k,v in oldresult.items():assert result[k]==v,(method,k)
    checks['025_reference_and_all_prior_category_metrics_exact']=True
    # Directly compare current selected readout to the previously saved predictions.
    from omni_semantic_map.shared_observations import observation_table
    from omni_semantic_map.consensus_category import classify
    cache=OLD/'category_holdout_v1/frontend/025/cache';f,frames,areas=observation_table(cache)
    text=np.load(cache/'text_features.npy');text/=np.linalg.norm(text,axis=1,keepdims=True);vocab=read(cache/'vocabulary.json')
    cls=read(RUN/'validation/025/cg/history2/classification.json')['objects']
    for obj,prior in zip(cls,rows):
        assert obj['object_id']==prior['object_id']
        now=classify(obj['observation_ids'],f,frames,areas,text,vocab)
        for k,v in now.items():assert prior['methods']['consensus_sqrt_area'][k]==v,k
    checks['025_main_category_predictions_exact']=True
    dump(RUN/'validation/acceptance.json',dict(complete=True,checks=checks,no_final_test_values_read=True,
        source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('paper_final_*.py')}))
    print('VALIDATION PASSED',checks,flush=True)


if __name__=='__main__':
    validate()
