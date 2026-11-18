import numpy as np
from omni_semantic_map.learned_observation_revision import revise,rollback
from omni_semantic_map.operation_revision import gated_revision,OP_FEATURES
from omni_semantic_map.operation_teacher import OperationTeacher
from omni_semantic_map.full_ovo_evaluation import voxel_metrics


def test_ungated_matches_previous_optimizer_and_recording_cannot_change_partition():
    rng=np.random.default_rng(31);w=rng.normal(size=(12,12));w=(w+w.T)/2;np.fill_diagonal(w,0)
    initial=np.repeat([1,2,3],4);features=rng.normal(size=(12,6));frames=np.arange(12);mass=np.arange(12)+1
    records=[];r=gated_revision(w,initial,features,frames,mass,recorder=lambda *a:records.append(a[3].copy()))
    old=revise(w,initial,allow_reassignment=True)
    np.testing.assert_array_equal(r['labels'],old['labels']);assert r['events']==old['events']
    assert all(v.shape==(len(OP_FEATURES),) for v in records)


def test_reject_all_keeps_native_and_approved_moves_are_reversible():
    w=np.array([[0,4,-8,-8],[4,0,-8,-8],[-8,-8,0,4],[-8,-8,4,0]],float)
    initial=np.ones(4,dtype=int);f=np.eye(4);frames=np.arange(4);mass=np.ones(4)
    r=gated_revision(w,initial,f,frames,mass,gate=lambda x:dict(accept=False))
    np.testing.assert_array_equal(r['labels'],initial);assert not r['events']
    r=gated_revision(w,initial,f,frames,mass,gate=lambda x:dict(accept=True))
    np.testing.assert_array_equal(rollback(r['labels'],r['events']),initial)


def test_counterfactual_rewards_match_full_domain_including_background_and_uncovered_gt():
    # Atoms own positions 0..7; final two GT points are outside source coverage.
    truth=np.array([1,1,1,2,2,2,0,0,2,3]);owners=np.array([0,0,1,2,2,3,1,3,-1,-1])
    counts=np.zeros((4,4),int)
    for g,o in zip(truth,owners):
        if o>=0:counts[o,g]+=1
    teacher=OperationTeacher(counts,np.bincount(truth,minlength=4));initial=np.array([1,1,2,2])
    def full(labels):
        prediction=np.zeros(len(truth),int);known=owners>=0;prediction[known]=labels[owners[known]]
        return voxel_metrics(truth,prediction)
    base=full(initial);fast=teacher.metrics(initial)
    assert fast['pq']==base['observed_voxel_instance_pq'] and fast['tp']==base['tp']
    for indices,target in [([0],2),([0,1],2),([1],3),([2,3],1)]:
        reward=teacher.reward(initial,np.array(indices),target);after=initial.copy();after[indices]=target;m=full(after)
        np.testing.assert_allclose(reward['delta_pq'],m['observed_voxel_instance_pq']-base['observed_voxel_instance_pq'])
        assert reward['delta_tp']==m['tp']-base['tp']


def test_hidden_empty_atom_does_not_create_an_evaluation_false_positive():
    t=OperationTeacher(np.array([[0,5],[0,0]]),np.array([0,5]));labels=np.array([1,1])
    r=t.reward(labels,[1],2);assert r['delta_pq']==0 and r['delta_predicted']==0
