import itertools
import numpy as np
import torch
import pytest
from omni_semantic_map.spatial_acceptance_readout import build_readout,hard_readout,first_two_readouts,expected_labels,slice_readout


def brute_transfer(base,labels,candidate,owners,query,radius):
    keep=owners>0;xyz=np.concatenate([base,candidate[keep]]);ids=np.r_[labels,owners[keep]];out=[]
    for q in query:
        if not len(xyz):out.append(0);continue
        d=np.linalg.norm(xyz-q,axis=1);i=d.argmin();out.append(ids[i] if d[i]<radius else 0)
    return np.asarray(out)


def test_hard_matches_every_subset_with_ties_and_missing_geometry():
    base=np.array([[0.,0,0],[2.,0,0]]);labels=np.array([1,2]);candidate=np.array([[1.,0,0],[1.,0,0],[3.,0,0]])
    query=np.array([[.5,0,0],[1.,0,0],[1.5,0,0],[3.,0,0],[5.,0,0]])
    a=build_readout(base,labels,candidate,query,1.)
    for decision in itertools.product([0,1],repeat=3):
        owners=np.array(decision)*np.array([3,4,5])
        np.testing.assert_array_equal(hard_readout(a,owners),brute_transfer(base,labels,candidate,owners,query,1.))


def test_expected_labels_match_exhaustive_categorical_sampling():
    base=np.array([[0.,0,0]]);candidate=np.array([[.3,0,0],[.5,0,0]]);query=np.array([[.4,0,0],[.6,0,0],[2.,0,0]])
    a=build_readout(base,[1],candidate,query,.8);ids=np.array([[2,3],[2,3]])
    p=np.array([[.2,.3],[.1,.4]]);want=np.zeros((len(query),4))
    for states in itertools.product(range(3),repeat=2):
        prob=1.;owner=[]
        for i,state in enumerate(states):
            prob*=1-p[i].sum() if state==0 else p[i,state-1];owner.append(0 if state==0 else ids[i,state-1])
        labels=hard_readout(a,np.array(owner));want[np.arange(len(query)),labels]+=prob
    actual=expected_labels(a,torch.tensor(p),ids).detach().numpy()
    np.testing.assert_allclose(actual,want,atol=1e-14);np.testing.assert_allclose(actual.sum(1),1)


def test_probability_endpoints_and_single_deletion_fallback():
    a=build_readout([[0.,0,0]],[1],[[.3,0,0],[.5,0,0]],[[.4,0,0],[.6,0,0]],.8)
    owners=np.array([2,3]);y,winner,backup=first_two_readouts(a,owners)
    for i in range(2):
        removed=owners.copy();removed[i]=0;expected=y.copy();expected[winner==i]=backup[winner==i]
        np.testing.assert_array_equal(hard_readout(a,removed),expected)
    for p in [np.zeros((2,1)),np.ones((2,1))]:
        marginal=expected_labels(a,torch.tensor(p),owners[:,None]).detach().numpy()
        np.testing.assert_array_equal(marginal.argmax(1),hard_readout(a,owners*p[:,0].astype(int)))
        np.testing.assert_array_equal(marginal.max(1),1)


def test_soft_gradients_and_chunking_are_exact():
    a=build_readout([[0.,0,0]],[1],[[.3,0,0],[.5,0,0]],[[.4,0,0],[.6,0,0],[2.,0,0]],.8)
    ids=np.array([[2,3],[2,3]]);p=torch.tensor([[.2,.3],[.1,.4]],dtype=torch.double,requires_grad=True)
    assert torch.autograd.gradcheck(lambda q:expected_labels(a,q,ids),(p,),eps=1e-6,atol=1e-6)
    whole=expected_labels(a,p,ids)
    chunks=torch.cat([expected_labels(slice_readout(a,0,1),p,ids),expected_labels(slice_readout(a,1,3),p,ids)])
    torch.testing.assert_close(whole,chunks)


def test_empty_map_and_rejected_candidate_are_not_null_geometry_anchors():
    a=build_readout(np.empty((0,3)),np.empty(0,int),[[0.,0,0]],[[0.,0,0]],.2)
    np.testing.assert_array_equal(hard_readout(a,[0]),[0]);np.testing.assert_array_equal(hard_readout(a,[7]),[7])
    a=build_readout([[.1,0,0]],[7],[[0.,0,0]],[[0.,0,0]],.2)
    np.testing.assert_array_equal(hard_readout(a,[0]),[7])


def test_invalid_categorical_mass_and_duplicate_owner_rejected():
    a=build_readout([[0.,0,0]],[1],[[.3,0,0]],[[.4,0,0]],.8)
    with pytest.raises(ValueError):expected_labels(a,torch.tensor([[.8,.3]]),np.array([[1,2]]))
    with pytest.raises(ValueError):expected_labels(a,torch.tensor([[.2,.3]]),np.array([[1,1]]))
