import numpy as np
import pytest
from omni_semantic_map.learned_observation_revision import revise,objective,rollback,source_ownership,signed_weights,pair_features


def test_negative_history_links_can_split_while_merge_only_cannot():
    w=np.array([[0,4,-8,-8],[4,0,-8,-8],[-8,-8,0,4],[-8,-8,4,0]],float)
    original=np.ones(4,dtype=int)
    merge=revise(w,original,allow_reassignment=False);new=revise(w,original,allow_reassignment=True)
    assert len(np.unique(merge['labels']))==1
    assert new['labels'][0]==new['labels'][1] and new['labels'][2]==new['labels'][3]
    assert new['labels'][0]!=new['labels'][2]
    assert new['objective_after']>merge['objective_after']
    np.testing.assert_array_equal(rollback(new['labels'],new['events']),original)


def test_unknown_links_never_merge_unobserved_objects():
    original=np.array([1,2,3]);r=revise(np.zeros((3,3)),original,allow_reassignment=True)
    np.testing.assert_array_equal(r['labels'],original);assert not r['events']


def test_positive_merge_and_reassignment_objective_accounting():
    w=np.array([[0,5,2],[5,0,2],[2,2,0]],float);original=np.array([1,2,3])
    r=revise(w,original,allow_reassignment=True);assert len(np.unique(r['labels']))==1
    assert r['objective_after']-r['objective_before']==pytest.approx(sum(e['gain'] for e in r['events']))


def test_source_owner_independent_of_revision_and_raw_ids_break_frame_ties():
    supports=[np.array([0,1]),np.array([1,2]),np.array([2,3])]
    owner=source_ownership(supports,[30,10,20],[0,0,1],5)
    np.testing.assert_array_equal(owner,[0,1,1,2,-1])
    for labels in [np.array([1,1,1]),np.array([1,2,3])]:
        rendered=np.zeros(5,int);active=owner>=0;rendered[active]=labels[owner[active]]
        np.testing.assert_array_equal(rendered>0,active)


def test_pair_geometry_uses_actual_support_and_valid_probabilities():
    xyz=np.array([[0.,0,0],[.1,0,0],[.2,0,0],[4.,0,0]])
    pairs,x=pair_features(xyz,[np.array([0,1]),np.array([1,2]),np.array([3])],np.array([[1.,0],[1.,0],[0,1]]),[0,1,2],[1,2,3])
    np.testing.assert_array_equal(pairs,[[0,1]]);assert x[0,1]==pytest.approx(.5)
    w=signed_weights(3,pairs,[.8],np.array([0,1,2]));assert w[0,1]>0 and w[0,2]==0
    with pytest.raises(ValueError):signed_weights(3,pairs,[1.1],np.array([0,1,2]))
