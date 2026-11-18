import numpy as np
import pytest
from omni_semantic_map.revision_authorization import margin_after_view_deletion,compatible_donors


def test_one_view_cannot_authorize_identity_change():
    assert not margin_after_view_deletion([7],[1.],[0.])['approved']
    assert not margin_after_view_deletion([],[],[])['approved']


def test_single_dominant_view_cannot_hide_conflict():
    r=margin_after_view_deletion([1,2,3],[1.,0.,0.],[0.,.2,.2])
    assert r['margin']>0 and r['worst_after_one_deletion']<0 and not r['approved']


def test_every_actual_deletion_matches_analytic_bound():
    rng=np.random.default_rng(17)
    for n in range(1,25):
        a,b=rng.random((2,n));r=margin_after_view_deletion(np.arange(n),a,b)
        expected=min([sum(a-b)]+[sum(np.delete(a-b,i)) for i in range(n)])
        assert abs(expected-r['worst_after_one_deletion'])<1e-12


def test_duplicate_observation_rejected():
    with pytest.raises(ValueError):margin_after_view_deletion([1,1],[1.,1.],[0.,0.])


def test_transitive_bridge_does_not_merge_conflicting_donors():
    yes=margin_after_view_deletion([1,2],[1.,1.],[0.,0.])
    no=margin_after_view_deletion([1,2],[0.,0.],[1.,1.])
    r=compatible_donors(1,[2,3],{(1,2):yes,(1,3):yes,(2,3):no})
    assert r['accepted']==[2] and r['rejected']==[3]


def test_missing_pair_evidence_is_abstention():
    yes=margin_after_view_deletion([1,2],[1.,1.],[0.,0.])
    r=compatible_donors(1,[2,3],{(1,2):yes})
    assert r['accepted']==[2] and r['rejected']==[3]
