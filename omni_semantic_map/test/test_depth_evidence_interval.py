import numpy as np
import pytest
from omni_semantic_map.depth_evidence_interval import fit_intervals,intervals,candidate_depth_agreement


def test_precise_consistent_shape_and_wrong_surface_rejection():
    p=np.full(300,10.);d=np.arange(300)%30;z=p+np.linspace(-.02,.02,300)
    m=fit_intervals(p,z,d);lo,hi,ok=intervals(p,d,m)
    assert ok.all() and candidate_depth_agreement(p,lo,hi,ok).all()
    assert not candidate_depth_agreement(p+1,lo,hi,ok).any()


def test_broad_uncertainty_does_not_authorize_observation():
    p=np.full(300,10.);d=np.ones(300);z=np.linspace(7,13,300)
    lo,hi,ok=intervals(p,d,fit_intervals(p,z,d))
    assert np.isfinite(lo).all() and not ok.any()
    assert not candidate_depth_agreement(p,lo,hi,ok).any()


def test_insufficient_calibration_is_unknown():
    p=np.ones(20);m=fit_intervals(p,p,np.ones(20))
    lo,hi,ok=intervals(p,np.ones(20),m)
    assert not np.isfinite(lo).any() and not ok.any()


def test_fixed_calibration_and_order_equivariance():
    p=np.linspace(5,20,200);z=p+.02;d=np.arange(200)%40
    m=fit_intervals(p,z,d);r=np.random.default_rng(0).permutation(200)
    assert fit_intervals(p[r],z[r],d[r])==m
    a=intervals(p,d,m);b=intervals(p[r],d[r],m)
    for x,y in zip(a,b):np.testing.assert_array_equal(x[r],y)


def test_invalid_calibration_rejected():
    with pytest.raises(ValueError):fit_intervals([0],[1],[0])
