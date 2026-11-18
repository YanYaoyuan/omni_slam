import numpy as np
import pytest
from omni_semantic_map.surface_gap_witness import MeasuredRayTubes, UNKNOWN, SUPPORTED, OCCLUDED, FREE


def test_unknown_occlusion_and_support_never_count_as_free():
    rays=MeasuredRayTubes([0,0,0],[[0,0,5]])
    np.testing.assert_array_equal(rays.classify([[0,0,3],[0,0,5],[0,0,7],[1,0,3],[0,0,0]]),[FREE,SUPPORTED,OCCLUDED,UNKNOWN,UNKNOWN])
    assert not MeasuredRayTubes([0,0,0],np.empty((0,3))).classify([[0,0,3]]).any()


def test_duplicate_collinear_far_return_cannot_carve_through_nearer_one():
    rays=MeasuredRayTubes([0,0,0],[[0,0,10],[0,0,5]])
    assert rays.classify([[0,0,7]])[0]==OCCLUDED
    assert rays.classify([[0,0,5]])[0]==SUPPORTED


def test_gap_requires_both_measured_endpoints_and_free_middle():
    a,b=np.array([[-.2,0,5]]),np.array([[.2,0,5]])
    gap,states=MeasuredRayTubes([0,0,0],[a[0],b[0],[0,0,10]]).gaps(a,b)
    assert gap[0];np.testing.assert_array_equal(states[:,0],[SUPPORTED,FREE,SUPPORTED])
    assert not MeasuredRayTubes([0,0,0],[a[0],b[0]]).gaps(a,b)[0][0]
    assert not MeasuredRayTubes([0,0,0],[a[0],[0,0,10]]).gaps(a,b)[0][0]
    assert not MeasuredRayTubes([0,0,0],[a[0],b[0],[0,0,5]]).gaps(a,b)[0][0]


def test_rigid_coordinates_and_scan_order_do_not_change_evidence():
    points=np.array([[-.2,0,5],[.2,0,5],[0,0,10]])
    q=np.array([[0,0,5],[0,0,12],[0,0,10],[1,0,8]])
    r=np.array([[0,-1,0],[1,0,0],[0,0,1]],float);t=np.array([70.,-12,4])
    a=MeasuredRayTubes([0,0,0],points).classify(q)
    b=MeasuredRayTubes(t,points[::-1]@r.T+t).classify(q@r.T+t)
    np.testing.assert_array_equal(a,b)


def test_overlapping_ray_support_vetoes_false_free_evidence():
    # Two close beams: one really hits the surface, another passes beside it.
    rays=MeasuredRayTubes([0,0,0],[[0,0,5],[.02,0,10]])
    assert rays.classify([[0,0,5]])[0]==SUPPORTED
    with pytest.raises(ValueError):MeasuredRayTubes([0,0,0],[[0,0,np.nan]])
