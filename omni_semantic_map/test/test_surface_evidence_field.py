import numpy as np
import pytest
from omni_semantic_map.surface_evidence_field import aggregate_views,solve_field,surface_graph


def test_query_duplicates_do_not_inflate_view_evidence_and_missing_is_unknown():
    args=dict(point_ids=np.array([0,0,0]),frame_ids=np.array([4,4,4]),object_ids=np.array([1,1,2]),support=np.array([.8,.7,.2]),visible_pairs=np.array([[0,4],[0,4],[1,5]]),num_points=3,num_objects=2)
    u,c=aggregate_views(**args)
    args.update(point_ids=np.array([0,0,0,0]),frame_ids=np.array([4,4,4,4]),object_ids=np.array([1,1,2,1]),support=np.array([.8,.7,.2,.8]))
    uu,cc=aggregate_views(**args);np.testing.assert_array_equal(u,uu);np.testing.assert_array_equal(c,cc)
    np.testing.assert_array_equal(c,[1,1,0]);np.testing.assert_allclose(u[0],[1/6,2/3,1/6])
    np.testing.assert_array_equal(u[1:],[ [1,0,0],[1,0,0] ])


def test_support_requires_measured_visibility():
    with pytest.raises(ValueError,match='without measured'):
        aggregate_views(np.array([0]),np.array([5]),np.array([1]),[.9],np.array([[0,4]]),1,1)


def test_joint_solution_changes_unsupported_middle_but_not_disconnected_unknown():
    unary=np.array([[0,1,0],[1,0,0],[0,0,1],[1,0,0]],float);reliability=np.array([1,0,1,0])
    edges=np.array([[0,1],[1,2]]);weights=np.ones(2)
    out,info=solve_field(unary,reliability,edges,weights)
    np.testing.assert_allclose(out[1],[0,.5,.5],atol=1e-8);np.testing.assert_array_equal(out[3],[1,0,0])
    np.testing.assert_allclose(out[0],[0,.75,.25],atol=1e-8)
    assert info['active_points']==3 and info['residual']<1e-8
    # This deliberately changes the old unary too: it is a coupled field, not
    # an append-only classifier. Export must control existing-map mutations.
    assert not np.array_equal(out[0],unary[0])


def test_zero_coupling_is_exact_independent_control_and_duplicates_are_invariant():
    u=np.array([[.2,.8],[.7,.3],[1,0]],float);r=np.array([2,1,0]);e=np.array([[0,1],[1,2]])
    independent,_=solve_field(u,r,e,np.ones(2),strength=0)
    np.testing.assert_allclose(independent,u)
    a,_=solve_field(u,r,e,np.ones(2));b,_=solve_field(u,r,np.r_[e,e[:,::-1]],np.ones(4))
    np.testing.assert_allclose(a,b)


def test_point_and_object_permutations_are_equivariant():
    u=np.array([[.1,.8,.1],[.5,.2,.3],[.2,.2,.6]]);r=np.array([1,2,1]);e=np.array([[0,1],[1,2]]);w=np.array([.3,.7])
    base,_=solve_field(u,r,e,w);order=np.array([2,0,1]);inverse=np.argsort(order);channels=[0,2,1]
    alternate,_=solve_field(u[order][:,channels],r[order],inverse[e],w)
    np.testing.assert_allclose(alternate,base[order][:,channels],atol=1e-9)


def test_no_observations_abstains_and_negative_weights_rejected():
    u=np.array([[0,1],[0,1]],float);e=np.array([[0,1]])
    p,info=solve_field(u,np.zeros(2),e,[1]);np.testing.assert_array_equal(p,[[1,0],[1,0]])
    assert info['active_points']==0
    with pytest.raises(ValueError):solve_field(u,np.ones(2),e,[-1])


def test_physical_graph_rigid_invariance_and_surface_weights_are_bounded():
    rng=np.random.RandomState(13);points=rng.uniform(-.3,.3,(80,3));points[:40,2]*=.05
    rotation=np.array([[0,-1,0],[0,0,1],[-1,0,0]],float)
    edges,euclidean,surface,info=surface_graph(points)
    other,ew,sw,_=surface_graph(points@rotation.T+[12,-7,19])
    np.testing.assert_array_equal(edges,other);np.testing.assert_allclose(euclidean,ew,atol=1e-12);np.testing.assert_allclose(surface,sw,atol=1e-12)
    assert np.all(surface>=0) and np.all(surface<=euclidean+1e-12)
    assert (np.abs(surface-euclidean)>1e-6).any()


def test_graph_does_not_bridge_unmeasured_distance_gap():
    xyz=np.array([[0,0,0],[.1,0,0],[3,0,0],[3.1,0,0]],float)
    edges,_,_,_=surface_graph(xyz)
    np.testing.assert_array_equal(edges,[[0,1],[2,3]])
