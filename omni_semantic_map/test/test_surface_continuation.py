import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation
from omni_semantic_map.surface_continuation import pure_anchor_labels,geometry_features,affinity_length


def test_unknown_and_conflicted_anchors_do_not_become_background():
    np.testing.assert_array_equal(pure_anchor_labels(5,[0,0,1,2,2],[0,0,8,3,4]),[0,8,-1,-1,-1])
    np.testing.assert_array_equal(pure_anchor_labels(3,[],[]),[-1,-1,-1])


def test_features_are_rigid_invariant_and_ignore_edge_order():
    rng=np.random.RandomState(7);xyz=rng.normal(size=(100,3));pairs=cKDTree(xyz).query_pairs(1.2,output_type='ndarray');a,b=pairs.T
    graph=csr_matrix((np.ones(len(a)),(a,b)),shape=(len(xyz),len(xyz)))
    first=geometry_features(xyz,graph);rot=Rotation.from_euler('xyz',[27,83,19],degrees=True).as_matrix()
    second=geometry_features(xyz@rot.T+[100,-250,350],graph.T)
    for name in ['left','right','degree','normal_valid']:np.testing.assert_array_equal(first[name],second[name])
    np.testing.assert_allclose(first['features'],second['features'],atol=2e-5,rtol=2e-5)


def test_isolated_and_linear_neighborhoods_are_not_valid_normals():
    xyz=np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0],[10,10,10]],float)
    a,b=np.triu_indices(4,1);graph=csr_matrix((np.ones(len(a)),(a,b)),shape=(5,5))
    data=geometry_features(xyz,graph)
    assert not data['normal_valid'].any() and np.isfinite(data['features']).all()
    np.testing.assert_array_equal(data['features'][:,1:4],0)


def test_affinity_changes_cost_continuously_without_cutting_edges():
    d=np.array([.1,.1,.1]);cost=affinity_length(d,[1,.5,0])
    assert cost[0]==d[0] and cost[0]<cost[1]<cost[2] and np.isfinite(cost).all()
    with pytest.raises(ValueError):affinity_length([.1],[1.01])
