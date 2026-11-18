import numpy as np
from omni_semantic_map.sparse_support_components import largest_measured_component


def test_sparse_surface_separated_from_background():
    surface=np.array([[x*.2,y*.2,1.] for x in range(5) for y in range(4)])
    background=surface[:8]+[0,0,4]
    p=np.concatenate([surface,background,[[30.,0,0]]])
    ids=largest_measured_component(p,.4,16)
    np.testing.assert_array_equal(ids,np.arange(20))
    # Rigid pose changes cannot change distance connectivity.
    R=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    np.testing.assert_array_equal(largest_measured_component(p@R+[20,-3,8],.4,16),ids)


def test_no_dense_cluster_does_not_return_all_noise():
    p=np.array([[i,0.,0.] for i in range(20)])
    assert len(largest_measured_component(p,.4,16))==0
    assert len(largest_measured_component(np.empty((0,3)),.4,16))==0
