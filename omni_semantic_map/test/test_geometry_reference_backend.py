import numpy as np
from geometry_reference_backend import frame_components
from omni_semantic_map.sparse_support_components import largest_measured_component


def test_shared_graph_components_equal_original_per_query_including_ties():
    rng=np.random.RandomState(24);xyz=np.r_[rng.normal(0,.07,(24,3)),rng.normal([2,0,0],.07,(24,3)),rng.uniform(-4,4,(20,3))]
    masks=np.r_[np.ones((1,len(xyz)),bool),rng.uniform(size=(20,len(xyz)))>.3]
    masks=np.r_[masks,masks[[0]],np.zeros((1,len(xyz)),bool)]
    values=frame_components(xyz,masks)
    for mask,actual in zip(masks,values):
        ix=np.flatnonzero(mask);expected=ix[largest_measured_component(xyz[ix],.4,16)]
        np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(values[0],np.arange(24))
