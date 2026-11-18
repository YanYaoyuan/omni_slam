import numpy as np
from omni_semantic_map.measured_query_cells import query_cell_ids


def test_cell_ids_are_causal_and_geometry_is_unchanged():
    xyz=np.array([[1.01,0,0],[1.08,0,0],[2.,0,0],[-10.,0,0],[1.1,0,0]])
    copy=xyz.copy();full=query_cell_ids(xyz)
    np.testing.assert_array_equal(full,[0,0,1,2,0]);np.testing.assert_array_equal(xyz,copy)
    for n in range(len(xyz)+1):np.testing.assert_array_equal(query_cell_ids(xyz[:n]),full[:n])
