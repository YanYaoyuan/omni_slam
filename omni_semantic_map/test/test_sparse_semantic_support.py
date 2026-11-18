import numpy as np
from omni_semantic_map.sparse_semantic_support import sparse_support


def test_image_epoch_transform_and_occlusion():
    # Image camera has moved one metre relative to the tracking cloud epoch.
    T=np.eye(4);T[0,3]=-1
    p=np.array([[1,0,1],[1,0,1.03],[1,0,2],[0,0,1],[1,0,-1],[np.nan,0,1]])
    r=sparse_support(p,T,np.eye(3),np.array([[7,8],[9,-1]]))
    np.testing.assert_array_equal(r['point_indices'],[0,1])
    np.testing.assert_array_equal(r['mask_ids'],[7,7])
    assert r['rejected_occluded']==1


def test_empty_and_unassigned_support():
    r=sparse_support(np.empty((0,3)),np.eye(4),np.eye(3),np.array([[-1]]))
    assert not len(r['point_indices'])
    r=sparse_support([[0,0,1]],np.eye(4),np.eye(3),np.array([[-1]]))
    assert len(r['visible_indices'])==1 and len(r['point_indices'])==0


def test_bounds_before_integer_conversion_and_exclusive_ownership():
    p=np.array([[-.1,0,1],[0,0,1],[.9,0,1],[1,0,1],[2,0,1]])
    r=sparse_support(p,np.eye(4),np.eye(3),np.array([[0,1]]))
    np.testing.assert_array_equal(r['point_indices'],[1,2,3])
    np.testing.assert_array_equal(r['mask_ids'],[0,0,1])
    assert len(np.unique(r['point_indices']))==len(r['point_indices'])
