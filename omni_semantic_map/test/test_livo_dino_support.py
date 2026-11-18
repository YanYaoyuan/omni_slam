import numpy as np
from omni_semantic_map.livo_dino_support import pool


def test_non_square_image_and_invalid_support():
    # Two tokens across a 4x2 image; bottom-right point must use right token.
    tokens=np.array([[[1.,0.],[0.,1.]]]);K=np.eye(3)
    points=np.array([[3.9,1.9,1.],[-.1,0,1],[4.,0,1],[0,0,-1],[np.nan,0,1]])
    desc,anchors,features=pool(tokens,points,np.eye(4),K,4,2)
    np.testing.assert_array_equal(desc,[0,1]);np.testing.assert_allclose(anchors,points[:1])
    np.testing.assert_array_equal(features,[[0,1]])


def test_image_epoch_projection_and_world_anchors():
    tokens=np.array([[[1.,0.],[0.,1.]]]);T=np.eye(4);T[0,3]=-2
    points=np.array([[2.,0,1],[2.1,0,1],[3.,0,1]])
    desc,anchors,features=pool(tokens,points,T,np.eye(3),2,1)
    np.testing.assert_allclose(desc,np.array([2.,1.])/np.sqrt(5),atol=1e-7)
    np.testing.assert_allclose(anchors,[[2.05,0,1],[3.,0,1]],atol=1e-6)
    assert features.shape==(2,2)


def test_empty_support_is_not_a_valid_descriptor():
    d,a,f=pool(np.ones((1,2,3)),np.empty((0,3)),np.eye(4),np.eye(3),2,1)
    assert np.count_nonzero(d)==0 and a.shape==(0,3) and f.shape==(0,3)
