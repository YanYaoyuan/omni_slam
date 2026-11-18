import numpy as np
import pytest
from omni_semantic_map.sensor_features import camera_pose_from_anchor,pool_measured_tokens
from omni_semantic_map.objects import transform_points
from omni_semantic_map.vbot_pose_revision import pair_partition_change


def test_camera_anchor_chain_preserves_world_points_after_realistic_pose_update():
    anchor=np.eye(4);anchor[:3,3]=[2,3,1]
    camera=np.array([[0,-1,0,.1],[0,0,-1,.2],[1,0,0,.3],[0,0,0,1.]])
    points=np.array([[1,2,3.],[2,1,4.]])
    camera_points=transform_points(points,camera)
    np.testing.assert_allclose(transform_points(camera_points,camera_pose_from_anchor(anchor,camera)),transform_points(points,anchor))
    anchor[0,3]+=1.7
    np.testing.assert_allclose(transform_points(camera_points,camera_pose_from_anchor(anchor,camera)),transform_points(points,anchor))


def test_pooling_uses_actual_image_height_not_flat_height():
    tokens=np.array([[[1.,0.]],[[0.,1.]]]);pixels=np.array([[10,10],[10,350]])
    mean,xyz,features=pool_measured_tokens(tokens,pixels,np.array([[0,0,1],[0,1,1]]),(360,640,3))
    np.testing.assert_allclose(features,np.eye(2));np.testing.assert_allclose(mean,[2**-.5,2**-.5],atol=1e-6)
    with pytest.raises(ValueError):pool_measured_tokens(tokens,np.array([[640,0]]),np.array([[0,0,1]]),(360,640))


def test_partition_change_ignores_numeric_id_renaming():
    assert pair_partition_change([1,1,2],[7,7,9])==dict(separated_observation_pairs=0,joined_observation_pairs=0)
    assert pair_partition_change([1,1,2],[1,2,2])==dict(separated_observation_pairs=1,joined_observation_pairs=1)
