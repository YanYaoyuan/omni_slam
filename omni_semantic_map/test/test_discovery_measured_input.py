import numpy as np
from omni_semantic_map.discovery_measured_input import robot_measured_visibility
from omni_semantic_map.sparse_lidar_support import MeasuredVoxelMap


def test_unmeasured_and_occluded_history_cannot_become_negative_query_evidence():
    frame=dict(height=3,width=3,intrinsic=[[1,0,1],[0,1,1],[0,0,1]],camera_from_tracking=np.eye(4),camera_pose=np.eye(4))
    raw=np.array([[0,0,1.]])
    history=np.array([[0,0,1.],[0,0,2.],[1,0,1.]])
    ids,pixels,stats=robot_measured_visibility(history,raw,frame)
    np.testing.assert_array_equal(ids,[0]);np.testing.assert_array_equal(pixels,[[1,1]])
    assert stats['unknown_or_depth_rejected']==2


def test_persistent_source_ids_do_not_restart_for_every_robot_scan():
    m=MeasuredVoxelMap(.05);p=np.array([[.01,.01,1.01]])
    m.append(p,np.zeros((1,3),np.uint8),1,np.array([7]),np.array([[1,1]]))
    m.append(p+.001,np.zeros((1,3),np.uint8),2,np.array([19]),np.array([[1,1]]))
    a=m.arrays();assert len(a['xyz'])==1
    np.testing.assert_array_equal(a['xyz'],p);np.testing.assert_array_equal(a['source_frame'],[1])
