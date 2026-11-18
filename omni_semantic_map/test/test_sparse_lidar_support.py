from types import SimpleNamespace
import numpy as np
from omni_semantic_map.sparse_lidar_support import MeasuredVoxelMap, project_measured_visibility, measured_frame_evidence


def test_sparse_pixels_use_nearest_center_and_unknown_abstains():
    cam = dict(H=2, W=3, fx=1., fy=1., cx=0., cy=0.)
    depth = np.array([[0., 1., 0.], [0., 0., 0.]])
    # u=.7 must use measured pixel 1, not unknown pixel 0.
    xyz = np.array([[.7, .1, 1.], [.2, .1, 1.], [1.4, .2, 2.], [-1., 0., 1.]])
    cfg = SimpleNamespace(depth_absolute_m=.05, depth_relative=.01)
    x, y, fov, visible = project_measured_visibility(xyz, np.eye(4), cam, depth, cfg)
    np.testing.assert_array_equal(x, [1, 0, 1, 0])
    np.testing.assert_array_equal(fov, [True, True, True, False])
    np.testing.assert_array_equal(visible, [True, False, False, False])


def test_voxel_representatives_preserve_first_measurement_and_provenance():
    m = MeasuredVoxelMap(.05)
    points = np.array([[.011, .012, 1.013], [.021, .022, 1.023], [-.001, 0., 1.]])
    colors = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
    pixels = np.array([[1, 1], [2, 2], [3, 3]])
    np.testing.assert_array_equal(m.append(points, colors, 9, np.array([14, 25, 36]), pixels), [0, 2])
    m.append(points[:1]+.001, colors[:1], 14, np.array([100]), pixels[:1])
    saved = m.arrays()
    np.testing.assert_array_equal(saved['xyz'], points[[0, 2]])
    np.testing.assert_array_equal(saved['source_frame'], [9, 9])
    np.testing.assert_array_equal(saved['source_raw_index'], [14, 36])
    np.testing.assert_array_equal(saved['colors'], colors[[0, 2]])


def test_measured_projection_roundtrip_with_nonidentity_pose():
    cam = dict(H=2, W=3, fx=1., fy=1., cx=0., cy=0.)
    pose = np.array([[0., -1., 0., 2.], [1., 0., 0., 3.], [0., 0., 1., -1.], [0., 0., 0., 1.]])
    camera_xyz = np.array([[.8, .1, 1.]])
    world = camera_xyz@pose[:3, :3].T+pose[:3, 3]
    depth = np.array([[0., 1., 0.], [0., 0., 0.]])
    result = project_measured_visibility(world, pose, cam, depth,
        SimpleNamespace(depth_absolute_m=.05, depth_relative=0.))
    assert result[-1].tolist() == [True]


def test_missing_depth_never_votes_and_duplicate_masks_do_not_multiply():
    cam = dict(H=1, W=3, fx=1., fy=1., cx=0., cy=0.)
    cfg = SimpleNamespace(depth_absolute_m=.05, depth_relative=0., boundary_pixels=0)
    xyz = np.array([[.7, 0., 1.], [.2, 0., 1.], [1.7, 0., 1.]])
    masks = np.array([[[False, True, False]], [[False, True, False]], [[False, False, True]]])
    result = measured_frame_evidence(xyz, np.array([1, 1, 1]), np.eye(4), cam,
        np.array([[0., 1., 1.]]), masks, np.array([1, 1, 2]), cfg)
    assert result[0].dtype == bool
    np.testing.assert_array_equal(result[0], [True, False, False])
    np.testing.assert_array_equal(result[1], [False, False, True])
