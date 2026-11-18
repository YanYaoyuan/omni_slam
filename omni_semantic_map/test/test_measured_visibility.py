import numpy as np
from omni_semantic_map.measured_visibility import camera_from_record, measured_depth
from omni_semantic_map.visibility_cleanup import CleanupConfig, project_visibility


def test_sparse_depth_preserves_unknown_and_nearest_surface():
    cam = dict(W=5, H=5, fx=2., fy=2., cx=2., cy=2.)
    points = np.array([[0, 0, 2], [0, 0, 3], [100, 0, 2], [0, 0, -.5]], float)
    depth = measured_depth(points, np.eye(4), cam)
    assert np.count_nonzero(depth) == 1 and depth[2, 2] == 2
    _, _, _, visible = project_visibility(points, np.eye(4), cam, depth, CleanupConfig())
    assert visible.tolist() == [True, False, False, False]


def test_actual_record_calibration_overrides_unrelated_template():
    r = dict(width=640, height=360, intrinsic=[[268, 0, 319], [0, 269, 180], [0, 0, 1]])
    cam = camera_from_record(r)
    assert cam == dict(W=640, H=360, fx=268., fy=269., cx=319., cy=180.)
