import numpy as np
import pytest
from omni_semantic_map.kitti360_input import project_lidar, load_poses, load_times


def test_sparse_occlusion_retains_exact_source():
    c = dict(K=np.eye(3), width=4, height=3, rectified_from_velodyne=np.eye(4))
    points = np.array([[0, 0, 4], [0, 0, 2], [2, 2, 2], [0, 0, -1], [np.nan, 0, 1]])
    result = project_lidar(points, c)
    assert result['source_point_index'][0, 0] == 1
    assert result['depth'][0, 0] == 2
    assert result['source_point_index'][1, 1] == 2
    assert np.count_nonzero(result['depth']) == 2
    assert (result['source_point_index'][result['depth'] == 0] == -1).all()


def test_empty_scan_and_pixel_boundary():
    c = dict(K=np.eye(3), width=2, height=2, rectified_from_velodyne=np.eye(4))
    assert not project_lidar(np.empty((0, 3)), c)['depth'].any()
    r = project_lidar(np.array([[-.49, 0, 1], [-.51, 0, 1], [1.5, 0, 1]]), c)
    assert r['selected_indices'].tolist() == [0]


def test_pose_file_does_not_invent_missing_frames(tmp_path):
    p = tmp_path/'poses.txt'; row = ' '.join(map(str, np.eye(4)[:3].ravel()))
    p.write_text('1 '+row+'\n69 '+row+'\n')
    assert list(load_poses(p)) == [1, 69]
    p.write_text('1 '+row+'\n1 '+row+'\n')
    with pytest.raises(ValueError): load_poses(p)


def test_timestamp_nanoseconds_preserved(tmp_path):
    p = tmp_path/'timestamps.txt'
    p.write_text('2013-05-28 12:00:00.000000001\n2013-05-28 12:00:00.000000003\n')
    assert np.diff(load_times(p)).tolist() == [2]
