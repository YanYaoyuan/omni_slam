import numpy as np
from omni_semantic_map.replica_benchmark import projection, truth_labels, pair_score


def test_replica_depth_scale_and_projection_roundtrip():
    depth = np.full((680, 1200), 2000, np.uint16)
    depth[0, 0] = 0
    depth[0, 8] = 6000
    points, valid, u, v = projection(depth)
    assert not valid[0, 0] and not valid[0, 1]
    assert valid[1, 1] and points[1, 1, 2] == 2.
    np.testing.assert_allclose(points[..., 0][valid]/points[..., 2][valid]*600+599.5, u[valid], atol=1e-4)
    np.testing.assert_allclose(points[..., 1][valid]/points[..., 2][valid]*600+339.5, v[valid], atol=1e-4)


def test_valid_instance_zero_survives_but_stuff_and_unknown_do_not():
    raw = np.array([[0, 8, 9, 10]], np.uint16)
    semantic = np.array([[20, 40, 0, 255]], np.uint8)
    np.testing.assert_array_equal(truth_labels(raw, semantic), [[1, 0, 0, 0]])


def test_pair_metrics_detect_merges_and_splits_independent_of_id_names():
    assert pair_score([1,1,2,2], [8,8,9,9])['f1'] == 1.
    merged = pair_score([1,1,2,2], [8,8,8,8])
    assert merged['recall'] == 1. and merged['precision'] == 1/3
    split = pair_score([1,1,1,1], [8,8,9,9])
    assert split['precision'] == 1. and split['recall'] == 1/3
