import numpy as np
import pytest
from omni_semantic_map.live_core import LatestSlot, mask_points, rigid_matrix, support_indices


def test_slow_inference_keeps_only_latest_pending_frame():
    slot = LatestSlot()
    for i in range(100):
        slot.put(i)
    assert slot.pop() == 99 and slot.pop() is None and slot.dropped == 99


def test_invalid_quaternion_and_missing_depth_cannot_create_box():
    with pytest.raises(ValueError):
        rigid_matrix([0, 0, 0], [0, 0, 0, 0])
    points = np.zeros((7, 3)); points[:, 2] = 2
    assert support_indices(points, np.ones((7, 2)), np.ones(7)*2,
                           np.ones(7, bool), np.ones((4, 4), bool)) is None


def test_projection_mask_rejects_outside_and_behind_camera():
    pixels = np.array([[1, 1], [-2, 2], [5, 1], [np.nan, np.nan], [2, 2]])
    assert mask_points(pixels, np.array([1, 1, 1, 1, 0], bool),
                       np.ones((4, 4), bool)).tolist() == [True, False, False, False, False]


def test_ambiguous_foreground_background_clusters_do_not_create_box():
    rng = np.random.default_rng(3)
    points = np.vstack([rng.normal([0, 0, 3], .03, (12, 3)),
                        rng.normal([0, 0, 6], .03, (12, 3))])
    assert support_indices(points, np.ones((24, 2)), points[:, 2],
                           np.ones(24, bool), np.ones((4, 4), bool)) is None
    selected = support_indices(points[:12], np.ones((12, 2)), points[:12, 2],
                               np.ones(12, bool), np.ones((4, 4), bool))
    assert len(selected) == 12
