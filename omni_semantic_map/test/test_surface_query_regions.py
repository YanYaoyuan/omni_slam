import numpy as np
import pytest
from omni_semantic_map.surface_query_regions import SurfaceQueryMemory, crop_grid, vertical_surface_weight


def test_ground_wall_and_joint_rigid_transform():
    x, y = np.meshgrid(np.linspace(-2, 2, 8), np.linspace(-2, 2, 8)); wall = np.c_[x.ravel(), np.zeros(x.size), y.ravel()]
    ground = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
    assert vertical_surface_weight(ground, [0, 0, 1]).max() < 1e-12
    assert vertical_surface_weight(wall, [0, 0, 1]).mean() > .4
    rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_allclose(vertical_surface_weight(wall, [0, 0, 1]), vertical_surface_weight(wall@rotation.T+[13, 20, 8], rotation@np.array([0, 0, 1])), atol=1e-10)


def test_memory_changes_region_choice_and_future_is_rejected():
    m = SurfaceQueryMemory(); boxes = np.array([[0, 0, 10, 10], [10, 0, 20, 10]])
    for frame in range(4): m.observe(frame, [0, 1], [True, False])
    m.observe(4, [0, 1], [False, False])
    result = m.rank(4, [0, 1], [[2, 2], [12, 2]], [1., .8], [False, False], boxes, 0)
    assert result['chosen']['current'] == 0 and result['chosen']['memory'] == 1
    with pytest.raises(ValueError): m.observe(4, [0], [False])
    with pytest.raises(ValueError): m.rank(5, [0], [[2, 2]], [1.], [False], boxes, 0)


def test_duplicate_measurements_never_add_frame_votes_or_crop_score():
    m = SurfaceQueryMemory(); m.observe(0, [0, 0, 1], [False, True, False])
    np.testing.assert_array_equal(m.seen, [1, 1]); np.testing.assert_array_equal(m.covered, [1, 0])
    boxes = crop_grid(30, 20)
    a = m.rank(0, [0, 1], [[3, 3], [22, 3]], [1., .5], [True, False], boxes, 0)
    b = m.rank(0, [0, 0, 1], [[3, 3], [3, 3], [22, 3]], [1., 1., .5], [True, True, False], boxes, 0)
    assert a == b


def test_no_current_surface_has_no_positive_utility():
    m = SurfaceQueryMemory(); m.observe(0, np.empty(0, np.int64), [])
    result = m.rank(0, [], np.empty((0, 2)), [], [], crop_grid(30, 20), 2)
    assert all(sum(v) == 0 for v in result['scores'].values())
