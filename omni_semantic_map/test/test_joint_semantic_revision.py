import numpy as np
import pytest
from omni_semantic_map.joint_semantic_revision import evidence, revised_class


def test_leave_cell_out_has_no_self_support():
    r = evidence([[0, 7], [1, 7], [0, 7], [1, 7]], [1, 1, 2, 2],
                 [[1, 0], [0, 1], [1, 0], [0, 1]])
    np.testing.assert_allclose(r['leave_cell_out_probability'], [[0, 1], [1, 0]])
    np.testing.assert_array_equal(r['semantic_conflict'], [True, True])
    solo = evidence([[0, 7], [0, 7]], [1, 2], [[1, 0], [1, 0]])
    assert solo['other_observation_frames'][0] == 0 and not solo['semantic_conflict'][0]


def test_view_balancing_and_uniform_evidence_abstain():
    r = evidence([[0, 7], [1, 7], [2, 7]], [1, 1, 2], [[1, 0], [1, 0], [0, 1]])
    np.testing.assert_allclose(r['object_probability'][0], [.5, .5])
    r = evidence([[0, 7], [1, 7]], [1, 1], [[.5, .5], [1, 0]])
    np.testing.assert_allclose(r['compatibility_ratio'], [1, 1])
    assert not r['semantic_conflict'].any()


def test_class_revision_does_not_imply_geometry_conflict():
    r = evidence([[0, 7], [1, 7], [0, 7], [1, 7]], [1, 1, 2, 2],
                 [[0, 1]] * 4)
    assert revised_class(r['object_probability'][0], previous=0) == 1
    assert not r['semantic_conflict'].any()
    assert revised_class([.5, .5], previous=1) == 1


def test_duplicates_and_nonfinite_rejected():
    with pytest.raises(ValueError): evidence([[0, 7], [0, 7]], [1, 1], [[1, 0], [1, 0]])
    with pytest.raises(ValueError): evidence([[0, 7]], [1], [[np.nan, 1]])


def test_random_leave_cell_out_matches_explicit_recomputation():
    rng = np.random.default_rng(62)
    for _ in range(40):
        rows = [(cell, owner, frame) for owner in [7, 13] for cell in range(6)
                for frame in range(4) if rng.random() > .4]
        a = np.asarray(rows); q = rng.dirichlet(np.ones(5), len(a))
        r = evidence(a[:, :2], a[:, 2], q)
        for i, (cell, owner) in enumerate(r['cell_owner']):
            take = (a[:, 1] == owner) & (a[:, 0] != cell)
            parts = [q[take & (a[:, 2] == frame)].mean(0) for frame in np.unique(a[take, 2])]
            expected = np.mean(parts, axis=0) if parts else np.zeros(5)
            np.testing.assert_allclose(r['leave_cell_out_probability'][i], expected, atol=1e-12)
