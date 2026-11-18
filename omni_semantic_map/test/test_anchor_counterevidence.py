import numpy as np
import pytest
from omni_semantic_map.anchor_counterevidence import counterevidence


def run(q, labels, frames, feature, owner=7, frame=0, chunk=1024):
    return counterevidence(np.asarray(q, np.float32).reshape(-1, 2),
                           np.full(len(q), owner), np.full(len(q), frame),
                           feature, labels, frames, chunk)


def test_excludes_self_and_retracts_only_majority():
    # Own frame supports the query, but both other frames refute it.
    labels = [7, -1] * 3
    frames = [0, 0, 1, 1, 2, 2]
    feature = [[1, 0], [-1, 0], [-1, 0], [1, 0], [-1, 0], [1, 0]]
    r = run([[1, 0], [-1, 0]], labels, frames, feature)
    np.testing.assert_array_equal(r['eligible'], [2, 2])
    np.testing.assert_array_equal(r['refute'], [2, 0])
    np.testing.assert_array_equal(r['retract'], [True, False])
    # Chunk boundaries cannot alter votes.
    other = run([[1, 0], [-1, 0]], labels, frames, feature, chunk=1)
    for name in r: np.testing.assert_allclose(r[name], other[name])


def test_missing_views_ties_and_other_foreground_abstain():
    r = run([[1, 0]], [7, -1], [1, 1], [[-1, 0], [1, 0]])
    assert r['eligible'][0] == 1 and not r['retract'][0]
    r = run([[1, 0]], [7, -1, 7, -1], [1, 1, 2, 2], [[1, 0]] * 4)
    assert r['eligible'][0] == 2 and r['refute'][0] == 0 and not r['retract'][0]
    r = run([[1, 0]], [7, 8, 7, 8], [1, 1, 2, 2], [[-1, 0], [1, 0]] * 2)
    assert r['eligible'][0] == 0 and not r['retract'][0]


def test_same_frame_competition_not_global_background_pool():
    # Perfect background match in frame 3 has no matching own-object view.
    r = run([[1, 0]], [7, -1, 7, -1, -1], [1, 1, 2, 2, 3],
            [[1, 0], [0, 1], [1, 0], [0, 1], [1, 0]])
    assert r['eligible'][0] == 2 and r['support'][0] == 2 and not r['retract'][0]


def test_duplicate_prototype_and_zero_feature_rejected():
    with pytest.raises(ValueError):
        run([[1, 0]], [7, 7], [1, 1], [[1, 0], [1, 0]])
    with pytest.raises(ValueError):
        run([[0, 0]], [7, -1], [1, 1], [[1, 0], [0, 1]])
