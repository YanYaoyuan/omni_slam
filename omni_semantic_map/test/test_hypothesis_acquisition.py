import numpy as np
import pytest
from omni_semantic_map.hypothesis_acquisition import (
    AcquisitionConfig, partition_disagreement, choose_views, symmetric_prompts,
    source_partition_from_logits)


def test_id_permutation_and_unknown_are_not_evidence():
    p = np.array([[1, 1, 2, 2, -1], [7, 7, 8, 8, 9]])
    assert not partition_disagreement(p).any()
    p[0, 2:4] = 1
    d = partition_disagreement(p)
    assert d[0, 2] == .25 and not d[4].any() and not np.diag(d).any()


def test_joint_evidence_beats_unbalanced_area_and_cannot_read_validation():
    f = np.repeat([0, 1], 10)
    p = np.array([np.ones(20, int), f+1])
    v = np.zeros((3, 20), bool)
    v[0, :10] = True; v[0, 10:12] = True  # area12, pairs20
    v[1, :5] = True; v[1, 10:15] = True  # area10, pairs25
    v[2] = True  # must never be read for selection
    cfg = AcquisitionConfig(views_per_event=1, min_visible_per_fragment=2)
    kwargs = dict(partitions=p, visibility=v, camera_centers=np.eye(3),
                  roles=['acquire', 'acquire', 'validate'], fragments=f, config=cfg)
    assert choose_views(**kwargs)['selected'] == [1]
    assert choose_views(**kwargs, policy='area')['selected'] == [0]


def test_missing_measurements_and_repeated_camera_abstain():
    f = np.repeat([0, 1], 6); p = np.array([np.ones(12, int), f+1])
    cfg = AcquisitionConfig()
    r = choose_views(p, np.zeros((2, 12), bool), np.zeros((2, 3)), ['acquire']*2, f, config=cfg)
    assert r['selected'] == []
    r = choose_views(p, np.ones((2, 12), bool), np.zeros((2, 3)), ['acquire']*2, f, config=cfg)
    assert r['selected'] == [0]


def test_symmetric_prompts_are_balanced_and_unknown_is_not_deletion():
    uv = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    coords, labels, anchors = symmetric_prompts(uv, np.ones(4, bool), np.array([0, 0, 1, 1]))
    assert coords.shape == (3, 4, 2)
    np.testing.assert_array_equal(labels[1]+labels[2], np.ones(4))
    np.testing.assert_array_equal(coords[0], coords[1])
    logits = np.array([[[1, 1, -1, 1]], [[3, 1, -1, -1]], [[1, 3, -2, 2]]])
    out = source_partition_from_logits(logits, uv)
    np.testing.assert_array_equal(out, [[1, 1, -1, 1], [1, 2, -1, 2]])
    with pytest.raises(ValueError): symmetric_prompts(np.zeros_like(uv), np.ones(4, bool), np.array([0,0,1,1]))
