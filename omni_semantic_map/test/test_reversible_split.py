import numpy as np
from omni_semantic_map.reversible_split import (
    EvidenceView, SplitConfig, propose_splits, validate_split,
    apply_splits, rollback, view_gain,
)


def fixture():
    rng = np.random.default_rng(1)
    xyz = np.r_[rng.normal([-.5, 0, 0], .02, (30, 3)),
                rng.normal([.5, 0, 0], .02, (30, 3))]
    labels = np.ones(60, np.int64)
    masks = np.array([np.arange(60) < 30, np.arange(60) >= 30])
    views = []
    for i in range(10):
        angle = i * .3
        views.append(EvidenceView(i, np.array([2*np.sin(angle), 2*np.cos(angle), 0.]),
                                  np.arange(60), masks.copy(), np.array([2*i, 2*i+1])))
    return xyz, labels, views


def test_recovery_and_exact_rollback():
    xyz, labels, views = fixture()
    proposals = propose_splits(xyz, labels, views)
    assert len(proposals) == 1
    decision = validate_split(proposals[0], labels, views)
    assert decision['accepted'] and decision['separating_views'] >= 2
    updated, lineage = apply_splits(labels, proposals, [True])
    assert set(updated) == {1, 2}
    assert np.array_equal(rollback(updated, lineage), labels)
    assert np.array_equal(labels, np.ones(60))


def test_repeated_camera_does_not_multiply_independent_evidence():
    xyz, labels, views = fixture()
    proposals = propose_splits(xyz, labels, views)
    for v in views[6:]:
        v.camera_center = views[6].camera_center.copy()
    result = validate_split(proposals[0], labels, views)
    assert result['distinct_views'] == 1
    assert not result['accepted']


def test_later_union_masks_reject_false_split_and_do_not_change_proposal():
    xyz, labels, views = fixture()
    original = propose_splits(xyz, labels, views)[0]
    for v in views[6:]:
        v.membership = np.ones((1, 60), bool)
        v.observation_ids = np.array([1000+v.frame_id])
    recomputed = propose_splits(xyz, labels, views)[0]
    assert np.array_equal(original['child'], recomputed['child'])
    decision = validate_split(recomputed, labels, views)
    assert not decision['accepted'] and decision['mean_gain'] < 0


def test_unknown_depth_and_overlapping_masks_never_force_split():
    xyz, labels, views = fixture()
    for v in views:
        v.indices = np.empty(0, np.int64)
        v.membership = np.empty((2, 0), bool)
    assert not propose_splits(xyz, labels, views)
    child = np.r_[np.zeros(30), np.ones(30)].astype(int)
    assert view_gain(np.ones((2, 60), bool), child, SplitConfig()) < 0
