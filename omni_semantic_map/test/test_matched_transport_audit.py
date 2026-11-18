import numpy as np
import pytest
from omni_semantic_map.matched_transport_audit import confidence, matched_selection, summarize_changes


def test_exact_object_budget_and_deterministic_ties():
    ids = np.array([8, 3, 5, 2, 9, 1])
    owner = np.array([2, 1, 1, 2, 2, 1])
    score = np.array([1., 0., 0., 1., 2., 0.])
    target = np.array([3, 5, 8]); target_owner = np.array([1, 1, 2])
    expected = [1, 3, 9]
    np.testing.assert_array_equal(matched_selection(ids, owner, score, target, target_owner), expected)
    np.testing.assert_array_equal(matched_selection(ids[::-1], owner[::-1], score[::-1], target, target_owner), expected)
    with pytest.raises(ValueError):
        matched_selection(ids, owner, score, [8], [1])
    with pytest.raises(ValueError):
        matched_selection(ids, owner, score, [88], [2])


def test_missing_competitor_and_zero_distances():
    d = np.array([[0, 0], [0, 2], [1, 3], [4, np.inf]])
    np.testing.assert_array_equal(confidence(d, 'normalized_margin'), [0, 1, .5, 1])
    np.testing.assert_array_equal(confidence(d, 'nearest_distance'), [0, 0, -1, -4])
    with pytest.raises(ValueError):
        confidence([[np.inf, np.inf]], 'nearest_distance')


def test_source_weight_and_object_macro_are_distinct():
    # Sources 0 and 1 share one cell but each counts in the actual map.
    row = summarize_changes(np.array([2, 2, 3, 4]), np.array([1, 1, 1, 2]), np.array([0, 0, 1, 0, 0]), 2)
    assert row['vulnerable_sources'] == 2
    assert row['vulnerable_fraction'] == .5
    assert row['mean_failure_fraction'] == .25
    assert row['macro_object_vulnerable_fraction'] == pytest.approx(1 / 3)
