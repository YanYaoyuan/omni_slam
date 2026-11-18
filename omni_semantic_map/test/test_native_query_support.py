import numpy as np
import pytest
from omni_semantic_map.native_query_support import exclusive_measured_queries


def test_packed_lookup_matches_full_dense_native_arbitration():
    rng = np.random.RandomState(12); masks = rng.rand(7, 4, 13) > .65; scores = np.array([.9, .2, .8, .2, .5, .7, .3]); y, x = np.indices((4, 13)); px = np.c_[x.ravel(), y.ravel()]
    dense = np.full((4, 13), -1)
    for q in np.argsort(-scores, kind='stable'):
        if scores[q] > .1: dense[masks[q] & (dense < 0)] = q
    actual = exclusive_measured_queries(np.packbits(masks, axis=2), scores, px, 13, .1)
    np.testing.assert_array_equal(actual, dense.ravel())


def test_deferred_queries_cannot_steal_high_confidence_pixels():
    masks = np.array([[[1, 0, 0, 1]], [[1, 1, 0, 0]], [[0, 1, 1, 1]]], bool); px = np.c_[np.arange(4), np.zeros(4, int)]; scores = np.array([.8, .2, .5]); packed = np.packbits(masks, axis=2)
    high = exclusive_measured_queries(packed, scores, px, 4, .5)
    low = exclusive_measured_queries(packed, scores, px, 4, .1)
    np.testing.assert_array_equal(low[high >= 0], high[high >= 0]); np.testing.assert_array_equal(low, [0, 2, 2, 0])


def test_empty_candidates_and_invalid_pixel():
    assert exclusive_measured_queries(np.empty((0, 2, 1), np.uint8), [], np.array([[0, 0]]), 4, .1)[0] == -1
    with pytest.raises(ValueError): exclusive_measured_queries(np.zeros((1, 2, 1), np.uint8), [.9], np.array([[4, 0]]), 4, .1)
