import numpy as np
import pytest
from omni_semantic_map.mask_observation_code import (
    unique_queries, fit_code, decode_code, unique_winner,
)


def test_exclusion_separates_anchors_sharing_a_large_mask():
    # Both objects share the large mask. The second mask excludes object 2.
    x = np.array([[1, 1, 1, 1], [1, 1, 0, 0]], bool)
    p, _ = fit_code(x, np.array([1, 1, 2, 2]), [1, 2])
    decoded = decode_code(np.array([[1, 1], [1, 0]], bool), p)['full']
    np.testing.assert_array_equal(unique_winner(decoded, [1, 2]), [1, 2])


def test_identical_codes_abstain_instead_of_low_id_winning():
    x = np.ones((1, 4), bool)
    p, _ = fit_code(x, np.array([1, 1, 2, 2]), [1, 2])
    np.testing.assert_array_equal(unique_winner(decode_code(x, p)['full'], [1, 2]), 0)


def test_duplicate_queries_and_joint_permutation_do_not_change_decisions():
    x = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]], bool)
    expanded = x[[2, 1, 0, 1, 2, 0]]
    a, _ = unique_queries(x)
    b, _ = unique_queries(expanded)
    labels = np.array([1, 1, 2, 2])
    p, _ = fit_code(a, labels, [1, 2])
    q, _ = fit_code(b, labels, [1, 2])
    np.testing.assert_allclose(decode_code(a, p)['full'], decode_code(b, q)['full'])
    np.testing.assert_array_equal(unique_winner(decode_code(a, p)['full'], [1, 2]), labels)


def test_repeated_measurements_with_cell_weights_do_not_change_code():
    x = np.array([[1, 0], [0, 1]], bool)
    p, _ = fit_code(x, [1, 2], [1, 2])
    q, n = fit_code(x[:, [0, 0, 0, 1]], [1, 1, 1, 2], [1, 2], [1/3, 1/3, 1/3, 1])
    np.testing.assert_allclose(p, q)
    np.testing.assert_allclose(n, 1)


def test_invalid_or_unobserved_anchors_are_not_silently_fitted():
    with pytest.raises(ValueError):
        fit_code(np.ones((2, 2), bool), [1, 1], [1, 2])
    with pytest.raises(ValueError):
        decode_code(np.empty((0, 1), bool), np.empty((2, 0)))
