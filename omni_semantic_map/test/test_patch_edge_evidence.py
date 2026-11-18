import numpy as np
import pytest
from omni_semantic_map.patch_edge_evidence import bilinear_edge_cosine, patch_weights


def test_gram_matches_direct_border_padded_sampling():
    rng = np.random.default_rng(19)
    grid = rng.normal(size=(3, 5, 17)).astype(np.float16)
    xy = np.r_[rng.uniform([0, 0], [101, 59], (30, 2)), [[0, 0], [100, 58]]]
    left = np.arange(len(xy)); right = left[::-1]
    got, valid = bilinear_edge_cosine(grid, xy, (59, 101), left, right)
    # Independent direct sampler, including half-pixel/border handling.
    uv = (xy+.5)*[5/101, 3/59]-.5
    out = []
    for x, y in uv:
        x0, y0 = int(np.floor(x)), int(np.floor(y)); dx, dy = x-x0, y-y0
        out.append(sum(grid[np.clip(y0+j, 0, 2), np.clip(x0+i, 0, 4)].astype(float)*wx*wy
                       for i, wx in [(0, 1-dx), (1, dx)] for j, wy in [(0, 1-dy), (1, dy)]))
    out = np.asarray(out); out /= np.linalg.norm(out, axis=1, keepdims=True)
    assert valid.all()
    np.testing.assert_allclose(got, (out[left]*out[right]).sum(1), atol=1e-12)
    swapped, _ = bilinear_edge_cosine(grid, xy, (59, 101), right, left)
    np.testing.assert_allclose(got, swapped, atol=1e-12)


def test_zero_features_are_missing_and_identical_features_are_one():
    xy = np.array([[0, 0], [8, 8]])
    zero, valid = bilinear_edge_cosine(np.zeros((2, 2, 4)), xy, (9, 9), [0], [1])
    assert not valid.any() and np.isnan(zero).all()
    same, valid = bilinear_edge_cosine(np.ones((2, 2, 4)), xy, (9, 9), [0], [1])
    assert valid.all(); np.testing.assert_allclose(same, 1)


def test_invalid_pixels_and_edges_rejected():
    for xy in [[[10, 0]], [[-1, 0]], [[np.nan, 0]]]:
        with pytest.raises(ValueError): patch_weights(xy, (10, 10), (2, 2))
    with pytest.raises(ValueError):
        bilinear_edge_cosine(np.ones((2, 2, 4)), [[0, 0]], (10, 10), [0], [1])
