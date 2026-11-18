import numpy as np
import pytest
from omni_semantic_map.mask_recall_revision import revise


def test_unknown_stays_unchanged_and_never_votes():
    z=np.zeros((5,8),np.uint8)
    assert not revise(z)['reject'].any()
    np.testing.assert_allclose(revise(z)['membership'],.9)


def test_partial_masks_preserve_observed_true_points_and_remove_never_supported():
    # Each true point is visible in every image but segmented in only 40%.
    z=np.ones((20,120),np.uint8)
    for t in range(20):
        z[t,np.flatnonzero(np.arange(100)%5<2)]=2
        z[t,:100]=np.roll(z[t,:100],t%5)
    result=revise(z)
    assert not result['reject'][:100].any()
    assert result['reject'][100:].all()
    assert np.all((result['view_recall']>.3)&(result['view_recall']<.6))


def test_unknown_rows_and_point_permutation_invariance():
    rng=np.random.default_rng(42);z=rng.integers(0,3,(9,32),dtype=np.uint8)
    perm=rng.permutation(z.shape[1]);a=revise(z)
    b=revise(np.vstack([z[:,perm],np.zeros((7,32),np.uint8)]))
    np.testing.assert_allclose(a['membership'][perm],b['membership'],atol=1e-10)


def test_invalid_input():
    with pytest.raises(ValueError):revise(np.array([[3]]))
    with pytest.raises(ValueError):revise(np.zeros((2,3)),mode='unknown')
