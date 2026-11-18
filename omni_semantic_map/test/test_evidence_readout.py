import numpy as np
import pytest
from omni_semantic_map.evidence_readout import aggregate, METHODS


def test_correlated_duplicates_do_not_dominate_density_readout():
    # Nine repeated camera directions versus one opposite direction: density
    # weighting gives these two directional groups approximately equal mass.
    f = np.array([[1., 0.]]*9+[[0., 1.]])
    p = np.array([[1., 0., 0.]]*9+[[-1., 0., 0.]])
    result = aggregate(f, np.ones(10), p, np.zeros(3), method='density')
    assert result['weights'][:9].sum() == pytest.approx(.5, abs=2e-6)
    assert result['effective_views'] == pytest.approx(2., abs=2e-6)
    assert aggregate(f, np.ones(10), p, np.zeros(3))['weights'][:9].sum() == pytest.approx(.9)


@pytest.mark.parametrize('method', METHODS)
def test_permutation_invariance_and_normalization(method):
    rng = np.random.default_rng(5)
    f = rng.random((15, 8)); p = rng.normal(size=(15, 3)); a = rng.integers(1, 500, 15)
    ids = np.arange(15)*3; perm = rng.permutation(15)
    x = aggregate(f, a, p, np.zeros(3), method=method, observation_ids=ids)
    y = aggregate(f[perm], a[perm], p[perm], np.zeros(3), method=method, observation_ids=ids[perm])
    np.testing.assert_allclose(x['descriptor'], y['descriptor'], atol=1e-12)
    np.testing.assert_allclose(x['weights'][perm], y['weights'], atol=1e-12)
    assert 1 <= x['effective_views'] <= 15
    assert np.linalg.norm(x['descriptor']) == pytest.approx(1)


def test_single_view_and_three_view_gate_are_explicit():
    for method in METHODS:
        x = aggregate([[3, 4]], [10], [[1, 0, 0]], [0, 0, 0], method=method)
        np.testing.assert_allclose(x['descriptor'], [.6, .8])
        assert x['effective_views'] == pytest.approx(1)
        assert x['retained_by_count'] == (method != 'mean_min3')


def test_convex_readout_cannot_fix_consistent_background_dominance():
    # First feature/text axis is background, second is foreground.
    f = np.array([[.9, .1], [.8, .2], [.7, .3]])
    p = np.eye(3)
    for method in METHODS:
        assert aggregate(f, [10, 20, 30], p, [0, 0, 0], method=method)['descriptor'].argmax() == 0


def test_invalid_geometry_and_nonfinite_features_fail():
    with pytest.raises(ValueError): aggregate([[1, 0]], [1], [[0, 0, 0]], [0, 0, 0])
    with pytest.raises(ValueError): aggregate([[np.nan, 0]], [1], [[1, 0, 0]], [0, 0, 0])
