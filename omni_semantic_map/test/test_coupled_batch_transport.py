import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from omni_semantic_map.coupled_batch_transport import native_library, solve


@pytest.fixture(scope='module')
def lib(tmp_path_factory): return native_library(tmp_path_factory.mktemp('coupled_native'))


def brute(graph, seeds, keep, tolerance=1e-10):
    seeds = seeds[keep]; n = graph.shape[0]
    if not len(seeds): return np.full(n, -1, np.int64), np.full(n, np.inf)
    paths = dijkstra(graph, indices=seeds[:, 0])
    owners = np.unique(seeds[:, 1])
    cost = np.array([paths[seeds[:, 1] == o].min(axis=0) for o in owners])
    order = np.argsort(cost, axis=0, kind='stable')
    first = cost[order[0], np.arange(n)]
    second = cost[order[1], np.arange(n)] if len(owners) > 1 else np.full(n, np.inf)
    return np.where(np.isfinite(first) & (first < second - tolerance), owners[order[0]], -1), first


def assert_exhaustive(graph, seeds, lib):
    r = solve(graph, seeds, lib); before, distance = brute(graph, seeds, np.ones(len(seeds), bool))
    np.testing.assert_array_equal(r['nearest_owner'], before)
    np.testing.assert_allclose(r['nearest_distance'], distance, rtol=1e-12, atol=1e-12)
    stable = before >= 0
    for batch in np.unique(seeds[:, 2]):
        after, d = brute(graph, seeds, seeds[:, 2] != batch)
        stable &= after == before
        ids = r['critical_batch'] == batch
        np.testing.assert_array_equal(r['after_critical_owner'][ids], after[ids])
        np.testing.assert_allclose(r['after_critical_distance'][ids], d[ids], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(r['stable_owner'], np.where(stable, before, -1))
    return r


def test_shared_batch_competitor_removal_recovers_rejected_point(lib):
    # At vertex0, owner1 is nearest. Competitor2 at distance1 belongs to the
    # same batch; owner1's second batch is at distance2. Holding competitor2
    # fixed rejects, while actually removing either entire batch preserves1.
    g = csr_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    seeds = np.array([[0, 1, 10], [1, 2, 10], [2, 1, 20]])
    r = assert_exhaustive(g, seeds, lib)
    assert r['stable_owner'][0] == 1


@pytest.mark.parametrize('scale', [1., 1e-10])
def test_random_quantized_graphs_all_batches(lib, scale):
    rng = np.random.RandomState(47)
    for _ in range(80):
        a = rng.randint(0, 5, (17, 17)).astype(float) * scale
        a[rng.rand(17, 17) < .8] = 0; a = np.maximum(a, a.T); np.fill_diagonal(a, 0)
        seeds = np.column_stack([rng.randint(0, 17, 26), rng.randint(0, 6, 26), rng.randint(0, 8, 26)])
        assert_exhaustive(csr_matrix(a), seeds, lib)


def test_third_frame_tie_must_not_be_discarded(lib):
    g = csr_matrix([[0, 1, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    seeds = np.array([[0, 1, 0], [1, 1, 1], [2, 2, 2]])
    r = assert_exhaustive(g, seeds, lib)
    assert r['nearest_owner'][0] == 1 and r['stable_owner'][0] == -1
    assert r['max_retained_frames'] >= 3 and r['nearest_owner'][3] == -1


def test_within_frame_ambiguous_owners_and_zero_length_edges(lib):
    g = csr_matrix((np.array([0., 0., 1., 1.]), ([0, 1, 1, 2], [1, 0, 2, 1])), shape=(3, 3))
    seeds = np.array([[0, 1, 0], [1, 2, 0], [2, 1, 1]])
    r = assert_exhaustive(g, seeds, lib)
    assert r['nearest_owner'][0] == -1


def test_seed_order_duplication_and_batch_renaming(lib):
    g = csr_matrix([[0, 1, 0], [1, 0, 2], [0, 2, 0]])
    s = np.array([[0, 1, 2], [1, 2, 2], [2, 1, 4]])
    a = solve(g, s, lib); renamed = s[::-1].copy(); renamed[:, 2] = 100 - renamed[:, 2]
    b = solve(g, np.r_[renamed, renamed], lib)
    for key in ['nearest_owner', 'stable_owner']: np.testing.assert_array_equal(a[key], b[key])


def test_empty_seeds_and_invalid_input(lib):
    g = csr_matrix((3, 3)); r = solve(g, np.empty((0, 3), np.int64), lib)
    assert (r['stable_owner'] == -1).all()
    with pytest.raises(ValueError): solve(g, [[0, -1, 0]], lib)
    with pytest.raises(ValueError): solve(g, [[0, 1, .5]], lib)
    with pytest.raises(ValueError): solve(csr_matrix([[0, -1], [-1, 0]]), [[0, 1, 0]], lib)
