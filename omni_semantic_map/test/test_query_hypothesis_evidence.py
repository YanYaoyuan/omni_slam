import numpy as np
import pytest
from omni_semantic_map.query_hypothesis_evidence import membership, retrieve


def test_weak_query_can_match_without_lowering_an_acceptance_threshold():
    h = membership([[1, 2, 3]], 10); q = membership([[1, 2, 3], [1, 4, 5]], 10)
    result = retrieve(h, [11], q, [11, 11], [.03, .9], [1, 2, 3, 4, 5])
    assert result['query'][0] == 0 and result['agreement'][0] == 1 and result['score'][0] == np.float32(.03)


def test_occluded_cells_abstain_and_large_background_masks_are_penalized():
    h = membership([[1, 2, 8, 9], [8, 9]], 10); q = membership([[1, 2], [1, 2, 3, 4]], 10)
    result = retrieve(h, [11, 11], q, [11, 11], [.2, .9], [1, 2, 3, 4])
    np.testing.assert_array_equal(result['query'], [0, -1]); np.testing.assert_array_equal(result['visible_cells'], [2, 0]); assert result['agreement'][0] == 1


def test_duplicates_class_constraints_and_native_suppression():
    h = membership([[1, 1, 2]], 6); q = membership([[1, 2], [1, 2, 2], [1, 2], [1, 2]], 6)
    result = retrieve(h, [26], q, [11, 26, 26, 26], [.9, .2, .4, 0], [1, 2])
    assert result['query'][0] == 2 and result['hypothesis_cells'][0] == 2
    with pytest.raises(ValueError, match='unmeasured'): retrieve(h, [26], q, [11, 26, 26, 26], [.9, .2, .4, 0], [1])


def test_sparse_reader_matches_dense_enumeration():
    rng = np.random.RandomState(8); hs = [np.flatnonzero(rng.rand(40) < .2) for _ in range(9)]; visible = np.flatnonzero(rng.rand(40) < .7); qs = [visible[rng.rand(len(visible)) < .3] for _ in range(12)]; hc = rng.randint(1, 3, 9); qc = rng.randint(1, 3, 12); scores = rng.rand(12)
    result = retrieve(membership(hs, 40), hc, membership(qs, 40), qc, scores, visible)
    for i, hyp in enumerate(hs):
        seen = set(hyp)&set(visible); options = []
        for j, query in enumerate(qs):
            intersection = seen&set(query)
            if hc[i] == qc[j] and intersection: options.append((len(intersection)/len(seen|set(query)), scores[j], -j))
        expected = -max(options)[2] if options else -1
        assert result['query'][i] == expected
