import numpy as np
import pytest

from omni_semantic_map.history_graph import build_history_graph, contrastive_bridge


def proposal(a, b, frame, dino=True):
    return dict(a=a, b=b, frame=frame, dino_pass=dino)


def test_transitive_coframe_conflict_and_reversible_edge_removal():
    nodes = {1: [1], 2: [2], 3: [1]}
    proposals = [proposal(1, 2, 3), proposal(2, 3, 4)]
    result = build_history_graph(nodes, proposals)
    assert result['groups'] == [[1, 2], [3]]
    assert result['decisions'][1]['reason'] == 'coframe_conflict'
    assert build_history_graph(nodes, proposals[1:])['groups'] == [[1], [2, 3]]
    assert nodes == {1: [1], 2: [2], 3: [1]}


def test_votes_are_distinct_frames_and_dino_is_a_subset():
    nodes = {1: [1], 2: [2]}
    proposals = [proposal(1, 2, 3), proposal(2, 1, 3), proposal(1, 2, 4, False)]
    assert build_history_graph(nodes, proposals, 2)['groups'] == [[1, 2]]
    assert build_history_graph(nodes, proposals, 2, True)['groups'] == [[1], [2]]
    assert build_history_graph(nodes, proposals[:2], 2)['groups'] == [[1], [2]]


def test_common_order_preserves_conservation_and_rejects_unknown_nodes():
    nodes = {1: [1], 2: [2], 3: [3]}
    edges = [proposal(1, 2, 5), proposal(2, 3, 4), proposal(2, 3, 6)]
    result = build_history_graph(nodes, edges)
    assert result['decisions'][0]['a'] == 2
    assert sorted(x for group in result['groups'] for x in group) == [1, 2, 3]
    with pytest.raises(ValueError):
        build_history_graph(nodes, [proposal(1, 4, 8)])


def test_contrastive_evidence_normalizes_and_requires_both_branches():
    q = np.array([1., 0.])
    assert contrastive_bridge(q, [[4., 0]], [[.9, .1]], [[0., 3]])['pass_vote']
    assert not contrastive_bridge(q, [[4., 0]], [[0., 1]], [[.9, .1]])['pass_vote']
    assert not contrastive_bridge(q, [[1., 0]], [[1., 0]], [[1., 0]])['pass_vote']
    assert contrastive_bridge(q, [[0., 0]], [[1., 0]], [[0., 1]])['reason'] == 'unknown'
    assert contrastive_bridge(q, [[1., 0]], [[1., 0]], np.empty((0, 2)))['reason'] == 'unknown'
