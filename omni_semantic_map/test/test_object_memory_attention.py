import numpy as np
import pytest
from omni_semantic_map.object_memory_attention import read_memory, rerank, select_memory


def test_view_selection_causal_unique_and_deterministic():
    frames = np.array([0, 0, 1, 2, 3])
    areas = np.array([5, 10, 2, 7, 1])
    assert select_memory([2, 0, 1, 1], 4, frames, areas) == [1, 2]
    with pytest.raises(ValueError, match='noncausal'):
        select_memory([1, 4], 4, frames, areas)
    areas[0] = 10
    assert select_memory([1, 0], 4, frames, areas) == [0]


def test_retrieval_recovers_a_view_hidden_by_averaging():
    q = np.array([1., 0.])
    h = np.array([[1., 0.], [0., 1.], [0., 1.]])
    r = read_memory(q, h)
    assert r['scores']['mean'] < .5
    assert r['scores']['attention'] > .99
    assert r['scores']['nearest'] == 1.
    assert r['weights'][0] > .99
    assert sum(r['weights']) == pytest.approx(1.)
    scaled = read_memory(q * 4, h * np.array([2, 3, 4])[:, None])
    assert scaled['scores'] == pytest.approx(r['scores'])


def test_temperature_limits_and_invalid_inputs():
    q = np.array([1., 0.])
    h = np.array([[1., 0.], [0., 1.]])
    assert read_memory(q, h, 1e6)['scores']['attention'] == pytest.approx(2**-.5, abs=1e-6)
    assert read_memory(q, h, 1e-6)['scores']['attention'] == pytest.approx(1.)
    assert read_memory(q, np.zeros((2, 2))) is None
    assert read_memory(q, np.array([[np.nan, 0.]])) is None
    with pytest.raises(ValueError):
        read_memory(q, h, 0)


def test_all_readers_share_history_and_invalid_fallback():
    event = dict(query=3, frame=2, candidates=[
        dict(members=[0, 1], is_native=False), dict(members=[2], is_native=True)])
    frames = np.array([0, 1, 0, 2]); areas = np.ones(4)
    features = np.array([[1., 0.], [0., 1.], [1., 0.], [1., 0.]])
    before = features.copy()
    r = rerank(event, features, frames, areas)
    assert r['memories'] == [[0, 1], [2]]
    assert r['choices']['nearest'] == 1  # tie stays native
    np.testing.assert_array_equal(before, features)
    features[0] = 0
    r = rerank(event, features, frames, areas)
    assert not r['supported']
    assert set(r['choices'].values()) == {1}


def test_future_features_cannot_change_decision():
    event = dict(query=2, frame=1, candidates=[
        dict(members=[0], is_native=True), dict(members=[1], is_native=False)])
    frames = np.array([0, 0, 1, 2]); areas = np.ones(4)
    features = np.array([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
    a = rerank(event, features, frames, areas)
    features[3] = [100., -100.]
    assert a == rerank(event, features, frames, areas)


def test_attention_can_follow_contamination_not_only_correct_matches():
    # A wrong object can contain a single highly similar view. No guarantee of
    # identity comes from softmax or memory alone; the experiment must count harm.
    q = np.array([1., 0.])
    wrong = np.array([[1., 0.], [0., 1.], [0., 1.]])
    right = np.array([[.8, .6]])
    a, b = read_memory(q, wrong), read_memory(q, right)
    assert a['scores']['mean'] < b['scores']['mean']
    assert a['scores']['attention'] > b['scores']['attention']


def test_diagnostic_identity_counts_unique_points_and_background():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / 'scripts/benchmarks/probe_object_memory_attention.py'
    spec = importlib.util.spec_from_file_location('probe', path)
    probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)
    labels = np.array([3, 3, 3, 3, 3, 0, 0, 0, 7])
    valid = np.ones(9, bool)
    s = probe.signature([0, 0, 1, 2, 3, 4, 5, 6, 7], labels, valid)
    assert s['valid_points'] == 8 and s['purity'] == .625 and not s['usable']
    assert probe.signature([0, 1, 2, 3, 4, 5, 6], labels, valid)['usable']
    valid[:5] = False
    assert not probe.signature(np.arange(9), labels, valid)['usable']
