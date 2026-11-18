import numpy as np
import pytest
from omni_semantic_map.replay_fusion import FusionObservation, ReplayableObjectMap


class SmallBackend:
    """Deterministic test double for ledger transactions, not algorithm accuracy."""
    def __init__(self):
        self.values = {}

    def integrate(self, observations, pose):
        for obs in observations:
            self.values[obs.observation_id] = 1 if pose[0, 3] < .5 else 2
        return {o.observation_id: self.values[o.observation_id] for o in observations}

    def labels(self, finalize=True):
        return dict(self.values)

    def export_objects(self, finalize=True):
        return [dict(observation_ids=list(self.values))]


def observation(oid, frame, points=None):
    points = np.zeros((20, 3)) if points is None else points
    # Advanced-indexed RGB supports can arrive in column-major memory layout.
    return FusionObservation(oid, frame, 1, points, np.asfortranarray(np.ones_like(points)*.5), [1., 0.], [1., 0.])


def test_observation_owns_input_arrays_and_rejects_invalid_data():
    points = np.zeros((20, 3))
    obs = observation(0, 0, points)
    points[:] = 9
    assert np.all(obs.points == 0)
    with pytest.raises(ValueError):
        obs.points[0, 0] = 1
    with pytest.raises(ValueError):
        observation(1, 0, np.full((20, 3), np.nan))


def test_causal_history_and_revisions_are_separate_and_export_is_detached():
    state = ReplayableObjectMap(SmallBackend)
    original = np.eye(4)
    state.add_frame(10, [observation(7, 10)], original)
    original[0, 3] = 3
    state.revise_poses({10: original})
    assert state.labels() == {7: 2}
    assert state.causal_labels() == {7: 1}
    assert state.events[0]['changed'] == [dict(observation_id=7, old=1, new=2)]
    state.export_objects()[0]['observation_ids'].clear()
    assert state.export_objects()[0]['observation_ids'] == [7]
    assert state.revise_poses({10: np.eye(4)}) == {7: 1}


def test_invalid_revision_and_duplicate_frames_do_not_change_state():
    state = ReplayableObjectMap(SmallBackend)
    state.add_frame(0, [observation(0, 0)], np.eye(4))
    for poses in ({}, {0: np.zeros((4, 4))}, {0: np.eye(4), 1: np.eye(4)}):
        with pytest.raises(ValueError):
            state.revise_poses(poses)
    with pytest.raises(ValueError):
        state.add_frame(0, [observation(1, 0)], np.eye(4))
    with pytest.raises(ValueError):
        state.add_frame(1, [observation(0, 1)], np.eye(4))
    assert state.labels() == {0: 1}
    assert state.revision == 0


def test_failed_backend_replay_does_not_publish_partial_revision():
    class FailingBackend(SmallBackend):
        def integrate(self, observations, pose):
            if pose[0, 3] > 5:
                raise RuntimeError('backend failure')
            return super().integrate(observations, pose)
    state = ReplayableObjectMap(FailingBackend)
    state.add_frame(0, [observation(0, 0)], np.eye(4))
    bad = np.eye(4)
    bad[0, 3] = 6
    with pytest.raises(RuntimeError):
        state.revise_poses({0: bad})
    assert state.labels() == state.causal_labels() == {0: 1}
    assert state.revision == 0


def test_failed_ingestion_rolls_back_backend_and_allows_retry():
    class FailingBackend(SmallBackend):
        def integrate(self, observations, pose):
            labels = super().integrate(observations, pose)
            if pose[0, 3] > 5:
                raise RuntimeError('partial mutation')
            return labels
    state = ReplayableObjectMap(FailingBackend)
    state.add_frame(0, [observation(0, 0)], np.eye(4))
    bad = np.eye(4)
    bad[0, 3] = 6
    with pytest.raises(RuntimeError):
        state.add_frame(1, [observation(1, 1)], bad)
    assert state.labels() == state.causal_labels() == {0: 1}
    state.add_frame(1, [observation(1, 1)], np.eye(4))
    assert state.labels() == {0: 1, 1: 1}


def test_native_fusion_revision_restores_identity_geometry_and_continues_ingestion():
    pytest.importorskip('conceptgraph')
    from omegaconf import OmegaConf
    from omni_semantic_map.replay_fusion import ConceptGraphsBackend
    cfg = OmegaConf.create(dict(device='cpu', spatial_sim_type='overlap', match_method='sim_sum',
        phys_bias=0., sim_threshold=1.2, min_points_threshold=16, downsample_voxel_size=.025,
        dbscan_remove_noise=True, dbscan_eps=.1, dbscan_min_points=10,
        obj_min_points=0, obj_min_detections=3, merge_overlap_thresh=.7,
        merge_visual_sim_thresh=.7, merge_text_sim_thresh=.7,
        denoise_interval=20, filter_interval=-1, merge_interval=-1))
    factory = lambda: ConceptGraphsBackend(cfg)
    xyz = np.array(np.meshgrid(*([np.linspace(0, .12, 7)]*3))).reshape(3, -1).T
    state, clean = ReplayableObjectMap(factory), ReplayableObjectMap(factory)
    shifted = np.eye(4)
    shifted[0,3] = .7
    before = {}
    for k in range(6):
        obs = observation(k, k, xyz)
        before[k] = np.eye(4) if k < 3 else shifted
        state.add_frame(k, [obs], before[k])
        clean.add_frame(k, [obs], np.eye(4))
    causal, drift = state.causal_labels(), state.labels()
    assert len(set(drift.values())) == 2
    assert state.revise_poses({k: np.eye(4) for k in range(6)}) == clean.labels()
    assert len(set(state.labels().values())) == 1
    np.testing.assert_allclose(state.export_objects()[0]['points'], clean.export_objects()[0]['points'])
    assert state.causal_labels() == causal
    assert state.revise_poses(before) == drift
    # Finalization is a snapshot: a query must not filter/mutate the live state.
    state.add_frame(6, [observation(6, 6, xyz)], shifted)
    assert state.causal_labels()[6] == causal[5]


def test_native_rejections_remain_addressable_and_do_not_modify_numpy_rng():
    pytest.importorskip('conceptgraph')
    from omegaconf import OmegaConf
    from omni_semantic_map.replay_fusion import ConceptGraphsBackend
    cfg = OmegaConf.create(dict(min_points_threshold=16, obj_min_points=0,
        obj_min_detections=3, merge_overlap_thresh=.7))
    np.random.seed(17)
    expected = np.random.normal(size=4)
    np.random.seed(17)
    backend = ConceptGraphsBackend(cfg)
    assert backend.integrate([observation(4, 0, np.zeros((2,3)))], np.eye(4)) == {4: 0}
    assert backend.labels() == {4: 0}
    np.testing.assert_array_equal(np.random.normal(size=4), expected)
