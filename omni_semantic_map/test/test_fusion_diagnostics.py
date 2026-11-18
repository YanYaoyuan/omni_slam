from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from omni_semantic_map.fusion_diagnostics import guarded_merge, membership_audit


def setup_merge(monkeypatch):
    pytest.importorskip('conceptgraph')
    import torch
    from conceptgraph.slam import utils
    from conceptgraph.slam.slam_classes import MapObjectList
    def merge(cfg, destination, source, run_dbscan):
        assert run_dbscan
        result = deepcopy(destination)
        result['observation_ids'] += source['observation_ids']
        return result
    monkeypatch.setattr(utils, 'merge_obj2_into_obj1', merge)
    cfg = SimpleNamespace(merge_overlap_thresh=.7, merge_visual_sim_thresh=.7,
                          merge_text_sim_thresh=.7)
    objects = MapObjectList([dict(observation_ids=[i], clip_ft=torch.tensor([1., 0.]),
                                  text_ft=torch.tensor([1., 0.])) for i in range(3)])
    return cfg, objects, utils


def test_consumed_source_cannot_join_two_retained_objects(monkeypatch):
    cfg, objects, utils = setup_merge(monkeypatch)
    overlap = np.zeros((3, 3)); overlap[0, 1] = .95; overlap[0, 2] = .9
    native = utils.merge_overlap_objects(cfg, deepcopy(objects), overlap)
    assert membership_audit(native)['cross_object_observations'] == 1
    guarded = guarded_merge(cfg, deepcopy(objects), overlap)
    assert [o['observation_ids'] for o in guarded] == [[1, 0], [2]]
    assert membership_audit(guarded)['repeated_memberships'] == 0


def test_chain_merge_conserves_every_source_exactly_once(monkeypatch):
    cfg, objects, utils = setup_merge(monkeypatch)
    overlap = np.zeros((3, 3))
    overlap[0, 1] = .95; overlap[1, 2] = .9; overlap[0, 2] = .85
    assert membership_audit(utils.merge_overlap_objects(cfg, deepcopy(objects), overlap))['repeated_memberships'] == 1
    assert guarded_merge(cfg, objects, overlap)[0]['observation_ids'] == [2, 1, 0]


def test_duplicate_input_fails_before_any_mutation(monkeypatch):
    cfg, objects, _ = setup_merge(monkeypatch)
    objects[2]['observation_ids'] = [0]
    before = deepcopy(objects)
    with pytest.raises(ValueError, match='disjoint'):
        guarded_merge(cfg, objects)
    assert [x['observation_ids'] for x in objects] == [x['observation_ids'] for x in before]


def test_disjoint_valid_native_merge_is_unchanged(monkeypatch):
    cfg, objects, utils = setup_merge(monkeypatch)
    overlap = np.zeros((3, 3)); overlap[0, 1] = .9
    native = utils.merge_overlap_objects(cfg, deepcopy(objects), overlap)
    guarded = guarded_merge(cfg, deepcopy(objects), overlap)
    assert [x['observation_ids'] for x in native] == [x['observation_ids'] for x in guarded]
