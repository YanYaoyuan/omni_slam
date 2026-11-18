import numpy as np
import pytest
import torch
from omni_semantic_map.persistent_source_memory import (
    PersistentSourceMemory, SourceFeatureLedger, sample_patch_features,
    sample_sources, measured_relations, retrieve_context)
from omni_semantic_map.joint_object_memory import source_instance_loss


def test_readout_is_chunk_invariant_and_training_reaches_encoder_and_persistent_state():
    torch.manual_seed(4); torch.set_num_threads(1)
    m = PersistentSourceMemory(8, 16, 2, 1, 4)
    x = torch.randn(10, 8); frame = torch.tensor([0]*5+[1]*5)
    relation = torch.zeros(10, 10, dtype=torch.int8)
    first = m(x, frame, relation)
    second = m(x+1, frame, relation, memory=first['memory'])
    q = torch.randn(13, 8)
    logits = m.decode_sources(q, second['memory'])
    chunks = torch.cat([m.decode_sources(v, second['memory']) for v in q.split(4)])
    torch.testing.assert_close(logits, chunks, atol=1e-5, rtol=1e-5)
    target = torch.tensor([8]*4+[9]*4+[0]*4+[-1])
    loss = source_instance_loss(logits, target, second['objectness'])['loss']
    loss.backward()
    for name in ['project.weight', 'frame_attention.0.in_proj_weight',
                 'global_attention.0.in_proj_weight', 'slot_read.in_proj_weight',
                 'token_read.in_proj_weight', 'new_slots']:
        assert dict(m.named_parameters())[name].grad.abs().sum() > 0, name
    assert not torch.allclose(m.decode_sources(q, first['memory']), logits)


def test_pixel_center_sampling_matches_torch_and_rejects_outside():
    grid = np.arange(3*5*4, dtype=np.float32).reshape(3, 5, 4)
    xy = np.array([[0, 0], [12, 8], [6, 4], [4, 1]])
    expected = torch.nn.functional.grid_sample(
        torch.tensor(grid.transpose(2, 0, 1)[None]),
        torch.tensor(((xy+.5)/[13, 9]*2-1).astype('float32'))[None, None],
        align_corners=False, padding_mode='border')[0, :, 0].T.numpy()
    np.testing.assert_allclose(sample_patch_features(grid, xy, (9, 13)), expected, atol=1e-5)
    with pytest.raises(ValueError):sample_patch_features(grid, [[13, 0]], (9, 13))


def test_ledger_is_causal_and_failure_atomic_and_snapshot_immutable():
    ledger = SourceFeatureLedger(np.zeros((4, 3)))
    p = np.zeros((2, 19)); p[:, 2] = 1
    grids = [np.ones((2, 2, 4)), np.full((2, 2, 4), 3.)]
    ledger.observe(0, [0, 1], [[0, 0], [1, 1]], p)
    before = ledger.packet([0], grids, (2, 2))
    with pytest.raises(ValueError):ledger.packet([2], grids, (2, 2))
    with pytest.raises(ValueError):ledger.observe(2, [0, 2], [[0, 0], [1, 1]], p)
    assert ledger.step == 0 and ledger.count[2] == 0
    p[:, 2] = 0; p[:, 13] = 1
    ledger.observe(1, [0, 2], [[1, 0], [0, 1]], p)
    assert before['frame_ids'].tolist() == [0]
    assert ledger.latest.tolist() == [1, 0, 1, -1]
    assert ledger.probability[0, 2] == ledger.probability[0, 13] == .5
    assert ledger.probability[2, 13] == 1


def test_relation_distinguishes_repeat_depth_conflict_and_unknown():
    xyz = np.array([[0, 0, 2.], [0, 0, 4.], [1, 0, 2.]])
    packet = dict(source_ids=np.array([0, 0, 1, 2]), frame_ids=np.array([0, 1, 1, 1]),
                  pixel_xy=np.array([[0, 0], [0, 0], [0, 0], [5, 0]]))
    rel = measured_relations(packet, xyz, [np.eye(4)]*2,
                            dict(fx=10, fy=10, cx=0, cy=0))
    assert rel[0, 1] == 1 and rel[0, 2] == -1 and rel[0, 3] == 0
    np.testing.assert_array_equal(rel, rel.T)
    assert not np.any(np.diag(rel))


def test_semantic_sampler_balances_without_duplicate_or_label_inputs():
    p = np.zeros((24, 19)); p[:8, 2] = 1; p[8:16, 13] = 1; p[16:, 0] = 1
    ids = sample_sources(np.arange(24), p, 9, 4)
    assert len(np.unique(ids)) == 9
    assert [int((p[ids].argmax(1) == k).sum()) for k in [2, 13, 0]] == [3, 3, 3]
    np.testing.assert_array_equal(ids, sample_sources(np.arange(24), p, 9, 4))


def test_exact_reobservation_retrieval_keeps_unseen_history_without_gt():
    previous = np.arange(20); current = np.arange(10,30)
    p = np.zeros((20,19)); p[:,2] = 1
    old, now = retrieve_context(previous, current, p, p, 8, 17)
    np.testing.assert_array_equal(old[:4], now[:4])
    assert np.isin(old[:4], np.arange(10,20)).all()
    assert len(np.unique(old)) == len(np.unique(now)) == 8
    assert np.isin(old, previous).all() and np.isin(now, current).all()
    empty = np.empty((0,19))
    old, now = retrieve_context(np.array([],int), current, empty, p, 8, 17)
    assert len(old) == 0 and len(now) == 8
