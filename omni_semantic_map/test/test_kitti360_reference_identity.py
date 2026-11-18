from types import SimpleNamespace
import numpy as np
import pytest
from omni_semantic_map.kitti360_reference_identity import encode_reference, MODES


TAX = {k: SimpleNamespace(trainId=t, name=n, ignoreInInst=i) for k, t, n, i in
       [(0, 255, 'unlabeled', True), (7, 0, 'road', False),
        (11, 2, 'building', False), (34, 2, 'garage', True), (26, 13, 'car', False)]}


def test_alias_collision_is_reversible_and_canonical_ids_preserved():
    s = np.array([34, 11, 26, 11, 34, 0, 7])
    ids = np.array([34013, 11013, 26013, 11000, 34000, 0, 7000])
    old, _ = encode_reference(s, ids, TAX)
    np.testing.assert_array_equal(old, [2013, 2013, 13013, 2000, 2000, 0, 0])
    unique, mapping = encode_reference(s, ids, TAX, 'raw_identity')
    np.testing.assert_array_equal(unique[[1, 2, 3]], old[[1, 2, 3]])
    assert len(set(unique[:5])) == 5
    for row in mapping:
        np.testing.assert_array_equal(unique[(s == row['raw_semantic']) &
                                            (ids == row['raw_instance'])], row['encoded_id'])
    perm = np.array([5, 2, 4, 1, 0, 6, 3])
    shuffled, _ = encode_reference(s[perm], ids[perm], TAX, 'raw_identity')
    np.testing.assert_array_equal(shuffled, unique[perm])


def test_target_only_marks_alias_void_without_deleting_positions():
    s, ids = np.array([11, 34, 26, 0]), np.array([11013, 34013, 26013, 0])
    out, mapping = encode_reference(s, ids, TAX, 'target_only')
    np.testing.assert_array_equal(out, [2013, 0, 13013, 0])
    assert len(out) == len(s)
    assert next(v for v in mapping if v['raw_semantic'] == 34)['ignore_in_instance']


def test_empty_and_no_collision_are_unchanged():
    for mode in MODES:
        out, mapping = encode_reference(np.array([], int), np.array([], int), TAX, mode)
        assert out.shape == (0,) and mapping == []
    s, ids = np.array([11, 26]), np.array([11013, 26013])
    for mode in MODES:
        np.testing.assert_array_equal(encode_reference(s, ids, TAX, mode)[0], [2013, 13013])


def test_invalid_schema_rejected():
    cases = [([11], [34013], 'upstream_train'), ([11], [11013], 'unknown'),
             ([99], [99001], 'target_only'), ([11], [11001, 11002], 'raw_identity')]
    for s, ids, mode in cases:
        with pytest.raises(ValueError):
            encode_reference(np.array(s), np.array(ids), TAX, mode)
    with pytest.raises(ValueError):
        encode_reference(np.array([11.]), np.array([11001]), TAX)
