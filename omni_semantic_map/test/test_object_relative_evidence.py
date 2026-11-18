import numpy as np
import pytest
from omni_semantic_map.object_relative_evidence import ObjectReferenceRanks


def test_leave_whole_cell_out_and_constant_shape():
    xyz = np.array([[.01, 0, 0], [.02, 0, 0], [.21, 0, 0], [.41, 0, 0]])
    f = np.array([[0., 7.], [1., 7.], [3., 7.], [5., 7.]])
    bank = ObjectReferenceRanks().fit(xyz, np.ones(4), f)
    # Query=2 in first cell: reference is 3,5, not the other point in that cell.
    np.testing.assert_array_equal(bank.transform([[.03, 0, 0]], [1], [[2, 7]]), [[-1, 0]])
    # Unseen cell sees representative1,3,5; 3 has centred midpoint rank0.
    np.testing.assert_array_equal(bank.transform([[.61, 0, 0]], [1], [[3, 7]]), [[0, 0]])


def test_reference_permutation_and_identical_density_duplicates():
    xyz = np.array([[.01, 0, 0], [.08, 0, 0], [.21, 0, 0], [.41, 0, 0]])
    f = np.array([[0.], [1.], [3.], [5.]])
    a = ObjectReferenceRanks().fit(xyz, np.ones(4), f)
    ix = np.array([3, 1, 1, 0, 2, 2, 2])
    b = ObjectReferenceRanks().fit(xyz[ix], np.ones(len(ix)), f[ix])
    np.testing.assert_array_equal(a.transform(xyz, np.ones(4), f), b.transform(xyz, np.ones(4), f))


def test_object_specific_monotone_offsets_and_identity_equivariance():
    xyz = np.array([[.01, 0, 0], [.21, 0, 0], [.41, 0, 0]] * 2)
    owners = np.array([1, 1, 1, 2, 2, 2]); f = np.array([[0., 3], [2, 3], [4, 3]] * 2)
    changed = f.copy(); changed[owners == 1] = changed[owners == 1] * 2 + 100
    changed[owners == 2] = changed[owners == 2] * 3 - 40
    a = ObjectReferenceRanks().fit(xyz, owners, f)
    b = ObjectReferenceRanks().fit(xyz, owners + 10, changed)
    np.testing.assert_array_equal(a.transform(xyz, owners, f), b.transform(xyz, owners + 10, changed))


def test_missing_support_is_explicit_and_invalid_fit_atomic():
    bank = ObjectReferenceRanks().fit([[0, 0, 0], [1, 0, 0]], [1, 1], [[0], [1]])
    with pytest.raises(ValueError): bank.fit([[0, 0, 0]], [2], [[3]])
    assert set(bank.references) == {1}
    with pytest.raises(ValueError): bank.transform([[0, 0, 0]], [2], [[3]])
    with pytest.raises(ValueError): bank.transform([[0, 0, 0]], [1], [[np.nan]])
