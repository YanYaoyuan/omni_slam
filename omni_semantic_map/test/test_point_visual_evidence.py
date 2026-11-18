import numpy as np
import pytest
from omni_semantic_map.point_visual_evidence import patch_indices, similarity_matrix, view_evidence, summarize_evidence


def test_same_patch_cannot_confirm_its_own_identity():
    sim = np.eye(3)
    a, b, valid, competing = view_evidence(sim, [0], [0, 0], [])
    assert not valid[0] and not competing[0] and a[0] == b[0] == 0


def test_shared_anchors_and_density_cannot_create_support():
    sim = similarity_matrix([[1, 0], [.8, .6], [0, 1], [-1, 0]])
    a = view_evidence(sim, [0, 1], [0, 1, 2], [2, 3])
    b = view_evidence(sim, [0, 1], [0, 0, 1, 2, 2], [2, 3, 3])
    for x, y in zip(a, b): np.testing.assert_array_equal(x, y)
    np.testing.assert_allclose(a[0], [.8, .8])
    np.testing.assert_allclose(a[1], [-1, -.8])


def test_multiple_views_are_frame_balanced_and_first_means_first():
    args = ([0, 0, 1], [12, 5, 9], [.9, .3, .7], [.2, .5, 0.], [True, True, False], [3, 1, 0])
    full = summarize_evidence(*args)
    first = summarize_evidence(*args, first_only=True)
    np.testing.assert_allclose(full[0, 1:5], [.6, .3, .9, .3], atol=1e-6)
    np.testing.assert_allclose(full[0, 6:10], [.25, -.2, .7, .5], atol=1e-6)
    assert first[0, 1] == np.float32(.3)
    assert full[1, 10] == 0 and np.all(full[2] == 0)
    with pytest.raises(ValueError, match='twice'):
        summarize_evidence([0, 0], [5, 5], [.9, .3], [0., 0.], [False, False], [2])


def test_pixel_centers_and_image_bounds():
    np.testing.assert_array_equal(patch_indices([[0, 0], [13, 13], [14, 0], [27, 27]], 28, 28, 2, 2), [0, 0, 1, 3])
    with pytest.raises(ValueError): patch_indices([[28, 0]], 28, 28, 2, 2)
