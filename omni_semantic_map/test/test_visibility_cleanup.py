import unittest
import numpy as np
from omni_semantic_map.visibility_cleanup import (
    CleanupConfig, frame_evidence, project_visibility, rejection_mask, statistical_outliers,
)


class VisibilityCleanupTests(unittest.TestCase):
    def setUp(self):
        self.cfg = CleanupConfig(boundary_pixels=0)
        self.cam = dict(H=8, W=8, fx=1., fy=1., cx=0., cy=0.)
        self.depth = np.ones((8, 8), np.float32)

    def test_visibility_abstains_for_occlusion_missing_and_outside(self):
        self.depth[3, 3] = 0
        xyz = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 1], [-1, 0, 1], [0, 0, -1.]])
        _, _, fov, visible = project_visibility(xyz, np.eye(4), self.cam, self.depth, self.cfg)
        np.testing.assert_array_equal(fov, [1, 1, 1, 0, 0])
        np.testing.assert_array_equal(visible, [1, 0, 0, 0, 0])

    def test_camera_to_world_translation(self):
        pose = np.eye(4); pose[:3, 3] = [10, 20, 30]
        x, y, _, visible = project_visibility(np.array([[11, 22, 31.]]), pose, self.cam, self.depth, self.cfg)
        self.assertEqual((x[0], y[0], bool(visible[0])), (1, 2, True))

    def test_unknown_pixels_and_absent_object_abstain(self):
        masks = np.zeros((2, 8, 8), bool); masks[0, 1, 1] = 1; masks[1, 2, 2] = 1
        xyz = np.array([[1, 1, 1], [2, 2, 1], [3, 3, 1], [2, 2, 1.]])
        p, n, *_ = frame_evidence(xyz, np.array([1, 1, 1, 3]), np.eye(4), self.cam, self.depth, masks, [1, 2], self.cfg)
        np.testing.assert_array_equal(p, [1, 0, 0, 0]); np.testing.assert_array_equal(n, [0, 1, 0, 0])

    def test_duplicate_masks_cannot_multiply_votes(self):
        masks = np.zeros((2, 8, 8), bool); masks[:, 1, 1] = 1
        p, n, *_ = frame_evidence(np.array([[1, 1, 1.]]), np.array([1]), np.eye(4), self.cam, self.depth, masks, [1, 1], self.cfg)
        self.assertEqual(int(p.sum()), 1); self.assertFalse(n.any())

    def test_boundary_is_unknown(self):
        masks = np.zeros((2, 8, 8), bool); masks[0, 2:6, 2:6] = 1; masks[1] = 1
        p, n, *_ = frame_evidence(np.array([[2, 3, 1.], [3, 3, 1.], [1, 3, 1.], [0, 3, 1.]]), np.ones(4, int), np.eye(4), self.cam, self.depth, masks, [1, 2], CleanupConfig(boundary_pixels=1))
        np.testing.assert_array_equal(p, [0, 1, 0, 0]); np.testing.assert_array_equal(n, [0, 0, 0, 1])

    def test_repeated_refutation_and_single_frame_ablation(self):
        p = np.array([0, 0, 1, 2, 0]); n = np.array([0, 1, 3, 3, 3])
        np.testing.assert_array_equal(rejection_mask(p, n, self.cfg), [0, 0, 1, 0, 1])
        np.testing.assert_array_equal(rejection_mask(p, n, self.cfg, single_frame=True), [0, 1, 1, 0, 1])

    def test_geometric_control_and_identity_independence(self):
        xyz = np.r_[np.zeros((30, 3)), [[100., 100., 100.]]]
        rejected = statistical_outliers(xyz, np.ones(31, int), self.cfg)
        self.assertEqual(np.flatnonzero(rejected).tolist(), [30])


if __name__ == '__main__':
    unittest.main()
