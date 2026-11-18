import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from map_quality import (
    surface_spread,
    temporal_events,
    trajectory_poses,
    planar_patches,
)
from revisit_geometry import loop_fit_mask


class MapQualityTest(unittest.TestCase):
    def test_reference_ground_fit_preserves_measured_slope(self):
        rng = np.random.default_rng(9)
        xy = rng.uniform(-2, 2, (1000, 2))
        points = np.column_stack([xy, -0.8 + 0.15 * xy[:, 0]])
        normal = np.array([-0.15, 0, 1.0])
        normal /= np.linalg.norm(normal)
        patches, _ = planar_patches(
            points, np.tile(normal, (len(points), 1)), [0, 0, 1.0]
        )
        ground = [p for p in patches if p["kind"] == "ground"]
        self.assertTrue(ground)
        self.assertGreater(abs(ground[0]["normal"] @ normal), 0.999)
        self.assertGreater(abs(ground[0]["normal"][0]), 0.1)

    def test_whole_scan_split_is_shared_across_modes_at_epoch_times(self):
        epoch = 1779359674.0014
        times = epoch + np.array([0, 1.9, 2, 3.9, 4, 5.9])
        mask = loop_fit_mask(times, epoch, 2)
        np.testing.assert_array_equal(mask, [True, True, False, False, True, True])
        self.assertFalse(np.any(mask & ~mask))
        self.assertTrue(loop_fit_mask(times, epoch).all())

    def test_double_wall_is_not_hidden_by_per_map_refitting(self):
        points = np.linspace(-0.02, 0.02, 200)
        before = surface_spread(points, points + 2.0)
        after = surface_spread(points, points + 0.03)
        self.assertAlmostEqual(before["abs_layer_gap_m"], 2.0)
        self.assertGreater(before["balanced_width90_m"], 2.0)
        self.assertLess(after["balanced_width90_m"], 0.1)
        # Constant global height changes do not count as a map improvement.
        shifted = surface_spread(points + 4, points + 6)
        self.assertAlmostEqual(shifted["abs_layer_gap_m"], 2)

    def test_repeated_edges_form_events_in_both_time_coordinates(self):
        stamps = np.array([0.0, 5.0, 100.0, 105.0, 200.0, 205.0])
        edges = [{"i": 2, "j": 0}, {"i": 3, "j": 1}, {"i": 4, "j": 0}, {"i": 5, "j": 2}]
        events = temporal_events(edges, stamps)
        self.assertEqual(sorted(map(len, events)), [1, 1, 2])

    def test_pose_sampling_does_not_extrapolate(self):
        tum = np.array([[10, 0, 0, 0, 0, 0, 0, 1], [12, 2, 0, 1, 0, 0, 0, 1.0]])
        pose = trajectory_poses(tum, [11])[0]
        np.testing.assert_allclose(pose[:3, 3], [1, 0, 0.5])
        with self.assertRaises(ValueError):
            trajectory_poses(tum, [9])


if __name__ == "__main__":
    unittest.main()
