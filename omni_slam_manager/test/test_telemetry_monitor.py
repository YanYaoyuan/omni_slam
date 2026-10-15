"""Unit tests for the pure SLAM telemetry quality tracking logic."""

import math
import unittest

from omni_slam_manager.telemetry_monitor import (
    FitnessTracker,
    PoseJumpTracker,
    PoseSample,
    PostSaveMotionDetector,
    yaw_from_quaternion,
)


def _pose(x=0.0, y=0.0, z=0.0, yaw=0.0, stamp=0.0) -> PoseSample:
    return PoseSample(x=x, y=y, z=z, yaw=yaw, stamp=stamp)


class YawFromQuaternionTests(unittest.TestCase):
    def test_identity(self):
        self.assertAlmostEqual(yaw_from_quaternion(0.0, 0.0, 0.0, 1.0), 0.0)

    def test_quarter_turn(self):
        # 90 degrees about z: q = (0, 0, sqrt(2)/2, sqrt(2)/2)
        yaw = yaw_from_quaternion(0.0, 0.0, math.sqrt(2) / 2, math.sqrt(2) / 2)
        self.assertAlmostEqual(yaw, math.pi / 2)

    def test_half_turn(self):
        yaw = yaw_from_quaternion(0.0, 0.0, 1.0, 0.0)
        self.assertAlmostEqual(yaw, math.pi)


class PoseJumpTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = PoseJumpTracker(
            position_threshold_m=0.5,
            yaw_threshold_rad=math.radians(45.0),
            refractory_sec=0.5,
        )

    def test_first_sample_is_not_a_jump(self):
        self.assertFalse(self.tracker.observe(_pose(stamp=0.0)))
        self.assertEqual(self.tracker.count, 0)

    def test_small_moves_do_not_count(self):
        self.tracker.observe(_pose(stamp=0.0))
        for index in range(1, 10):
            self.assertFalse(
                self.tracker.observe(_pose(x=index * 0.05, stamp=index * 0.1))
            )
        self.assertEqual(self.tracker.count, 0)

    def test_position_jump_counts_once(self):
        self.tracker.observe(_pose(stamp=0.0))
        self.assertTrue(self.tracker.observe(_pose(x=2.0, stamp=1.0)))
        # same discontinuity on the next frame: refractory suppresses it
        self.assertFalse(self.tracker.observe(_pose(x=2.0, stamp=1.1)))
        # settled at the new position: no further jumps
        self.assertFalse(self.tracker.observe(_pose(x=2.0, stamp=2.0)))
        self.assertEqual(self.tracker.count, 1)

    def test_yaw_jump_counts(self):
        self.tracker.observe(_pose(stamp=0.0))
        self.assertTrue(
            self.tracker.observe(_pose(yaw=math.radians(60.0), stamp=1.0))
        )
        self.assertEqual(self.tracker.count, 1)

    def test_yaw_delta_wraps_across_pi(self):
        # -10 deg to +10 deg is a 20 deg turn, not 340 deg.
        self.tracker.observe(_pose(yaw=math.radians(-10.0), stamp=0.0))
        self.assertFalse(
            self.tracker.observe(_pose(yaw=math.radians(10.0), stamp=1.0))
        )
        self.assertEqual(self.tracker.count, 0)

    def test_second_jump_after_refractory_counts(self):
        self.tracker.observe(_pose(stamp=0.0))
        self.tracker.observe(_pose(x=2.0, stamp=1.0))
        self.assertTrue(self.tracker.observe(_pose(x=5.0, stamp=2.0)))
        self.assertEqual(self.tracker.count, 2)

    def test_non_finite_samples_are_ignored(self):
        self.tracker.observe(_pose(stamp=0.0))
        self.assertFalse(self.tracker.observe(_pose(x=math.nan, stamp=1.0)))
        self.assertEqual(self.tracker.count, 0)
        # the non-finite sample must not have become the new baseline
        self.assertTrue(self.tracker.observe(_pose(x=2.0, stamp=2.0)))
        self.assertEqual(self.tracker.count, 1)

    def test_reset(self):
        self.tracker.observe(_pose(stamp=0.0))
        self.tracker.observe(_pose(x=2.0, stamp=1.0))
        self.tracker.reset()
        self.assertEqual(self.tracker.count, 0)
        # first sample after a reset is a new baseline
        self.assertFalse(self.tracker.observe(_pose(x=3.0, stamp=2.0)))

    def test_invalid_thresholds_raise(self):
        with self.assertRaises(ValueError):
            PoseJumpTracker(position_threshold_m=0.0)
        with self.assertRaises(ValueError):
            PoseJumpTracker(yaw_threshold_rad=0.0)
        with self.assertRaises(ValueError):
            PoseJumpTracker(refractory_sec=-1.0)


class FitnessTrackerTests(unittest.TestCase):
    def test_nan_before_first_sample(self):
        self.assertTrue(math.isnan(FitnessTracker().value))

    def test_updates_on_finite_values(self):
        tracker = FitnessTracker()
        tracker.observe(0.25)
        self.assertEqual(tracker.value, 0.25)
        tracker.observe(0.10)
        self.assertEqual(tracker.value, 0.10)

    def test_ignores_non_finite(self):
        tracker = FitnessTracker()
        tracker.observe(0.25)
        tracker.observe(float("nan"))
        tracker.observe(float("inf"))
        self.assertEqual(tracker.value, 0.25)

    def test_reset_back_to_nan(self):
        tracker = FitnessTracker()
        tracker.observe(0.25)
        tracker.reset()
        self.assertTrue(math.isnan(tracker.value))


if __name__ == "__main__":
    unittest.main()

class PostSaveMotionDetectorTests(unittest.TestCase):
    def test_disarmed_detector_never_reports(self):
        detector = PostSaveMotionDetector()
        self.assertFalse(detector.observe(_pose(x=100.0)))

    def test_stationary_streaming_after_save_stays_clean(self):
        detector = PostSaveMotionDetector(position_threshold_m=0.3)
        detector.arm(_pose())
        for stamp in range(50):
            self.assertFalse(detector.observe(_pose(x=0.01, yaw=0.01, stamp=stamp)))
        self.assertTrue(detector.armed)

    def test_translation_or_rotation_after_save_marks_dirty_once(self):
        detector = PostSaveMotionDetector(
            position_threshold_m=0.3, yaw_threshold_rad=math.radians(10.0)
        )
        detector.arm(_pose())
        self.assertTrue(detector.observe(_pose(x=0.31)))
        self.assertFalse(detector.armed)
        self.assertFalse(detector.observe(_pose(x=5.0)))
        detector.arm(_pose(yaw=math.pi - 0.05))
        self.assertTrue(detector.observe(_pose(yaw=-math.pi + 0.15)))

    def test_missing_or_non_finite_baseline_uses_first_finite_sample(self):
        detector = PostSaveMotionDetector(position_threshold_m=0.3)
        detector.arm(_pose(x=math.nan))
        self.assertFalse(detector.observe(_pose(x=2.0)))
        self.assertFalse(detector.observe(_pose(x=math.inf)))
        self.assertTrue(detector.observe(_pose(x=2.5)))

    def test_invalid_thresholds_are_rejected(self):
        with self.assertRaises(ValueError):
            PostSaveMotionDetector(position_threshold_m=0.0)
        with self.assertRaises(ValueError):
            PostSaveMotionDetector(yaw_threshold_rad=0.0)


class ManagerPostSaveDirtyContractTests(unittest.TestCase):
    """manager_node.py imports rclpy, so check its wiring statically."""

    def setUp(self):
        from pathlib import Path

        self.source = (
            Path(__file__).resolve().parents[1]
            / "omni_slam_manager"
            / "manager_node.py"
        ).read_text(encoding="utf-8")

    def test_save_arms_detector_before_requesting_snapshot(self):
        arm = self.source.index("self._post_save_motion.arm(self._latest_pose)")
        trigger = self.source.index("self._map_save_client.call_async(Trigger.Request())")
        self.assertLess(arm, trigger)

    def test_motion_in_map_ready_restores_dirty_and_mapping(self):
        start = self.source.index("def _odom_callback")
        end = self.source.index("def _initial_pose_callback", start)
        callback = self.source[start:end]
        self.assertIn("SlamState.MAP_READY", callback)
        self.assertIn("self._post_save_motion.observe(sample)", callback)
        self.assertIn("self._mapping_dirty = True", callback)
        self.assertIn("SlamState.MAPPING", callback)
