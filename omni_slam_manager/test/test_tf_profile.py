from pathlib import Path
import tempfile
import unittest

from omni_slam_manager.tf_profile import DEFAULT_FRAMES, load_tf_frames


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class TfProfileTest(unittest.TestCase):
    def test_empty_path_uses_omni_defaults(self):
        self.assertEqual(load_tf_frames(""), DEFAULT_FRAMES)

    def test_loads_manager_owned_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "robot.yaml"
            path.write_text(
                """omni_tf_manager:
  ros__parameters:
    frames.map: omni_test_map
    frames.odom: omni_test_odom
    frames.base: omni_test_base
    frames.tracking: omni_test_imu
    frames.icp_sensor: omni_test_lidar
""",
                encoding="utf-8",
            )
            frames = load_tf_frames(str(path))
            self.assertEqual(frames["tracking"], "omni_test_imu")

    def test_rejects_non_omni_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(
                """omni_tf_manager:
  ros__parameters:
    frames.map: map
    frames.odom: omni_odom
    frames.base: omni_base
    frames.tracking: omni_imu
    frames.icp_sensor: omni_lidar
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "frames.map"):
                load_tf_frames(str(path))

    def test_manager_launch_selects_robot_specific_tf_profiles(self):
        source = (PACKAGE_ROOT / "launch" / "manager.launch.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            '"omni_dog": tf_manager_share / "config" / "omni_dog.yaml"', source
        )
        self.assertIn(
            '"vbot": tf_manager_share / "config" / "omni_vbot_dog.yaml"', source
        )
        self.assertIn(
            '"matrix": tf_manager_share / "config" / "matrix_xgw.yaml"', source
        )
        self.assertIn('"robot_profile": robot_profile', source)
        self.assertIn("requested_tf_profile or", source)


if __name__ == "__main__":
    unittest.main()
