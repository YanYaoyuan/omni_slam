import importlib.util
import unittest


ROS_AVAILABLE = importlib.util.find_spec("rclpy") is not None


@unittest.skipUnless(ROS_AVAILABLE, "ROS 2 Python runtime is not installed")
class RosContractTests(unittest.TestCase):
    def test_generated_interfaces_and_manager_entrypoint_import(self):
        from omni_slam_interfaces.action import SaveMap, StartLocalization
        from omni_robot_interfaces.srv import ListMaps
        from omni_slam_interfaces.srv import StartMapping, StopLocalization, StopMapping
        from omni_slam_manager.manager_node import ROBOT_PROFILES, SlamManagerNode, main
        from omni_tf_manager.msg import SlamStatus

        self.assertEqual(SlamStatus.STATE_LOCALIZED, 7)
        self.assertEqual(SaveMap.Feedback.STAGE_VERIFYING_CHECKSUM, 3)
        localization_goal = StartLocalization.Goal()
        self.assertEqual(localization_goal.initial_x, 0.0)
        self.assertEqual(localization_goal.initial_y, 0.0)
        self.assertEqual(localization_goal.initial_z, 0.0)
        self.assertEqual(localization_goal.initial_yaw, 0.0)
        self.assertIsNotNone(StartMapping.Request())
        self.assertIsNotNone(StopMapping.Request())
        self.assertIsNotNone(StopLocalization.Request())
        self.assertEqual(
            set(ListMaps.Response.get_fields_and_field_types()),
            {"map_ids", "map_versions", "map_checksums", "created_at", "size_bytes"},
        )
        self.assertIn("vbot.launch.py", ROBOT_PROFILES["vbot"]["mapping_command"])
        self.assertIn(
            "vbot_relocalization.launch.py",
            ROBOT_PROFILES["vbot"]["localization_command"],
        )
        self.assertEqual(
            ROBOT_PROFILES["vbot"]["imu_topic"], "/omni/sensors/imu/data"
        )
        self.assertEqual(
            ROBOT_PROFILES["omni_dog"]["imu_topic"],
            "/omni/sensors/imu/data",
        )
        self.assertIn(
            "matrix_sim.yaml", ROBOT_PROFILES["matrix"]["mapping_config"]
        )
        self.assertIn(
            "config_file:={mapping_config_file}",
            ROBOT_PROFILES["matrix"]["mapping_command"],
        )
        self.assertIn(
            "config_file:={localization_config_file}",
            ROBOT_PROFILES["matrix"]["localization_command"],
        )
        self.assertIn(
            "ros2 launch fast_lio omni_dog.launch.py",
            ROBOT_PROFILES["matrix"]["mapping_command"],
        )
        self.assertIn(
            "ros2 launch fast_lio omni_dog_relocalization.launch.py",
            ROBOT_PROFILES["matrix"]["localization_command"],
        )
        self.assertNotIn(
            "external_runtime", ROBOT_PROFILES["matrix"]["mapping_command"]
        )
        self.assertNotIn(
            "run_matrix_", ROBOT_PROFILES["matrix"]["localization_command"]
        )
        self.assertEqual(
            ROBOT_PROFILES["matrix"]["imu_topic"],
            "/omni/sensors/imu/data",
        )
        self.assertTrue(callable(main))
        self.assertTrue(issubclass(SlamManagerNode, object))


if __name__ == "__main__":
    unittest.main()
