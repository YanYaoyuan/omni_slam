import importlib.util
import unittest


ROS_AVAILABLE = importlib.util.find_spec("rclpy") is not None


@unittest.skipUnless(ROS_AVAILABLE, "ROS 2 Python runtime is not installed")
class RosContractTests(unittest.TestCase):
    def test_generated_interfaces_and_manager_entrypoint_import(self):
        from omni_slam_interfaces.action import SaveMap, StartLocalization
        from omni_slam_interfaces.msg import SlamStatus
        from omni_slam_interfaces.srv import StartMapping, StopLocalization, StopMapping
        from omni_slam_manager.manager_node import ROBOT_PROFILES, SlamManagerNode, main

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
        self.assertTrue(callable(main))
        self.assertTrue(issubclass(SlamManagerNode, object))


if __name__ == "__main__":
    unittest.main()
