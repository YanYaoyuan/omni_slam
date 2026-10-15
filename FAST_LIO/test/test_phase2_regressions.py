"""Static regressions for the Phase 2 SLAM telemetry and cancel contract.

Like the Phase 0 tests, these check source contracts without importing ROS
so they run in lightweight CI and through ``colcon test``.
"""

from pathlib import Path
import re
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent


def _manager_source() -> str:
    return (
        REPO_ROOT
        / "omni_slam_manager"
        / "omni_slam_manager"
        / "manager_node.py"
    ).read_text(encoding="utf-8")


def _icp_source() -> str:
    return (
        REPO_ROOT / "icp_relocalization" / "src" / "icp_node.cpp"
    ).read_text(encoding="utf-8")


class SaveMapCancelRegressionTests(unittest.TestCase):
    def test_save_map_accepts_cancel(self):
        source = _manager_source()
        self.assertNotIn("cancel_callback=lambda _: CancelResponse.REJECT", source)
        save_server = source[source.index("_save_action = ActionServer("):]
        save_server = save_server[: save_server.index(")") + 1]
        self.assertIn("CancelResponse.ACCEPT", save_server)

    def test_save_map_honors_cancel_in_execute(self):
        source = _manager_source()
        execute = source[source.index("def _execute_save_map("):]
        execute = execute[: execute.index("\n    def ")]
        self.assertRegex(execute, r"goal_handle\.is_cancel_requested")
        self.assertRegex(execute, r"wait_with_cancel\(")
        self.assertRegex(execute, r"_cancel_save\(goal_handle, result\)")

    def test_save_map_wait_gets_callable_cancel_flag(self):
        # rclpy's is_cancel_requested is a property: the wait must receive
        # a zero-arg callable, never the bare attribute (a bool).
        source = _manager_source()
        execute = source[source.index("def _execute_save_map("):]
        execute = execute[: execute.index("\n    def ")]
        self.assertRegex(
            execute, r"lambda:\s*goal_handle\.is_cancel_requested"
        )
        # The original bug: passing the property value directly.
        self.assertNotRegex(
            execute, r"future\.done,\s*goal_handle\.is_cancel_requested\b"
        )

    def test_save_cancel_rolls_back_to_mapping(self):
        source = _manager_source()
        self.assertRegex(source, r"REASON_SAVE_CANCELLED\s*=\s*1104")
        cancel = source[source.index("def _cancel_save("):]
        cancel = cancel[: cancel.index("\n    def ")]
        self.assertRegex(cancel, r"SlamState\.MAPPING")
        self.assertRegex(cancel, r"goal_handle\.canceled\(\)")
        self.assertNotIn("import_pcd", cancel)


class TelemetryWiringRegressionTests(unittest.TestCase):
    def test_sensor_telemetry_uses_compatible_qos(self):
        source = _manager_source()
        self.assertIn("qos_profile_sensor_data", source)
        subscription_block = source[
            source.index("telemetry_qos = qos_profile_sensor_data"):
            source.index("self._map_save_client = self.create_client(")
        ]
        self.assertEqual(subscription_block.count("telemetry_qos"), 4)

    def test_status_reports_real_pose_jump_and_fitness(self):
        source = _manager_source()
        publish = source[source.index("def _publish_status("):]
        publish = publish[: publish.index("\n    def ")]
        self.assertRegex(publish, r"message\.pose_jump_count\s*=\s*self\._pose_jumps\.count")
        self.assertRegex(publish, r"message\.fitness_score\s*=\s*float\(self\._fitness\.value\)")
        self.assertNotRegex(publish, r"pose_jump_count\s*=\s*0\b")
        self.assertNotRegex(publish, r"fitness_score\s*=\s*math\.nan")

    def test_odom_callback_feeds_jump_tracker(self):
        source = _manager_source()
        odom = source[source.index("def _odom_callback("):]
        odom = odom[: odom.index("\n    def ")]
        self.assertRegex(odom, r"self\._pose_jumps\.observe\(")
        self.assertRegex(odom, r"yaw_from_quaternion\(")

    def test_icp_fitness_subscription_exists(self):
        source = _manager_source()
        self.assertRegex(source, r"icp_fitness_topic")
        self.assertRegex(source, r"self\._fitness\.observe\(float\(message\.data\)\)")

    def test_localization_feedback_carries_fitness(self):
        source = _manager_source()
        execute = source[source.index("def _execute_localization("):]
        execute = execute[: execute.index("\n    def ")]
        self.assertRegex(execute, r"feedback\.fitness_score\s*=\s*float\(fitness\)")
        self.assertNotRegex(execute, r"feedback\.fitness_score\s*=\s*math\.nan")

    def test_session_reset_clears_telemetry_trackers(self):
        source = _manager_source()
        reset = source[source.index("def _reset_telemetry("):]
        reset = reset[: reset.index("\n    def ")]
        self.assertIn("self._pose_jumps.reset()", reset)
        self.assertIn("self._fitness.reset()", reset)


class IcpFitnessPublisherRegressionTests(unittest.TestCase):
    def test_icp_node_publishes_fitness_on_every_computation(self):
        icp = REPO_ROOT / "icp_relocalization" / "src" / "icp_node.cpp"
        if not icp.is_file():
            self.skipTest("icp_relocalization source not in this layout")
        source = _icp_source()
        self.assertRegex(
            source, r"create_publisher<std_msgs::msg::Float64>\(icp_fitness_topic,\s*10\)"
        )
        self.assertRegex(source, r"this->declare_parameter\(\"icp_fitness_topic\",")
        # both scan callbacks (pointcloud2 + livox) compute a fitness and
        # each computation must be followed by a publish
        computations = source.count("icp.getFitnessScore(")
        self.assertEqual(2, computations)
        self.assertEqual(computations, source.count("fitness_pub_->publish("))
        for match in re.finditer(
            r"icp\.getFitnessScore\([^)]*\);", source
        ):
            tail = source[match.end(): match.end() + 400]
            self.assertRegex(
                tail,
                r"fitness_msg\.data\s*=\s*fitness_score;\s*fitness_pub_->publish\(fitness_msg\);",
            )

    def test_icp_requires_one_consistent_pose_cluster(self):
        source = _icp_source()
        self.assertIn("RelocalizationConvergenceGate", source)
        self.assertIn("convergence_gate_->Observe(quality_accepted, candidate)", source)
        self.assertIn("convergence_gate_->Estimate()", source)
        self.assertIn("initGuess = candidate", source)
        self.assertNotIn("converged_count++", source)

        launch_source = (
            PACKAGE_ROOT / "launch" / "omni_dog_relocalization.launch.py"
        ).read_text(encoding="utf-8")
        for parameter in (
            "candidate_position_tolerance_m",
            "candidate_angle_tolerance_rad",
        ):
            self.assertIn(f"DeclareLaunchArgument(\n        '{parameter}'", launch_source)
            self.assertIn(f"'{parameter}': ParameterValue(", launch_source)


class StateMachineCancelPathRegressionTests(unittest.TestCase):
    def test_saving_can_rollback_to_mapping(self):
        state_machine = (
            REPO_ROOT
            / "omni_slam_manager"
            / "omni_slam_manager"
            / "state_machine.py"
        ).read_text(encoding="utf-8")
        saving_block = state_machine[state_machine.index(
            "SlamState.SAVING: frozenset("
        ):]
        saving_block = saving_block[: saving_block.index(")")]
        self.assertIn("SlamState.MAPPING", saving_block)


if __name__ == "__main__":
    unittest.main()


class IcpResultQualityRegressionTests(unittest.TestCase):
    """SLAM-03: ICP must not claim perfect confidence or mislabel frames."""

    def setUp(self):
        icp = REPO_ROOT / "icp_relocalization" / "src" / "icp_node.cpp"
        if not icp.is_file():
            self.skipTest("icp_relocalization source not in this layout")
        self.source = _icp_source()

    def test_result_publishes_cluster_covariance(self):
        start = self.source.index("void publish_relocalization_result(")
        end = self.source.index("publisher_->publish(pose_msg);", start)
        publish = self.source[start:end]
        self.assertIn("convergence_gate_->Covariance(fitness_score)", publish)
        self.assertIn("pose_msg.pose.covariance.begin()", publish)

    def test_pointcloud_input_frame_must_match_result_frame(self):
        start = self.source.index("void cloud_callback(")
        end = self.source.index("pcl::fromROSMsg(*msg, *input_cloud);", start)
        guard = self.source[start:end]
        self.assertIn("msg->header.frame_id != sensor_frame", guard)
        self.assertIn("convergence_gate_->Reset();", guard)
        self.assertIn("return;", guard)
