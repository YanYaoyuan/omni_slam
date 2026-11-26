"""Static regressions for safety-critical Phase 0 SLAM fixes.

These tests deliberately avoid importing ROS 2 launch modules so they can run in
lightweight CI and developer environments as well as through ``colcon test``.
"""

import ast
from pathlib import Path
import re
import tempfile
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _function_body(source: str, signature_pattern: str) -> str:
    """Return a C++ function body using brace matching."""

    match = re.search(signature_pattern, source)
    assert match is not None, f"Function not found: {signature_pattern}"
    start = source.find("{", match.end())
    assert start >= 0, f"Function body not found: {signature_pattern}"

    depth = 0
    for index in range(start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start + 1:index]
    raise AssertionError(f"Unterminated function body: {signature_pattern}")


class Phase0RegressionTests(unittest.TestCase):

    def test_map_save_service_propagates_writer_result(self):
        source = (PACKAGE_ROOT / "src" / "laserMapping.cpp").read_text(
            encoding="utf-8"
        )

        save_wrapper = _function_body(source, r"\bbool\s+save_to_pcd\s*\(\s*\)")
        self.assertIn("!mapping_backend_", save_wrapper)
        self.assertRegex(save_wrapper, r"saved\s*=\s*mapping_backend_->save\(")
        self.assertRegex(save_wrapper, r"return\s+saved\s*;")

        callback = _function_body(source, r"\bvoid\s+map_save_callback\s*\(")
        self.assertRegex(
            callback, r"res->success\s*=\s*save_to_pcd\s*\(\s*\)\s*;"
        )
        self.assertNotRegex(callback, r"res->success\s*=\s*true\s*;")

    def test_pcd_writer_rejects_empty_cloud_and_uses_atomic_replace(self):
        source = (PACKAGE_ROOT / "src" / "laserMapping.cpp").read_text(
            encoding="utf-8"
        )
        writer = _function_body(source, r"\bstatic\s+bool\s+save_cloud_to_pcd\s*\(")

        self.assertIn("!cloud || cloud->empty()", writer)
        self.assertIn("fast_lio::atomic_replace_file", writer)
        self.assertRegex(
            writer,
            r"pcd_writer\.writeBinary\(temporary_path,\s*\*cloud\)\s*==\s*0",
        )
        self.assertNotRegex(writer, r"writeBinary\(\s*path\s*,")

    def test_fast_lio_is_pose_only_under_omni_tf_authority(self):
        source = (PACKAGE_ROOT / "src" / "laserMapping.cpp").read_text(
            encoding="utf-8"
        )
        publish_odometry = _function_body(
            source, r"\bvoid\s+publish_odometry\s*\("
        )
        self.assertIn('Node("omni_fast_lio"', source)
        self.assertIn('declare_parameter<bool>("publish.tf_en", false)', source)
        self.assertIn("if (publish_tf)", publish_odometry)
        self.assertIn("tf_broadcaster_->sendTransform", publish_odometry)

    def test_legacy_relocalization_launch_starts_icp_only_in_delay_group(self):
        launch_file = PACKAGE_ROOT / "launch" / "relocalization.launch.py"
        source = launch_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(launch_file))

        direct_icp_actions = []
        timer_icp_actions = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_action"
                and len(node.args) == 1
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "icp_node"
            ):
                direct_icp_actions.append(node)

            if isinstance(node.func, ast.Name) and node.func.id == "TimerAction":
                actions = next(
                    (
                        keyword.value
                        for keyword in node.keywords
                        if keyword.arg == "actions"
                    ),
                    None,
                )
                if isinstance(actions, ast.List):
                    timer_icp_actions.extend(
                        item
                        for item in actions.elts
                        if isinstance(item, ast.Name) and item.id == "icp_node"
                    )

        self.assertFalse(
            direct_icp_actions, "icp_node must not also be added directly"
        )
        self.assertEqual(
            len(timer_icp_actions), 1, "icp_node must appear once in TimerAction"
        )
        self.assertNotIn("/home/sentry_ws", source)
        self.assertIn("{'map_path':map_path}", source)
        self.assertIn("{'prior_map_path':map_path}", source)
        self.assertIn("OpaqueFunction(function=_validate_relocalization_inputs)", source)
        self.assertLess(
            source.index("OpaqueFunction(function=_validate_relocalization_inputs)"),
            source.index("ld.add_action(start_rviz)"),
        )
        self.assertNotIn("transform_publisher", source)

    def test_product_relocalization_validates_map_before_starting_nodes(self):
        launch_file = PACKAGE_ROOT / "launch" / "omni_dog_relocalization.launch.py"
        source = launch_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(launch_file))

        validator_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_validate_prior_map_path"
        )
        validator_module = ast.Module(
            body=[ast.Import(names=[ast.alias(name="os")]), validator_node],
            type_ignores=[],
        )
        ast.fix_missing_locations(validator_module)
        namespace = {}
        exec(compile(validator_module, str(launch_file), "exec"), namespace)
        validate = namespace["_validate_prior_map_path"]

        for invalid in ("", "relative/map.pcd", "/definitely/missing/map.pcd"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(RuntimeError):
                    validate(invalid)

        with tempfile.TemporaryDirectory() as directory:
            empty_map = Path(directory) / "empty.pcd"
            empty_map.touch()
            with self.assertRaises(RuntimeError):
                validate(str(empty_map))

            valid_map = Path(directory) / "map.pcd"
            valid_map.write_text("VERSION .7\nPOINTS 1\nDATA ascii\n0 0 0 1\n", encoding="ascii")
            self.assertEqual(validate(str(valid_map)), str(valid_map.resolve()))

        function_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "OpaqueFunction"
        ]
        self.assertEqual(len(function_calls), 1)
        self.assertLess(source.index("OpaqueFunction(function=_validate_relocalization_inputs)"),
                        source.index("ld.add_action(global_pointcloud_node)"))
        self.assertNotIn("map_odom_trans", source)
        self.assertNotIn("global_odometry_node", source)

    def test_vbot_relocalization_uses_vanjee_topics_and_frames(self):
        config = (PACKAGE_ROOT / "config" / "vbot_relocalization.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("locate_in_prior_map: true", config)
        self.assertIn('odom_frame_id: "omni_odom"', config)
        self.assertIn('sensor_frame_id: "omni_imu_link"', config)
        self.assertIn('base_frame_id: "omni_base_link"', config)
        self.assertIn("tf_en: false", config)
        self.assertIn('lid_topic: "/omni/sensors/lidar/points"', config)
        self.assertIn('imu_topic: "/omni/sensors/imu/data"', config)
        self.assertIn("lidar_type: 6", config)
        self.assertIn("pcd_save_en: false", config)

        wrapper = (
            PACKAGE_ROOT / "launch" / "vbot_relocalization.launch.py"
        ).read_text(encoding="utf-8")
        self.assertIn("'vbot_relocalization.yaml'", wrapper)
        self.assertIn(
            "('pointcloud_topic', '/omni/sensors/lidar/points'", wrapper
        )
        self.assertIn("('tracking_frame', 'omni_imu_link'", wrapper)

        product_launch = (
            PACKAGE_ROOT / "launch" / "omni_dog_relocalization.launch.py"
        ).read_text(encoding="utf-8")
        self.assertIn("tracking_frame = LaunchConfiguration('tracking_frame')", product_launch)
        self.assertIn("'publish.tf_en': False", product_launch)
        self.assertNotIn("global_odometry_node", product_launch)
        self.assertNotIn("map_odom_trans", product_launch)

    def test_global_cloud_waits_for_same_stamp_transform(self):
        source = (
            PACKAGE_ROOT.parent
            / "icp_relocalization"
            / "src"
            / "global_pointcloud_publisher.cpp"
        ).read_text(encoding="utf-8")
        launch_source = (
            PACKAGE_ROOT / "launch" / "omni_dog_relocalization.launch.py"
        ).read_text(encoding="utf-8")

        self.assertIn('"transform_timeout_ms", 250', source)
        self.assertIn(
            "rclcpp::Duration::from_seconds(transform_timeout_sec_)", source
        )
        self.assertIn("{'transform_timeout_ms': 250}", launch_source)

    def test_matrix_profiles_require_a_stable_imu_initialization_window(self):
        source = (PACKAGE_ROOT / "src" / "laserMapping.cpp").read_text(
            encoding="utf-8"
        )
        imu_source = (PACKAGE_ROOT / "src" / "IMU_Processing.hpp").read_text(
            encoding="utf-8"
        )
        self.assertIn('"mapping.imu_init_min_samples"', source)
        self.assertIn("set_init_sample_count(imu_init_min_samples)", source)
        self.assertIn("init_iter_num > init_sample_count_", imu_source)
        for name in ("matrix_sim.yaml", "matrix_sim_relocalization.yaml"):
            config = (PACKAGE_ROOT / "config" / name).read_text(encoding="utf-8")
            self.assertIn("imu_init_min_samples: 500", config)

    def test_matrix_profiles_use_session_relative_reference_odometry(self):
        source = (PACKAGE_ROOT / "src" / "laserMapping.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn('"common.reference_odom_topic"', source)
        self.assertIn("reference_odom_tracker_.reset()", source)
        self.assertIn("apply_reference_odometry(Measures.lidar_end_time)", source)
        self.assertIn("if (reference_odom_enabled_)", source)

        for name in ("matrix_sim.yaml", "matrix_sim_relocalization.yaml"):
            config = (PACKAGE_ROOT / "config" / name).read_text(encoding="utf-8")
            self.assertIn('reference_odom_topic: "/odom/mujoco_odom"', config)
            self.assertIn('reference_odom_parent_frame_id: "world"', config)
            self.assertIn(
                'reference_odom_pose_frame_id: "omni_base_link"', config
            )

        # 外部真值只能属于仿真 profile，不能悄悄改变真机定位语义。
        for name in (
            "omni_dog.yaml",
            "omni_dog_relocalization.yaml",
            "vbot.yaml",
            "vbot_relocalization.yaml",
        ):
            config = (PACKAGE_ROOT / "config" / name).read_text(encoding="utf-8")
            self.assertNotIn("reference_odom_topic:", config)

    def test_real_dog_profiles_consume_only_manager_normalized_sensors(self):
        for name in (
            "omni_dog.yaml",
            "omni_dog_relocalization.yaml",
            "vbot.yaml",
            "vbot_relocalization.yaml",
        ):
            with self.subTest(name=name):
                config = (PACKAGE_ROOT / "config" / name).read_text(encoding="utf-8")
                self.assertIn(
                    'lid_topic: "/omni/sensors/lidar/points"', config
                )
                self.assertIn('imu_topic: "/omni/sensors/imu/data"', config)
                self.assertNotIn('lid_topic: "/front_lidar"', config)
                self.assertNotIn('lid_topic: "/lidar_points"', config)

    def test_prior_map_failure_throws_before_sensor_processing(self):
        source = (PACKAGE_ROOT / "src" / "laserMapping.cpp").read_text(
            encoding="utf-8"
        )
        validation = source.index("// Register the one-shot ICP result")
        subscriptions = source.index("/*** ROS subscribe initialization ***/")
        self.assertLess(validation, subscriptions)
        validation_block = source[validation:subscriptions]
        self.assertIn("prior_map_path.empty()", validation_block)
        # Seed reception may precede map I/O, but sensor processing cannot.
        self.assertIn('create_subscription<nav_msgs::msg::Odometry>("/icp_result"', validation_block)
        # The prior map is indexed into bounded tiles, never loaded whole.
        self.assertIn("fast_lio::SpatialMap", validation_block)
        self.assertNotIn("loadPCDFile", validation_block)
        self.assertIn("throw std::invalid_argument", validation_block)
        self.assertNotIn("catch (", validation_block)
        self.assertNotIn("return;", validation_block)

    def test_relocalization_launch_exit_contract_is_fail_closed(self):
        launch_contracts = {
            "omni_dog_relocalization.launch.py": (
                "global_pointcloud_node",
                "fast_lio_node",
            ),
            "relocalization.launch.py": (
                "fast_lio_node",
            ),
        }

        class FakeShutdown:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeExitEvent:
            def __init__(self, returncode):
                self.returncode = returncode

        for filename, critical_nodes in launch_contracts.items():
            with self.subTest(launch=filename):
                launch_file = PACKAGE_ROOT / "launch" / filename
                source = launch_file.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(launch_file))

                assignments = {
                    node.targets[0].id: node.value
                    for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "generate_launch_description"
                    for node in node.body
                    if isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Call)
                }

                self.assertNotIn("map_odom_trans", assignments)
                self.assertNotIn("global_odometry_node", assignments)

                for variable_name in critical_nodes:
                    node_call = assignments[variable_name]
                    on_exit = next(
                        keyword.value
                        for keyword in node_call.keywords
                        if keyword.arg == "on_exit"
                    )
                    self.assertIsInstance(on_exit, ast.Call)
                    self.assertIsInstance(on_exit.func, ast.Name)
                    self.assertEqual("Shutdown", on_exit.func.id)

                icp_on_exit = next(
                    keyword.value
                    for keyword in assignments["icp_node"].keywords
                    if keyword.arg == "on_exit"
                )
                self.assertIsInstance(icp_on_exit, ast.Name)
                self.assertEqual("_shutdown_on_failed_icp", icp_on_exit.id)

                helper_nodes = [
                    node
                    for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name in ("_icp_exit_failed", "_shutdown_on_failed_icp")
                ]
                helper_module = ast.Module(body=helper_nodes, type_ignores=[])
                ast.fix_missing_locations(helper_module)
                namespace = {"Shutdown": FakeShutdown}
                exec(compile(helper_module, str(launch_file), "exec"), namespace)

                self.assertFalse(namespace["_icp_exit_failed"](0))
                self.assertTrue(namespace["_icp_exit_failed"](1))
                self.assertTrue(namespace["_icp_exit_failed"](-6))
                self.assertEqual(
                    [], namespace["_shutdown_on_failed_icp"](FakeExitEvent(0), None)
                )
                failure_actions = namespace["_shutdown_on_failed_icp"](
                    FakeExitEvent(1), None
                )
                self.assertEqual(1, len(failure_actions))
                self.assertIsInstance(failure_actions[0], FakeShutdown)
                self.assertIn("return code 1", failure_actions[0].kwargs["reason"])


if __name__ == "__main__":
    unittest.main()
