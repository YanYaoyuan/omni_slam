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
        self.assertRegex(save_wrapper, r"return\s+save_cloud_to_pcd\s*\(")

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
            source.index("ld.add_action(map_odom_trans)"),
        )

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
                        source.index("ld.add_action(map_odom_trans)"))

    def test_prior_map_failure_throws_before_ros_endpoints(self):
        source = (PACKAGE_ROOT / "src" / "laserMapping.cpp").read_text(
            encoding="utf-8"
        )
        validation = source.index("// Relocalization must fail before any ROS endpoint")
        subscriptions = source.index("/*** ROS subscribe initialization ***/")
        self.assertLess(validation, subscriptions)
        validation_block = source[validation:subscriptions]
        self.assertIn("prior_map_path.empty()", validation_block)
        self.assertIn("load_result < 0 || prior_map->empty()", validation_block)
        self.assertIn("feats_down_prior_map->empty()", validation_block)
        self.assertGreaterEqual(validation_block.count("throw std::"), 4)
        self.assertNotIn("return;", validation_block)

    def test_relocalization_launch_exit_contract_is_fail_closed(self):
        launch_contracts = {
            "omni_dog_relocalization.launch.py": (
                "map_odom_trans",
                "global_odometry_node",
                "global_pointcloud_node",
                "fast_lio_node",
            ),
            "relocalization.launch.py": (
                "map_odom_trans",
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
