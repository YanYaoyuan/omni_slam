"""Exercise real manager callbacks without requiring a ROS runtime or processes."""

import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from omni_slam_manager.state_machine import SlamState, SlamStateMachine


def _load_manager():
    if importlib.util.find_spec("rclpy") is not None:
        from omni_slam_manager import manager_node
        return manager_node

    # Only ROS imports are stubbed; all lifecycle and state-machine code is real.
    modules = {
        name: Mock()
        for name in (
            "ament_index_python.packages", "rclpy", "rclpy.action",
            "rclpy.callback_groups", "rclpy.executors",
            "rclpy.impl.implementation_singleton", "rclpy.qos",
            "nav_msgs.msg", "sensor_msgs.msg", "std_msgs.msg", "std_srvs.srv",
            "omni_robot_interfaces.srv", "omni_slam_interfaces.srv",
            "omni_tf_manager.msg",
        )
    }
    modules["rclpy.node"] = SimpleNamespace(Node=object)
    modules["action_msgs.msg"] = SimpleNamespace(
        GoalStatus=SimpleNamespace(STATUS_CANCELED=5, STATUS_ABORTED=6)
    )
    action = SimpleNamespace(Result=SimpleNamespace, Feedback=SimpleNamespace)
    modules["omni_slam_interfaces.action"] = SimpleNamespace(
        StartLocalization=action, SaveMap=action
    )
    path = Path(__file__).resolve().parents[1] / "omni_slam_manager" / "manager_node.py"
    spec = importlib.util.spec_from_file_location(
        "omni_slam_manager._localization_test_manager", path
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


manager_node = _load_manager()


class _GoalHandle:
    def __init__(self):
        self.request = SimpleNamespace(
            map_id="test-map", map_version=1, timeout_sec=1.0,
            initial_x=0.0, initial_y=0.0, initial_z=0.0, initial_yaw=0.0,
        )
        self.is_cancel_requested = False
        self.status = 2  # STATUS_EXECUTING
        self.terminal = None
        self.feedbacks = []

    def publish_feedback(self, feedback):
        self.feedbacks.append(feedback)

    def succeed(self):
        self.terminal = "succeeded"

    def abort(self):
        self.terminal = "aborted"

    def canceled(self):
        self.terminal = "canceled"


class LocalizationOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.node = manager_node.SlamManagerNode.__new__(manager_node.SlamManagerNode)
        node = self.node
        node._lock = threading.RLock()
        node._machine = SlamStateMachine()
        node._localization_session = None
        node._external_runtime_mode = "none"
        node._stop_timeout = 1.0
        node._process = Mock()
        node._process.reap_exit.return_value = None
        node._mapping_dirty = False
        node._mapping_command = "mapping"
        node._localization_command = "localization"
        node._map_source_path = Path("/tmp/test-map.pcd")
        node._mapping_config_file = node._localization_config_file = ""
        node._tf_frames = dict.fromkeys(
            ("map", "odom", "base", "tracking", "icp_sensor"), "omni_test"
        )
        node._select_asset = Mock(return_value=SimpleNamespace(pcd_path="test.pcd"))
        node._publish_status = Mock()
        node._pose_jumps = Mock()
        node._fitness = SimpleNamespace(value=float("nan"), reset=lambda: None)
        self.now = 0.0
        node._now_steady = lambda: self.now
        for target, name, value in (
            (manager_node.rclpy, "ok", True), (manager_node.time, "sleep", None)
        ):
            patcher = patch.object(target, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _stop(self):
        response = self.node._stop_localization_service(None, SimpleNamespace())
        self.assertTrue(response.accepted)

    def _start_mapping(self):
        response = self.node._start_mapping(
            SimpleNamespace(session_id="replacement"), SimpleNamespace()
        )
        self.assertTrue(response.accepted)

    def _assert_atomic_stop(self):
        original = self.node._stop_localization

        def stop(reason, message):
            self.assertTrue(self.node._lock._is_owned())
            original(reason, message)

        return patch.object(self.node, "_stop_localization", side_effect=stop)

    def test_cancel_stops_own_process_atomically(self):
        handle = _GoalHandle()
        handle.is_cancel_requested = True
        with self._assert_atomic_stop():
            result = self.node._execute_localization(handle)
        self.assertEqual(handle.terminal, "canceled")
        self.assertEqual(result.reason_code, manager_node.REASON_LOCALIZATION_CANCELLED)
        self.assertEqual(self.node._machine.snapshot.state, SlamState.STOPPED)
        self.node._process.stop.assert_called_once()

    def test_timeout_stops_own_process_atomically(self):
        handle = _GoalHandle()
        with patch.object(
            manager_node.time, "sleep",
            side_effect=lambda _: setattr(self, "now", 2.0),
        ):
            with self._assert_atomic_stop():
                result = self.node._execute_localization(handle)
        self.assertEqual(handle.terminal, "aborted")
        self.assertEqual(result.reason_code, manager_node.REASON_LOCALIZATION_TIMEOUT)
        self.assertEqual(self.node._machine.snapshot.state, SlamState.STOPPED)
        self.node._process.stop.assert_called_once()

    def test_success_does_not_stop_process(self):
        handle = _GoalHandle()
        with patch.object(
            manager_node.time, "sleep",
            side_effect=lambda _: self.node._machine.transition(SlamState.LOCALIZED),
        ):
            result = self.node._execute_localization(handle)
        self.assertTrue(result.success)
        self.assertEqual(handle.terminal, "succeeded")
        self.node._process.stop.assert_not_called()

    def test_start_failure_is_reported(self):
        handle = _GoalHandle()
        self.node._process.start.side_effect = manager_node.ProcessSupervisorError("failed")
        result = self.node._execute_localization(handle)
        self.assertFalse(result.success)
        self.assertEqual(handle.terminal, "aborted")
        self.assertEqual(result.reason_code, manager_node.REASON_PROCESS_START_FAILED)
        self.assertEqual(self.node._machine.snapshot.state, SlamState.ERROR)
        self.node._process.stop.assert_not_called()

    def test_process_failure_is_reported(self):
        handle = _GoalHandle()
        with patch.object(
            manager_node.time, "sleep",
            side_effect=lambda _: self.node._machine.fail(
                manager_node.REASON_PROCESS_EXITED, "exited"
            ),
        ):
            result = self.node._execute_localization(handle)
        self.assertFalse(result.success)
        self.assertEqual(handle.terminal, "aborted")
        self.assertEqual(result.reason_code, manager_node.REASON_PROCESS_EXITED)
        self.node._process.stop.assert_not_called()

    def test_stop_service_ends_waiting_action_without_timeout(self):
        handle = _GoalHandle()
        # Keep the clock fixed: stop must end the Action before its deadline.
        with patch.object(
            manager_node.time, "sleep", side_effect=self._stop_on_first_sleep
        ):
            with self._assert_atomic_stop():
                result = self.node._execute_localization(handle)
        self.assertFalse(result.success)
        self.assertEqual(result.final_state, int(SlamState.STOPPED))
        self.assertEqual(handle.terminal, "aborted")
        self.node._process.stop.assert_called_once()

    def _stop_on_first_sleep(self, _duration):
        if self.node._process.stop.call_count:
            self.fail("stopped Action kept waiting")
        self._stop()

    def test_old_action_cannot_affect_replacement_runtime(self):
        for replacement in ("mapping", "localization", "localized"):
            for cause in ("cancel", "timeout"):
                with self.subTest(replacement=replacement, cause=cause):
                    self._replacement_case(replacement, cause)

    def test_mapping_stop_after_error_invalidates_old_action(self):
        for cause in ("cancel", "timeout"):
            with self.subTest(cause=cause):
                self._replacement_case("mapping", cause, stop_mapping=True)

    def _replacement_case(self, replacement, cause, stop_mapping=False):
        # Pause old and new callbacks independently at their normal polling waits.
        node = self.node
        node._machine = SlamStateMachine()
        node._process.reset_mock()
        self.now = 0.0
        handles = {name: _GoalHandle() for name in ("old", "new")}
        waiting = {name: threading.Event() for name in handles}
        resume = {name: threading.Event() for name in handles}
        results, errors, threads = {}, [], []

        def sleep(_duration):
            name = threading.current_thread().name
            waiting[name].set()
            if not resume[name].wait(3.0):
                raise AssertionError("test did not release localization callback")

        def execute(name):
            try:
                results[name] = node._execute_localization(handles[name])
            except BaseException as error:
                errors.append(error)

        def start(name):
            thread = threading.Thread(
                target=execute, args=(name,), name=name, daemon=True
            )
            threads.append(thread)
            thread.start()
            self.assertTrue(waiting[name].wait(3.0), errors)

        with patch.object(manager_node.time, "sleep", side_effect=sleep):
            try:
                start("old")
                if stop_mapping:
                    with node._lock:
                        node._machine.fail(manager_node.REASON_PROCESS_EXITED, "exited")
                    response = node._stop_mapping(
                        SimpleNamespace(discard_unsaved=True), SimpleNamespace()
                    )
                    self.assertTrue(response.accepted)
                else:
                    self._stop()
                if replacement == "mapping":
                    self._start_mapping()
                    expected_state = SlamState.STARTING_MAPPING
                else:
                    start("new")
                    expected_state = SlamState.RELOCALIZING
                    if replacement == "localized":
                        with node._lock:
                            node._machine.transition(SlamState.LOCALIZED)
                        resume["new"].set()
                        threads[-1].join(3.0)
                        self.assertTrue(results["new"].success)
                        expected_state = SlamState.LOCALIZED
                snapshot = node._machine.snapshot
                if cause == "cancel":
                    handles["old"].is_cancel_requested = True
                else:
                    self.now = 2.0
                resume["old"].set()
                threads[0].join(3.0)
                self.assertFalse(threads[0].is_alive())
                self.assertFalse(errors, errors)
                self.assertFalse(results["old"].success)
                self.assertEqual(results["old"].final_state, int(SlamState.STOPPED))
                self.assertEqual(node._machine.snapshot, snapshot)
                self.assertEqual(node._machine.snapshot.state, expected_state)
                node._process.stop.assert_called_once()
            finally:
                # Let a still-relocalizing replacement complete, even on failure.
                with node._lock:
                    if node._machine.snapshot.state == SlamState.RELOCALIZING:
                        node._machine.transition(SlamState.LOCALIZED)
                with patch.object(
                    manager_node.rclpy, "ok",
                    return_value=node._machine.snapshot.state == SlamState.LOCALIZED,
                ):
                    for event in resume.values():
                        event.set()
                    for thread in threads:
                        thread.join(3.0)
                        self.assertFalse(thread.is_alive())
        self.assertFalse(errors, errors)


if __name__ == "__main__":
    unittest.main()
