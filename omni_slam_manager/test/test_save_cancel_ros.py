"""ROS-backed tests for SaveMap cancellation (needs the ROS 2 runtime)."""

import importlib.util
from pathlib import Path
import tempfile
import threading
import time
import unittest

ROS_AVAILABLE = importlib.util.find_spec("rclpy") is not None


class _StubFuture:
    """A service-call future that completes after ``delay`` seconds.

    ``delay=None`` never completes (simulates a hung map_save service).
    """

    def __init__(self, delay=None, result=None):
        self._done = False
        self._result = result
        if delay is not None:
            threading.Timer(delay, self._complete).start()

    def _complete(self) -> None:
        self._done = True

    def done(self) -> bool:
        return self._done

    def result(self):
        return self._result


class _CancelOnResultFuture:
    """Completes immediately; reading the result also flips the goal's
    cancel flag. Simulates a cancel arriving in the microsecond window
    between the service response and the manager's import step —
    deterministic where real timers cannot target that window."""

    def __init__(self, goal_handle, result):
        self._goal_handle = goal_handle
        self._result = result

    def done(self) -> bool:
        return True

    def result(self):
        self._goal_handle.cancel_now()
        return self._result


class _StubSaveClient:
    def __init__(self, future: _StubFuture):
        self._future = future

    def wait_for_service(self, timeout_sec=None) -> bool:
        return True

    def call_async(self, request):
        return self._future


class _StubAsset:
    map_id = "tmap"
    version = 1
    pcd_path = "/tmp/stub-map.pcd"
    checksum = "0" * 64


class _StubStore:
    def __init__(self):
        self.import_called = False

    def import_pcd(self, map_id, source, **kwargs):
        self.import_called = True
        return _StubAsset()

    def verify(self, asset) -> None:
        pass


class _GoalHandle:
    """Minimal stand-in for an rclpy action goal handle."""

    def __init__(self, goal, cancel_at=None):
        self.request = goal
        self._cancel_at = cancel_at
        self._force_cancel = False
        self._start = time.monotonic()
        self.terminal = None
        self.feedbacks = []

    def cancel_now(self) -> None:
        self._force_cancel = True

    @property
    def is_cancel_requested(self) -> bool:
        return self._force_cancel or (
            self._cancel_at is not None
            and time.monotonic() - self._start >= self._cancel_at
        )

    def publish_feedback(self, feedback) -> None:
        self.feedbacks.append(feedback)

    def succeed(self) -> None:
        self.terminal = "succeeded"

    def abort(self) -> None:
        self.terminal = "aborted"

    def canceled(self) -> None:
        self.terminal = "canceled"


@unittest.skipUnless(ROS_AVAILABLE, "ROS 2 Python runtime is not installed")
class SaveMapCancelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import rclpy

        cls._temporary_directory = tempfile.TemporaryDirectory()
        root = Path(cls._temporary_directory.name)
        if not rclpy.ok():
            rclpy.init(
                args=[
                    "--ros-args",
                    "-p",
                    f"map_root:={root / 'maps'}",
                    "-p",
                    f"map_source_path:={root / 'work' / 'map.pcd'}",
                ]
            )

    @classmethod
    def tearDownClass(cls):
        import rclpy

        if rclpy.ok():
            rclpy.shutdown()
        cls._temporary_directory.cleanup()

    def _make_node(self):
        from omni_slam_manager.manager_node import SlamManagerNode
        from omni_slam_manager.state_machine import SlamState

        node = SlamManagerNode()
        node._store = _StubStore()
        node._machine.transition(SlamState.STARTING_MAPPING)
        node._machine.transition(SlamState.MAPPING)
        return node

    def _make_goal(self):
        from omni_slam_interfaces.action import SaveMap

        return SaveMap.Goal(map_id="tmap", calibration_hash="", make_current=True)

    def _success_response(self):
        from std_srvs.srv import Trigger

        return Trigger.Response(success=True, message="ok")

    def test_cancel_while_waiting_cancels_without_import(self):
        from omni_slam_manager.manager_node import REASON_SAVE_CANCELLED
        from omni_slam_manager.state_machine import SlamState

        node = self._make_node()
        try:
            node._map_save_client = _StubSaveClient(_StubFuture())  # hangs
            handle = _GoalHandle(self._make_goal(), cancel_at=0.2)
            started = time.monotonic()
            result = node._execute_save_map(handle)
            elapsed = time.monotonic() - started

            self.assertLess(elapsed, 2.0)
            self.assertEqual(handle.terminal, "canceled")
            self.assertFalse(result.success)
            self.assertEqual(result.reason_code, REASON_SAVE_CANCELLED)
            # rollback: SAVING -> MAPPING, and the atomic store is untouched
            self.assertEqual(node._machine.snapshot.state, SlamState.MAPPING)
            self.assertFalse(node._store.import_called)
        finally:
            node.destroy_node()

    def test_cancel_after_service_response_cancels_before_import(self):
        from omni_slam_manager.manager_node import REASON_SAVE_CANCELLED
        from omni_slam_manager.state_machine import SlamState

        node = self._make_node()
        try:
            # Reading the (immediately-ready) result flips the cancel flag,
            # so the cancel lands exactly in the response->import window.
            handle = _GoalHandle(self._make_goal())
            node._map_save_client = _StubSaveClient(
                _CancelOnResultFuture(handle, self._success_response())
            )
            result = node._execute_save_map(handle)

            self.assertEqual(handle.terminal, "canceled")
            self.assertEqual(result.reason_code, REASON_SAVE_CANCELLED)
            self.assertEqual(node._machine.snapshot.state, SlamState.MAPPING)
            self.assertFalse(node._store.import_called)
        finally:
            node.destroy_node()

    def test_uncancelled_save_still_succeeds(self):
        from omni_slam_manager.manager_node import REASON_NONE
        from omni_slam_manager.state_machine import SlamState

        node = self._make_node()
        try:
            node._map_save_client = _StubSaveClient(
                _StubFuture(delay=0.1, result=self._success_response())
            )
            handle = _GoalHandle(self._make_goal())
            result = node._execute_save_map(handle)

            self.assertEqual(handle.terminal, "succeeded")
            self.assertTrue(result.success)
            self.assertEqual(result.reason_code, REASON_NONE)
            self.assertEqual(node._machine.snapshot.state, SlamState.MAP_READY)
            self.assertTrue(node._store.import_called)
        finally:
            node.destroy_node()


if __name__ == "__main__":
    unittest.main()
