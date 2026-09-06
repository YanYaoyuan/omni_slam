import signal
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from omni_slam_manager.process_supervisor import (
    ProcessSupervisor,
    ProcessSupervisorError,
    expand_command,
)


class ProcessSupervisorTests(unittest.TestCase):
    def test_command_expansion_is_argument_safe(self):
        command = expand_command(
            "ros2 launch pkg file.launch.py map_path:={map_path}",
            {"map_path": "/tmp/maps/plant a/map.pcd"},
        )
        self.assertEqual(command[-1], "map_path:=/tmp/maps/plant a/map.pcd")

    def test_unknown_placeholder_is_rejected(self):
        with self.assertRaises(ProcessSupervisorError):
            expand_command("echo {missing}")

    def test_vbot_localization_arguments_are_expanded(self):
        command = expand_command(
            "ros2 launch fast_lio vbot_relocalization.launch.py "
            "map_path:={map_path} initial_x:={initial_x} initial_yaw:={initial_yaw}",
            {
                "map_path": "/userdata/omni/maps/factory a/map.pcd",
                "initial_x": "2.5",
                "initial_yaw": "1.57",
            },
        )
        self.assertIn("map_path:=/userdata/omni/maps/factory a/map.pcd", command)
        self.assertIn("initial_x:=2.5", command)
        self.assertIn("initial_yaw:=1.57", command)

    def test_start_and_stop_process_group(self):
        supervisor = ProcessSupervisor()
        supervisor.start(f'{sys.executable} -c "import time; time.sleep(30)"')
        self.assertTrue(supervisor.running)
        self.assertIsNotNone(supervisor.stop(timeout_sec=0.2))
        self.assertFalse(supervisor.running)

    def test_stop_escalates_sigint_then_sigterm_then_sigkill(self):
        class SignalDrivenProcess:
            pid = 4242

            def __init__(self):
                self.returncode = None

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                if self.returncode is None:
                    if timeout is not None:
                        raise subprocess.TimeoutExpired("fake", timeout)
                    raise AssertionError("SIGKILL should make the fake process exit")
                return self.returncode

        process = SignalDrivenProcess()
        supervisor = ProcessSupervisor()
        supervisor._process = process
        delivered = []

        def deliver(_process_group, signum):
            delivered.append(signum)
            if signum == signal.SIGKILL:
                process.returncode = -signal.SIGKILL

        with patch(
            "omni_slam_manager.process_supervisor.os.killpg", side_effect=deliver
        ):
            self.assertEqual(supervisor.stop(timeout_sec=0.0), -signal.SIGKILL)

        self.assertEqual(
            delivered, [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]
        )
        self.assertFalse(supervisor.running)

    def test_stop_does_not_escalate_after_clean_sigint_exit(self):
        class InterruptibleProcess:
            pid = 4343
            returncode = None

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                del timeout
                return self.returncode

        process = InterruptibleProcess()
        supervisor = ProcessSupervisor()
        supervisor._process = process
        delivered = []

        def deliver(_process_group, signum):
            delivered.append(signum)
            process.returncode = -signum

        with patch(
            "omni_slam_manager.process_supervisor.os.killpg", side_effect=deliver
        ):
            self.assertEqual(supervisor.stop(timeout_sec=0.0), -signal.SIGINT)

        self.assertEqual(delivered, [signal.SIGINT])

    def test_natural_exit_is_observable(self):
        supervisor = ProcessSupervisor()
        supervisor.start(f'{sys.executable} -c "pass"')
        deadline = time.monotonic() + 2.0
        while supervisor.return_code is None and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(supervisor.return_code, 0)
