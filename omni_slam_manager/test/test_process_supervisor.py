import sys
import time
import unittest

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

    def test_start_and_stop_process_group(self):
        supervisor = ProcessSupervisor()
        supervisor.start(f'{sys.executable} -c "import time; time.sleep(30)"')
        self.assertTrue(supervisor.running)
        self.assertIsNotNone(supervisor.stop(timeout_sec=0.2))
        self.assertFalse(supervisor.running)

    def test_natural_exit_is_observable(self):
        supervisor = ProcessSupervisor()
        supervisor.start(f'{sys.executable} -c "pass"')
        deadline = time.monotonic() + 2.0
        while supervisor.return_code is None and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(supervisor.return_code, 0)
