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

    def test_natural_exit_is_observable(self):
        supervisor = ProcessSupervisor()
        supervisor.start(f'{sys.executable} -c "pass"')
        deadline = time.monotonic() + 2.0
        while supervisor.return_code is None and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(supervisor.return_code, 0)
