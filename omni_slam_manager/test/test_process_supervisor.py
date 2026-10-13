import os
import select
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

    def test_stop_escalates_only_until_group_exits(self):
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
                    raise AssertionError("the final signal should make the process exit")
                return self.returncode

        signals = [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]
        for final_signal in (signal.SIGTERM, signal.SIGKILL):
            with self.subTest(final_signal=final_signal):
                process = SignalDrivenProcess()
                supervisor = ProcessSupervisor()
                supervisor._process = process
                delivered = []

                def deliver(_process_group, signum):
                    if signum:
                        delivered.append(signum)
                    elif process.returncode is not None:
                        raise ProcessLookupError
                    if signum == final_signal:
                        process.returncode = -final_signal

                with patch(
                    "omni_slam_manager.process_supervisor.os.killpg",
                    side_effect=deliver,
                ):
                    self.assertEqual(supervisor.stop(timeout_sec=0.0), -final_signal)

                self.assertEqual(delivered, signals[:signals.index(final_signal) + 1])
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
            if not signum:
                raise ProcessLookupError
            delivered.append(signum)
            process.returncode = -signum

        with patch(
            "omni_slam_manager.process_supervisor.os.killpg", side_effect=deliver
        ):
            self.assertEqual(supervisor.stop(timeout_sec=0.0), -signal.SIGINT)

        self.assertEqual(delivered, [signal.SIGINT])

    def test_missing_group_is_not_signalled_again(self):
        supervisor = ProcessSupervisor()
        supervisor.start("{python} -c pass", {"python": sys.executable})
        process = supervisor._process
        process.wait(timeout=2.0)
        with patch(
            "omni_slam_manager.process_supervisor.os.killpg",
            side_effect=ProcessLookupError,
        ) as killpg:
            self.assertEqual(supervisor.stop(timeout_sec=0.0), 0)
        killpg.assert_called_once_with(process.pid, signal.SIGINT)

    def _start_with_stubborn_child(self, natural_exit=False):
        code = """
import os, signal, sys, time
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
if os.fork() == 0:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    os.write(1, b"ready")
    time.sleep(30)
    os._exit(0)
if sys.argv[1] == "exit":
    sys.exit(7)
time.sleep(30)
"""
        supervisor = ProcessSupervisor()
        popen = subprocess.Popen
        with patch(
            "omni_slam_manager.process_supervisor.subprocess.Popen",
            side_effect=lambda *args, **kwargs: popen(
                *args, **kwargs, stdout=subprocess.PIPE
            ),
        ):
            supervisor.start(
                "{python} -c {code} {mode}",
                {
                    "python": sys.executable,
                    "code": code,
                    "mode": "exit" if natural_exit else "wait",
                },
            )
        process = supervisor._process

        def cleanup():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=3.0)

        self.addCleanup(cleanup)
        self.addCleanup(process.stdout.close)
        self.assertTrue(select.select([process.stdout], [], [], 3.0)[0])
        self.assertEqual(process.stdout.read(5), b"ready")
        return supervisor, process.stdout

    def _assert_child_exited(self, output):
        # EOF proves exit even if init has not reaped the child's zombie.
        self.assertTrue(select.select([output], [], [], 3.0)[0])
        self.assertEqual(output.read(1), b"")

    def test_stop_kills_child_after_group_leader_exits_on_sigint(self):
        supervisor, output = self._start_with_stubborn_child()
        self.assertEqual(supervisor.stop(timeout_sec=0.05), 0)
        self._assert_child_exited(output)
        self.assertFalse(supervisor.running)
        self.assertEqual(supervisor.command, [])

    def test_reap_exit_cleans_children_before_forgetting_group(self):
        supervisor, output = self._start_with_stubborn_child(natural_exit=True)
        supervisor._process.wait(timeout=3.0)
        stop = supervisor.stop
        with patch.object(supervisor, "stop", side_effect=lambda: stop(0.05)):
            self.assertEqual(supervisor.reap_exit(), 7)
        self._assert_child_exited(output)
        self.assertIsNone(supervisor.reap_exit())

    def test_restart_cleans_children_of_previous_group(self):
        supervisor, output = self._start_with_stubborn_child(natural_exit=True)
        supervisor._process.wait(timeout=3.0)
        stop = supervisor.stop
        with patch.object(supervisor, "stop", side_effect=lambda: stop(0.05)):
            supervisor.start("{python} -c pass", {"python": sys.executable})
        self.addCleanup(supervisor.stop, 0.05)
        self._assert_child_exited(output)

    def test_natural_exit_is_observable(self):
        supervisor = ProcessSupervisor()
        supervisor.start(f'{sys.executable} -c "pass"')
        deadline = time.monotonic() + 2.0
        while supervisor.return_code is None and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(supervisor.return_code, 0)

