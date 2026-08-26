from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class ControllerRelocationTest(unittest.TestCase):
    def test_s100_bin_symlink_resolves_runtime_setup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            source_controller = (
                Path(__file__).resolve().parents[1] / "scripts" / "omni_slam_ctl"
            )
            runtime = tmp_path / "userdata" / "omni_navi" / "omni_slam"
            controller = runtime / "lib" / "omni_slam_manager" / "omni_slam_ctl"
            runtime_bin = runtime / "bin"
            fake_bin = tmp_path / "fake-bin"
            data_root = tmp_path / "data"

            controller.parent.mkdir(parents=True)
            runtime_bin.mkdir(parents=True)
            fake_bin.mkdir()
            shutil.copy2(source_controller, controller)
            controller.chmod(0o755)
            bin_controller = runtime_bin / "omni_slam_ctl"
            bin_controller.symlink_to(
                "../lib/omni_slam_manager/omni_slam_ctl"
            )

            setup = runtime / "setup.bash"
            setup.write_text(
                f'export PATH="{fake_bin}:$PATH"\n'
                'export OMNI_TEST_SETUP_LOADED=1\n',
                encoding="utf-8",
            )

            ros2 = fake_bin / "ros2"
            ros2.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-} ${2:-}" == "service list" ]]; then
  printf '%s\\n' /omni/slam/start_mapping
  exit 0
fi
if [[ "${1:-}" == "launch" ]]; then
  exec -a omni_slam_manager sleep 30
fi
exit 0
""",
                encoding="utf-8",
            )
            ros2.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "OMNI_DATA_ROOT": str(data_root),
                    "OMNI_BOARD_ENV": str(tmp_path / "missing-board-env.sh"),
                }
            )

            for entry_kind in ("symlink", "dereferenced-copy"):
                with self.subTest(entry_kind=entry_kind):
                    if entry_kind == "dereferenced-copy":
                        bin_controller.unlink()
                        shutil.copy2(controller, bin_controller)

                    start = subprocess.run(
                        [str(bin_controller), "manager-start"],
                        env=env,
                        text=True,
                        capture_output=True,
                        timeout=10,
                        check=False,
                    )
                    self.assertEqual(start.returncode, 0, start.stderr)
                    self.assertNotIn(
                        f"setup file not found: {runtime.parent / 'setup.bash'}",
                        start.stderr,
                    )
                    self.assertTrue((data_root / "run" / "manager.pid").is_file())

                    stop = subprocess.run(
                        [str(bin_controller), "manager-stop"],
                        env=env,
                        text=True,
                        capture_output=True,
                        timeout=10,
                        check=False,
                    )
                    self.assertEqual(stop.returncode, 0, stop.stderr)


if __name__ == "__main__":
    unittest.main()
