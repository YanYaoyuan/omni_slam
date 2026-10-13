"""Lightweight build fixtures; run with python3 .github/ci/test_delivery.py.

PyYAML is supplied by the ROS build image. No Docker, ROS build, or network is
used: execute the production provisioning/packaging shell with local fixtures.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[2]


def workflow(name):
    return yaml.load(
        (ROOT / ".github/workflows" / name).read_text(), Loader=yaml.BaseLoader
    )


def write(path, text="", executable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if executable:
        path.chmod(0o755)


class DeliveryTests(unittest.TestCase):
    def test_release_forwards_only_declared_checkout_secret(self):
        release = workflow("release.yml")
        for platform in ("x86", "orin", "s100"):
            with self.subTest(platform=platform):
                build = workflow(f"build-{platform}.yml")
                self.assertEqual(
                    build["on"]["workflow_call"]["secrets"],
                    {"OMNI_CI_REPO_TOKEN": {"required": "true"}},
                )
                self.assertEqual(
                    release["jobs"][platform]["secrets"],
                    {"OMNI_CI_REPO_TOKEN": "${{ secrets.OMNI_CI_REPO_TOKEN }}"},
                )

    def test_checksums_survive_flattened_release_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / "release-assets"
            release.mkdir()
            for platform in ("x86", "orin", "s100"):
                with self.subTest(platform=platform):
                    workspace = root / platform
                    write(workspace / "install/payload", platform)
                    write(workspace / "cross/runtime/payload", platform)
                    write(
                        workspace / ".github/ci/write_manifest.sh",
                        (ROOT / ".github/ci/write_manifest.sh").read_text(),
                        executable=True,
                    )
                    steps = workflow(f"build-{platform}.yml")["jobs"]["build"]["steps"]
                    package = next(s for s in steps if s.get("name", "").startswith("Package"))
                    # S100's preceding binary/profile checks belong to the real
                    # build; this fixture exercises the entire archive step.
                    script = package["run"][package["run"].index('version="'):]
                    result = subprocess.run(
                        ["bash", "-euo", "pipefail", "-c", script],
                        cwd=workspace,
                        env={**os.environ, "GITHUB_WORKSPACE": str(workspace),
                             "ROS_WS": str(workspace), "GITHUB_SHA": "a" * 40,
                             "VERSION_INPUT": "v1.2.3", "runtime_root": str(workspace / "cross/runtime")},
                        text=True, capture_output=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    archives = list((workspace / "dist").glob("*.tar.gz"))
                    self.assertEqual(len(archives), 1)
                    archive = archives[0]
                    checksum = Path(f"{archive}.sha256")
                    self.assertEqual(checksum.read_text().split()[1], archive.name)
                    upload = next(s for s in steps if "upload-artifact" in s.get("uses", ""))
                    self.assertIn("dist/*.sha256", upload["with"]["path"])
                    for artifact in (archive, checksum):
                        shutil.copy(artifact, release)
            verify = next(
                s["run"] for s in workflow("release.yml")["jobs"]["publish"]["steps"]
                if s.get("name") == "Verify checksums"
            )
            result = subprocess.run(["bash", "-e", "-c", verify], cwd=root, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            next(release.glob("*.tar.gz")).write_text("corrupted")
            result = subprocess.run(["bash", "-e", "-c", verify], cwd=root, capture_output=True)
            self.assertNotEqual(result.returncode, 0)

    def test_s100_provisions_target_gtsam_with_cold_and_pcl_only_caches(self):
        for pcl_cached in (False, True):
            with self.subTest(pcl_cached=pcl_cached):
                self.run_s100_fixture(pcl_cached=pcl_cached)

    def test_s100_rejects_host_gtsam(self):
        self.run_s100_fixture(pcl_cached=True, architecture="x86-64", succeeds=False)

    def run_s100_fixture(self, pcl_cached, architecture="ARM aarch64", succeeds=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = root / "ros/humble"
            prefix.mkdir(parents=True)
            cache = root / "work/cache/ros-pcl-arm64"
            workspace = root / "work/cc_ws/tros_ws"
            workspace.mkdir(parents=True)
            if pcl_cached:
                write(prefix / "share/pcl_ros/cmake/pcl_rosConfig.cmake")
                write(cache / "ros-humble-pcl-ros_1_arm64.deb")
            binaries = root / "bin"
            write(binaries / "curl", '#!/bin/bash\ntouch "${@: -1}"\n', True)
            write(binaries / "apt-get", '''#!/bin/bash
printf '%s\\n' "$*" >> "$APT_LOG"
for argument in "$@"; do
  case "$argument" in
    *:arm64) touch "${argument%:arm64}_1_arm64.deb" ;;
  esac
done
''', True)
            write(binaries / "dpkg-deb", '''#!/bin/bash
prefix="$3/opt/ros/humble"
case "$2" in
  *pcl-conversions_*)
    mkdir -p "$prefix/share/pcl_conversions/cmake"
    touch "$prefix/share/pcl_conversions/cmake/pcl_conversionsConfig.cmake" ;;
  *pcl-ros_*)
    mkdir -p "$prefix/share/pcl_ros/cmake" "$prefix/lib"
    touch "$prefix/share/pcl_ros/cmake/pcl_rosConfig.cmake"
    touch "$prefix/lib/libpcl_ros_tf.a" "$prefix/lib/libpcd_to_pointcloud_lib.so" ;;
  *gtsam_*)
    mkdir -p "$prefix/lib/cmake/GTSAM" "$prefix/include/gtsam/nonlinear"
    touch "$prefix/lib/cmake/GTSAM/GTSAMConfig.cmake"
    touch "$prefix/include/gtsam/nonlinear/ISAM2.h" "$prefix/lib/libgtsam.so" ;;
esac
''', True)
            write(binaries / "file", '''#!/bin/bash
path="${@: -1}"
test -f "$path" || exit 1
case "$path" in
  *libgtsam.so) echo "ELF $GTSAM_ARCH" ;;
  *) echo 'ELF ARM aarch64' ;;
esac
''', True)
            source = (ROOT / "scripts/build_s100_cross.sh").read_text()
            script = source.split('echo "=== Complete ARM64 ROS PCL', 1)[1]
            script = 'echo "=== Complete ARM64 ROS PCL' + script.split(
                'echo "=== Activate ROS 2 build environment ==="', 1
            )[0]
            # Redirect hard-coded container locations to the isolated fixture.
            script = script.replace("/opt/ros/humble", str(prefix))
            script = script.replace("/mnt/s100-cross", str(root / "work"))
            script = script.replace("/tmp/s100-", str(root / "s100-"))
            # Mock deb extraction retains /opt/ros/humble beneath its staging root.
            script = script.replace('${ros_extract_root}' + str(prefix), '${ros_extract_root}/opt/ros/humble')
            env = {**os.environ, "PATH": f"{binaries}:{os.environ['PATH']}",
                   "APT_LOG": str(root / "apt.log"), "GTSAM_ARCH": architecture}
            result = subprocess.run(
                ["bash", "-euo", "pipefail", "-c", script], cwd=workspace,
                env=env, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode == 0, succeeds, result.stdout + result.stderr)
            log = (root / "apt.log").read_text()
            self.assertIn("APT::Architecture=arm64", log)
            self.assertIn("ros-humble-gtsam:arm64", log)
            self.assertNotIn(" install ", log)
            if succeeds:
                self.assertTrue((prefix / "include/gtsam/nonlinear/ISAM2.h").is_file())
                # A completed target installation must not download again.
                (root / "apt.log").unlink()
                result = subprocess.run(
                    ["bash", "-euo", "pipefail", "-c", script], cwd=workspace,
                    env=env, text=True, capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse((root / "apt.log").exists())


if __name__ == "__main__":
    unittest.main()
