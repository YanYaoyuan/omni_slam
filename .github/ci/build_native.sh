#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/humble/setup.bash
rm -rf build install log
colcon build --merge-install \
  --packages-select fast_lio icp_relocalization omni_slam_interfaces omni_slam_manager \
  --event-handlers console_direct+ \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DUSE_LIVOX_CUSTOM_MSG=OFF
source install/setup.bash
colcon test --packages-select fast_lio icp_relocalization omni_slam_manager \
  --event-handlers console_direct+
colcon test-result --verbose
