#!/usr/bin/env bash
# Native build only. Source the robot's existing Omni SDK/workspace first.
set -euo pipefail
TRIAL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRIAL_ROS_SETUP=${TRIAL_ROS_SETUP:-}
if [[ -z "$TRIAL_ROS_SETUP" ]]; then
  if [[ -f /opt/tros/humble/setup.bash ]]; then
    TRIAL_ROS_SETUP=/opt/tros/humble/setup.bash
  else
    TRIAL_ROS_SETUP=/opt/ros/humble/setup.bash
  fi
fi
[[ -f "$TRIAL_ROS_SETUP" ]] || { echo "Missing Humble setup: $TRIAL_ROS_SETUP" >&2; exit 1; }
set +u
# shellcheck disable=SC1090
source "$TRIAL_ROS_SETUP"
set -u
for package in omni_tf_manager omni_robot_interfaces; do
  ros2 pkg prefix "$package" >/dev/null || {
    echo "Source the existing Omni workspace/SDK containing $package first." >&2; exit 1;
  }
done
cd "$TRIAL_ROOT"
MAKEFLAGS="-j${TRIAL_MAKE_JOBS:-2}" colcon build \
  --base-paths omni_slam_interfaces FAST_LIVO2 omni_slam_manager omni_semantic_map \
  --packages-select omni_slam_interfaces fast_livo omni_slam_manager omni_semantic_map \
  --executor sequential --build-base build/trial --install-base install/trial \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DFAST_LIVO_USE_LIVOX=OFF
echo "Built native $(uname -m) overlay. Source $TRIAL_ROOT/install/trial/setup.bash"
