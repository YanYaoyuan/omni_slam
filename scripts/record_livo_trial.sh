#!/usr/bin/env bash
# Inputs must already be canonical and the camera image must already be rectified.
set -euo pipefail
if [[ $# -ne 1 ]]; then echo "Usage: $0 /absolute/new/bag_directory" >&2; exit 2; fi
[[ "$1" = /* && ! -e "$1" ]] || { echo "Bag output must be absolute and new" >&2; exit 2; }
exec ros2 bag record -o "$1" \
  /omni/sensors/lidar/points /omni/sensors/imu/data \
  /omni/sensors/camera/image_rect /omni/sensors/camera/camera_info \
  /state_estimation /omni/slam/livo/semantic_frames \
  /omni/slam/status /omni/semantic/status /omni/semantic/boxes \
  /omni/semantic/people /omni/semantic/ground /omni/semantic/image /tf /tf_static
