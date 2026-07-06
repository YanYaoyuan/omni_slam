#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROS_DISTRO_TARGET=${ROS_DISTRO_TARGET:-humble}
LIVOX_SETUP=${LIVOX_SETUP:-$HOME/ws_livox/install/setup.bash}
WORKERS=${WORKERS:-1}
MAKE_JOBS=${MAKE_JOBS:-2}

if [[ ! -f "/opt/ros/$ROS_DISTRO_TARGET/setup.bash" ]]; then
  echo "[build_native_orin] ERROR: missing /opt/ros/$ROS_DISTRO_TARGET/setup.bash" >&2
  exit 1
fi
source "/opt/ros/$ROS_DISTRO_TARGET/setup.bash"

if [[ -f "$LIVOX_SETUP" ]]; then
  source "$LIVOX_SETUP"
else
  echo "[build_native_orin] WARNING: Livox setup not found: $LIVOX_SETUP" >&2
fi

cd "$ROOT_DIR"
MAKEFLAGS="-j$MAKE_JOBS" colcon build \
  --symlink-install \
  --parallel-workers "$WORKERS" \
  --packages-select fast_lio icp_relocalization \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
