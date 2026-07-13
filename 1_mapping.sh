#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup_dog3_env.sh"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/install/local_setup.bash"

mkdir -p "$SCRIPT_DIR/maps"
cd "$SCRIPT_DIR"

echo "启动 omni_slam 建图"
echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
echo "地图保存：$SCRIPT_DIR/maps/map.pcd（停止建图时 Ctrl+C 保存）"

ros2 launch fast_lio omni_dog.launch.py use_sim_time:=false "$@"
