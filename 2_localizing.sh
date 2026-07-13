#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup_dog3_env.sh"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/install/local_setup.bash"

MAP_PATH="${MAP_PATH:-$SCRIPT_DIR/maps/map.pcd}"
FITNESS_SCORE_THRE="${FITNESS_SCORE_THRE:-2.5}"
MAX_CORRESPONDENCE_DISTANCE="${MAX_CORRESPONDENCE_DISTANCE:-1.0}"
CONVERGED_COUNT_THRE="${CONVERGED_COUNT_THRE:-5}"

if [ $# -gt 0 ] && [[ "$1" != *":="* ]] && [ -f "$1" ]; then
  MAP_PATH="$1"
  shift
fi

if [ ! -f "$MAP_PATH" ]; then
  echo "错误：地图不存在：$MAP_PATH"
  echo "先建图，或用：MAP_PATH=/path/to/map.pcd bash 2_localizing.sh"
  exit 2
fi

cd "$SCRIPT_DIR"

echo "启动 slam 定位"
echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
echo "地图：$MAP_PATH"

ros2 launch fast_lio omni_dog_relocalization.launch.py \
  map_path:="$MAP_PATH" \
  fitness_score_thre:="$FITNESS_SCORE_THRE" \
  max_correspondence_distance:="$MAX_CORRESPONDENCE_DISTANCE" \
  converged_count_thre:="$CONVERGED_COUNT_THRE" \
  "$@"
