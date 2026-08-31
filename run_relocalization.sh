#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/setup_env.sh"
MAP_PATH=${MAP_PATH:-$SCRIPT_DIR/maps/omni_dog_map.pcd}

exec ros2 launch fast_lio omni_dog_relocalization.launch.py \
  map_path:="$MAP_PATH" "$@"
