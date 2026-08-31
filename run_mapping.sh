#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/setup_env.sh"

exec ros2 launch fast_lio omni_dog.launch.py "$@"
