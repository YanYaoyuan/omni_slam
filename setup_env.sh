#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-24}
export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_zenoh_cpp}

set +u
source /opt/ros/humble/setup.bash
source "$SCRIPT_DIR/install/setup.bash"
set -u
