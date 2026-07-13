#!/usr/bin/env bash

if [ -f /opt/ros/humble/setup.bash ]; then
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
elif [ -f /app/opt/ros/humble/local_setup.bash ]; then
  # shellcheck disable=SC1091
  source /app/opt/ros/humble/local_setup.bash
else
  echo "错误：未找到 ROS 2 Humble 环境" >&2
  return 2 2>/dev/null || exit 2
fi

if [ -f /app/idl_msgs/local_setup.bash ]; then
  # shellcheck disable=SC1091
  source /app/idl_msgs/local_setup.bash
elif [ -f /app/idl_msgs/setup.bash ]; then
  # shellcheck disable=SC1091
  source /app/idl_msgs/setup.bash
fi

if [ -f /app/script/env.sh ]; then
  # shellcheck disable=SC1091
  source /app/script/env.sh
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS_DIR="$SCRIPT_DIR/.deps"
if [ -d "$DEPS_DIR/venv/bin" ]; then
  export PATH="$DEPS_DIR/venv/bin:$PATH"
fi
if [ -d "$DEPS_DIR/venv/lib/python3.10/site-packages" ]; then
  export PYTHONPATH="$DEPS_DIR/venv/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
fi
if [ -d "$DEPS_DIR/root/usr" ]; then
  export CMAKE_PREFIX_PATH="$DEPS_DIR/ros_sdk_overlay:$DEPS_DIR/root/usr${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
  export CPATH="$DEPS_DIR/root/usr/include${CPATH:+:$CPATH}"
  export LIBRARY_PATH="$DEPS_DIR/sdk_sysroot/usr/lib/aarch64-linux-gnu:$DEPS_DIR/root/usr/lib/aarch64-linux-gnu${LIBRARY_PATH:+:$LIBRARY_PATH}"
  export LD_LIBRARY_PATH="$DEPS_DIR/sdk_sysroot/usr/lib/aarch64-linux-gnu:$DEPS_DIR/root/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_zenoh_cpp}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"

if [ -z "${ZENOH_SESSION_CONFIG_URI:-}" ] && [ -f /app_param/zenoh/s100_session.json5 ]; then
  export ZENOH_SESSION_CONFIG_URI=/app_param/zenoh/s100_session.json5
fi
export ZENOH_ROUTER_CHECK_ATTEMPTS="${ZENOH_ROUTER_CHECK_ATTEMPTS:-0}"
