#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_DIR/FAST_LIO/config/matrix_sim.yaml}"
LIDAR_TOPIC="${LIDAR_TOPIC:-/livox/lidar}"
IMU_TOPIC="${IMU_TOPIC:-/livox/imu}"
ODOM_TOPIC="${ODOM_TOPIC:-/odom/mujoco_odom}"
SYNTH_IMU=0
START_ZENOH=1
RVIZ="${RVIZ:-false}"

usage() {
    cat <<EOF
Usage:
  ./run_matrix_mapping.sh [options] [extra ros2 launch args]

Options:
  --synth-imu          Publish ${IMU_TOPIC} from ${ODOM_TOPIC} if MATRiX has no IMU topic.
  --no-zenoh          Do not auto-start rmw_zenohd.
  --lidar-topic TOPIC  Override LiDAR topic. Default: ${LIDAR_TOPIC}
  --imu-topic TOPIC    Override IMU topic. Default: ${IMU_TOPIC}
  --odom-topic TOPIC   Override odometry topic for --synth-imu. Default: ${ODOM_TOPIC}
  -h, --help          Show this help.

Environment:
  ROS_DOMAIN_ID defaults to 89.
  RMW_IMPLEMENTATION defaults to rmw_zenoh_cpp.
  CONFIG_FILE can override the FAST-LIO yaml.
EOF
}

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --synth-imu)
            SYNTH_IMU=1
            shift
            ;;
        --no-zenoh)
            START_ZENOH=0
            shift
            ;;
        --lidar-topic)
            LIDAR_TOPIC="${2:?missing lidar topic}"
            shift 2
            ;;
        --imu-topic)
            IMU_TOPIC="${2:?missing imu topic}"
            shift 2
            ;;
        --odom-topic)
            ODOM_TOPIC="${2:?missing odom topic}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-89}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_zenoh_cpp}"

set +u
source /opt/ros/humble/setup.bash
source "$SCRIPT_DIR/install/setup.bash"
set -u

cleanup_pids=()
cleanup() {
    for pid in "${cleanup_pids[@]:-}"; do
        if kill -0 "$pid" >/dev/null 2>&1; then
            kill -TERM "$pid" >/dev/null 2>&1 || true
        fi
    done
    if [[ -n "${RUNTIME_CONFIG_FILE:-}" && -f "$RUNTIME_CONFIG_FILE" ]]; then
        rm -f "$RUNTIME_CONFIG_FILE"
    fi
}
trap cleanup EXIT INT TERM

topic_exists() {
    ros2 topic list 2>/dev/null | grep -Fxq "$1"
}

wait_for_topic() {
    local topic="$1"
    local timeout="${2:-30}"
    local start
    start=$(date +%s)
    while ! topic_exists "$topic"; do
        if (( $(date +%s) - start >= timeout )); then
            echo "[ERROR] timed out waiting for topic: $topic" >&2
            echo "        Check that MATRiX is running with ROS_DOMAIN_ID=$ROS_DOMAIN_ID and RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION" >&2
            return 1
        fi
        sleep 1
    done
}

if [[ "$RMW_IMPLEMENTATION" == "rmw_zenoh_cpp" && "$START_ZENOH" == "1" ]]; then
    if ! pgrep -f "rmw_zenoh_cpp.*rmw_zenohd" >/dev/null 2>&1; then
        echo "[INFO] starting rmw_zenohd"
        ros2 run rmw_zenoh_cpp rmw_zenohd >/tmp/matrix_rmw_zenohd.log 2>&1 &
        cleanup_pids+=("$!")
        sleep 2
    fi
fi

echo "[INFO] ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "[INFO] RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
echo "[INFO] waiting for LiDAR: $LIDAR_TOPIC"
wait_for_topic "$LIDAR_TOPIC" 45

if [[ "$SYNTH_IMU" == "1" ]]; then
    echo "[INFO] starting synthetic IMU bridge: $ODOM_TOPIC -> $IMU_TOPIC"
    python3 "$SCRIPT_DIR/tools/matrix_odom_to_imu.py" \
        --ros-args \
        -p odom_topic:="$ODOM_TOPIC" \
        -p imu_topic:="$IMU_TOPIC" \
        -p imu_frame_id:="livox_frame" &
    cleanup_pids+=("$!")
fi

echo "[INFO] waiting for IMU: $IMU_TOPIC"
wait_for_topic "$IMU_TOPIC" 45

echo "[INFO] LiDAR type:"
ros2 topic info "$LIDAR_TOPIC" || true
echo "[INFO] IMU type:"
ros2 topic info "$IMU_TOPIC" || true

RUNTIME_CONFIG_FILE=$(mktemp /tmp/matrix_fastlio_XXXXXX.yaml)
python3 "$SCRIPT_DIR/tools/matrix_configure_fastlio.py" \
    --input "$CONFIG_FILE" \
    --output "$RUNTIME_CONFIG_FILE" \
    --lidar-topic "$LIDAR_TOPIC" \
    --imu-topic "$IMU_TOPIC"

exec ros2 launch fast_lio mapping.launch.py \
    config_file:="$RUNTIME_CONFIG_FILE" \
    use_sim_time:=false \
    rviz:="$RVIZ" \
    "${EXTRA_ARGS[@]}"
