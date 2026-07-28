#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_DIR/FAST_LIO/config/matrix_sim_relocalization.yaml}"
MAP_PATH="${MAP_PATH:-$SCRIPT_DIR/maps/matrix_sim_scans.pcd}"
LIDAR_TOPIC="${LIDAR_TOPIC:-/livox/lidar}"
IMU_TOPIC="${IMU_TOPIC:-/livox/imu}"
ODOM_TOPIC="${ODOM_TOPIC:-/odom/mujoco_odom}"
SYNTH_IMU=0
START_ZENOH=1
INITIAL_X="${INITIAL_X:-0.0}"
INITIAL_Y="${INITIAL_Y:-0.0}"
INITIAL_Z="${INITIAL_Z:-0.0}"
INITIAL_YAW="${INITIAL_YAW:-0.0}"
MAX_CORRESPONDENCE_DISTANCE="${MAX_CORRESPONDENCE_DISTANCE:-1.0}"
FITNESS_SCORE_THRE="${FITNESS_SCORE_THRE:-0.2}"
CONVERGED_COUNT_THRE="${CONVERGED_COUNT_THRE:-10}"

usage() {
    cat <<EOF
Usage:
  ./run_matrix_relocalization.sh [options] [extra ros2 launch args]

Options:
  --map PATH           Prior PCD map. Default: ${MAP_PATH}
  --synth-imu          Publish ${IMU_TOPIC} from ${ODOM_TOPIC} if MATRiX has no IMU topic.
  --no-zenoh           Do not auto-start rmw_zenohd.
  --lidar-topic TOPIC  Override LiDAR topic. Default: ${LIDAR_TOPIC}
  --imu-topic TOPIC    Override IMU topic. Default: ${IMU_TOPIC}
  --odom-topic TOPIC   Override odometry topic for --synth-imu. Default: ${ODOM_TOPIC}
  --initial-x VALUE    Initial x guess in map frame. Default: ${INITIAL_X}
  --initial-y VALUE    Initial y guess in map frame. Default: ${INITIAL_Y}
  --initial-z VALUE    Initial z guess in map frame. Default: ${INITIAL_Z}
  --initial-yaw VALUE  Initial yaw guess in radians. Default: ${INITIAL_YAW}
  -h, --help           Show this help.

Environment:
  ROS_DOMAIN_ID defaults to 89.
  RMW_IMPLEMENTATION defaults to rmw_zenoh_cpp.
EOF
}

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --map)
            MAP_PATH="${2:?missing map path}"
            shift 2
            ;;
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
        --initial-x)
            INITIAL_X="${2:?missing initial x}"
            shift 2
            ;;
        --initial-y)
            INITIAL_Y="${2:?missing initial y}"
            shift 2
            ;;
        --initial-z)
            INITIAL_Z="${2:?missing initial z}"
            shift 2
            ;;
        --initial-yaw)
            INITIAL_YAW="${2:?missing initial yaw}"
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

if [[ ! -f "$MAP_PATH" ]]; then
    echo "[ERROR] map file not found: $MAP_PATH" >&2
    exit 1
fi

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
echo "[INFO] map: $MAP_PATH"
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

RUNTIME_CONFIG_FILE=$(mktemp /tmp/matrix_fastlio_relocalization_XXXXXX.yaml)
python3 "$SCRIPT_DIR/tools/matrix_configure_fastlio.py" \
    --input "$CONFIG_FILE" \
    --output "$RUNTIME_CONFIG_FILE" \
    --lidar-topic "$LIDAR_TOPIC" \
    --imu-topic "$IMU_TOPIC" \
    --prior-map-path "$MAP_PATH" \
    --map-file-path "$MAP_PATH"

exec ros2 launch fast_lio omni_dog_relocalization.launch.py \
    config_file:="$RUNTIME_CONFIG_FILE" \
    map_path:="$MAP_PATH" \
    pointcloud_topic:="$LIDAR_TOPIC" \
    initial_x:="$INITIAL_X" \
    initial_y:="$INITIAL_Y" \
    initial_z:="$INITIAL_Z" \
    initial_yaw:="$INITIAL_YAW" \
    max_correspondence_distance:="$MAX_CORRESPONDENCE_DISTANCE" \
    fitness_score_thre:="$FITNESS_SCORE_THRE" \
    converged_count_thre:="$CONVERGED_COUNT_THRE" \
    "${EXTRA_ARGS[@]}"
