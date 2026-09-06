#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_DIR/FAST_LIO/config/matrix_sim.yaml}"
LIDAR_TOPIC="${LIDAR_TOPIC:-/livox/lidar}"
IMU_TOPIC="${IMU_TOPIC:-/livox/imu}"
FAST_LIO_LIDAR_TOPIC="${FAST_LIO_LIDAR_TOPIC:-/omni/sensors/lidar/points}"
FAST_LIO_IMU_TOPIC="${FAST_LIO_IMU_TOPIC:-/omni/sensors/imu/data}"
ODOM_TOPIC="${ODOM_TOPIC:-/odom/mujoco_odom}"
SYNTH_IMU=0
START_ZENOH=1
TF_PROFILE_FILE="${TF_PROFILE_FILE:-$SCRIPT_DIR/../omni_tf_manager/config/matrix_xgw.yaml}"
START_TF_MANAGER=1
START_SLAM_MANAGER=1
SLAM_MAP_ID="${SLAM_MAP_ID:-matrix_sim}"

usage() {
    cat <<EOF
Usage:
  ./run_matrix_mapping.sh [options] [extra ros2 launch args]

Options:
  --synth-imu          Publish ${IMU_TOPIC} from ${ODOM_TOPIC} if MATRiX has no IMU topic.
  --no-zenoh          Do not auto-start rmw_zenohd.
  --no-tf-manager     Do not start omni_tf_manager (only for externally managed runs).
  --no-slam-manager   Do not publish the typed /omni/slam/status observation.
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
        --no-tf-manager)
            START_TF_MANAGER=0
            shift
            ;;
        --no-slam-manager)
            START_SLAM_MANAGER=0
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
if [[ -f "$SCRIPT_DIR/../omni_tf_manager/install/setup.bash" ]]; then
    source "$SCRIPT_DIR/../omni_tf_manager/install/setup.bash"
fi
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
if [[ "$START_TF_MANAGER" == "1" ]]; then
    if [[ ! -f "$TF_PROFILE_FILE" ]]; then
        echo "[ERROR] omni_tf_manager profile not found: $TF_PROFILE_FILE" >&2
        exit 1
    fi
    echo "[INFO] starting omni_tf_manager in standalone mapping mode"
    ros2 launch omni_tf_manager omni_tf_manager.launch.py \
        config_file:="$TF_PROFILE_FILE" \
        mode:=profile \
        slam_state_required:=false \
        fallback_slam_mode:=mapping &
    cleanup_pids+=("$!")
fi
if [[ "$START_SLAM_MANAGER" == "1" ]]; then
    if ros2 node list 2>/dev/null | grep -Fxq "/omni_slam_manager"; then
        echo "[ERROR] /omni_slam_manager is already running; refuse duplicate SlamStatus owner" >&2
        exit 1
    fi
    echo "[INFO] starting omni_slam_manager in external mapping observation mode"
    ros2 run omni_slam_manager omni_slam_manager \
        --ros-args \
        -p tf_profile_path:="$TF_PROFILE_FILE" \
        -p map_root:=/tmp/omni_matrix_maps \
        -p map_source_path:=/tmp/omni_slam_work/map.pcd \
        -p external_runtime_mode:=mapping \
        -p external_map_id:="$SLAM_MAP_ID" \
        -p external_map_version:=1 &
    cleanup_pids+=("$!")
fi
echo "[INFO] waiting for LiDAR: $LIDAR_TOPIC"
wait_for_topic "$LIDAR_TOPIC" 45

if [[ "$SYNTH_IMU" == "1" ]]; then
    echo "[INFO] starting synthetic IMU bridge: $ODOM_TOPIC -> $IMU_TOPIC"
    python3 "$SCRIPT_DIR/tools/matrix_odom_to_imu.py" \
        --ros-args \
        -p odom_topic:="$ODOM_TOPIC" \
        -p imu_topic:="$IMU_TOPIC" \
        -p imu_frame_id:="livox_imu_link" &
    cleanup_pids+=("$!")
fi

echo "[INFO] waiting for IMU: $IMU_TOPIC"
wait_for_topic "$IMU_TOPIC" 45
echo "[INFO] waiting for normalized LiDAR: $FAST_LIO_LIDAR_TOPIC"
wait_for_topic "$FAST_LIO_LIDAR_TOPIC" 15
echo "[INFO] waiting for normalized IMU: $FAST_LIO_IMU_TOPIC"
wait_for_topic "$FAST_LIO_IMU_TOPIC" 15

echo "[INFO] LiDAR type:"
ros2 topic info "$LIDAR_TOPIC" || true
echo "[INFO] IMU type:"
ros2 topic info "$IMU_TOPIC" || true

RUNTIME_CONFIG_FILE=$(mktemp /tmp/matrix_fastlio_XXXXXX.yaml)
python3 "$SCRIPT_DIR/tools/matrix_configure_fastlio.py" \
    --input "$CONFIG_FILE" \
    --output "$RUNTIME_CONFIG_FILE" \
    --lidar-topic "$FAST_LIO_LIDAR_TOPIC" \
    --imu-topic "$FAST_LIO_IMU_TOPIC"

ros2 launch fast_lio omni_dog.launch.py \
    config_file:="$RUNTIME_CONFIG_FILE" \
    use_sim_time:=false \
    "${EXTRA_ARGS[@]}"
