#!/usr/bin/env python3
"""Record a lio_map route from global lidar odometry as body-center points."""

import argparse
import math
import os
import queue
import signal
import sys
import threading
import time
from datetime import datetime
from typing import List, Optional, Tuple

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.utilities import remove_ros_args


Point = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]

DEFAULT_ODOMETRY_TOPIC = "/state_estimation_global"
DEFAULT_FRAME_ID = "lio_map"
DEFAULT_CHILD_FRAME_ID = "livox_frame"


def script_dir() -> str:
    return os.path.dirname(os.path.realpath(__file__))


def quaternion_multiply(first: Quaternion, second: Quaternion) -> Quaternion:
    x1, y1, z1, w1 = first
    x2, y2, z2, w2 = second
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def normalize_quaternion(quaternion: Quaternion) -> Quaternion:
    norm = math.sqrt(sum(value * value for value in quaternion))
    if not math.isfinite(norm) or norm < 1.0e-9:
        raise ValueError("invalid quaternion")
    return tuple(value / norm for value in quaternion)  # type: ignore[return-value]


def inverse_quaternion(quaternion: Quaternion) -> Quaternion:
    x, y, z, w = normalize_quaternion(quaternion)
    return (-x, -y, -z, w)


def quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> Quaternion:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return normalize_quaternion(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )
    )


def rotate_vector(quaternion: Quaternion, vector: Point) -> Point:
    q = normalize_quaternion(quaternion)
    vector_quaternion: Quaternion = (vector[0], vector[1], vector[2], 0.0)
    rotated = quaternion_multiply(
        quaternion_multiply(q, vector_quaternion), inverse_quaternion(q)
    )
    return (rotated[0], rotated[1], rotated[2])


def sensor_pose_to_body_position(
    sensor_position: Point,
    world_from_sensor: Quaternion,
    body_to_sensor_translation: Point,
    body_to_sensor_rotation: Quaternion,
) -> Point:
    world_from_body = normalize_quaternion(
        quaternion_multiply(
            normalize_quaternion(world_from_sensor),
            inverse_quaternion(body_to_sensor_rotation),
        )
    )
    world_offset = rotate_vector(world_from_body, body_to_sensor_translation)
    return (
        sensor_position[0] - world_offset[0],
        sensor_position[1] - world_offset[1],
        sensor_position[2] - world_offset[2],
    )


def distance_xy(first: Point, second: Point) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def distance_3d(first: Point, second: Point) -> float:
    return math.sqrt(
        (first[0] - second[0]) ** 2
        + (first[1] - second[1]) ** 2
        + (first[2] - second[2]) ** 2
    )


def point_line(point: Point) -> str:
    return f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n"


class EnterWatcher(threading.Thread):
    def __init__(self, events: "queue.Queue[str]") -> None:
        super().__init__(daemon=True)
        self.events = events

    def run(self) -> None:
        while True:
            line = sys.stdin.readline()
            if line == "":
                return
            self.events.put("waypoint")


class PathRecorder(Node):
    def __init__(self, arguments: argparse.Namespace) -> None:
        super().__init__("record_global_body_path")
        self.arguments = arguments
        self.topic = arguments.topic
        self.output = os.path.abspath(arguments.output)
        self.paths_dir = os.path.abspath(arguments.paths_dir)
        self.segment_index = 1
        self.total_count = 0
        self.rejected_outliers = 0
        self.last_recorded: Optional[Point] = None
        self.last_observed: Optional[Point] = None
        self.last_observed_time: Optional[float] = None
        self.last_line: Optional[str] = None
        self.current_segment: List[str] = []
        self.segment_has_new_points = False
        self.first_message_seen = False
        self.last_message_time: Optional[float] = None
        self.last_frame_error_time = 0.0
        self.wait_state = ""
        self.body_to_sensor_translation: Point = (
            arguments.body_to_sensor_x,
            arguments.body_to_sensor_y,
            arguments.body_to_sensor_z,
        )
        self.body_to_sensor_rotation = quaternion_from_rpy(
            arguments.body_to_sensor_roll,
            arguments.body_to_sensor_pitch,
            arguments.body_to_sensor_yaw,
        )

        self.prepare_outputs()
        reliability = (
            ReliabilityPolicy.RELIABLE
            if arguments.qos == "reliable"
            else ReliabilityPolicy.BEST_EFFORT
        )
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=200,
            reliability=reliability,
        )
        self.subscription = self.create_subscription(
            Odometry, self.topic, self.on_odometry, qos
        )
        self.wait_timer = self.create_timer(1.0, self.report_topic_state)
        self.global_file = open(
            self.output, "x", buffering=1, encoding="utf-8"
        )
        self.write_header(self.global_file)

        print("开始录制全局机体路径")
        print(f"输入里程计：{self.topic}")
        print(
            f"输入坐标：{arguments.frame} -> {arguments.child_frame}"
        )
        print("输出坐标：lio_map -> scan_base_link")
        print(f"全局路径：{self.output}")
        print(f"分段路径：{self.paths_dir}/pathN.txt")
        print(f"XY 最小记录间隔：{arguments.distance:.3f} m")
        print("按 Enter 保存当前路径段；按 Ctrl+C 保存最后一段并退出")
        self.report_topic_state(force=True)

    def write_header(self, output_file) -> None:
        args = self.arguments
        output_file.write("# omni_slam global body path v1\n")
        output_file.write("# frame_id: lio_map\n")
        output_file.write("# child_frame_id: scan_base_link\n")
        output_file.write(f"# source_topic: {self.topic}\n")
        output_file.write(
            "# body_to_sensor_xyz: "
            f"{args.body_to_sensor_x:.6f} "
            f"{args.body_to_sensor_y:.6f} "
            f"{args.body_to_sensor_z:.6f}\n"
        )
        output_file.write(
            "# body_to_sensor_rpy: "
            f"{args.body_to_sensor_roll:.6f} "
            f"{args.body_to_sensor_pitch:.6f} "
            f"{args.body_to_sensor_yaw:.6f}\n"
        )
        output_file.write("# columns: x y z\n")

    def prepare_outputs(self) -> None:
        if os.path.exists(self.output):
            raise FileExistsError(
                f"output already exists, refusing to overwrite: {self.output}"
            )
        os.makedirs(os.path.dirname(self.output), exist_ok=True)
        if os.path.exists(self.paths_dir) and os.listdir(self.paths_dir):
            raise FileExistsError(
                f"paths directory is not empty: {self.paths_dir}"
            )
        os.makedirs(self.paths_dir, exist_ok=True)

    def report_topic_state(self, force: bool = False) -> None:
        publisher_count = self.count_publishers(self.topic)
        if publisher_count == 0:
            state = "waiting_topic"
            message = f"等待 {self.topic}..."
        elif (
            self.last_message_time is None
            or time.monotonic() - self.last_message_time > 3.0
        ):
            state = "waiting_message"
            message = (
                f"已发现 {self.topic}，但未收到匹配 QoS/frame 的位姿..."
            )
        else:
            self.wait_state = "recording"
            return
        if force or state != self.wait_state:
            print(message, flush=True)
        self.wait_state = state

    def valid_frames(self, message: Odometry) -> bool:
        now = time.monotonic()
        if message.header.frame_id != self.arguments.frame:
            if now - self.last_frame_error_time >= 2.0:
                self.last_frame_error_time = now
                self.get_logger().error(
                    "拒绝位姿：frame_id=%r，要求=%r"
                    % (message.header.frame_id, self.arguments.frame)
                )
            return False
        if message.child_frame_id != self.arguments.child_frame:
            if now - self.last_frame_error_time >= 2.0:
                self.last_frame_error_time = now
                self.get_logger().error(
                    "拒绝位姿：child_frame_id=%r，要求=%r"
                    % (message.child_frame_id, self.arguments.child_frame)
                )
            return False
        return True

    def body_position(self, message: Odometry) -> Point:
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        sensor_position: Point = (
            float(position.x),
            float(position.y),
            float(position.z),
        )
        world_from_sensor: Quaternion = (
            float(orientation.x),
            float(orientation.y),
            float(orientation.z),
            float(orientation.w),
        )
        values = sensor_position + world_from_sensor
        if not all(math.isfinite(value) for value in values):
            raise ValueError("pose contains a non-finite value")
        return sensor_pose_to_body_position(
            sensor_position,
            world_from_sensor,
            self.body_to_sensor_translation,
            self.body_to_sensor_rotation,
        )

    def is_outlier(self, point: Point, receive_time: float) -> bool:
        if self.last_observed is None or self.last_observed_time is None:
            self.last_observed = point
            self.last_observed_time = receive_time
            return False
        dt = max(1.0e-3, receive_time - self.last_observed_time)
        step = distance_3d(point, self.last_observed)
        self.last_observed = point
        self.last_observed_time = receive_time
        return step > self.arguments.max_speed * dt + self.arguments.jump_margin

    def on_odometry(self, message: Odometry) -> None:
        receive_time = time.monotonic()
        self.last_message_time = receive_time
        self.wait_state = "recording"
        if not self.valid_frames(message):
            return
        try:
            point = self.body_position(message)
        except ValueError as error:
            self.get_logger().error(f"拒绝位姿：{error}")
            return
        if self.is_outlier(point, receive_time):
            self.rejected_outliers += 1
            self.get_logger().warning(
                "拒绝疑似位姿跳变："
                f"x={point[0]:.3f}, y={point[1]:.3f}, z={point[2]:.3f}"
            )
            return

        if not self.first_message_seen:
            self.first_message_seen = True
            print("已收到合法全局位姿，开始记录", flush=True)
        if (
            self.last_recorded is not None
            and distance_xy(point, self.last_recorded)
            < self.arguments.distance
        ):
            return

        line = point_line(point)
        self.global_file.write(line)
        self.current_segment.append(line)
        self.segment_has_new_points = True
        self.last_recorded = point
        self.last_line = line
        self.total_count += 1
        print(
            f"记录 {self.total_count}: "
            f"x={point[0]:.3f}, y={point[1]:.3f}, z={point[2]:.3f}",
            flush=True,
        )

    def save_segment(self, reason: str) -> bool:
        if self.last_line is None:
            print("\n尚未收到合法位姿，未保存路径段", flush=True)
            return False
        if not self.segment_has_new_points:
            print("\n上一段之后没有新路径，未重复保存", flush=True)
            return False

        output = os.path.join(
            self.paths_dir, f"path{self.segment_index}.txt"
        )
        temporary = f"{output}.tmp"
        with open(temporary, "x", encoding="utf-8") as segment_file:
            self.write_header(segment_file)
            segment_file.writelines(self.current_segment)
            segment_file.flush()
            os.fsync(segment_file.fileno())
        os.replace(temporary, output)
        print(
            f"\n已保存 {output}（{len(self.current_segment)} 点，{reason}）",
            flush=True,
        )
        self.segment_index += 1
        self.current_segment = [self.last_line]
        self.segment_has_new_points = False
        return True

    def close_files(self) -> None:
        self.save_segment("结束录制")
        if not self.global_file.closed:
            self.global_file.flush()
            os.fsync(self.global_file.fileno())
            self.global_file.close()


def default_output_paths(
    output: Optional[str], paths_dir: Optional[str]
) -> Tuple[str, str]:
    if output:
        resolved_output = os.path.abspath(output)
        resolved_paths = os.path.abspath(
            paths_dir
            if paths_dir
            else os.path.join(os.path.dirname(resolved_output), "paths")
        )
        return resolved_output, resolved_paths
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(script_dir(), "recordings", run_name)
    return (
        os.path.join(run_dir, "globalpath.txt"),
        os.path.join(run_dir, "paths"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "记录 /state_estimation_global，并用标定外参转换为 "
            "lio_map 下的 scan_base_link 路径。"
        )
    )
    parser.add_argument("--topic", default=DEFAULT_ODOMETRY_TOPIC)
    parser.add_argument("--frame", default=DEFAULT_FRAME_ID)
    parser.add_argument("--child-frame", default=DEFAULT_CHILD_FRAME_ID)
    parser.add_argument("--output")
    parser.add_argument("--paths-dir")
    parser.add_argument(
        "--distance",
        type=float,
        default=0.30,
        help="XY 平面相邻记录点最小间距，单位 m",
    )
    parser.add_argument(
        "--qos",
        choices=("reliable", "best_effort"),
        default="best_effort",
    )
    parser.add_argument("--max-speed", type=float, default=3.0)
    parser.add_argument("--jump-margin", type=float, default=0.50)
    parser.add_argument("--body-to-sensor-x", type=float, default=0.13011)
    parser.add_argument("--body-to-sensor-y", type=float, default=-0.02329)
    parser.add_argument("--body-to-sensor-z", type=float, default=0.17598)
    parser.add_argument("--body-to-sensor-roll", type=float, default=0.0)
    parser.add_argument("--body-to-sensor-pitch", type=float, default=0.0)
    parser.add_argument("--body-to-sensor-yaw", type=float, default=0.0)
    arguments = parser.parse_args(remove_ros_args(args=sys.argv)[1:])
    for name in ("distance", "max_speed", "jump_margin"):
        value = getattr(arguments, name)
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name.replace('_', '-')} 必须是有限非负数")
    if arguments.max_speed <= 0.0:
        parser.error("--max-speed 必须大于 0")
    arguments.output, arguments.paths_dir = default_output_paths(
        arguments.output, arguments.paths_dir
    )
    return arguments


def main() -> int:
    arguments = parse_args()
    events: "queue.Queue[str]" = queue.Queue()
    stop_requested = {"value": False}

    def request_stop(_signum, _frame) -> None:
        stop_requested["value"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    EnterWatcher(events).start()

    rclpy.init(args=sys.argv)
    recorder: Optional[PathRecorder] = None
    try:
        recorder = PathRecorder(arguments)
        while rclpy.ok() and not stop_requested["value"]:
            rclpy.spin_once(recorder, timeout_sec=0.1)
            while True:
                try:
                    event = events.get_nowait()
                except queue.Empty:
                    break
                if event == "waypoint":
                    recorder.save_segment("Enter 途径点")
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except (OSError, ValueError) as error:
        print(f"录制失败：{error}", file=sys.stderr)
        return 2
    finally:
        if recorder is not None:
            recorder.close_files()
            print(
                f"全局路径已保存：{recorder.output} "
                f"（{recorder.total_count} 点，"
                f"拒绝跳变 {recorder.rejected_outliers} 次）"
            )
            recorder.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
