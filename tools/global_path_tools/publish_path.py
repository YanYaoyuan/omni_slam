#!/usr/bin/env python3
"""Validate, simplify and publish a recorded omni_map body path once."""

import argparse
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.utilities import remove_ros_args


Point = Tuple[float, float, float]
EXPECTED_FRAME_ID = "omni_map"
EXPECTED_BODY_FRAME_ID = "omni_base_link"


def script_dir() -> str:
    return os.path.dirname(os.path.realpath(__file__))


def latest_recording() -> str:
    recordings_dir = os.path.join(script_dir(), "recordings")
    candidates: List[str] = []
    if os.path.isdir(recordings_dir):
        for name in os.listdir(recordings_dir):
            candidate = os.path.join(
                recordings_dir, name, "globalpath.txt"
            )
            if os.path.isfile(candidate):
                candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(
            f"未找到录制文件，请使用 --input；搜索目录：{recordings_dir}"
        )
    return max(candidates, key=os.path.getmtime)


def read_points(path: str) -> Tuple[List[Point], Dict[str, str]]:
    points: List[Point] = []
    metadata: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                content = line[1:].strip()
                if ":" in content:
                    key, value = content.split(":", 1)
                    metadata[key.strip()] = value.strip()
                continue
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(
                    f"{path}:{line_number}: 每行必须是 x y z"
                )
            point = tuple(float(value) for value in fields)
            if not all(math.isfinite(value) for value in point):
                raise ValueError(
                    f"{path}:{line_number}: 存在非有限数值"
                )
            points.append(point)  # type: ignore[arg-type]
    if len(points) < 2:
        raise ValueError(f"路径至少需要 2 个点，实际 {len(points)} 个")
    frame_id = metadata.get("frame_id")
    child_frame_id = metadata.get("child_frame_id")
    if frame_id and frame_id != EXPECTED_FRAME_ID:
        raise ValueError(
            f"路径元数据 frame_id={frame_id!r}，要求 {EXPECTED_FRAME_ID!r}"
        )
    if child_frame_id and child_frame_id != EXPECTED_BODY_FRAME_ID:
        raise ValueError(
            "路径元数据 child_frame_id="
            f"{child_frame_id!r}，要求 {EXPECTED_BODY_FRAME_ID!r}"
        )
    return points, metadata


def validate_metadata(
    metadata: Dict[str, str], allow_legacy: bool
) -> None:
    frame_id = metadata.get("frame_id")
    child_frame_id = metadata.get("child_frame_id")
    if frame_id is None or child_frame_id is None:
        if allow_legacy:
            return
        raise ValueError(
            "路径缺少 frame_id/child_frame_id 元数据。"
            "它可能是旧版 lio_odom 或雷达中心路径；请重新录制。"
            "只有确认坐标正确时才能使用 --allow-legacy"
        )


def distance_xy(first: Point, second: Point) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def distance_3d(first: Point, second: Point) -> float:
    return math.sqrt(
        (first[0] - second[0]) ** 2
        + (first[1] - second[1]) ** 2
        + (first[2] - second[2]) ** 2
    )


def path_length(points: List[Point]) -> float:
    return sum(
        distance_3d(points[index - 1], points[index])
        for index in range(1, len(points))
    )


def point_segment_distance(
    point: Point, start: Point, end: Point
) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dz = end[2] - start[2]
    length_squared = dx * dx + dy * dy + dz * dz
    if length_squared < 1.0e-12:
        return distance_3d(point, start)
    ratio = (
        (point[0] - start[0]) * dx
        + (point[1] - start[1]) * dy
        + (point[2] - start[2]) * dz
    ) / length_squared
    ratio = min(1.0, max(0.0, ratio))
    projection = (
        start[0] + ratio * dx,
        start[1] + ratio * dy,
        start[2] + ratio * (end[2] - start[2]),
    )
    return distance_3d(point, projection)


def simplify_rdp(points: List[Point], tolerance: float) -> List[Point]:
    if len(points) <= 2 or tolerance <= 0.0:
        return list(points)
    keep = {0, len(points) - 1}
    pending = [(0, len(points) - 1)]
    while pending:
        start_index, end_index = pending.pop()
        max_distance = -1.0
        split_index = -1
        for index in range(start_index + 1, end_index):
            distance = point_segment_distance(
                points[index], points[start_index], points[end_index]
            )
            if distance > max_distance:
                max_distance = distance
                split_index = index
        if split_index >= 0 and max_distance > tolerance:
            keep.add(split_index)
            pending.append((start_index, split_index))
            pending.append((split_index, end_index))
    return [points[index] for index in sorted(keep)]


def remove_duplicate_points(points: List[Point]) -> List[Point]:
    result = [points[0]]
    for point in points[1:]:
        if distance_3d(point, result[-1]) > 1.0e-6:
            result.append(point)
    return result


def densify(points: List[Point], max_spacing: float) -> List[Point]:
    if max_spacing <= 0.0:
        return list(points)
    result = [points[0]]
    for start, end in zip(points, points[1:]):
        distance = distance_3d(start, end)
        segment_count = max(1, int(math.ceil(distance / max_spacing)))
        for index in range(1, segment_count + 1):
            ratio = index / segment_count
            result.append(
                (
                    start[0] + ratio * (end[0] - start[0]),
                    start[1] + ratio * (end[1] - start[1]),
                    start[2] + ratio * (end[2] - start[2]),
                )
            )
    return result


def prepare_points(
    raw_points: List[Point], arguments: argparse.Namespace
) -> List[Point]:
    points = remove_duplicate_points(raw_points)
    points = simplify_rdp(points, arguments.simplify_tolerance)
    points = densify(points, arguments.max_spacing)
    if arguments.reverse:
        points.reverse()
    if arguments.z_mode == "constant":
        points = [(point[0], point[1], arguments.z) for point in points]
    if len(points) < 2:
        raise ValueError("简化后路径少于 2 个点")
    if len(points) > arguments.max_points:
        raise ValueError(
            f"简化后仍有 {len(points)} 点，超过 --max-points "
            f"{arguments.max_points}；请增大 --simplify-tolerance 或 "
            "--max-spacing，或分段发布"
        )
    return points


class GlobalPathFilePublisher(Node):
    def __init__(
        self, points: List[Point], path_file: str, arguments: argparse.Namespace
    ) -> None:
        super().__init__("omni_global_path_file_publisher")
        self.points = points
        self.path_file = path_file
        self.arguments = arguments
        self.current_body_position: Optional[Point] = None
        self.published = False
        self.last_body_error_time = 0.0
        self.last_start_error_time = 0.0

        path_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        odom_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.publisher = self.create_publisher(
            Path, arguments.topic, path_qos
        )
        self.body_subscription = self.create_subscription(
            Odometry,
            arguments.body_topic,
            self.on_body_odometry,
            odom_qos,
        )
        self.timer = self.create_timer(0.2, self.try_publish)
        self.get_logger().info(
            f"路径已加载：{path_file}，{len(points)} 点，"
            f"等待 {arguments.topic} 订阅者和合法机体位姿"
        )

    def on_body_odometry(self, message: Odometry) -> None:
        if (
            message.header.frame_id != EXPECTED_FRAME_ID
            or message.child_frame_id != EXPECTED_BODY_FRAME_ID
        ):
            now = time.monotonic()
            if now - self.last_body_error_time >= 2.0:
                self.last_body_error_time = now
                self.get_logger().error(
                    "拒绝机体位姿：收到 "
                    f"{message.header.frame_id} -> "
                    f"{message.child_frame_id}，要求 "
                    f"{EXPECTED_FRAME_ID} -> {EXPECTED_BODY_FRAME_ID}"
                )
            return
        position = message.pose.pose.position
        point = (float(position.x), float(position.y), float(position.z))
        if not all(math.isfinite(value) for value in point):
            return
        self.current_body_position = point

    def validate_start(self) -> bool:
        if self.arguments.skip_start_check:
            return True
        if self.current_body_position is None:
            return False
        distance = distance_xy(self.current_body_position, self.points[0])
        if distance > self.arguments.start_tolerance:
            now = time.monotonic()
            if now - self.last_start_error_time >= 2.0:
                self.last_start_error_time = now
                self.get_logger().error(
                    f"机器人距离路径起点 {distance:.2f}m，超过 "
                    f"{self.arguments.start_tolerance:.2f}m；拒绝发布。"
                    "请移动到起点、使用 --reverse，或确认安全后使用 "
                    "--skip-start-check"
                )
            return False
        return True

    def try_publish(self) -> None:
        if self.published:
            return
        if self.publisher.get_subscription_count() == 0:
            return
        if not self.validate_start():
            return
        self.publish_path()
        self.published = True

    def publish_path(self) -> None:
        stamp = self.get_clock().now().to_msg()
        message = Path()
        message.header.stamp = stamp
        message.header.frame_id = EXPECTED_FRAME_ID
        for index, point in enumerate(self.points):
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = point[2]
            if index + 1 < len(self.points):
                neighbor = self.points[index + 1]
                yaw = math.atan2(
                    neighbor[1] - point[1],
                    neighbor[0] - point[0],
                )
            else:
                neighbor = self.points[index - 1]
                yaw = math.atan2(
                    point[1] - neighbor[1],
                    point[0] - neighbor[0],
                )
            pose.pose.orientation.z = math.sin(yaw * 0.5)
            pose.pose.orientation.w = math.cos(yaw * 0.5)
            message.poses.append(pose)
        self.publisher.publish(message)
        self.get_logger().info(
            f"已单次发布 {len(self.points)} 点路径到 "
            f"{self.arguments.topic}，frame={EXPECTED_FRAME_ID}。"
            "节点继续运行以保留 transient-local 路径；需要重新发布时请重启。"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="校验、简化并单次发布录制的 omni_map 机体路径"
    )
    parser.add_argument(
        "--input",
        help="路径文件；不指定时自动选择 recordings 中最新 globalpath.txt",
    )
    parser.add_argument("--topic", default="/planning/global_path")
    parser.add_argument(
        "--body-topic", default="/omni/tf_manager/body_odom_global"
    )
    parser.add_argument("--simplify-tolerance", type=float, default=0.15)
    parser.add_argument("--max-spacing", type=float, default=2.0)
    parser.add_argument("--max-points", type=int, default=300)
    parser.add_argument("--start-tolerance", type=float, default=1.0)
    parser.add_argument("--skip-start-check", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument(
        "--allow-legacy",
        action="store_true",
        help="允许缺少 frame/body 元数据的旧路径；默认拒绝",
    )
    parser.add_argument(
        "--z-mode", choices=("recorded", "constant"), default="recorded"
    )
    parser.add_argument(
        "--z",
        type=float,
        default=0.0,
        help="--z-mode constant 时使用的 omni_map 机体中心高度",
    )
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args(remove_ros_args(args=sys.argv)[1:])
    for name in (
        "simplify_tolerance",
        "max_spacing",
        "start_tolerance",
        "z",
    ):
        if not math.isfinite(getattr(arguments, name)):
            parser.error(f"--{name.replace('_', '-')} 必须是有限数")
    if (
        arguments.simplify_tolerance < 0.0
        or arguments.max_spacing <= 0.0
        or arguments.start_tolerance < 0.0
        or arguments.max_points < 2
    ):
        parser.error("路径简化、间距、起点容差或最大点数参数无效")
    return arguments


def main() -> int:
    arguments = parse_args()
    try:
        path_file = (
            os.path.abspath(arguments.input)
            if arguments.input
            else latest_recording()
        )
        raw_points, metadata = read_points(path_file)
        validate_metadata(metadata, arguments.allow_legacy)
        points = prepare_points(raw_points, arguments)
    except (OSError, ValueError) as error:
        print(f"路径检查失败：{error}", file=sys.stderr)
        return 2

    print(
        f"路径检查通过：{path_file}\n"
        f"原始点数：{len(raw_points)}，发布点数：{len(points)}\n"
        f"路径长度：{path_length(points):.2f}m，"
        f"frame={EXPECTED_FRAME_ID}，body={EXPECTED_BODY_FRAME_ID}"
    )
    if arguments.check:
        return 0

    rclpy.init(args=sys.argv)
    node = GlobalPathFilePublisher(points, path_file, arguments)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
