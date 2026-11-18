"""Bounded ROS 2 semantic trial: latest-frame inference, measured boxes and ground."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py.point_cloud2 import read_points, create_cloud_xyz32
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from omni_slam_interfaces.msg import LivoSemanticFrame

from .live_core import LatestSlot, rigid_matrix, mask_points, support_indices
from .live_models import TrialModels
from .objects import project, transform_points, associate
from .geometry import observed_bounds, bounds_corners, BOX_EDGES, local_ground


def seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def matrix(message):
    p = getattr(message, "position", None)
    q = getattr(message, "orientation", None)
    if p is None:
        p, q = message.translation, message.rotation
    return rigid_matrix([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])


class SemanticTrial(Node):
    def __init__(self):
        super().__init__("omni_semantic_trial")
        defaults = dict(weights="", ground_model="", device="cpu", threads=2,
            output_directory="", max_frames=300, max_objects=1000, max_result_age_s=20.0,
            score=.7, input_topic="/omni/slam/livo/semantic_frames")
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.p = {key: self.get_parameter(key).value for key in defaults}
        if not self.p["output_directory"] or not self.p["weights"]:
            raise ValueError("explicit weights and a new output_directory are required")
        if self.p["max_frames"] < 1 or self.p["max_objects"] < 1 or self.p["threads"] < 1:
            raise ValueError("trial limits and threads must be positive")
        if not 0 < self.p["score"] <= 1 or self.p["max_result_age_s"] <= 0:
            raise ValueError("invalid score or result age")
        self.output = Path(self.p["output_directory"]).resolve()
        if self.output.exists() and any(self.output.iterdir()):
            raise ValueError("output directory must be new or empty; use one semantic session per LIVO run")
        self.models = TrialModels(self.p["weights"], self.p["ground_model"],
                                  self.p["device"], self.p["threads"], self.p["score"])
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "evidence").mkdir()
        (self.output / "run.json").write_text(json.dumps(
            {"parameters": self.p, "models": self.models.provenance}, indent=2))
        self.bridge = CvBridge(); self.slot = LatestSlot()
        self.pool = ThreadPoolExecutor(max_workers=1); self.future = None
        self.objects = []; self.poses = {}; self.frame = None
        self.processed = 0; self.received = 0; self.rejected = 0
        self.last_id = -1; self.last_stamp = -1.; self.last_receive = None
        self.state = "waiting_for_visual_keyframes"; self.error = ""
        self.last_wall_s = 0.; self.last_age_s = 0.
        input_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.subscription = self.create_subscription(LivoSemanticFrame,
            self.p["input_topic"], self.receive, input_qos)
        self.box_pub = self.create_publisher(MarkerArray, "/omni/semantic/boxes", latched)
        self.people_pub = self.create_publisher(MarkerArray, "/omni/semantic/people", input_qos)
        self.ground_pub = self.create_publisher(PointCloud2, "/omni/semantic/ground", input_qos)
        self.image_pub = self.create_publisher(Image, "/omni/semantic/image", input_qos)
        self.status_pub = self.create_publisher(String, "/omni/semantic/status", latched)
        # Health and completion handling must keep running when replay /clock stops.
        self.wall_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(.1, self.tick, clock=self.wall_clock)
        self.create_timer(1., self.status, clock=self.wall_clock)
        self.get_logger().info("Semantic trial ready; explicit local models, one pending frame")

    def receive(self, msg):
        self.received += 1; self.last_receive = time.monotonic()
        k = msg.keyframe; stamp = seconds(k.header.stamp)
        age = stamp - seconds(msg.image.header.stamp)
        try:
            if self.state in ("limit_reached", "session_mismatch", "error"):
                return
            if self.frame is not None and (k.id <= self.last_id or stamp <= self.last_stamp or k.header.frame_id != self.frame):
                self.state = "session_mismatch"
                raise ValueError("LIVO session reset or out-of-order frame; start a new semantic session")
            if (not k.visual_update or not msg.image.data or age < -.001 or age > .1
                    or msg.image.header.stamp != k.image_stamp
                    or msg.camera_info.header != msg.image.header
                    or k.cloud.header.frame_id != k.tracking_frame
                    or k.cloud.header.stamp != k.header.stamp):
                raise ValueError("invalid image/cloud epochs or frame IDs")
            if not k.header.frame_id or not k.tracking_frame or not msg.image.header.frame_id:
                raise ValueError("empty frame ID")
            if msg.camera_info.width != msg.image.width or msg.camera_info.height != msg.image.height:
                raise ValueError("CameraInfo dimensions disagree with image")
            intrinsic = np.asarray(msg.camera_info.k).reshape(3, 3)
            if (not np.isfinite(intrinsic).all() or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
                    or not np.allclose(intrinsic[2], [0, 0, 1]) or np.any(np.asarray(msg.camera_info.d) != 0)):
                raise ValueError("expected rectified pinhole image")
            matrix(k.pose); matrix(msg.camera_from_tracking)
        except ValueError as exc:
            self.rejected += 1; self.error = str(exc)
            return
        self.frame = k.header.frame_id; self.last_id = k.id; self.last_stamp = stamp
        self.slot.put(msg)

    def infer(self, msg):
        began = time.monotonic(); k = msg.keyframe
        image = self.bridge.imgmsg_to_cv2(msg.image, "bgr8")
        # read_points handles PointCloud2 field offsets, padding and byte order.
        raw = read_points(k.cloud, field_names=("x", "y", "z"), skip_nans=True)
        if isinstance(raw, np.ndarray) and raw.dtype.names:
            points = np.column_stack([raw[name].ravel() for name in ("x", "y", "z")])
        else:
            points = np.asarray(list(raw), dtype=float).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) > 30000:
            points = points[np.linspace(0, len(points)-1, 30000).astype(int)]
        pose = matrix(k.pose)
        pixels, depth, valid = project(points, matrix(msg.camera_from_tracking),
                                       np.asarray(msg.camera_info.k).reshape(3, 3))
        detections, ground_mask = self.models(image)
        annotated = image.copy()
        annotated[ground_mask] = (.5 * annotated[ground_mask] + .5 * np.array([80, 220, 90])).astype(np.uint8)
        observations = []; people = []
        evidence = f"evidence/keyframe_{k.id:06d}.jpg"
        for det in detections:
            # Exclude all detected object masks from ground candidates.
            ground_mask[det["mask"]] = False
            selected = support_indices(points, pixels, depth, valid, det["mask"])
            x1, y1, x2, y2 = det["box"].astype(int)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 180, 60), 1)
            cv2.putText(annotated, det["label"], (x1, max(y1-4, 12)), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 180, 60), 1)
            if selected is None:
                continue
            support = points[selected]; bounds = observed_bounds(support)
            if bounds is None:
                continue
            if det["label"] == "person":
                people.append(transform_points(bounds_corners(bounds["min"], bounds["max"]), pose))
                continue
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], det["mask"].astype(np.uint8),
                [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
            hist /= max(hist.sum(), 1)
            observations.append(dict(keyframe_id=int(k.id), stamp=seconds(k.header.stamp),
                image_stamp=seconds(msg.image.header.stamp), label=det["label"], score=det["score"],
                box=det["box"].tolist(), local_position=np.median(support, axis=0).tolist(),
                support_points=len(support), local_support_bounds=bounds,
                appearance=hist.tolist(), evidence=evidence))
        world = transform_points(points, pose)
        candidates = mask_points(pixels, valid, ground_mask)
        accepted, _ = local_ground(world, candidates, pose[:3, 3], seed=int(k.id))
        return dict(message=msg, observations=observations, people=people,
                    ground=world[accepted], image=annotated, evidence=evidence,
                    pose=pose, wall_s=time.monotonic()-began)

    def tick(self):
        if self.future is not None and self.future.done():
            try:
                result = self.future.result()
                self.last_wall_s = result["wall_s"]
                self.last_age_s = self.get_clock().now().nanoseconds * 1e-9 - seconds(result["message"].keyframe.header.stamp)
                if self.state not in ("session_mismatch", "error"):
                    if -.1 <= self.last_age_s <= self.p["max_result_age_s"]:
                        self.publish_result(result)
                    else:
                        self.rejected += 1; self.state = "stale_result"
            except Exception as exc:
                self.state = "error"; self.error = str(exc)
                self.get_logger().error(f"Semantic processing stopped: {exc}")
            self.future = None
        if self.state in ("error", "session_mismatch", "limit_reached"):
            self.slot.pop()
            return
        if self.processed >= self.p["max_frames"] or len(self.objects) >= self.p["max_objects"]:
            self.state = "limit_reached"; self.slot.pop()
            return
        if self.future is None:
            msg = self.slot.pop()
            if msg is not None:
                self.state = "processing"; self.future = self.pool.submit(self.infer, msg)

    def publish_result(self, result):
        msg = result["message"]; k = msg.keyframe
        self.poses[int(k.id)] = result["pose"]
        associate(self.objects, result["observations"], self.poses)
        self.processed += 1
        if not cv2.imwrite(str(self.output / result["evidence"]), result["image"]):
            raise IOError("failed to save evidence")
        document = dict(schema_version=1, frame=self.frame, map_revision="online_frontend",
            objects=self.objects, poses={str(i): p.tolist() for i, p in self.poses.items()},
            limitations=["Measured surface boxes, not complete dimensions", "No online loop correction",
                         "Ground is not traversability", "COCO trial: no doors; no person identity tracking"])
        temporary = self.output / "objects.json.tmp"
        temporary.write_text(json.dumps(document, ensure_ascii=False))
        temporary.replace(self.output / "objects.json")
        with (self.output / "frames.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(keyframe_id=int(k.id), stamp=seconds(k.header.stamp),
                image_stamp=seconds(msg.image.header.stamp), inference_s=result["wall_s"],
                ground_points=len(result["ground"]), person_boxes=len(result["people"]),
                observations=result["observations"])) + "\n")
        cloud = create_cloud_xyz32(k.header, result["ground"].astype(np.float32).tolist())
        self.ground_pub.publish(cloud)
        output_image = self.bridge.cv2_to_imgmsg(result["image"], "bgr8")
        output_image.header = msg.image.header; self.image_pub.publish(output_image)
        markers = MarkerArray(); clear = Marker(); clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        people_markers = MarkerArray(); people_markers.markers.append(clear)
        boxes = [(o["id"], np.asarray(o["box3d"]["corners"]),
            (0.2, .65, 1.) if o["confirmed"] else (1., .65, .15), False)
            for o in self.objects if o.get("box3d")]
        boxes.extend((f"person_{i}", c, (.95, .3, .8), True) for i, c in enumerate(result["people"]))
        for i, (label, corners, color, transient) in enumerate(boxes):
            box = Marker(); box.header = k.header; box.ns = "observed_boxes"; box.id = i
            box.type = Marker.LINE_LIST; box.action = Marker.ADD; box.pose.orientation.w = 1.
            box.scale.x = .035; box.color.r, box.color.g, box.color.b = color; box.color.a = 1.
            # Expire people relative to acquisition time, not inference completion.
            if transient:
                remaining = 2. - max(0., self.last_age_s)
                if remaining <= 0:
                    continue
                box.lifetime.sec = int(remaining); box.lifetime.nanosec = int((remaining % 1) * 1e9)
            for a, b in BOX_EDGES:
                for point in (corners[a], corners[b]):
                    box.points.append(Point(x=float(point[0]), y=float(point[1]), z=float(point[2])))
            target = people_markers if transient else markers
            target.markers.append(box)
            text = Marker(); text.header = k.header; text.ns = "object_ids"; text.id = i
            text.type = Marker.TEXT_VIEW_FACING; text.action = Marker.ADD
            text.pose.orientation.w = 1.; center = corners.mean(axis=0)
            text.pose.position = Point(x=float(center[0]), y=float(center[1]), z=float(corners[:, 2].max()+.2))
            text.scale.z = .22; text.color = box.color; text.lifetime = box.lifetime
            text.text = label; target.markers.append(text)
        self.box_pub.publish(markers); self.people_pub.publish(people_markers)
        self.state = "ready"; self.error = ""

    def status(self):
        state = self.state
        if self.last_receive is not None and time.monotonic()-self.last_receive > 5 and state not in ("error", "session_mismatch", "limit_reached"):
            state = "no_recent_visual_keyframes"
        data = dict(state=state, received=self.received, processed=self.processed,
                    dropped_pending=self.slot.dropped, rejected=self.rejected,
                    objects=len(self.objects), last_inference_s=self.last_wall_s,
                    last_result_age_s=self.last_age_s, error=self.error)
        self.status_pub.publish(String(data=json.dumps(data)))
        temp = self.output / "status.json.tmp"
        temp.write_text(json.dumps(data, indent=2)); temp.replace(self.output / "status.json")

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


def main(args=None):
    rclpy.init(args=args); node = None
    try:
        node = SemanticTrial(); rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close(); node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
