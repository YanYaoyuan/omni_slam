"""ROS 2 orchestration node for mapping, map assets and localization."""

from __future__ import annotations

import math
from pathlib import Path
import threading
import time
from typing import Optional

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, PointCloud2
from std_srvs.srv import Trigger

from omni_slam_interfaces.action import SaveMap, StartLocalization
from omni_slam_interfaces.msg import SlamStatus
from omni_slam_interfaces.srv import StartMapping, StopLocalization, StopMapping

from .map_store import MapAsset, MapStore, MapStoreError
from .process_supervisor import ProcessSupervisor, ProcessSupervisorError
from .state_machine import InvalidTransition, SlamState, SlamStateMachine


REASON_NONE = 0
REASON_INVALID_STATE = 1001
REASON_PROCESS_START_FAILED = 1002
REASON_PROCESS_EXITED = 1003
REASON_MAP_SAVE_UNAVAILABLE = 1101
REASON_MAP_SAVE_FAILED = 1102
REASON_MAP_IMPORT_FAILED = 1103
REASON_MAP_INVALID = 1201
REASON_LOCALIZATION_TIMEOUT = 1202
REASON_LOCALIZATION_CANCELLED = 1203


ROBOT_PROFILES = {
    "omni_dog": {
        "mapping_command": (
            "ros2 launch fast_lio omni_dog.launch.py "
            "map_path:={map_source_path}"
        ),
        "localization_command": (
            "ros2 launch fast_lio omni_dog_relocalization.launch.py "
            "map_path:={map_path} initial_x:={initial_x} initial_y:={initial_y} "
            "initial_z:={initial_z} initial_yaw:={initial_yaw}"
        ),
        "imu_topic": "/front_lidar/imu",
    },
    "vbot": {
        "mapping_command": (
            "ros2 launch fast_lio vbot.launch.py "
            "map_path:={map_source_path}"
        ),
        "localization_command": (
            "ros2 launch fast_lio vbot_relocalization.launch.py "
            "map_path:={map_path} initial_x:={initial_x} initial_y:={initial_y} "
            "initial_z:={initial_z} initial_yaw:={initial_yaw}"
        ),
        "imu_topic": "/lidar_imu",
    },
}


class SlamManagerNode(Node):
    def __init__(self) -> None:
        super().__init__("omni_slam_manager")
        self._callbacks = ReentrantCallbackGroup()
        self._lock = threading.RLock()
        self._machine = SlamStateMachine()
        self._process = ProcessSupervisor()

        self.declare_parameter("robot_profile", "omni_dog")
        robot_profile = str(self.get_parameter("robot_profile").value)
        if robot_profile not in ROBOT_PROFILES:
            raise ValueError(
                f"unsupported robot_profile '{robot_profile}'; "
                f"expected one of {sorted(ROBOT_PROFILES)}"
            )
        profile = ROBOT_PROFILES[robot_profile]

        self.declare_parameter("map_root", "/var/lib/omni/maps")
        self.declare_parameter("map_source_path", "/tmp/omni_slam_work/map.pcd")
        self.declare_parameter(
            "mapping_command",
            profile["mapping_command"],
        )
        self.declare_parameter(
            "localization_command",
            profile["localization_command"],
        )
        self.declare_parameter("map_save_service", "/map_save")
        self.declare_parameter("cloud_topic", "/cloud_registered_global")
        self.declare_parameter("imu_topic", profile["imu_topic"])
        self.declare_parameter("odom_topic", "/state_estimation_global")
        self.declare_parameter("initial_pose_topic", "/icp_result")
        self.declare_parameter("status_period_ms", 500)
        self.declare_parameter("sensor_fresh_timeout_ms", 1000)
        self.declare_parameter("localization_lost_timeout_ms", 3000)
        self.declare_parameter("process_stop_timeout_sec", 5.0)
        self.declare_parameter("map_save_timeout_sec", 30.0)

        self._map_source_path = Path(
            self.get_parameter("map_source_path").value
        ).expanduser().resolve()
        if not self._map_source_path.is_absolute():
            raise ValueError("map_source_path must be absolute")
        self._map_source_path.parent.mkdir(parents=True, exist_ok=True)
        self._store = MapStore(self.get_parameter("map_root").value)

        self._mapping_command = str(self.get_parameter("mapping_command").value)
        self._localization_command = str(
            self.get_parameter("localization_command").value
        )
        self._stop_timeout = float(self.get_parameter("process_stop_timeout_sec").value)
        self._save_timeout = float(self.get_parameter("map_save_timeout_sec").value)
        self._fresh_timeout = (
            float(self.get_parameter("sensor_fresh_timeout_ms").value) / 1000.0
        )
        self._lost_timeout = (
            float(self.get_parameter("localization_lost_timeout_ms").value) / 1000.0
        )
        if self._fresh_timeout <= 0.0 or self._lost_timeout < self._fresh_timeout:
            raise ValueError("localization timeouts are invalid")

        self._last_cloud: Optional[float] = None
        self._last_imu: Optional[float] = None
        self._last_odom: Optional[float] = None
        self._last_initial_pose: Optional[float] = None
        self._pose_covariance_norm = math.nan
        self._active_asset: Optional[MapAsset] = None
        self._mapping_dirty = False
        self._mapping_session_id = ""

        self.get_logger().info(f"Using SLAM robot profile: {robot_profile}")

        status_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._status_publisher = self.create_publisher(
            SlamStatus, "/omni/slam/status", status_qos
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("cloud_topic").value),
            self._cloud_callback,
            10,
            callback_group=self._callbacks,
        )
        self.create_subscription(
            Imu,
            str(self.get_parameter("imu_topic").value),
            self._imu_callback,
            20,
            callback_group=self._callbacks,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self._odom_callback,
            20,
            callback_group=self._callbacks,
        )
        self.create_subscription(
            PoseWithCovarianceStamped,
            str(self.get_parameter("initial_pose_topic").value),
            self._initial_pose_callback,
            10,
            callback_group=self._callbacks,
        )
        self._map_save_client = self.create_client(
            Trigger,
            str(self.get_parameter("map_save_service").value),
            callback_group=self._callbacks,
        )
        self.create_service(
            StartMapping,
            "/omni/slam/start_mapping",
            self._start_mapping,
            callback_group=self._callbacks,
        )
        self.create_service(
            StopMapping,
            "/omni/slam/stop_mapping",
            self._stop_mapping,
            callback_group=self._callbacks,
        )
        self.create_service(
            StopLocalization,
            "/omni/slam/stop_localization",
            self._stop_localization_service,
            callback_group=self._callbacks,
        )
        self._save_action = ActionServer(
            self,
            SaveMap,
            "/omni/slam/save_map",
            execute_callback=self._execute_save_map,
            goal_callback=self._save_goal,
            cancel_callback=lambda _: CancelResponse.REJECT,
            callback_group=self._callbacks,
        )
        self._localization_action = ActionServer(
            self,
            StartLocalization,
            "/omni/slam/start_localization",
            execute_callback=self._execute_localization,
            goal_callback=self._localization_goal,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=self._callbacks,
        )
        period = max(100, int(self.get_parameter("status_period_ms").value)) / 1000.0
        self.create_timer(period, self._monitor, callback_group=self._callbacks)
        self._publish_status()

    @staticmethod
    def _now_steady() -> float:
        return time.monotonic()

    def _cloud_callback(self, _: PointCloud2) -> None:
        with self._lock:
            self._last_cloud = self._now_steady()

    def _imu_callback(self, _: Imu) -> None:
        with self._lock:
            self._last_imu = self._now_steady()

    def _odom_callback(self, message: Odometry) -> None:
        with self._lock:
            self._last_odom = self._now_steady()
            covariance = message.pose.covariance
            self._pose_covariance_norm = math.sqrt(sum(value * value for value in covariance))

    def _initial_pose_callback(self, _: PoseWithCovarianceStamped) -> None:
        with self._lock:
            self._last_initial_pose = self._now_steady()

    def _start_mapping(self, request, response):
        with self._lock:
            if self._machine.snapshot.state != SlamState.STOPPED:
                return self._service_reject(response, "SLAM is not stopped")
            if not request.session_id.strip():
                return self._service_reject(response, "session_id is required")
            self._reset_telemetry()
            self._mapping_dirty = True
            self._mapping_session_id = request.session_id.strip()
            self._active_asset = None
            self._machine.transition(SlamState.STARTING_MAPPING)
            try:
                self._process.start(
                    self._mapping_command,
                    {"map_source_path": str(self._map_source_path)},
                )
            except ProcessSupervisorError as error:
                self._machine.fail(REASON_PROCESS_START_FAILED, str(error))
                return self._service_reject(response, str(error), REASON_PROCESS_START_FAILED)
        response.accepted = True
        response.reason_code = REASON_NONE
        response.message = "mapping process started; waiting for fresh telemetry"
        self._publish_status()
        return response

    def _stop_mapping(self, request, response):
        with self._lock:
            state = self._machine.snapshot.state
            if state not in (
                SlamState.STARTING_MAPPING,
                SlamState.MAPPING,
                SlamState.MAP_READY,
                SlamState.ERROR,
            ):
                return self._service_reject(response, "mapping is not running")
            if self._mapping_dirty and not request.discard_unsaved:
                return self._service_reject(
                    response, "unsaved mapping data exists; save it or confirm discard"
                )
            if state != SlamState.ERROR:
                self._machine.transition(SlamState.STOPPING)
            else:
                self._machine.transition(SlamState.STOPPING)
            self._process.stop(self._stop_timeout)
            self._machine.transition(SlamState.STOPPED)
            self._mapping_dirty = False
        response.accepted = True
        response.reason_code = REASON_NONE
        response.message = "mapping stopped"
        self._publish_status()
        return response

    @staticmethod
    def _service_reject(response, message: str, reason: int = REASON_INVALID_STATE):
        response.accepted = False
        response.reason_code = reason
        response.message = message
        return response

    def _stop_localization_service(self, request, response):
        del request
        with self._lock:
            state = self._machine.snapshot.state
            if state not in (
                SlamState.STARTING_LOCALIZATION,
                SlamState.RELOCALIZING,
                SlamState.LOCALIZED,
                SlamState.DEGRADED,
                SlamState.LOST,
                SlamState.ERROR,
            ):
                return self._service_reject(response, "localization is not running")
        self._stop_localization(REASON_NONE, "localization stopped by request")
        response.accepted = True
        response.reason_code = REASON_NONE
        response.message = "localization stopped"
        self._publish_status()
        return response

    def _save_goal(self, goal_request) -> GoalResponse:
        try:
            MapStore.validate_map_id(goal_request.map_id)
        except MapStoreError:
            return GoalResponse.REJECT
        with self._lock:
            return (
                GoalResponse.ACCEPT
                if self._machine.snapshot.state in (SlamState.MAPPING, SlamState.MAP_READY)
                else GoalResponse.REJECT
            )

    def _execute_save_map(self, goal_handle):
        result = SaveMap.Result()
        with self._lock:
            if self._machine.snapshot.state not in (SlamState.MAPPING, SlamState.MAP_READY):
                return self._abort_save(goal_handle, result, "mapping is not ready")
            self._machine.transition(SlamState.SAVING)
        self._publish_save_feedback(
            goal_handle,
            SaveMap.Feedback.STAGE_REQUESTING_FAST_LIO_SAVE,
            0.1,
            "requesting FAST-LIO map save",
        )
        if not self._map_save_client.wait_for_service(timeout_sec=3.0):
            return self._save_failed(
                goal_handle, result, REASON_MAP_SAVE_UNAVAILABLE, "map_save service unavailable"
            )
        future = self._map_save_client.call_async(Trigger.Request())
        if not self._wait_future(future, self._save_timeout):
            return self._save_failed(
                goal_handle, result, REASON_MAP_SAVE_FAILED, "map_save service timed out"
            )
        try:
            response = future.result()
        except Exception as error:
            return self._save_failed(
                goal_handle, result, REASON_MAP_SAVE_FAILED, f"map_save call failed: {error}"
            )
        if response is None or not response.success:
            message = "map_save failed" if response is None else response.message
            return self._save_failed(
                goal_handle, result, REASON_MAP_SAVE_FAILED, message
            )

        self._publish_save_feedback(
            goal_handle,
            SaveMap.Feedback.STAGE_IMPORTING_ASSET,
            0.5,
            "importing immutable map asset",
        )
        try:
            asset = self._store.import_pcd(
                goal_handle.request.map_id,
                self._map_source_path,
                calibration_hash=goal_handle.request.calibration_hash,
                metadata={"mapping_session_id": self._mapping_session_id},
                publish_current=goal_handle.request.make_current,
            )
            self._store.verify(asset)
        except (OSError, MapStoreError) as error:
            return self._save_failed(
                goal_handle, result, REASON_MAP_IMPORT_FAILED, str(error)
            )

        self._publish_save_feedback(
            goal_handle,
            SaveMap.Feedback.STAGE_VERIFYING_CHECKSUM,
            1.0,
            "map checksum verified",
        )
        with self._lock:
            self._active_asset = asset
            self._mapping_dirty = False
            self._machine.transition(SlamState.MAP_READY)
        result.success = True
        result.map_id = asset.map_id
        result.map_version = asset.version
        result.map_path = str(asset.pcd_path)
        result.map_checksum = asset.checksum
        result.reason_code = REASON_NONE
        result.message = "map asset saved and verified"
        goal_handle.succeed()
        self._publish_status()
        return result

    def _save_failed(self, goal_handle, result, reason: int, message: str):
        with self._lock:
            if self._machine.snapshot.state == SlamState.SAVING:
                self._machine.transition(
                    SlamState.MAPPING, reason_code=reason, reason_text=message
                )
        return self._abort_save(goal_handle, result, message, reason)

    @staticmethod
    def _abort_save(goal_handle, result, message: str, reason: int = REASON_INVALID_STATE):
        result.success = False
        result.reason_code = reason
        result.message = message
        goal_handle.abort()
        return result

    @staticmethod
    def _publish_save_feedback(goal_handle, stage: int, progress: float, message: str):
        feedback = SaveMap.Feedback()
        feedback.stage = stage
        feedback.progress = progress
        feedback.message = message
        goal_handle.publish_feedback(feedback)

    def _localization_goal(self, goal_request) -> GoalResponse:
        initial_guess = (
            goal_request.initial_x,
            goal_request.initial_y,
            goal_request.initial_z,
            goal_request.initial_yaw,
        )
        if not all(math.isfinite(float(value)) for value in initial_guess):
            return GoalResponse.REJECT
        with self._lock:
            if self._machine.snapshot.state != SlamState.STOPPED:
                return GoalResponse.REJECT
        try:
            self._select_asset(goal_request.map_id, goal_request.map_version)
        except (OSError, MapStoreError):
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _execute_localization(self, goal_handle):
        result = StartLocalization.Result()
        try:
            asset = self._select_asset(
                goal_handle.request.map_id, goal_handle.request.map_version
            )
        except (OSError, MapStoreError) as error:
            return self._finish_localization_failure(
                goal_handle, result, REASON_MAP_INVALID, str(error)
            )
        timeout = max(1.0, float(goal_handle.request.timeout_sec))
        started = self._now_steady()
        with self._lock:
            if self._machine.snapshot.state != SlamState.STOPPED:
                return self._finish_localization_failure(
                    goal_handle, result, REASON_INVALID_STATE, "SLAM is not stopped"
                )
            self._reset_telemetry()
            self._active_asset = asset
            self._machine.transition(SlamState.STARTING_LOCALIZATION)
            try:
                self._process.start(
                    self._localization_command,
                    {
                        "map_path": str(asset.pcd_path),
                        "initial_x": str(float(goal_handle.request.initial_x)),
                        "initial_y": str(float(goal_handle.request.initial_y)),
                        "initial_z": str(float(goal_handle.request.initial_z)),
                        "initial_yaw": str(float(goal_handle.request.initial_yaw)),
                    },
                )
                self._machine.transition(SlamState.RELOCALIZING)
            except ProcessSupervisorError as error:
                self._machine.fail(REASON_PROCESS_START_FAILED, str(error))
                return self._finish_localization_failure(
                    goal_handle, result, REASON_PROCESS_START_FAILED, str(error)
                )
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self._stop_localization(REASON_LOCALIZATION_CANCELLED, "localization cancelled")
                result.success = False
                result.final_state = int(SlamState.STOPPED)
                result.reason_code = REASON_LOCALIZATION_CANCELLED
                result.message = "localization cancelled"
                goal_handle.canceled()
                return result
            with self._lock:
                snapshot = self._machine.snapshot
            if snapshot.state == SlamState.LOCALIZED:
                result.success = True
                result.final_state = int(snapshot.state)
                result.reason_code = REASON_NONE
                result.message = "localization established"
                goal_handle.succeed()
                return result
            if snapshot.state == SlamState.ERROR:
                return self._finish_localization_failure(
                    goal_handle,
                    result,
                    snapshot.reason_code or REASON_PROCESS_EXITED,
                    snapshot.reason_text or "localization process failed",
                )
            elapsed = self._now_steady() - started
            if elapsed >= timeout:
                self._stop_localization(
                    REASON_LOCALIZATION_TIMEOUT, "localization timed out"
                )
                return self._finish_localization_failure(
                    goal_handle,
                    result,
                    REASON_LOCALIZATION_TIMEOUT,
                    "localization timed out",
                )
            feedback = StartLocalization.Feedback()
            feedback.state = int(snapshot.state)
            feedback.elapsed_sec = float(elapsed)
            feedback.fitness_score = math.nan
            feedback.message = snapshot.reason_text
            goal_handle.publish_feedback(feedback)
            time.sleep(0.1)
        return self._finish_localization_failure(
            goal_handle, result, REASON_PROCESS_EXITED, "ROS shutdown"
        )

    def _select_asset(self, map_id: str, version: int) -> MapAsset:
        return self._store.current(map_id) if version == 0 else self._store.version(map_id, version)

    def _stop_localization(self, reason: int, message: str) -> None:
        with self._lock:
            state = self._machine.snapshot.state
            if state not in (SlamState.STOPPED, SlamState.STOPPING):
                try:
                    self._machine.transition(
                        SlamState.STOPPING, reason_code=reason, reason_text=message
                    )
                except InvalidTransition:
                    self._machine.fail(reason, message)
                    self._machine.transition(SlamState.STOPPING)
            self._process.stop(self._stop_timeout)
            if self._machine.snapshot.state == SlamState.STOPPING:
                self._machine.transition(SlamState.STOPPED)

    @staticmethod
    def _finish_localization_failure(goal_handle, result, reason: int, message: str):
        result.success = False
        result.final_state = int(SlamState.ERROR)
        result.reason_code = reason
        result.message = message
        if goal_handle.status not in (GoalStatus.STATUS_CANCELED, GoalStatus.STATUS_ABORTED):
            goal_handle.abort()
        return result

    @staticmethod
    def _wait_future(future, timeout: float) -> bool:
        completed = threading.Event()
        future.add_done_callback(lambda _: completed.set())
        return completed.wait(timeout=max(0.0, timeout))

    def _monitor(self) -> None:
        with self._lock:
            exit_code = self._process.reap_exit()
            state = self._machine.snapshot.state
            if exit_code is not None and state not in (SlamState.STOPPED, SlamState.STOPPING):
                self._machine.fail(
                    REASON_PROCESS_EXITED, f"SLAM process exited with code {exit_code}"
                )
            else:
                now = self._now_steady()
                fresh = self._telemetry_fresh(now, self._fresh_timeout)
                if state == SlamState.STARTING_MAPPING and fresh:
                    self._machine.transition(SlamState.MAPPING)
                elif state == SlamState.RELOCALIZING:
                    if (
                        fresh
                        and self._last_initial_pose is not None
                        and self._last_odom is not None
                        and self._last_odom >= self._last_initial_pose
                    ):
                        self._machine.transition(SlamState.LOCALIZED)
                elif state == SlamState.LOCALIZED and not fresh:
                    self._machine.transition(
                        SlamState.DEGRADED,
                        reason_code=REASON_LOCALIZATION_TIMEOUT,
                        reason_text="SLAM telemetry is stale",
                    )
                elif state == SlamState.DEGRADED:
                    if fresh:
                        self._machine.transition(SlamState.LOCALIZED)
                    elif not self._telemetry_fresh(now, self._lost_timeout):
                        self._machine.transition(
                            SlamState.LOST,
                            reason_code=REASON_LOCALIZATION_TIMEOUT,
                            reason_text="SLAM telemetry exceeded lost timeout",
                        )
        self._publish_status()

    def _telemetry_fresh(self, now: float, timeout: float) -> bool:
        return all(
            stamp is not None and now - stamp <= timeout
            for stamp in (self._last_cloud, self._last_imu, self._last_odom)
        )

    def _reset_telemetry(self) -> None:
        self._last_cloud = None
        self._last_imu = None
        self._last_odom = None
        self._last_initial_pose = None
        self._pose_covariance_norm = math.nan

    def _age_ms(self, stamp: Optional[float], now: float) -> float:
        return math.nan if stamp is None else max(0.0, (now - stamp) * 1000.0)

    def _publish_status(self) -> None:
        with self._lock:
            snapshot = self._machine.snapshot
            asset = self._active_asset
            now = self._now_steady()
            message = SlamStatus()
            message.stamp = self.get_clock().now().to_msg()
            message.mode = int(snapshot.mode)
            message.state = int(snapshot.state)
            message.map_id = "" if asset is None else asset.map_id
            message.map_version = 0 if asset is None else asset.version
            message.map_checksum = "" if asset is None else asset.checksum
            message.initialized = snapshot.state in (
                SlamState.MAPPING,
                SlamState.SAVING,
                SlamState.MAP_READY,
                SlamState.LOCALIZED,
                SlamState.DEGRADED,
            ) and self._telemetry_fresh(now, self._fresh_timeout)
            message.lidar_age_ms = self._age_ms(self._last_cloud, now)
            message.imu_age_ms = self._age_ms(self._last_imu, now)
            message.odom_age_ms = self._age_ms(self._last_odom, now)
            message.fitness_score = math.nan
            message.pose_covariance_norm = self._pose_covariance_norm
            message.pose_jump_count = 0
            message.reason_code = snapshot.reason_code
            message.reason_text = snapshot.reason_text
            message.status_sequence = snapshot.sequence
        self._status_publisher.publish(message)

    def destroy_node(self):
        self._save_action.destroy()
        self._localization_action.destroy()
        self._process.stop(self._stop_timeout)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SlamManagerNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
