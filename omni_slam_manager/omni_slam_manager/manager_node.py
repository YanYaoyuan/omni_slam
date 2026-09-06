"""ROS 2 orchestration node for mapping, map assets and localization."""

from __future__ import annotations

import math
import os
from pathlib import Path
import threading
import time
from typing import Optional

from ament_index_python.packages import get_package_share_directory
import rclpy
from action_msgs.msg import GoalStatus
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.impl.implementation_singleton import rclpy_implementation
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, PointCloud2
from std_msgs.msg import Float64
from std_srvs.srv import Trigger

from omni_robot_interfaces.srv import ListMaps
from omni_slam_interfaces.action import SaveMap, StartLocalization
from omni_slam_interfaces.srv import StartMapping, StopLocalization, StopMapping
from omni_tf_manager.msg import SlamStatus

from .cancel_wait import wait_with_cancel
from .instance_lock import InstanceLock
from .map_store import MapAsset, MapStore, MapStoreError, MapStoreImportCancelled
from .process_supervisor import ProcessSupervisor, ProcessSupervisorError
from .state_machine import InvalidTransition, SlamState, SlamStateMachine
from .telemetry_monitor import (
    FitnessTracker,
    PoseJumpTracker,
    PoseSample,
    yaw_from_quaternion,
)
from .tf_profile import load_tf_frames


REASON_NONE = 0
REASON_INVALID_STATE = 1001
REASON_PROCESS_START_FAILED = 1002
REASON_PROCESS_EXITED = 1003
REASON_MAP_SAVE_UNAVAILABLE = 1101
REASON_MAP_SAVE_FAILED = 1102
REASON_MAP_IMPORT_FAILED = 1103
REASON_SAVE_CANCELLED = 1104
REASON_MAP_INVALID = 1201
REASON_LOCALIZATION_TIMEOUT = 1202
REASON_LOCALIZATION_CANCELLED = 1203


ROBOT_PROFILES = {
    "omni_dog": {
        "mapping_command": (
            "ros2 launch fast_lio omni_dog.launch.py "
            "map_path:={map_source_path} map_frame:={map_frame} odom_frame:={odom_frame} "
            "base_frame:={base_frame} tracking_frame:={tracking_frame} "
            "lidar_frame:={icp_sensor_frame}"
        ),
        "localization_command": (
            "ros2 launch fast_lio omni_dog_relocalization.launch.py "
            "map_path:={map_path} initial_x:={initial_x} initial_y:={initial_y} "
            "initial_z:={initial_z} initial_yaw:={initial_yaw} "
            "map_frame:={map_frame} odom_frame:={odom_frame} "
            "base_frame:={base_frame} tracking_frame:={tracking_frame} "
            "lidar_frame:={icp_sensor_frame}"
        ),
        "imu_topic": "/omni/sensors/imu/data",
    },
    "vbot": {
        "mapping_command": (
            "ros2 launch fast_lio vbot.launch.py "
            "map_path:={map_source_path} map_frame:={map_frame} odom_frame:={odom_frame} "
            "base_frame:={base_frame} tracking_frame:={tracking_frame} "
            "lidar_frame:={icp_sensor_frame}"
        ),
        "localization_command": (
            "ros2 launch fast_lio vbot_relocalization.launch.py "
            "map_path:={map_path} initial_x:={initial_x} initial_y:={initial_y} "
            "initial_z:={initial_z} initial_yaw:={initial_yaw} "
            "map_frame:={map_frame} odom_frame:={odom_frame} "
            "base_frame:={base_frame} tracking_frame:={tracking_frame} "
            "lidar_frame:={icp_sensor_frame}"
        ),
        "imu_topic": "/omni/sensors/imu/data",
    },
    "matrix": {
        # MATRiX 与真机复用同一套 FAST-LIO launch，只替换受版本管理的传感器
        # 参数。进程仍由本 Manager 的 ProcessSupervisor 创建和回收，不能再由
        # Bridge 或 APP 调用 run_matrix_*.sh 形成第二个运行时 owner。
        "mapping_command": (
            "ros2 launch fast_lio omni_dog.launch.py "
            "config_file:={mapping_config_file} map_path:={map_source_path} "
            "map_frame:={map_frame} odom_frame:={odom_frame} "
            "base_frame:={base_frame} tracking_frame:={tracking_frame} "
            "lidar_frame:={icp_sensor_frame} use_sim_time:=false"
        ),
        "localization_command": (
            "ros2 launch fast_lio omni_dog_relocalization.launch.py "
            "config_file:={localization_config_file} map_path:={map_path} "
            "pointcloud_topic:=/omni/sensors/lidar/points "
            "initial_x:={initial_x} initial_y:={initial_y} "
            "initial_z:={initial_z} initial_yaw:={initial_yaw} "
            "map_frame:={map_frame} odom_frame:={odom_frame} "
            "base_frame:={base_frame} tracking_frame:={tracking_frame} "
            "lidar_frame:={icp_sensor_frame}"
        ),
        "mapping_config": "matrix_sim.yaml",
        "localization_config": "matrix_sim_relocalization.yaml",
        "imu_topic": "/omni/sensors/imu/data",
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

        mapping_config_default = ""
        localization_config_default = ""
        if robot_profile == "matrix":
            # 仅 Matrix profile 需要额外配置文件。其他机器人不能因未安装可选的
            # Matrix FAST-LIO 资源而在 Manager 构造阶段失败。
            fast_lio_share = Path(get_package_share_directory("fast_lio"))
            mapping_config_default = str(
                fast_lio_share / "config" / profile["mapping_config"]
            )
            localization_config_default = str(
                fast_lio_share / "config" / profile["localization_config"]
            )

        self.declare_parameter("tf_profile_path", "")
        tf_profile_path = str(self.get_parameter("tf_profile_path").value)
        tf_frames = load_tf_frames(tf_profile_path)
        self.declare_parameter("frames.map", tf_frames["map"])
        self.declare_parameter("frames.odom", tf_frames["odom"])
        self.declare_parameter("frames.base", tf_frames["base"])
        self.declare_parameter("frames.tracking", tf_frames["tracking"])
        self.declare_parameter("frames.icp_sensor", tf_frames["icp_sensor"])
        self._tf_frames = {
            role: str(self.get_parameter(f"frames.{role}").value)
            for role in tf_frames
        }
        invalid_frames = [
            frame for frame in self._tf_frames.values() if not frame.startswith("omni_")
        ]
        if invalid_frames or len(set(self._tf_frames.values())) != len(self._tf_frames):
            raise ValueError("core TF frames must be unique and begin with 'omni_'")

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
        self.declare_parameter("mapping_config_file", mapping_config_default)
        self.declare_parameter(
            "localization_config_file", localization_config_default
        )
        self.declare_parameter("map_save_service", "/map_save")
        self.declare_parameter("cloud_topic", "/cloud_registered_global")
        self.declare_parameter("imu_topic", profile["imu_topic"])
        self.declare_parameter("odom_topic", "/omni/tf_manager/body_odom_global")
        self.declare_parameter("initial_pose_topic", "/icp_result")
        self.declare_parameter("status_period_ms", 500)
        self.declare_parameter("sensor_fresh_timeout_ms", 1000)
        self.declare_parameter("localization_lost_timeout_ms", 3000)
        self.declare_parameter("process_stop_timeout_sec", 5.0)
        self.declare_parameter("map_save_timeout_sec", 30.0)
        self.declare_parameter("icp_fitness_topic", "/icp_fitness")
        self.declare_parameter("pose_jump_position_threshold_m", 0.5)
        self.declare_parameter("pose_jump_yaw_threshold_deg", 45.0)
        self.declare_parameter("pose_jump_refractory_sec", 0.5)
        self.declare_parameter("external_runtime_mode", "none")
        self.declare_parameter("external_map_id", "")
        self.declare_parameter("external_map_version", 0)

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
        self._mapping_config_file = str(
            self.get_parameter("mapping_config_file").value
        )
        self._localization_config_file = str(
            self.get_parameter("localization_config_file").value
        )
        if robot_profile == "matrix":
            for parameter_name, config_path in (
                ("mapping_config_file", self._mapping_config_file),
                ("localization_config_file", self._localization_config_file),
            ):
                path = Path(config_path).expanduser()
                if (
                    not path.is_absolute()
                    or not path.is_file()
                    or not os.access(path, os.R_OK)
                ):
                    raise ValueError(
                        f"{parameter_name} must reference a readable installed "
                        "FAST-LIO yaml"
                    )
                if parameter_name == "mapping_config_file":
                    self._mapping_config_file = str(path.resolve())
                else:
                    self._localization_config_file = str(path.resolve())
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
        self._pose_jumps = PoseJumpTracker(
            position_threshold_m=float(
                self.get_parameter("pose_jump_position_threshold_m").value
            ),
            yaw_threshold_rad=math.radians(
                float(self.get_parameter("pose_jump_yaw_threshold_deg").value)
            ),
            refractory_sec=float(
                self.get_parameter("pose_jump_refractory_sec").value
            ),
        )
        self._fitness = FitnessTracker()
        self._active_asset: Optional[MapAsset] = None
        self._mapping_dirty = False
        self._mapping_session_id = ""
        self._external_runtime_mode = str(
            self.get_parameter("external_runtime_mode").value
        ).strip().lower()
        if self._external_runtime_mode not in ("none", "mapping", "localization"):
            raise ValueError(
                "external_runtime_mode must be one of: none, mapping, localization"
            )
        self._external_map_id = str(self.get_parameter("external_map_id").value).strip()
        self._external_map_version = int(
            self.get_parameter("external_map_version").value
        )
        if self._external_map_version < 0:
            raise ValueError("external_map_version must be non-negative")

        # MATRiX 脚本直接启动 FAST-LIO，SLAM Manager 不应再创建第二个后端进程。
        # 观察模式只依据标准遥测推进状态机，仍由本节点独占 SlamStatus 发布权。
        if self._external_runtime_mode == "mapping":
            self._mapping_dirty = True
            self._mapping_session_id = "external-runtime"
            self._machine.transition(
                SlamState.STARTING_MAPPING,
                reason_text="observing externally managed mapping runtime",
            )
        elif self._external_runtime_mode == "localization":
            self._machine.transition(
                SlamState.STARTING_LOCALIZATION,
                reason_text="observing externally managed localization runtime",
            )
            self._machine.transition(
                SlamState.RELOCALIZING,
                reason_text="waiting for external ICP and fresh telemetry",
            )

        self.get_logger().info(f"Using SLAM robot profile: {robot_profile}")
        self.get_logger().info(
            "Using omni_tf_manager frame contract: "
            + ", ".join(f"{role}={frame}" for role, frame in self._tf_frames.items())
        )

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
            Odometry,
            str(self.get_parameter("initial_pose_topic").value),
            self._initial_pose_callback,
            10,
            callback_group=self._callbacks,
        )
        self.create_subscription(
            Float64,
            str(self.get_parameter("icp_fitness_topic").value),
            self._icp_fitness_callback,
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
        self.create_service(
            ListMaps,
            "/omni/slam/maps/list",
            self._list_maps,
            callback_group=self._callbacks,
        )
        self._save_action = ActionServer(
            self,
            SaveMap,
            "/omni/slam/save_map",
            execute_callback=self._execute_save_map,
            goal_callback=self._save_goal,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
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
            now = self._now_steady()
            self._last_odom = now
            covariance = message.pose.covariance
            self._pose_covariance_norm = math.sqrt(sum(value * value for value in covariance))
            pose = message.pose.pose
            self._pose_jumps.observe(
                PoseSample(
                    x=pose.position.x,
                    y=pose.position.y,
                    z=pose.position.z,
                    yaw=yaw_from_quaternion(
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w,
                    ),
                    stamp=now,
                )
            )

    def _initial_pose_callback(self, _: Odometry) -> None:
        with self._lock:
            self._last_initial_pose = self._now_steady()

    def _list_maps(self, _request, response):
        """只发布校验通过的 current 地图三元组，供 Mission 对外转发。"""
        for asset in self._store.list_current():
            response.map_ids.append(asset.map_id)
            response.map_versions.append(asset.version)
            response.map_checksums.append(asset.checksum)
            response.created_at.append(asset.created_at)
            response.size_bytes.append(asset.size_bytes)
        return response

    def _icp_fitness_callback(self, message: Float64) -> None:
        with self._lock:
            self._fitness.observe(float(message.data))

    def _start_mapping(self, request, response):
        with self._lock:
            if self._external_runtime_mode != "none":
                return self._service_reject(
                    response, "SLAM process lifecycle is externally managed"
                )
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
                    {
                        "map_source_path": str(self._map_source_path),
                        "map_frame": self._tf_frames["map"],
                        "odom_frame": self._tf_frames["odom"],
                        "base_frame": self._tf_frames["base"],
                        "tracking_frame": self._tf_frames["tracking"],
                        "icp_sensor_frame": self._tf_frames["icp_sensor"],
                        "mapping_config_file": self._mapping_config_file,
                    },
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
            if self._external_runtime_mode != "none":
                return self._service_reject(
                    response, "SLAM process lifecycle is externally managed"
                )
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
            if self._external_runtime_mode != "none":
                return self._service_reject(
                    response, "SLAM process lifecycle is externally managed"
                )
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
        if goal_handle.is_cancel_requested:
            return self._cancel_save(goal_handle, result)
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
        # rclpy exposes is_cancel_requested as a *property*: pass a
        # zero-arg callable, not the bool value.
        if not wait_with_cancel(
            future.done,
            lambda: goal_handle.is_cancel_requested,
            self._save_timeout,
        ):
            if goal_handle.is_cancel_requested:
                return self._cancel_save(goal_handle, result)
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
        if goal_handle.is_cancel_requested:
            return self._cancel_save(goal_handle, result)

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
                cancel_requested=lambda: goal_handle.is_cancel_requested,
            )
        except MapStoreImportCancelled:
            return self._cancel_save(goal_handle, result)
        except (OSError, MapStoreError) as error:
            return self._save_failed(
                goal_handle, result, REASON_MAP_IMPORT_FAILED, str(error)
            )

        # import_pcd 返回即表示版本（以及可选 current）已经原子提交。提交点之后的
        # cancel 不能再报告 CANCELED，否则客户端看到的终态会与磁盘事实矛盾；该
        # 请求按已提交成功返回，后续删除/切换必须使用显式资产管理操作。
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

    def _cancel_save(self, goal_handle, result: SaveMap.Result) -> SaveMap.Result:
        # MapStore 只会在不可取消提交点之前抛出 MapStoreImportCancelled；因此走到
        # 这里时不会存在已发布的新版本，仅需把运行状态退回 MAPPING。
        with self._lock:
            if self._machine.snapshot.state == SlamState.SAVING:
                self._machine.transition(
                    SlamState.MAPPING,
                    reason_code=REASON_SAVE_CANCELLED,
                    reason_text="map save canceled before the asset commit point",
                )
        result.success = False
        result.reason_code = REASON_SAVE_CANCELLED
        result.message = "map save canceled before commit; no map version was published"
        goal_handle.canceled()
        self._publish_status()
        return result

    @staticmethod
    def _publish_save_feedback(goal_handle, stage: int, progress: float, message: str):
        feedback = SaveMap.Feedback()
        feedback.stage = stage
        feedback.progress = progress
        feedback.message = message
        goal_handle.publish_feedback(feedback)

    def _localization_goal(self, goal_request) -> GoalResponse:
        if self._external_runtime_mode != "none":
            return GoalResponse.REJECT
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
                        "map_frame": self._tf_frames["map"],
                        "odom_frame": self._tf_frames["odom"],
                        "base_frame": self._tf_frames["base"],
                        "tracking_frame": self._tf_frames["tracking"],
                        "icp_sensor_frame": self._tf_frames["icp_sensor"],
                        "localization_config_file": self._localization_config_file,
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
                fitness = self._fitness.value
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
            feedback.fitness_score = float(fitness)
            feedback.message = snapshot.reason_text
            goal_handle.publish_feedback(feedback)
            time.sleep(0.1)
        return self._finish_localization_failure(
            goal_handle, result, REASON_PROCESS_EXITED, "ROS shutdown"
        )

    def _select_asset(self, map_id: str, version: int) -> MapAsset:
        if version == 0:
            return self._store.current(map_id)
        return self._store.version(map_id, version)

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
                elif (
                    state == SlamState.LOST
                    and self._external_runtime_mode == "localization"
                    and fresh
                ):
                    # 外部后端可能在不重启 Manager 的情况下恢复。先回到重定位态，
                    # 下一次新的 ICP 结果和随后 odom 会再次证明 LOCALIZED。
                    self._last_initial_pose = None
                    self._machine.transition(
                        SlamState.RELOCALIZING,
                        reason_text="external telemetry recovered; waiting for ICP",
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
        self._pose_jumps.reset()
        self._fitness.reset()

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
            message.map_id = (
                self._external_map_id if asset is None else asset.map_id
            )
            message.map_version = (
                self._external_map_version if asset is None else asset.version
            )
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
            message.fitness_score = float(self._fitness.value)
            message.pose_covariance_norm = self._pose_covariance_norm
            message.pose_jump_count = self._pose_jumps.count
            message.reason_code = snapshot.reason_code
            message.reason_text = snapshot.reason_text
            message.status_sequence = snapshot.sequence
        self._status_publisher.publish(message)

    def destroy_node(self):
        self._save_action.destroy()
        self._localization_action.destroy()
        self._process.stop(self._stop_timeout)
        return super().destroy_node()


def _spin_until_context_shutdown(executor: MultiThreadedExecutor) -> None:
    """运行执行器，并只吞掉由 ROS context 正常关闭产生的退出异常。"""
    try:
        executor.spin()
    except ExternalShutdownException:
        # 其他节点触发 context shutdown 时，rclpy 会用该异常唤醒执行器。
        pass
    except rclpy_implementation.RCLError:
        # ROS 2 Humble 在 SIGINT 与 WaitSet 初始化并发时可能抛出 RCLError。
        # 仅当 context 已关闭才视为正常退出，避免掩盖真实的 RCL 故障。
        if rclpy.ok():
            raise


def main(args=None) -> None:
    lock_path = os.environ.get(
        "OMNI_SLAM_MANAGER_LOCK_FILE", "/tmp/omni/slam-manager.lock"
    )
    # 锁必须早于 rclpy.init 和任何 ROS provider 创建，否则两个并发实例会在
    # 发现彼此前短暂同时发布状态并接受建图命令。
    with InstanceLock(lock_path):
        node = None
        executor = None
        initialized = False
        try:
            rclpy.init(args=args)
            initialized = True
            node = SlamManagerNode()
            executor = MultiThreadedExecutor(num_threads=4)
            executor.add_node(node)
            _spin_until_context_shutdown(executor)
        finally:
            try:
                if executor is not None:
                    executor.shutdown()
            finally:
                try:
                    if node is not None:
                        node.destroy_node()
                finally:
                    # 节点构造或执行器注册失败时同样清理 ROS context；单例锁则在
                    # 整个清理流程完成后由 with 语句最后释放。
                    if initialized and rclpy.ok():
                        rclpy.shutdown()
