# omni_slam — Module Interface & Capability Specification

| Item | Value |
| --- | --- |
| Repository | `YanYaoyuan/omni_slam` |
| Audited commit | `1f8ff78` (2026-10-09) |
| Packages | `omni_slam_manager` (Python, control plane), `omni_slam_interfaces` (IDL), `fast_lio` (C++, LIO + loop-closure backend), `icp_relocalization` (C++, prior-map ICP) |
| Executables | `omni_slam_manager` (node `/omni_slam_manager`); `fastlio_mapping` (FAST-LIO, started as a child process); ICP nodes (`icp_node`, `sac_ia_gicp`, `global_pointcloud_publisher`); CLI `omni_slam_ctl` (diagnostics only) |
| Platforms | x86_64, NVIDIA Orin, RDK S100 (CI artefact `omni-slam-rdk-s100-aarch64`) |

## 1. Responsibility

**Owns:** LiDAR-inertial mapping; prior-map relocalization; the SLAM state machine; publication of
`SlamStatus`; the immutable versioned map store; and supervision of the FAST-LIO/ICP child processes.

**Does not own:** any TF edge in the managed tree (`publish.tf_en=false`, so `omni_tf_manager` is the
TF authority); deciding *when* to map or localize (the Mission Manager does that); or navigation.

In the product, the manager is started by `omni_mission_manager/inspection_runtime.launch.py`.
`omni_slam_ctl` is a stand-alone diagnostic entry point and must not run beside the Mission control plane.

## 2. Capabilities

| # | Capability | Status | Evidence |
| --- | --- | --- | --- |
| C1 | FAST-LIO2 mapping (Livox `CustomMsg` or `PointCloud2`, plus IMU) | Implemented | `FAST_LIO/src/laserMapping.cpp:657-671` |
| C2 | Localization in a prior map: FAST-LIO `locate_in_prior_map` with ICP initial alignment | Implemented | `laserMapping.cpp:600`, `icp_relocalization/src/icp_node.cpp` |
| C3 | Explicit-SLAM state machine with a closed transition table (12 states, 3 modes) | Implemented | `omni_slam_manager/state_machine.py:28-75` |
| C4 | Telemetry-driven health: LOCALIZED → DEGRADED (stale > 1 s) → LOST (stale > 3 s) | Implemented | `manager_node.py:870-905` |
| C5 | Versioned, immutable map store: `v%06d` directories, SHA-256 manifest, fsync, atomic `current` symlink | Implemented | `omni_slam_manager/map_store.py:93-205` |
| C6 | Map save as an Action with stages request → import → verify, cancellable before publish | Implemented | `SaveMap.action`, `manager_node.py:391` |
| C7 | Child-process supervision (start, monitor, bounded stop) | Implemented | `process_supervisor.py` |
| C8 | Single manager instance (lock) | Implemented | `instance_lock.py` |
| C9 | Pose-jump detection (0.5 m / 45°, 0.5 s refractory) reported as `pose_jump_count` | Implemented | `manager_node.py:200-202` |
| C10 | Background loop closure: keyframes (1 m / 15° / 5 s), descriptor candidates, ICP + overlap verification, GTSAM iSAM2 pose graph | **Implemented (since `ea48572`, 2026-09-24)**. Corrects the exported map and preview only, never live FAST-LIO state. Enabled in `vbot.yaml`, default off | `FAST_LIO/src/loop_closure_backend.cpp`, `include/fast_lio/loop_closure_backend.hpp:8-9` |
| C11 | Scalable map tiles and cache for ICP (20 m tiles, 80 m active radius) | Implemented | `icp_node.cpp` `map_cache.*` |
| C12 | Mapping-session body path for route recording during mapping | Implemented (opt-in `publish.mapping_body_path_en`) | `laserMapping.cpp:672-674` |
| C13 | External/observer mode (stand-alone MATRiX scripts start FAST-LIO; the manager only observes) | Implemented | `manager_node.py:275-300` |
| C14 | Online relocalization after LOST without a manager restart | Partial (only in `external_runtime_mode=localization`) | `manager_node.py:896-905` |
| C15 | Live global-map correction from loop closure fed back into localization | Not implemented (by design in C10) | — |

## 3. ROS interfaces

### 3.1 `omni_slam_manager` (node `/omni_slam_manager`)

| Kind | Name (parameter) | Type | Semantics |
| --- | --- | --- | --- |
| Pub | `/omni/slam/status` (fixed) | `omni_tf_manager/SlamStatus` | reliable, transient_local. Every `status_period_ms` (500, minimum 100). Sole publisher |
| Srv | `/omni/slam/start_mapping` | `omni_slam_interfaces/StartMapping` | `session_id` (optional) → `accepted`, `reason_code`, `message` |
| Srv | `/omni/slam/stop_mapping` | `omni_slam_interfaces/StopMapping` | `discard_unsaved` |
| Srv | `/omni/slam/stop_localization` | `omni_slam_interfaces/StopLocalization` | |
| Srv | `/omni/slam/maps/list` | `omni_robot_interfaces/ListMaps` | Map catalogue from the store. Mission re-exposes it as `/omni/maps/list` |
| Action | `/omni/slam/save_map` | `omni_slam_interfaces/SaveMap` | Goal `map_id`, `calibration_hash`, `make_current`. Feedback stage 1–3 with `progress`. Result `map_version`, `map_path`, `map_checksum` |
| Action | `/omni/slam/start_localization` | `omni_slam_interfaces/StartLocalization` | Goal `map_id`, `map_version`, `timeout_sec`, initial x/y/z/yaw. Feedback `state`, `elapsed_sec`, `fitness_score` |
| Client | `/map_save` (`map_save_service`) | `std_srvs/Trigger` | FAST-LIO save request |
| Sub | `/cloud_registered_global` (`cloud_topic`) | `sensor_msgs/PointCloud2` | SensorData. Freshness |
| Sub | `/omni/sensors/imu/data` (`imu_topic`) | `sensor_msgs/Imu` | SensorData. Freshness |
| Sub | `/omni/tf_manager/body_odom_global` (`odom_topic`) | `nav_msgs/Odometry` | SensorData. Freshness and pose jumps |
| Sub | `/icp_result` (`initial_pose_topic`) | `nav_msgs/Odometry` | depth 10. Evidence for the RELOCALIZING → LOCALIZED transition |
| Sub | `/icp_fitness` (`icp_fitness_topic`) | `std_msgs/Float64` | `fitness_score` |

### 3.2 `fast_lio` (`laserMapping.cpp`)

| Kind | Name | Type | Notes |
| --- | --- | --- | --- |
| Sub | `common.lid_topic` (default `/livox/lidar`; product `/omni/sensors/lidar/points`) | `livox_ros_driver2/CustomMsg` or `PointCloud2` | depth 20 or SensorData |
| Sub | `common.imu_topic` (default `/livox/imu`; product `/omni/sensors/imu/data`) | `sensor_msgs/Imu` | depth 1000 |
| Sub | `/icp_result` | `nav_msgs/Odometry` | Localization mode: initial pose |
| Sub | `common.reference_odom_topic` | `nav_msgs/Odometry` | Optional reference odometry (MATRiX) |
| Pub | `/cloud_registered`, `/cloud_registered_body`, `/cloud_effected`, `/ikd_tree` | `PointCloud2` | Per the `publish.*_en` flags |
| Pub | `/Laser_map` | `PointCloud2` | ≤ `publish.map_max_points` (300000) |
| Pub | `/Odometry` | `nav_msgs/Odometry` | IMU-body pose (product remaps to `/state_estimation` for the TF Manager) |
| Pub | `/path` | `nav_msgs/Path` | |
| Pub | `/omni/slam/mapping_body_path` | `nav_msgs/Path` | `publish.mapping_body_path_en` |
| Srv | `map_save` | `std_srvs/Trigger` | Writes the PCD and archive |

TF publication is disabled (`publish.tf_en=false`, `common.send_odom_base_tf=false`).

### 3.3 `icp_relocalization`

| Node | Kind | Name | Type |
| --- | --- | --- | --- |
| `icp_node` | Sub | `livox_topic` / `pointcloud_topic` | `CustomMsg` / `PointCloud2` |
| | Sub | `initialpose` | `geometry_msgs/PoseWithCovarianceStamped` |
| | Pub | `icp_result` | `nav_msgs/Odometry` (`T_map_lidar`, `frame_id=map_frame_id`, `child=sensor_frame_id`) |
| | Pub | `icp_fitness_topic` (`icp_fitness`) | `std_msgs/Float64` |
| | Pub | `prior_map`, `transformed_cloud` | `PointCloud2` |
| `sac_ia_gicp` | Pub | `/relocalization_result` | `PoseWithCovarianceStamped` |
| | Pub | `/icp_cloud`, `/template_cloud` | `PointCloud2` |
| `global_pointcloud_publisher` | Sub/Pub | cloud in, global cloud out | `PointCloud2` |

ICP parameters: `initial_{x,y,z,a}`, `solver_max_iter` (75), `max_correspondence_distance` (0.1),
`RANSAC_outlier_rejection_threshold` (1.0), `map_path`, `map_frame_id`, `sensor_frame_id`,
`fitness_score_thre`, `map_voxel_leaf_size` / `cloud_voxel_leaf_size` (0.1), `converged_count_thre` (20),
`candidate_position_tolerance_m` (0.35), `candidate_angle_tolerance_rad` (0.1745), `pcl_type` (`livox`),
`map_cache.{tile_size=20, max_cached_points=1e6, max_region_points=2e6, active_radius=80, directory}`.

## 4. State machine

```text
STOPPED → STARTING_MAPPING → MAPPING ⇄ SAVING → MAP_READY → (MAPPING | SAVING | STARTING_LOCALIZATION)
STOPPED | MAP_READY → STARTING_LOCALIZATION → RELOCALIZING → LOCALIZED ⇄ DEGRADED → LOST → RELOCALIZING
any → STOPPING → STOPPED;  any → ERROR → STOPPING | STOPPED
```

Mode is derived from state: the MAPPING states map to `MODE_MAPPING`, the localization states map to
`MODE_LOCALIZATION`, and everything else maps to `MODE_STOPPED`. `status_sequence` is strictly increasing.

- RELOCALIZING → LOCALIZED needs fresh telemetry, at least one ICP pose, and odometry newer than that pose.
- LOCALIZED → DEGRADED when telemetry is older than `sensor_fresh_timeout_ms` (1000).
- DEGRADED → LOST beyond `localization_lost_timeout_ms` (3000).

### Reason codes (`SlamStatus.reason_code`, SLAM action/service results)

| Code | Name |
| --- | --- |
| 0 | `REASON_NONE` |
| 1001 / 1002 / 1003 | `INVALID_STATE` / `PROCESS_START_FAILED` / `PROCESS_EXITED` |
| 1101 / 1102 / 1103 / 1104 | `MAP_SAVE_UNAVAILABLE` / `MAP_SAVE_FAILED` / `MAP_IMPORT_FAILED` / `SAVE_CANCELLED` |
| 1201 / 1202 / 1203 | `MAP_INVALID` / `LOCALIZATION_TIMEOUT` / `LOCALIZATION_CANCELLED` |

## 5. Parameters (`omni_slam_manager`)

| Parameter | Default | Notes |
| --- | --- | --- |
| `robot_profile` | `omni_dog` | `omni_dog` \| `vbot` \| `matrix`. Selects the FAST-LIO launch commands and the IMU topic |
| `tf_profile_path` | `""` | TF Manager profile. Frames are read from it and must be `omni_*` and unique |
| `frames.{map,odom,base,tracking,icp_sensor}` | from the TF profile | |
| `map_root` | `/var/lib/omni/maps` | Map store root |
| `map_source_path` | `/tmp/omni_slam_work/map.pcd` | FAST-LIO output to import |
| `mapping_command` / `localization_command` | per profile | Template strings with `{map_path}`, `{initial_*}` and `{*_frame}` |
| `mapping_config_file` / `localization_config_file` | `""` (matrix: `fast_lio/config/matrix_sim*.yaml`) | |
| `map_save_service` | `/map_save` | |
| `cloud_topic` / `imu_topic` / `odom_topic` / `initial_pose_topic` / `icp_fitness_topic` | see §3.1 | |
| `status_period_ms` | `500` | |
| `sensor_fresh_timeout_ms` / `localization_lost_timeout_ms` | `1000` / `3000` | |
| `process_stop_timeout_sec` / `map_save_timeout_sec` | `5.0` / `30.0` | |
| `pose_jump_position_threshold_m` / `pose_jump_yaw_threshold_deg` / `pose_jump_refractory_sec` | `0.5` / `45` / `0.5` | |
| `external_runtime_mode` / `external_map_id` / `external_map_version` | `none` / `""` / `0` | `none` \| `mapping` \| `localization` (observer mode) |

`manager.launch.py` arguments: `robot_profile` (`omni_dog`), `map_root` (`/var/lib/omni/maps`),
`tf_profile_path` (defaults to the matching `omni_tf_manager/config/*.yaml`), and `start_tf_manager` (`true`).

FAST-LIO profiles: `FAST_LIO/config/{omni_dog,vbot,matrix_sim,mid360}.yaml` and their `*_relocalization.yaml` variants.

## 6. Map store layout

```text
<map_root>/<map_id>/
  versions/v000001/{map.pcd, manifest.json}   # immutable after publish
  versions/.tmp-v00000N-<uuid>/                # staging, removed on cancel/failure
  current -> versions/v00000N                  # atomic symlink swap
```

`manifest.json` holds the SHA-256 checksum (`checksum_algorithm: sha256`), the version, the calibration
hash and the optional `body_path`. The identity tuple `map_id/map_version/map_checksum` is indivisible.

## 7. Dependencies

`rclpy`, `omni_tf_manager` (SlamStatus and profiles), `omni_robot_interfaces` (ListMaps),
`omni_slam_interfaces`; FAST-LIO: PCL, Eigen, GTSAM ≥ 4.2, OpenSSL, `pcl_ros`, and optionally
`livox_ros_driver2` (required for the `full-slam` profile).

## 8. Known gaps and documentation/code discrepancies

1. **The loop-closure backend exists.** `omni_navi/docs/ARCHITECTURE.md` §3.1/§6 and
   `MODULE_GAP_ANALYSIS.md` still say "关键帧、回环检测、位姿图和全局地图优化后端尚缺" (keyframe,
   loop-closure, pose-graph and global map optimisation backends are missing). Since `ea48572`
   (2026-09-24), offline/background loop closure with an iSAM2 pose graph is implemented for exported maps.
   Live correction of localization is still missing.
2. SLAM reason codes are 1001–1203. The `omni_robot_interfaces` README reserves 2000–2999 for SLAM.
3. `SlamStatus` is still owned by `omni_tf_manager` (V1 exception).
4. Quality gating of localization happens in `icp_node`. It publishes `icp_result` only after the
   solver converges with `fitness < fitness_score_thre` and `converged_count_thre` (20) consecutive
   candidates agree in SE(3) (`icp_node.cpp:214-240`). The manager then trusts any `icp_result`.
   After LOCALIZED, the fitness value is reported but never re-evaluated, so a later bad match only
   shows up as a pose jump.
5. Real-time recovery from LOST in managed mode requires a new `StartLocalization`.
