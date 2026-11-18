# omni_slam_manager

> 产品运行时不要单独执行 `omni_slam_ctl manager-start`。机器人只启动 Bridge
> 与 Mission 两个顶层入口，Mission 的 `inspection_runtime.launch.py` 创建本
> 控制面并按模式管理 FAST-LIO/ICP；以下直接命令仅用于模块开发和故障诊断。

This package is the engineering control plane for FAST-LIO and publishes the
typed state consumed by `omni_tf_manager`. `MapStore` is its immutable,
crash-safe map asset store.

Layout:

```text
<map_root>/<map_id>/
├── current -> versions/v000002
└── versions/
    ├── v000001/
    │   ├── map.pcd
    │   └── manifest.json
    └── v000002/
        ├── map.pcd
        └── manifest.json
```

An import copies and hashes the PCD into a staging directory, fsyncs the PCD,
manifest and directory, atomically publishes the immutable version directory,
then atomically switches `current`.  Failed imports never switch the active
map.  `MapStore.current()` verifies SHA-256 by default before returning a map.

The ROS state machine, typed status and actions will be layered on this storage
primitive; callers should not modify files inside `versions/`.

Start the manager once and leave it running. For Vbot on S100:

```bash
ros2 launch omni_slam_manager manager.launch.py \
  robot_profile:=vbot \
  map_root:=/userdata/omni/maps \
  map_source_path:=/userdata/omni/work/map.pcd
```

The manager selects `vbot.launch.py`, `vbot_relocalization.launch.py` and
the matching `omni_vbot_dog.yaml` TF profile. The default `omni_dog` robot
selects `omni_dog.yaml`. An explicit `tf_profile_path` still overrides this
selection.

MATRiX 仿真使用相同的 manager-owned 生命周期，不再由 Bridge 单独拉起建图或
重定位脚本：

```bash
ros2 launch omni_slam_manager manager.launch.py \
  robot_profile:=matrix \
  map_root:=/tmp/omni/maps \
  map_source_path:=/tmp/omni/work/map.pcd
```

`matrix` profile 选择 `matrix_xgw.yaml`、`matrix_sim.yaml` 与
`matrix_sim_relocalization.yaml`。TF Manager 只启动一次；后续通过 typed API 切换
建图与定位时，仅 FAST-LIO/ICP 子进程组发生变化。

`tf_profile_path` is the single source of truth for map, odom, base and IMU
tracking frame names. The launch also starts `omni_tf_manager`; the SLAM child
processes only publish pose data and never publish managed TF edges.
FAST-LIO consumes only `/omni/sensors/lidar/points` and
`/omni/sensors/imu/data`; the TF profile owns each vendor-topic alias.

Stable internal control-plane endpoints:

- `/omni/slam/start_mapping` (`StartMapping`)
- `/omni/slam/stop_mapping` (`StopMapping`)
- `/omni/slam/save_map` (`SaveMap` action)
- `/omni/slam/start_localization` (`StartLocalization` action)
- `/omni/slam/stop_localization` (`StopLocalization`)
- `/omni/slam/maps/list` (`omni_robot_interfaces/srv/ListMaps`，仅供 Mission
  转发；APP 使用 `/omni/maps/list`)
- `/omni/slam/status` (`omni_tf_manager/msg/SlamStatus`, reliable
  transient-local heartbeat)

The manager starts child launch processes without a shell and terminates their
entire process group on stop, cancellation or shutdown.  Mapping is not marked
ready until cloud, IMU and odometry telemetry are all fresh.  Localization is
not marked `LOCALIZED` until an ICP initial pose has been followed by fresh
odometry.

## Vbot control flow

The S100 runtime installs `omni_slam_ctl`, which wraps all calls below and
keeps the manager in the background. For normal board operation, prefer:

```bash
omni_slam_ctl manager-start
omni_slam_ctl mapping-start vbot_001
omni_slam_ctl map-save factory_a
omni_slam_ctl mapping-stop
omni_slam_ctl localization-start factory_a
omni_slam_ctl logs-follow
```

Run `omni_slam_ctl help` for stop, status, initial-pose and log commands. The
raw ROS 2 commands remain documented below for integration with other nodes.

Start mapping:

```bash
ros2 service call /omni/slam/start_mapping \
  omni_slam_interfaces/srv/StartMapping \
  "{session_id: vbot_001}"
```

Save the current map as a verified, versioned asset named `factory_a`:

```bash
ros2 action send_goal --feedback /omni/slam/save_map \
  omni_slam_interfaces/action/SaveMap \
  "{map_id: factory_a, calibration_hash: '', make_current: true}"
```

Stop mapping after the save succeeds:

```bash
ros2 service call /omni/slam/stop_mapping \
  omni_slam_interfaces/srv/StopMapping \
  "{discard_unsaved: false}"
```

Start localization with the current version of `factory_a`. Initial position
is in meters and yaw is in radians:

```bash
ros2 action send_goal --feedback /omni/slam/start_localization \
  omni_slam_interfaces/action/StartLocalization \
  "{map_id: factory_a, map_version: 0, timeout_sec: 60.0, initial_x: 0.0, initial_y: 0.0, initial_z: 0.0, initial_yaw: 0.0}"
```

Stop localization:

```bash
ros2 service call /omni/slam/stop_localization \
  omni_slam_interfaces/srv/StopLocalization "{}"
```

Observe the durable status heartbeat at any time:

```bash
ros2 topic echo /omni/slam/status
```

## 可选 VBot LIVO 建图后端

`mapping_backend:=fast_livo` 可选择原生 Humble LIVO，默认仍为 `fast_lio`。此选项目前针对 VBot 建图；已有定位命令保持 FAST-LIO/ICP 路径，不等于已经实现视觉地图重定位。

LIVO 模式必须提供 `mapping_config_file` 和包含相机安装变换的 `tf_profile_path`。Manager 校验 IMU–LiDAR、Camera–LiDAR 外参与 TF 配置一致后启动；不能把本批录包的安装方向直接套用真机默认配置。每次开始会创建独立关键帧运行目录，停止和保存仍走既有 Manager 服务/action。

```bash
ros2 launch omni_slam_manager manager.launch.py \
  robot_profile:=vbot mapping_backend:=fast_livo \
  mapping_config_file:=$PWD/build/experiments/semantic_20261106/managed_config/livo.yaml \
  tf_profile_path:=$PWD/build/experiments/semantic_20261106/managed_config/tf.yaml \
  map_root:=$PWD/build/experiments/my_managed_maps \
  map_source_path:=$PWD/build/experiments/my_managed_work/map.pcd \
  use_sim_time:=true
```

这份配置专用于 `rosbag2245` 标定和回放，输入为规范化 LiDAR/IMU 与已校正的 BGR8 图像；实时相机应使用其匹配标定并关闭仿真时钟。`prepare_managed_vbot.py` 可从录包标定生成相互一致的配置，不修改原有真机 TF 文件。

`/state_estimation` 保持 `T_odom_tracking`，机体转换仍由唯一 TF Manager 完成。`/omni/slam/livo/keyframes` 使用 `omni_slam_interfaces/msg/LivoKeyframe`，包含会话内 ID、最终位姿、tracking frame 中完整雷达视野点云、实际图像时间及视觉更新标志。缺图时继续输出 LIO 位姿、非空点云和无视觉关键帧。

`/map_save` 原子保存当前前端关键帧地图，供既有 SaveMap action 导入版本化资产。建图模式下 TF Manager 的 map→odom 为恒等关系；该保存动作不在在线链路中运行离线回环优化。离线校正地图及语义对象的导入对齐需另行处理。
