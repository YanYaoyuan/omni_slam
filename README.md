# Omni SLAM

> `omni_slam_ctl` 是独立模块诊断入口，不是巡检产品的启动方式。完整系统由
> `omni_mission_manager` 统一启动 TF/SLAM 控制面并按需拉起建图、定位与 Planner，
> 禁止与该控制面并行再启动第二个 SLAM Manager。

Omni SLAM 是基于 FAST-LIO2 的 ROS 2 建图与先验地图重定位工程，支持 x86_64、
NVIDIA Orin 和 D-Robotics RDK S100。VBot 驱动发布 `/lidar_points` 点云和
`/lidar_imu` IMU；`omni_tf_manager` 校验并规范为
`/omni/sensors/lidar/points` 与 `/omni/sensors/imu/data` 后交给 FAST-LIO，
常驻管理器统一控制建图、地图保存和定位。

![Relocalization](./reloc.gif)

## S100 成果物部署

从 GitHub Actions 下载 `omni-slam-rdk-s100-aarch64`，解压完整目录到板端，
例如：

```bash
cd /userdata/omni/omni_slam
source setup.bash
```

成果物已经包含 `bin/`、`config/`、`lib/`、`share/` 和 `setup.bash`。
板端仍需提供 RDK OS 以及 `/opt/tros/humble` 或 `/opt/ros/humble`。

首次部署后可以确认接口和程序是否可见：

```bash
ros2 pkg prefix fast_lio
ros2 pkg prefix omni_slam_manager
command -v omni_slam_ctl
```

## VBot 快速使用

`omni_slam_ctl` 会自动加载 `/app/script/env.sh`（存在时）和成果物的
`setup.bash`。管理器以后台进程运行，建图和定位进程由它启动、监控和停止，
退出 SSH 后仍会继续运行。

### 1. 启动常驻管理器

```bash
omni_slam_ctl manager-start
omni_slam_ctl manager-status
```

重复执行 `manager-start` 不会启动第二个管理器。

### 2. 建图并保存

```bash
# session_id 可省略，省略时自动按时间生成
omni_slam_ctl mapping-start vbot_001

# 保存为版本化地图，并设置成当前版本
omni_slam_ctl map-save factory_a

# 保存成功后停止建图
omni_slam_ctl mapping-stop
```

如果明确不需要本次未保存的建图结果：

```bash
omni_slam_ctl mapping-discard
```

地图默认保存在 `/userdata/omni/maps/<map_id>/versions/`。每次执行
`map-save` 都会生成新版本并校验文件，定位时版本 `0` 表示使用当前版本。

### 3. 启动和停止定位

零初始位姿定位：

```bash
omni_slam_ctl localization-start factory_a
```

指定初始位姿：

```bash
# 参数顺序：map_id x y yaw [z [version [timeout]]]
# x/y/z 单位为米，yaw 单位为弧度
omni_slam_ctl localization-start factory_a 1.2 -0.5 1.57
```

停止定位：

```bash
omni_slam_ctl localization-stop
```

### 4. 状态和日志

```bash
# 输出一次 SLAM 状态
omni_slam_ctl status

# 查看最近 100 行或指定行数
omni_slam_ctl logs
omni_slam_ctl logs 300

# 持续跟踪管理器、FAST-LIO 和重定位节点的合并日志
omni_slam_ctl logs-follow
```

日志默认写入 `/userdata/omni/log/manager-YYYYmmdd-HHMMSS.log`，
`manager-current.log` 始终指向本次运行日志。运行目录和 PID 文件位于
`/userdata/omni/run/`。

停止管理器及其当前建图/定位子进程：

```bash
omni_slam_ctl manager-stop
```

查看全部命令：

```bash
omni_slam_ctl help
```

可通过环境变量调整默认目录和机器人配置，例如：

```bash
export OMNI_DATA_ROOT=/userdata/omni
export OMNI_ROBOT_PROFILE=vbot
omni_slam_ctl manager-start
```

支持的变量包括 `OMNI_MAP_ROOT`、`OMNI_WORK_DIR`、`OMNI_RUN_DIR`、
`OMNI_LOG_DIR`、`OMNI_BOARD_ENV` 和 `OMNI_SETUP`。

## VBot 传感器配置

- 公共 FAST-LIO 参数：[FAST_LIO/config/vbot.yaml](FAST_LIO/config/vbot.yaml)
- 雷达与 IMU 标定：[FAST_LIO/config/vbot_calib.yaml](FAST_LIO/config/vbot_calib.yaml)
- 建图入口：[FAST_LIO/launch/vbot.launch.py](FAST_LIO/launch/vbot.launch.py)
- 定位入口：[FAST_LIO/launch/vbot_relocalization.launch.py](FAST_LIO/launch/vbot_relocalization.launch.py)

启动前应确认驱动正在发布传感器数据：

```bash
ros2 topic hz /lidar_points
ros2 topic hz /lidar_imu
ros2 topic echo /lidar_points --once --field header
ros2 topic echo /lidar_imu --once --field header
```

标定文件必须以实际 VBot 的雷达—IMU 外参为准，示例值不能替代实机标定。
2026-08-26 的 S100/VBot 实机记录确认 `/lidar_points` 与 `/lidar_imu` 的
`header.frame_id` 均为 `vita_lidar`。结合原先已直接消费这两个 payload 的
FAST-LIO 配置与 `vbot_calib.yaml` 外参，`omni_vbot_dog.yaml` 已审核并启用两个
规范化消息别名。该 profile 仍保持 `shadow`，只恢复规范化传感器消息，不接管
已有 `/tf`；完成现场 TF 发布器审计后再切换 `authority`。

## 管理接口

控制脚本封装了以下稳定 ROS 2 接口，其他程序也可以直接调用：

- `/omni/slam/start_mapping`：`StartMapping.srv`
- `/omni/slam/stop_mapping`：`StopMapping.srv`
- `/omni/slam/save_map`：`SaveMap.action`
- `/omni/slam/start_localization`：`StartLocalization.action`
- `/omni/slam/stop_localization`：`StopLocalization.srv`
- `/omni/slam/maps/list`：`omni_robot_interfaces/srv/ListMaps`（Mission 内部使用）
- `/omni/slam/status`：`omni_tf_manager/msg/SlamStatus`

原始 `ros2 service/action` 调用示例见
[omni_slam_manager/README.md](omni_slam_manager/README.md)。

## 构建与测试

x86_64 构建由 `.github/workflows/build-x86.yml` 执行。S100 使用官方 TROS
交叉编译环境，也可以在本地复现：

```bash
./scripts/test_s100_local.sh
```

S100 构建和打包细节见：

- [scripts/build_s100_cross.sh](scripts/build_s100_cross.sh)
- [scripts/package_s100_runtime.sh](scripts/package_s100_runtime.sh)

NVIDIA Orin 部署说明见 [DEPLOY_ORIN.md](DEPLOY_ORIN.md)，路径记录工具见
[tools/global_path_tools/README.md](tools/global_path_tools/README.md)。
