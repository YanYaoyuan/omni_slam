# Omni SLAM Orin NX Deployment

本文档记录 `omni_slam` 在 Nvidia Orin NX 上部署、打包和 systemd 托管的推荐方式。

目标环境：

- Ubuntu 22.04
- ROS 2 Humble
- Nvidia Jetson Orin NX / aarch64
- LiDAR topic: `/front_lidar`
- IMU topic: `/front_lidar/imu`
- FAST-LIO package: `fast_lio`
- Omni dog 配置: `FAST_LIO/config/omni_dog.yaml`

## 1. 推荐部署方式

优先推荐 **在 Orin NX 上原生编译**。原因是 ROS 2、PCL、Eigen、Python、Livox driver 和 JetPack 系统库之间 ABI 依赖较多，原生编译最稳。

推荐目录：

```bash
/opt/omni_slam
├── src
│   └── omni_slam
├── install
├── log
└── maps
```

运行用户建议使用普通用户，例如 `robot`，不要长期用 root 跑 SLAM。

## 2. Orin 原生部署

### 2.1 安装基础依赖

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake git python3-colcon-common-extensions python3-rosdep \
  ros-humble-pcl-ros ros-humble-pcl-conversions \
  ros-humble-tf2 ros-humble-tf2-ros ros-humble-tf2-eigen ros-humble-tf2-geometry-msgs \
  ros-humble-geometry-msgs ros-humble-nav-msgs ros-humble-sensor-msgs \
  ros-humble-std-msgs ros-humble-std-srvs ros-humble-visualization-msgs \
  libeigen3-dev libpcl-dev libomp-dev
```

`fast_lio` 目前仍依赖 `livox_ros_driver2`。如果你的实际点云来自 `sensor_msgs/msg/PointCloud2`，运行时不一定使用 Livox CustomMsg，但编译时仍需要该包存在。

安装或放置 `livox_ros_driver2` 后，确认：

```bash
source /opt/ros/humble/setup.bash
ros2 pkg prefix livox_ros_driver2
```

### 2.2 拉取源码

```bash
sudo mkdir -p /opt/omni_slam/src
sudo chown -R $USER:$USER /opt/omni_slam

cd /opt/omni_slam/src
git clone git@github.com:YanYaoyuan/omni_slam.git
```

如果机器没有 SSH key，也可以使用 HTTPS：

```bash
git clone https://github.com/YanYaoyuan/omni_slam.git
```

### 2.3 编译

```bash
cd /opt/omni_slam
source /opt/ros/humble/setup.bash
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

只重新编译 FAST-LIO：

```bash
colcon build --packages-select fast_lio --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

### 2.4 验证 topic 和 TF

```bash
ros2 topic list
ros2 topic echo --once /front_lidar --field header
ros2 topic echo --once /front_lidar --field fields
ros2 topic echo --once /front_lidar/imu --field header
ros2 run tf2_tools view_frames
```

当前 `omni_dog.yaml` 按验证过的 rosbag 设置：

```yaml
sensor_frame_id: "livox_frame"
base_frame_id: "livox_frame"
send_odom_base_tf: false
lid_topic: "/front_lidar"
imu_topic: "/front_lidar/imu"
lidar_type: 5
timestamp_unit: 3
extrinsic_T: [0.0, 0.0, 0.0]
extrinsic_R: [1.0, 0.0, 0.0,
              0.0, 1.0, 0.0,
              0.0, 0.0, 1.0]
```

如果实机 TF/外参和 bag 不同，需要改 `FAST_LIO/config/omni_dog.yaml`。例如实机如果确认 LiDAR 在 IMU 坐标系下的标定是：

```yaml
extrinsic_T: [-0.011, -0.02329, 0.04412]
extrinsic_R: [1.0, 0.0, 0.0,
              0.0, 1.0, 0.0,
              0.0, 0.0, 1.0]
```

则不要继续使用 bag 的 identity 外参。

### 2.5 启动建图

```bash
cd /opt/omni_slam
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch fast_lio omni_dog.launch.py
```

输出：

- `/state_estimation`
- `/cloud_registered`
- `/cloud_registered_body`
- `/Laser_map`
- `/path`

保存地图：

```bash
ros2 service call /map_save std_srvs/srv/Trigger "{}"
```

默认配置中 `map_file_path` 是：

```bash
./omni_dog_map.pcd
```

如果由 systemd 启动，工作目录一般设置为 `/opt/omni_slam`，地图会保存在：

```bash
/opt/omni_slam/omni_dog_map.pcd
```

### 2.6 rosbag 验证

回放 bag 时建议使用 QoS override：

```bash
ros2 bag play /path/to/bag \
  --qos-profile-overrides-path /opt/omni_slam/src/omni_slam/FAST_LIO/config/omni_dog_qos.yaml
```

另一个终端启动：

```bash
cd /opt/omni_slam
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch fast_lio omni_dog.launch.py
```

## 3. 交叉编译

### 3.1 建议

对 ROS 2 + PCL 项目，最稳的是在 Orin 上原生编译。交叉编译可行，但容易踩以下问题：

- host 和 target 的 ROS 2 安装路径、Python 包、PCL ABI 不一致
- `rosidl` 生成消息时混用 host/target Python
- `livox_ros_driver2`、PCL、Eigen、OpenMP 依赖需要 target sysroot
- JetPack 自带系统库版本和普通 Ubuntu arm64 镜像不完全一致

推荐顺序：

1. Orin 原生编译并打包 install 目录
2. 使用 arm64 Docker/QEMU 构建
3. 使用 sysroot + CMake toolchain 真交叉编译

### 3.2 方式 A: 在 Orin 上构建后打包

这是推荐方式。

```bash
cd /opt/omni_slam
source /opt/ros/humble/setup.bash
colcon build --merge-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

打包：

```bash
VERSION=$(date +%Y%m%d_%H%M)
tar --numeric-owner -czf omni_slam_orin_${VERSION}.tar.gz \
  install \
  src/omni_slam/FAST_LIO/config/omni_dog.yaml \
  src/omni_slam/FAST_LIO/config/omni_dog_relocalization.yaml \
  src/omni_slam/FAST_LIO/config/omni_dog_qos.yaml \
  src/omni_slam/FAST_LIO/launch/omni_dog.launch.py \
  src/omni_slam/FAST_LIO/launch/omni_dog_relocalization.launch.py
```

部署到另一台 Orin：

```bash
sudo mkdir -p /opt/omni_slam
sudo tar -xzf omni_slam_orin_<VERSION>.tar.gz -C /opt/omni_slam
```

注意：目标 Orin 仍需安装相同 ROS 2 Humble、PCL、Livox driver 等运行依赖。

### 3.3 方式 B: Docker/QEMU arm64 构建

在 x86 主机上使用 arm64 容器构建，适合 CI 或离线产物生成。

准备 buildx：

```bash
docker buildx create --use --name omni_builder
docker buildx inspect --bootstrap
```

示例 `Dockerfile.orin`：

```dockerfile
FROM arm64v8/ros:humble-ros-base-jammy

RUN apt-get update && apt-get install -y \
    build-essential cmake git python3-colcon-common-extensions python3-rosdep \
    ros-humble-pcl-ros ros-humble-pcl-conversions \
    ros-humble-tf2-ros ros-humble-tf2-eigen ros-humble-tf2-geometry-msgs \
    ros-humble-visualization-msgs \
    libeigen3-dev libpcl-dev libomp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws
COPY . /ws/src/omni_slam

SHELL ["/bin/bash", "-lc"]
RUN source /opt/ros/humble/setup.bash && \
    colcon build --merge-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

构建：

```bash
docker buildx build \
  --platform linux/arm64 \
  -f Dockerfile.orin \
  -t omni_slam:orin \
  --load .
```

从容器导出 install：

```bash
CID=$(docker create omni_slam:orin)
docker cp ${CID}:/ws/install ./install_orin
docker rm ${CID}
tar -czf omni_slam_orin_install.tar.gz install_orin
```

注意：该方式生成的是通用 Ubuntu arm64 产物，不一定完全等同 JetPack rootfs。若遇到运行时库不匹配，请改用 Orin 原生构建或 sysroot 构建。

### 3.4 方式 C: sysroot + toolchain 交叉编译

在 x86 Ubuntu 主机准备 Orin sysroot：

```bash
mkdir -p ~/sysroots/orin
rsync -aHAX --numeric-ids robot@<ORIN_IP>:/lib ~/sysroots/orin/
rsync -aHAX --numeric-ids robot@<ORIN_IP>:/usr ~/sysroots/orin/
rsync -aHAX --numeric-ids robot@<ORIN_IP>:/opt/ros ~/sysroots/orin/opt/
```

安装交叉编译器：

```bash
sudo apt install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

示例 `toolchain-aarch64.cmake`：

```cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_SYSROOT "$ENV{ORIN_SYSROOT}")
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

set(CMAKE_FIND_ROOT_PATH "${CMAKE_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
```

构建示例：

```bash
export ORIN_SYSROOT=$HOME/sysroots/orin
source /opt/ros/humble/setup.bash

colcon build \
  --merge-install \
  --cmake-args \
    -DCMAKE_TOOLCHAIN_FILE=$PWD/toolchain-aarch64.cmake \
    -DCMAKE_BUILD_TYPE=Release
```

如果 `rosidl` 或 Python 相关步骤失败，优先改用 Orin 原生构建。这类失败通常不是源码问题，而是 ROS 2 交叉编译链路需要更完整的 host/target 分离配置。

## 4. systemd 服务

### 4.1 环境文件

创建目录：

```bash
sudo mkdir -p /etc/omni_slam
```

示例 `/etc/omni_slam/omni_slam.env`：

```bash
ROS_DISTRO=humble
ROS_DOMAIN_ID=0
RMW_IMPLEMENTATION=rmw_fastrtps_cpp
OMNI_SLAM_WS=/opt/omni_slam
OMNI_SLAM_CONFIG=/opt/omni_slam/src/omni_slam/FAST_LIO/config/omni_dog.yaml
```

如果 DDS 网络发现不稳定，可按现场网络再补 Fast DDS profile。

### 4.2 启动脚本

创建 `/opt/omni_slam/start_omni_slam.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/${ROS_DISTRO:-humble}/setup.bash
source ${OMNI_SLAM_WS:-/opt/omni_slam}/install/setup.bash

cd ${OMNI_SLAM_WS:-/opt/omni_slam}

exec ros2 launch fast_lio omni_dog.launch.py \
  config_file:=${OMNI_SLAM_CONFIG:-/opt/omni_slam/src/omni_slam/FAST_LIO/config/omni_dog.yaml}
```

设置权限：

```bash
chmod +x /opt/omni_slam/start_omni_slam.sh
```

### 4.3 建图服务

创建 `/etc/systemd/system/omni-slam.service`：

```ini
[Unit]
Description=Omni SLAM Fast-LIO mapping
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=robot
Group=robot
EnvironmentFile=/etc/omni_slam/omni_slam.env
WorkingDirectory=/opt/omni_slam
ExecStart=/opt/omni_slam/start_omni_slam.sh
Restart=on-failure
RestartSec=3
TimeoutStopSec=10
KillSignal=SIGINT

# Keep logs in journald. Increase if point cloud logs are verbose.
LogRateLimitIntervalSec=0

[Install]
WantedBy=multi-user.target
```

启用：

```bash
sudo systemctl daemon-reload
sudo systemctl enable omni-slam.service
sudo systemctl start omni-slam.service
```

查看状态：

```bash
systemctl status omni-slam.service
journalctl -u omni-slam.service -f
```

停止：

```bash
sudo systemctl stop omni-slam.service
```

### 4.4 重定位服务

如果需要开机进入先验地图定位，可以单独创建重定位脚本。

`/opt/omni_slam/start_omni_relocalization.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/${ROS_DISTRO:-humble}/setup.bash
source ${OMNI_SLAM_WS:-/opt/omni_slam}/install/setup.bash

cd ${OMNI_SLAM_WS:-/opt/omni_slam}

exec ros2 launch fast_lio omni_dog_relocalization.launch.py \
  config_file:=/opt/omni_slam/src/omni_slam/FAST_LIO/config/omni_dog_relocalization.yaml \
  map_path:=${OMNI_SLAM_MAP:-/opt/omni_slam/maps/omni_dog_map.pcd} \
  initial_x:=${OMNI_INITIAL_X:-0.0} \
  initial_y:=${OMNI_INITIAL_Y:-0.0} \
  initial_z:=${OMNI_INITIAL_Z:-0.0} \
  initial_yaw:=${OMNI_INITIAL_YAW:-0.0}
```

`/etc/omni_slam/omni_relocalization.env`：

```bash
ROS_DISTRO=humble
ROS_DOMAIN_ID=0
RMW_IMPLEMENTATION=rmw_fastrtps_cpp
OMNI_SLAM_WS=/opt/omni_slam
OMNI_SLAM_MAP=/opt/omni_slam/maps/omni_dog_map.pcd
OMNI_INITIAL_X=0.0
OMNI_INITIAL_Y=0.0
OMNI_INITIAL_Z=0.0
OMNI_INITIAL_YAW=0.0
```

`/etc/systemd/system/omni-relocalization.service`：

```ini
[Unit]
Description=Omni SLAM Fast-LIO relocalization
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=robot
Group=robot
EnvironmentFile=/etc/omni_slam/omni_relocalization.env
WorkingDirectory=/opt/omni_slam
ExecStart=/opt/omni_slam/start_omni_relocalization.sh
Restart=on-failure
RestartSec=3
TimeoutStopSec=10
KillSignal=SIGINT
LogRateLimitIntervalSec=0

[Install]
WantedBy=multi-user.target
```

启用：

```bash
sudo systemctl daemon-reload
sudo systemctl enable omni-relocalization.service
sudo systemctl start omni-relocalization.service
```

建图服务和重定位服务不要同时启用，除非你明确要并行跑两套节点。

## 5. 离线发布包结构

推荐发布包：

```bash
omni_slam_release/
├── install/
├── config/
│   ├── omni_dog.yaml
│   ├── omni_dog_relocalization.yaml
│   └── omni_dog_qos.yaml
├── systemd/
│   ├── omni-slam.service
│   └── omni-relocalization.service
├── scripts/
│   ├── start_omni_slam.sh
│   └── start_omni_relocalization.sh
└── maps/
    └── omni_dog_map.pcd
```

打包：

```bash
tar -czf omni_slam_release_$(date +%Y%m%d_%H%M).tar.gz omni_slam_release
```

部署脚本可以做这些事：

```bash
sudo mkdir -p /opt/omni_slam /etc/omni_slam
sudo rsync -a install /opt/omni_slam/
sudo rsync -a maps /opt/omni_slam/
sudo install -m 755 scripts/start_omni_slam.sh /opt/omni_slam/start_omni_slam.sh
sudo install -m 755 scripts/start_omni_relocalization.sh /opt/omni_slam/start_omni_relocalization.sh
sudo install -m 644 systemd/omni-slam.service /etc/systemd/system/omni-slam.service
sudo install -m 644 systemd/omni-relocalization.service /etc/systemd/system/omni-relocalization.service
sudo systemctl daemon-reload
```

## 6. 常见问题

### 6.1 有 topic 但 SLAM 不动

检查 QoS：

```bash
ros2 topic info -v /front_lidar/imu
```

rosbag 回放时加：

```bash
--qos-profile-overrides-path /opt/omni_slam/src/omni_slam/FAST_LIO/config/omni_dog_qos.yaml
```

### 6.2 一直 waiting for IMU

通常是点内 timestamp 单位错了。当前 `omni_dog.yaml` 使用：

```yaml
timestamp_unit: 3
```

适配帧内纳秒偏移，如 `0 .. 100239557`。

### 6.3 地图乱或漂得很夸张

优先检查：

- `/front_lidar.header.frame_id`
- `/front_lidar/imu.header.frame_id`
- `sensor_frame_id`
- `base_frame_id`
- `extrinsic_T`
- `extrinsic_R`

如果 LiDAR 和 IMU 都在同一个 `livox_frame`，使用 identity 外参。如果实机发布的是独立 IMU frame，就要使用真实 LiDAR-IMU 标定。

### 6.4 systemd 启动失败但手动启动正常

检查：

```bash
journalctl -u omni-slam.service -n 200 --no-pager
systemctl cat omni-slam.service
```

常见原因：

- `User=robot` 不存在
- `/opt/omni_slam/install/setup.bash` 不存在
- 没有 source `/opt/ros/humble/setup.bash`
- systemd 工作目录不同，导致地图保存路径和手动启动不一致
- 传感器驱动服务还没启动

如果依赖传感器驱动服务，例如 `livox-driver.service`，可以在 unit 里加入：

```ini
After=network-online.target livox-driver.service
Requires=livox-driver.service
```

### 6.5 CPU 占用过高

FAST-LIO 在 aarch64 上已启用 OpenMP 分支。可以通过 systemd 环境变量限制线程：

```ini
Environment=OMP_NUM_THREADS=3
```

或在环境文件中加入：

```bash
OMP_NUM_THREADS=3
```

线程数建议根据 Orin NX 供电模式和散热实测调整。
