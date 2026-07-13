# omni_slam 启动命令

目录：

```bash
cd /home/robot/1_slam_20260709
```

## 建图

```bash
bash 1_mapping.sh
```

停止建图：在建图终端按 `Ctrl+C`。

地图默认保存到：

```text
/home/robot/1_slam_20260709/maps/map.pcd
```

## 定位

```bash
bash 2_localizing.sh
```

默认使用地图：

```text
/home/robot/1_slam_20260709/maps/map.pcd
```

指定其他地图：

```bash
MAP_PATH=/path/to/xxx.pcd bash 2_localizing.sh
```

## 检查雷达和 IMU

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=24
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
ros2 topic list | grep -E "/front_lidar|/front_lidar/imu"
```

输入话题：

```text
/front_lidar
/front_lidar/imu
```

定位输出：

```text
/odometry
```
