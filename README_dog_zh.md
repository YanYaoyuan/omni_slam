# 轮式机器狗 omni_slam 使用

目录：`/home/robot/1_slam_20260709`

## 编译

```bash
cd /home/robot/1_slam_20260709
bash build_on_dog.sh
```

## 建图

```bash
cd /home/robot/1_slam_20260709
bash 1_mapping.sh
```

停止建图时按 `Ctrl+C`，地图保存到：

```text
/home/robot/1_slam_20260709/maps/map.pcd
```

## 定位

```bash
cd /home/robot/1_slam_20260709
bash 2_localizing.sh
```

默认使用：

```text
/home/robot/1_slam_20260709/maps/map.pcd
```

换地图：

```bash
MAP_PATH=/path/to/xxx.pcd bash 2_localizing.sh
```

## 话题

输入：

```text
/front_lidar
/front_lidar/imu
```

定位输出：

```text
/odometry
```

建图输出：

```text
/state_estimation
```

默认：

```text
ROS_DOMAIN_ID=24
RMW_IMPLEMENTATION=rmw_zenoh_cpp
```

如果终端已经设置了其他值，脚本会使用已有值。
