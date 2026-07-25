# lio_map 全局路径录制与发布

这套工具与当前 `omni_slam + SCAN-Planner` 接口一致：

```text
录制输入：
/state_estimation_global
  lio_map -> livox_frame

工具按标定外参转换后保存：
  lio_map -> scan_base_link

发布输出：
/planning/global_path
  nav_msgs/msg/Path, frame_id=lio_map
```

## 1. 录制前提

建议先完成建图，保存最终 PCD，然后使用该 PCD 启动实时重定位。确认：

```bash
ros2 topic echo /state_estimation_global --once
ros2 run tf2_ros tf2_echo lio_map lio_odom
```

必须看到：

```text
/state_estimation_global.header.frame_id = lio_map
/state_estimation_global.child_frame_id  = livox_frame
```

不要录制 `/state_estimation` 的 `lio_odom` 路径后再修改成 `lio_map` 标签。

## 2. 录制

手动控制机器狗沿希望复现的路线行走：

```bash
cd /path/to/omni_slam
./tools/global_path_tools/1_record_path.sh
```

默认行为：

- 使用标定的 `scan_base_link -> livox_frame` 平移
  `[0.13011, -0.02329, 0.17598]`；
- 将雷达全局位姿换算成机体中心全局路径；
- XY 移动超过 `0.30m` 才记录新点；
- 拒绝疑似瞬时位姿跳变；
- 按 Enter 保存一个 `pathN.txt` 分段；
- Ctrl+C 保存最后一段并退出；
- 每次录制创建新的时间目录，不覆盖历史数据。

输出示例：

```text
tools/global_path_tools/recordings/20260725_153000/
├── globalpath.txt
└── paths/
    ├── path1.txt
    └── path2.txt
```

如标定外参变化，可通过参数覆盖：

```bash
./tools/global_path_tools/1_record_path.sh \
  --body-to-sensor-x 0.13011 \
  --body-to-sensor-y -0.02329 \
  --body-to-sensor-z 0.17598
```

## 3. 发布

先启动 SLAM 重定位和 SCAN-Planner。SCAN-Planner 必须运行
`navi_mode=3` 并订阅 `/planning/global_path`。

发布最近一次完整录制：

```bash
./tools/global_path_tools/2_publish_path.sh
```

发布指定文件或分段：

```bash
./tools/global_path_tools/2_publish_path.sh \
  --input /absolute/path/to/path1.txt
```

发布器会：

1. 校验路径元数据必须为 `lio_map -> scan_base_link`；
2. 使用 RDP 默认将三维路径简化到 `0.15m` 空间误差；
3. 确保相邻发布点不超过 `2.0m`；
4. 检查机器人距离路径起点不超过 `1.0m`；
5. 等待 SCAN-Planner 订阅者后只发布一次；
6. 保持节点运行，用 transient-local 保存最后路径。

只检查、不发布：

```bash
./tools/global_path_tools/2_publish_path.sh --check
```

旧版纯 xyz 文件没有记录 frame 和机体/雷达语义，默认会被拒绝。建议使用新版重新
录制；只有人工确认旧文件已经是 `lio_map` 下机体中心坐标时，才可使用
`--allow-legacy`。

反向执行路径：

```bash
./tools/global_path_tools/2_publish_path.sh --reverse
```

如果确认环境安全但机器人不在路径起点附近：

```bash
./tools/global_path_tools/2_publish_path.sh --skip-start-check
```

不建议在无人看护时跳过起点检查。

## 4. 高度处理

默认发布录制的机体中心 z，适合地图中确实存在坡道或高低变化的路线。

如果场地完全平坦，并确认 z 变化来自 SLAM 高度漂移，可以固定机体中心高度：

```bash
./tools/global_path_tools/2_publish_path.sh \
  --z-mode constant \
  --z 0.25
```

这里的 z 是 `lio_map` 下的机体中心高度，不是雷达高度。

## 5. 使用注意

- 录制和执行必须使用同一份最终 PCD 地图和同一套标定外参；
- 开始发布前，机器人应位于录制路径起点附近；
- 长路线建议按 Enter 分段，逐段发布；
- `/planning/global_path` 是路线参考，局部动态避障由 SCAN-Planner 完成；
- 需要重新发布路径时，Ctrl+C 退出发布器后重新启动即可。
