# S100 机器狗：LIVO 与语义建图试跑

本轮交付是源码试跑包。S100 运行 ROS 2 Humble/TROS 的 LIVO，配套电脑运行语义推理与 RViz。当前未连接机器狗，未完成 S100 编译、性能或现场标定验证；不把 x86 二进制打包给 S100，也不宣称模型已适配 S100 BPU。

## 工程边界

| 部分 | 运行位置 | 内容 |
| --- | --- | --- |
| `fast_livo` | S100 | 点云/IMU/校正图像融合，原生关键帧归档，发布原子语义输入 |
| `omni_slam_manager` | S100 既有控制面 | 选择 `fast_livo` 建图后端；TF 仍归 `omni_tf_manager` |
| `omni_semantic_map semantic_trial` | 配套电脑 | 本地模型、实测三维框、地面验证、对象与证据保存 |
| RViz | 配套电脑 | 点云、三维框、地面和语义图像 |

语义推理与定位进程分离。输入 DDS 为 best effort、depth 1，节点只保留一个正在推理的帧和一个最新待处理帧；忙时替换旧的待处理帧，计入丢帧统计。S100 仍承担关键帧图像/点云序列化与网络发送开销，须实测，不等于零负担。只有存在语义订阅者时才制作语义输入包。

原子输入 `LivoSemanticFrame` 包含关键帧 ID、雷达时刻的 tracking 点云与位姿、实际相机时刻的无标注校正图像、CameraInfo，以及从该关键帧到相机曝光时刻的投影变换。使用 LIVO 自身的短时旋转/平移补偿，保留其近似与标定误差。原有 `LivoKeyframe` 消息未改变。

在线试跑使用已有本地 Mask R-CNN COCO 权重：车辆、花盆、消火栓、行人。它与上一轮室外 Grounding DINO 离线检测器不同；本轮不包含门/建筑入口识别，也不继承人工审查后的消防检测结果。当前仅用单关键帧点云做掩码内深度聚类，小目标缺少深度时不生成框；尚未加入离线版的过去一秒点云融合。消火栓识别质量需要重新核验。行人仅显示当前观测框，不关联身份、不加入静态对象表；车辆重复观测也不证明车辆静止。

可选 SegFormer Cityscapes 模型提取道路、人行道、terrain 候选，再用局部激光平面验证。绿色地面不代表可通行，未生成导航代价地图。在线坐标为前端 `omni_odom`，本轮未接入离线回环或发布 `map→odom`。

## 源码包与依赖

打包：

```bash
python3 scripts/package_livo_trial.py --output dist/omni_slam_s100_trial.tar.gz
```

包内有源码、启动文件、RViz 配置、本说明和逐文件 SHA256 清单；排除 bag、PCD、报告视频、模型、编译产物。原实验结果不移动。模型单独拷贝到电脑，明确传入路径，不自动下载。

S100 原生编译需要现有 Omni SDK/工作空间（`omni_tf_manager`、`omni_robot_interfaces` 等），以及 Humble/TROS 的 rclcpp、消息生成器、PCL、Eigen、OpenCV、cv_bridge、OpenMP、colcon。先加载现有 SDK，再执行：

```bash
bash scripts/build_livo_trial.sh
source install/trial/setup.bash
ros2 interface show omni_slam_interfaces/msg/LivoSemanticFrame
ros2 pkg prefix fast_livo
```

脚本优先 `/opt/tros/humble/setup.bash`，否则 `/opt/ros/humble/setup.bash`；可设置 `TRIAL_ROS_SETUP`。默认单包顺序构建、编译并行度 2，独立放在 `build/trial`、`install/trial`，不覆盖已部署安装目录。不需要在 S100 安装 PyTorch。

电脑端构建相同接口和 `omni_semantic_map`。ROS Python 需要 rclpy、cv_bridge、sensor_msgs_py、NumPy、SciPy、OpenCV；推理需要相互匹配的 PyTorch/Torchvision。地面模型还需要 Transformers，本地已验证版本为 4.49.0。使用兼容 ROS Humble Python 3.10 的环境，不要用 pip 升级系统 NumPy 来替换 ROS 依赖。若使用虚拟环境，创建时使用 `--system-site-packages`，并使用该环境的 Python 显式启动语义节点。

## 标定与输入检查

必须提供本机匹配的 `livo.yaml`、`tf.yaml`；仓库的 `vbot_bag_example.yaml` 是录包示例，不能直接当作实机配置。`prepare_managed_vbot.py` 生成的文件也是录包配置，默认仿真时间，不应直接上机。

现场确认雷达、IMU、相机安装标定后，配置应满足：

- `use_sim_time: false`；`imu.acceleration_scale: 1.0` 仅适用于已确认 m/s² 的 VBot IMU。
- `common.img_en: 1`、`vio.lidar_timed_fusion: true`、`runtime.keyframe_archive: true`。
- `publish.tf_en: false`；雷达/IMU/相机变换与唯一 TF profile 一致。
- 图像是 `sensor_msgs/Image`，已按本机 CameraInfo 校正，尺寸与 `camera.*` 一致；本试跑配置使用 `scale: 1.0`、零畸变参数。
- 规范输入为 `/omni/sensors/lidar/points`、`/omni/sensors/imu/data`、`/omni/sensors/camera/image_rect`，同步发布对应 `/omni/sensors/camera/camera_info`。

VBot 的 H.265 不能直接交给 LIVO。本轮没有把录包解码器伪装成实时驱动；若现场仅有 `/image_left_raw/h265`，先让现有相机驱动/解码链路输出上述带原始采集时间的校正图像和 CameraInfo。当前相邻 `omni_stereo` 配置主要输出深度/点云，不能据此假设已经存在校正彩色图像输出。到这一步前检查应失败，不能绕过。

只读检查（配置路径替换为实际文件）：

```bash
ros2 run omni_semantic_map semantic_preflight \
  --config /absolute/site/livo.yaml --tf-profile /absolute/site/tf.yaml \
  --observe-seconds 10 --output /tmp/livo_preflight.json
```

检查配置/TF 一致性、时钟、消息连续性、frame_id、点字段、图像尺寸和 CameraInfo。退出码 0 表示这些自动检查通过，2 表示未通过；仍需现场看激光投影是否贴合真实物体，自动检查不能证明外参真实准确。

## 启动与观察

两端必须使用可互通的 DDS 网络、相同 `ROS_DOMAIN_ID`、相同接口构建，均设 `ROS_LOCALHOST_ONLY=0`。实机使用系统时间；不要将实验用 `ROS_LOCALHOST_ONLY=1` 或 `use_sim_time=true` 带上机。先确认电脑能看到板端关键帧，测量包接收与丢帧，当前未验证跨机器网络。

生产系统仍由 Mission 管理唯一 SLAM/TF 控制面。以下是独立模块试跑命令，仅在没有既有 Mission/SLAM Manager 同时运行时使用；不另起第二套控制面：

```bash
ros2 launch omni_slam_manager manager.launch.py \
  robot_profile:=vbot mapping_backend:=fast_livo \
  mapping_config_file:=/absolute/site/livo.yaml \
  tf_profile_path:=/absolute/site/tf.yaml \
  map_root:=/userdata/omni/trial/maps \
  map_source_path:=/userdata/omni/trial/work/map.pcd
```

另一个板端终端启动本次建图：

```bash
ros2 service call /omni/slam/start_mapping omni_slam_interfaces/srv/StartMapping \
  '{session_id: livo_trial_001}'
```

电脑端（每次使用新的结果目录；模型路径明确指定）：

```bash
ros2 launch omni_semantic_map trial.launch.py \
  weights:=/absolute/models/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth \
  ground_model:=/absolute/models/segformer_cityscapes \
  output_directory:=/absolute/runs/livo_trial_001 device:=cpu

rviz2 -d "$(ros2 pkg prefix omni_semantic_map)/share/omni_semantic_map/config/trial.rviz"
ros2 topic echo /omni/semantic/status
```

虚拟环境可用 `python -m omni_semantic_map.live_node --ros-args -p weights:=... -p ground_model:=... -p output_directory:=...`，确保先 source 相同 ROS overlay。CUDA 仅在电脑环境已验证时设 `device:=cuda`；节点不会自动改用 GPU。

蓝框为满足重复观测门槛的记录，橙框为暂定，紫框为短时行人。框是实际观测表面的范围，不是完整物体尺寸。地面点云按关键帧发布，RViz 保留最近 30 秒。静态框持久显示，行人使用独立非持久 topic，不能在后来连接 RViz 时重新出现旧行人。

默认处理最多 300 个语义帧或约 1000 条对象记录后进入 `limit_reached`，LIVO 继续运行。这是短程试跑保护，不是长期部署配置。语义断流超过 5 秒报告 `no_recent_visual_keyframes`；输入会话重启、ID/时间倒退则进入 `session_mismatch`，需新建语义会话，避免混合两次地图。推理失败只停止语义处理并报错，不给 LIVO 反馈状态修改。

录包可在电脑端执行 `bash scripts/record_livo_trial.sh /absolute/new/bag_directory`。结束时先通过现有 Manager 保存地图、停止建图，再停止语义进程和录包；`objects.json` 原子保存，`evidence/`、`frames.jsonl`、`status.json`、`run.json` 保留坐标、证据和运行情况。语义文件目前不自动进入 Manager 的版本化地图资产，保存时应与本次 LIVO 归档一同留存。

## 首轮现场验收

先原地静止初始化，再人工遥控短程慢走并返回起点；本包不发布任何运动命令。记录相机可用时的视觉更新、断图后的 LIO 连续输出、点云/图像投影、三维框与地面、CPU/内存/温度、语义延迟与丢帧。首轮以链路和位置可解释性为目标，不宣称 Z 漂移或全局闭环已经解决。

本地验证结果另见 `docs/experiments/s100_trial_preparation_2026-11-06.md`。S100 原生构建、跨机器 DDS、现场标定、热稳定性和模型加速均待实机验证。
