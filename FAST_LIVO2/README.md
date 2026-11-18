# FAST-LIVO2 — ROS 2 Humble

基于 HKU MaRS 官方 FAST-LIVO2 的本地原生 Humble 移植，支持 VBot WLR-722
`PointCloud2`（含逐点绝对秒时间戳）+ 雷达 IMU + 校正后的相机图像。
上游算法说明见 [README_UPSTREAM.md](README_UPSTREAM.md)，准确来源和固定提交见
[UPSTREAM.md](UPSTREAM.md)，许可条款见 [LICENSE](LICENSE)。

当前已接入 Manager 的可选 LIVO 建图后端、地图保存和关键帧归档；默认后端仍为 FAST-LIO。
离线回环尚未反馈在线状态，LIVO 先验地图定位和 S100/Orin 实机尚未验证。
[S100 语义试跑说明](../docs/deployment/s100_livo_semantic_trial.md)包含独立语义消费者、RViz 与输入审计入口。

本地新增了保留整帧雷达更新节奏的视觉辅助模式；上游风格的图像切帧模式保留作对照。
使用下述专用参数配置进行新实验，不能把最初的短片段链路验证当作稳定性结论。

## 构建

环境：Ubuntu 22.04、ROS 2 Humble、Eigen、PCL、OpenCV、OpenMP、cv_bridge。
Sophus 与 Vikit 按上游要求固定在 `third_party/` 中，无需安装到系统目录。

在 `omni_slam` 根目录执行：

```bash
source /opt/ros/humble/setup.bash
colcon --log-base log/livo_humble build \
  --base-paths omni_slam_interfaces FAST_LIVO2 \
  --packages-select omni_slam_interfaces fast_livo \
  --build-base build/livo_humble --install-base install/livo_humble \
  --parallel-workers 1 --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
source install/livo_humble/setup.bash
colcon --log-base log/livo_humble test \
  --build-base build/livo_humble --install-base install/livo_humble \
  --packages-select fast_livo
colcon test-result --test-result-base build/livo_humble
```

`FAST_LIVO_USE_LIVOX` 默认关闭；若有 `livox_ros_driver2`，可用
`-DFAST_LIVO_USE_LIVOX=ON` 构建 CustomMsg 输入。该可选路径未使用本批 VBot 数据测试。

## 历史短片段链路实验

数据转换依赖 Python `mcap`、`mcap_ros2`、`av`（PyAV）、NumPy、OpenCV、SciPy、PyYAML
以及 Humble 的 `rosbag2_py`、MCAP 存储插件。程序只读取原始录包。

```bash
source /opt/ros/humble/setup.bash
python3 FAST_LIVO2/scripts/prepare_vbot_bag.py \
  /home/user/vbot/data/roabag2336 \
  --output build/experiments/vbot_trial --duration 60 \
  --width 640 --height 360 --image-stride 3

source install/livo_humble/setup.bash
python3 FAST_LIVO2/scripts/run_bag_experiment.py \
  build/experiments/vbot_trial \
  --executable install/livo_humble/fast_livo/lib/fast_livo/fastlivo_mapping \
  --mode lio --domain 91 --rate 1.0
python3 FAST_LIVO2/scripts/run_bag_experiment.py \
  build/experiments/vbot_trial \
  --executable install/livo_humble/fast_livo/lib/fast_livo/fastlivo_mapping \
  --mode livo --domain 92 --rate 1.0
python3 FAST_LIVO2/scripts/run_bag_experiment.py \
  build/experiments/vbot_trial \
  --executable install/livo_humble/fast_livo/lib/fast_livo/fastlivo_mapping \
  --mode lidar_scheduled --domain 93 --rate 1.0
python3 FAST_LIVO2/scripts/analyze_experiment.py \
  build/experiments/vbot_trial --output build/experiments/vbot_trial/analysis
```

实验使用独立 ROS domain 和 localhost，不发布运动指令。已有输出目录会拒绝覆盖。
三种模式应顺序运行，以减少资源争用对处理耗时统计的影响。

- `lio`：FAST-LIVO2 的纯 LIO 模式，整帧雷达更新；它不是本仓 FAST-LIO2 的执行结果。
- `livo`：按图像时间切分点云，运行几何和光度更新。
- `lidar_scheduled`：保留同样的图像时间及点云切分，只关闭光度 EKF 更新，用于隔离视觉约束的影响。

转换脚本从录包内 `/stereo_left/camera_info` 和 `/tf_static` 推导标定。
默认选择原始左目视频，按八参数 rational 模型去畸变后输出 pinhole 图像。
CameraInfo 的空 R/P 不作有效校正参数使用。解码 PTS 取自视频消息采集时刻，
不替换为接收或回放时刻；初始时间偏移设为 0，尚不能证明硬件同步精度。

生成目录包含：`bag/`、三套 ROS 2 YAML、`input_audit.json`、图像样例、厂商里程计对照，
以及各模式的 `trajectory.txt`、`trajectory_final.tum`、`timing.csv`、`metrics.json`、
节点日志和 `Log/pcd/` 地图。厂商 odometry 不作为独立真值。

## 全部录包离线实验

长录包使用流式入口 `fastlivo_offline`，避免生成大量未压缩中间图像。Python 只解码
一次视频，向三个独立进程发送同一组传感器事件；C++ 直接调用正常节点的回调及估计器，
每条输入后同步处理可用测量，结束时校验传输数量。此路径不经过 DDS 传感器投递；
节点仍初始化 ROS 2 通信。三个进程共享 CPU，耗时不能当作独占 CPU 的速度基准。
录包之间串行运行。默认启用几何局部地图滑窗：半径 60 个 0.5 m 体素（每轴约 ±30 m），
移动 5 m 时清理窗口外体素；`--local-map-half-size 0` 可关闭。视觉特征及图像地图
尚未增加同样的裁剪，长包仍需观察峰值内存。不要在内存评估完成前并行启动多份长包。

```bash
source /opt/ros/humble/setup.bash
source install/livo_humble/setup.bash
python3 FAST_LIVO2/scripts/inspect_full_bags.py /home/user/vbot/data \
  --output build/experiments/full_trial/inventory.json
python3 FAST_LIVO2/scripts/run_full_bags.py \
  --inventory build/experiments/full_trial/inventory.json \
  --output build/experiments/full_trial/results \
  --executable install/livo_humble/fast_livo/lib/fast_livo/fastlivo_offline \
  --image-stride 3
MPLCONFIGDIR=/tmp/livo_matplotlib python3 FAST_LIVO2/scripts/analyze_full_bags.py \
  build/experiments/full_trial/results --output build/experiments/full_trial/analysis
```

不传 `--limit-seconds` 时读取每份录包完整时长，并核对 LiDAR、IMU、视频包数量。
所有视频包都送入解码器，默认每 3 个解码帧选择 1 个，约 10 Hz、640×360。
`--image-stride 1` 可用约 30 Hz，但本批 5 Hz 雷达在该切帧频率的关闭视觉对照中出现
不稳定，尚不作为推荐配置。编码流起始缺少参考帧时，视频包数可能多于解码帧数，
两者分别记录，不补造图像。

录包接收顺序按未修改的采样时间作 1 秒有界重排；超出窗口的迟到数据会报错。
采样时间早于录包起点的首个不完整扫描及 IMU 单独计入排除数量，保持与短片段实验
一致。EOF 后不补送后继扫描或 IMU，末尾尚未触发更新的图像保留在队列中，不外推数据。
完整读取不等于每条输入都形成有效状态，报告包含最终轨迹覆盖、初始化和 EOF 待处理量。
传输计数也不等于估计器内部使用量：原回调会忽略首条雷达之前的 IMU/图像。
实时接收顺序下的初始化与缓存仍需单独验证，本轮离线排序不证明在线时序已稳定。

输出额外包含 `status.json`、`progress.json`、逐包 `input_audit.json`、
`batch_result.json` 和逐模式 `offline_transport.json`。正常结束的模式可断点续跑，
失败模式重跑前会保留旧目录；更换程序、录包清单、时长、尺寸、抽帧或滑窗配置须使用新的输出目录。
多文件 MCAP 目前逐文件独立初始化，不把分卷自动拼成一次连续会话。

## 保留整帧雷达更新的视觉辅助实验

新实验使用 `config/vbot_lidar_timed_overrides.yaml`，通过 `--params-override` 合并参数。
它是实验参数覆盖文件，不是完整的 ROS launch 配置；每包完整配置由运行脚本生成。

```bash
source /opt/ros/humble/setup.bash
python3 FAST_LIVO2/scripts/run_full_bags.py \
  --inventory build/experiments/full_trial/inventory.json \
  --output build/experiments/full_trial/lidar_timed \
  --executable install/livo_humble/fast_livo/lib/fast_livo/fastlivo_offline \
  --image-stride 1 --params-override FAST_LIVO2/config/vbot_lidar_timed_overrides.yaml
```

- 雷达/IMU 按完整扫描更新，保持约 5 Hz 几何节奏；图像不再切分雷达扫描。
- 接收约 30 Hz 图像，每个雷达更新选择采样时间不晚于扫描结束、年龄不超过 40 ms 的最新图像，约 5 Hz 执行视觉更新。
- 图像保留真实采样时刻；用 IMU 积分补偿旋转、当前速度的一阶模型补偿短时平移，在雷达时刻更新融合状态。平移近似和未标定的时间偏移仍限制精度。
- 没有可用图像时继续输出 LIO；`lidar_scheduled` 在此配置下表示同一整帧雷达节奏且关闭光度更新的对照。
- 暂时关闭在线曝光估计，光度观测方差设为 10000；属于保守实验参数，不能据此宣称精度提升。

`timing.csv` 新增 `image_age_ms`。RGB 图像输出使用实际相机采样时间，融合里程计使用雷达更新时刻。
该模式的 TUM 导出在每个时刻保留最后状态；有视觉时保留视觉更新后状态，无视觉时保留 LIO 状态。

可用以下检查验证相机配置开启而没有图像时的行为（`--source` 为已完成实验中的单包目录）：

```bash
python3 FAST_LIVO2/test/check_camera_outage.py \
  --source build/experiments/full_trial/lidar_timed/roabag2336 \
  --executable install/livo_humble/fast_livo/lib/fast_livo/fastlivo_offline \
  --output build/experiments/full_trial/camera_outage
```

原数据的投影审计使用 `scripts/audit_vbot_projection.py --inventory ... --output ...`。
其图像叠加不做运动补偿，只能检查粗略几何关系，不能替代外参/时延精标定。

## 高度一致性与离线回环地图

视觉定位与语义目标建图开发采用上一轮完整关键帧 LIVO 基线，见
[当前主线与接入范围](../docs/experiments/fast_livo2_vbot_mainline.md)。
`config/vbot_livo_baseline.json` 固定六包程序、配置、地图和轨迹的哈希，供复现与后续语义输入核对；
它是实验清单，不是启动配置。下文整帧留出与删边结果仅作诊断，不替换该全量基线。

对于已经使用 m/s² 的 IMU，新增 `imu.acceleration_scale: 1.0`，避免用运动起步时的
短时均值再次缩放加速度。默认值 0 保留上游自动归一化；其他正数表示显式单位换算系数。
扫描去畸变与高频 IMU 传播共用此系数。该参数不会解决运动初始化带来的重力方向误差。

`config/vbot_si_overrides.yaml` 在整帧融合基础上启用 SI 单位及原生关键帧归档，
关闭大幅累计点云保存。归档按 2 s、1 m 或 15° 任一条件采样，包含 IMU 坐标系下
完整雷达视野 XYZRGB、位姿、速度、重力和比力。相机视野外使用灰色；灰色不代表语义类别。

```bash
source /opt/ros/humble/setup.bash
python3 FAST_LIVO2/scripts/run_full_bags.py \
  --inventory build/experiments/full_20261105/inventory.json \
  --output build/experiments/my_si_trial \
  --executable install/livo_humble/fast_livo/lib/fast_livo/fastlivo_offline \
  --modes livo --image-stride 1 \
  --params-override FAST_LIVO2/config/vbot_si_overrides.yaml
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/livo_matplotlib \
python3 FAST_LIVO2/scripts/optimize_keyframe_map.py \
  --source build/experiments/my_si_trial/20260414_4/livo \
  --output build/experiments/my_si_maps/20260414_4
```

后端额外需要 Open3D（本轮 0.19）、SciPy、NumPy、Matplotlib。先用静止段比力估计
统一的重力方向，再用邻近关键帧组成局部子图，检索远时回访，进行多高度初值的三维 ICP。
双向重叠、留出点残差、几何可观测性和多解检查通过后，才加入带不确定回环边的位姿图。
没有地面平坦或高度归零约束。没有可靠回环时，仅变换坐标系，不宣称修复漂移。

输出 `map_gravity_only.pcd`、`map_loop_corrected.pcd`、两套 TUM 轨迹、对比图及
`result.json`。地图使用关键帧采样，并以 0.15 m 体素保存。四分之一已验证回访边不加入
优化，用于内部一致性检查；它们仍共享数据和局部几何，不能代替独立真值。
后端当前只修正离线地图，不反馈前端状态、不发布在线 map→odom，也未接入 Manager。
本轮全部六包的结果与限制见
[高度优化实验](../docs/experiments/fast_livo2_vbot_vertical_2026-11-05.md)。

地图复核可用 `optimize_keyframe_map.py --loop-fit-half-period-s 2`，按共同的原始估计
起点将2秒时间块交替划为回环拟合与复核扫描。默认0保持原有全部关键帧候选路径。
划分只影响回环配准的点云，不丢弃前端输入或地图导出点。
`audit_map_quality.py --lio-source ... --lio-maps ... --livo-source ... --livo-maps ... --output ...`
核对除图像开关及输出路径外的参数、估计时刻和后端指纹，聚合回访事件，再用两组图中
未参与回环拟合的同一批LIO扫描检查几何。目标平面与点对应只建立一次，四种地图位姿
都使用相同样本；该指标隔离位姿一致性，不单独评价不同前端的去畸变精度。
`extract_revisit_images.py --inventory ... --audit ...`只读提取采集时间邻近的图像证据。
本轮复核结果与待隔离回环见[地图复核报告](../docs/experiments/fast_livo2_vbot_map_audit_2026-11-05.md)。

## ROS 2 接口

```bash
ros2 launch fast_livo mapping.launch.py \
  config:="$PWD/build/experiments/vbot_trial/livo.yaml" use_sim_time:=true
```

直接启动时，应将配置中的 `runtime.output_directory` 设置为新的运行目录，以保留已有实验结果。

`config` 必须使用 ROS 2 `ros__parameters` YAML。旧 ROS 1 文件只保留在
`config/upstream_ros1` 和 `launch/upstream_ros1` 供来源对照。

输入 Topic 由 `common.lid_topic`、`common.imu_topic`、`common.img_topic` 配置；
默认 SensorData QoS，输入图像为 `sensor_msgs/msg/Image`。H.265 在准备阶段解码，
节点自身不订阅压缩视频。`camera.*` 描述校正图像的尺寸、内参与缩放，
`camera.distortion` 使用五参数模型；八参数原始图像须在上游正确校正。

输出包括 `Odometry`、`path`、`cloud_registered` 和 `rgb_img`。位姿和点云使用
采集/估计时刻；LIVO 在视觉更新后发布融合位姿。姿态对应 IMU tracking frame，
不是 base_link。`publish.tf_en` 默认 false，避免接管现有 TF authority。
`runtime.output_directory` 控制全部运行文件位置，默认 `/tmp/fast_livo`。

`trajectory.txt` 的最后一列是更新类型：1 为 VIO，2 为 LIO。LIVO 的顺序更新可在
同一时刻产生两条内部记录，`trajectory_final.tum` 只保留最终 VIO 状态。
`timing.csv` 记录各更新的处理时间、点数、视觉支持点数以及本次位置/旋转修正量；
这些时间不包含视频解码、DDS 传输和 IMU 传播的全部开销。

## 本次验证范围

已完成的构建、测试与实测结果记录在
[实验报告](../docs/experiments/fast_livo2_humble_vbot_2026-11-05.md)。
全部录包的运行完整性、轨迹失败样本、内存及重复性检查见
[全量实验报告](../docs/experiments/fast_livo2_humble_vbot_full_2026-11-05.md)。
后续代码修复、整帧雷达调度与新的完整回归见
[VBot 优化实验](../docs/experiments/fast_livo2_vbot_optimization_2026-11-05.md)。
单段录包跑通和非零视觉修正证明链路可执行；定位精度、环境泛化、相机断流降级、
跨会话视觉重定位和长期内存行为仍需专门评价。

## 项目接入与语义原型

[Manager 的 LIVO 后端](../omni_slam_manager/README.md)已接入现有位姿、TF 与地图保存契约；
[语义对象包](../omni_semantic_map/README.md)提供基于完整 LIVO 地图的车辆关联和位置/证据查询。
`managed_mapping.launch.py` 禁止覆盖非空运行目录，关闭自身 TF 发布并启用原生关键帧归档。
关键帧消息中的位姿为 `T_odom_tracking`，点云在 tracking frame，图像时刻单独保留，不能用点云时刻冒充相机时刻。

新增消息需一起构建：

```bash
colcon build --base-paths omni_slam_interfaces FAST_LIVO2 omni_slam_manager omni_semantic_map \
  --packages-select omni_slam_interfaces fast_livo omni_slam_manager omni_semantic_map \
  --build-base build/livo_integration --install-base install/livo_integration
source install/livo_integration/setup.bash
```

[接入、缺图回退、全量回归与目标查询实测](../docs/experiments/livo_semantic_integration_2026-11-06.md)。
