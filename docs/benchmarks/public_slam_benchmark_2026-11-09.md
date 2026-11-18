# 公开 SLAM 基准实测：omni_slam

后续已完成官方原版 FAST-LIO2 / FAST-LIVO2 对照，见 [统一采样的四组比较](omni_slam_vs_upstream_2026-11-09.md)。本报告保留首次两模式评估口径。

日期：2026-11-09。第一批固定配置对照，使用完整 NTU VIRAL `eee_03`、`nya_01` 输入，分别运行 **omni_slam（LIO）** 和 **omni_slam（LIVO）**。本文表格中的 LIO / LIVO 均指 omni_slam 的对应模式。

## 1. 本次回答什么问题

我们要先验证语义建图所依赖的定位底座：接入相机后，当前版本能否在独立公开数据上完整运行，位置误差是否改善，额外计算和内存开销是多少。本轮没有启用语义检测、地面分割或离线回环修正；不能用这些数值代表整个语义建图系统的精度，也不能代表 S100 性能。

两组实验使用同一个冻结可执行文件。LIO 与 LIVO 配置仅图像开关和输出目录不同。这里的 omni_slam（LIO）是我们基于 FAST-LIVO2 开发的集成版关闭视觉更新后的消融对照，并非另一个 FAST-LIO2 仓库的独立算法。采用 NTU 传感器标定和预处理设置，保留此前工程的完整扫描调度及保守视觉权重。因工程行为已有修改，结果不作为 FAST-LIVO2 原论文复现或排行榜成绩。

## 2. 主结果

ATE 为位置绝对轨迹误差的均方根，单位 **cm**，越小越好；改善率为 `(LIO − LIVO) / LIO`。

| 序列 / 场景 | LIO ATE RMSE | LIVO ATE RMSE | LIVO 改善率 |
|---|---:|---:|---:|
| eee_03 · EEE 停车场 | 6.84 | 4.06 | +40.70% |
| nya_01 · 南洋礼堂室内 | 2.77 | 2.76 | +0.04% |

室内 `nya_01` 的 LIO/LIVO ATE 分别为 **2.765 / 2.764 cm**，差值约 **0.01 mm**，实际应视为持平，不能据此宣称视觉带来改善。该模式确实执行了 1,248 次视觉更新，其中 1,247 次有点支持；并非误把关闭相机的结果记成 LIVO。但图像如何被调度和保守权重如何影响贡献，还需另设消融实验验证。

本轮结论是：当前集成版在两条公开序列均完成全量运行，视觉收益依赖场景，尚无证据证明普遍优于 LIO。

首序列的收益分布不均：固定同一次全程对齐，评估起始 10 秒的 LIO/LIVO RMSE 为 **24.07 / 12.15 cm**；其后区间为 **2.99 / 2.67 cm**。因此不能把全程约 41% 的下降解释为每个路段均有同等提升。正式主表包含全部有效样本，没有删掉误差较大的开头，也没有分段重新对齐。

以下指标均为 cm。Z 是 **Leica 参考坐标系下、全程刚体对齐后的 Z 残差**；不等价于未对齐的重力方向累计漂移。1 s / 10 s 指标是位置位移增量误差，缺少姿态真值，不能称为完整 SE(3) RPE。

| 序列 | 模式 | ATE 中位数 | ATE P95 | ATE 最大值 | Z RMSE | 1 s 位移 RMSE | 10 s 位移 RMSE |
|---|---|---:|---:|---:|---:|---:|---:|
| eee_03 | LIO | 2.69 | 6.29 | 34.57 | 1.36 | 3.55 | 8.03 |
| eee_03 | LIVO | 2.33 | 6.06 | 17.27 | 1.14 | 2.78 | 5.17 |
| nya_01 | LIO | 2.51 | 4.24 | 9.78 | 1.18 | 1.62 | 2.95 |
| nya_01 | LIVO | 2.51 | 4.24 | 9.85 | 1.18 | 1.62 | 2.94 |

### 轨迹与误差图

图中黑线为 Leica 位置参考，蓝色为 LIO，橙色为 LIVO。每幅图依次展示 XY 轨迹、位置误差、参考坐标系 Z 和 Z 残差。

![eee_03 轨迹与误差](../reports/public_slam_benchmark_2026-11-09/eee_03/trajectory_errors.png)

![nya_01 轨迹与误差](../reports/public_slam_benchmark_2026-11-09/nya_01/trajectory_errors.png)

## 3. 全量输入与有效评估覆盖

完整读入每条序列的选定传感器，没有只截取效果好的片段。有效真值覆盖由时间戳关联决定，不能把未覆盖的部分算成已验证精度。下表的参考路程只覆盖有效评估样本。

| 序列 | 输入时长 s | IMU / 雷达 / 图像条数 | 两模式共有输出位姿 | 有效评估位姿 / 占比 | 评估跨度 s | 参考路程 m |
|---|---:|---|---:|---:|---:|---:|
| eee_03 | 181.35 | 70475 / 1814 / 1813 | 1812 | 1507 (83.2%) | 150.59 | 127.76 |
| nya_01 | 394.91 | 153239 / 3950 / 3947 | 3948 | 3911 (99.1%) | 391.99 | 160.02 |

转换保留原始消息顺序及传感器 header 时间，未运行官方 restamp 工具。选用 `/imu/imu`、`/os1_cloud_node1/points` 和 `/left/image_raw`；右相机、第二雷达、UWB 和真值未输入估计器。LIO 为便于统一传输核对也读入图像记录，但图像开关关闭，视觉更新数为零。

原始 ROS1 bag 经 `rosbags==0.10.11` 转为现有 native 输入流；IMU、点云转换为 ROS2 Humble CDR，保留 Ouster 的逐点 `t`（ns）和 `ring`。左图像用 NTU 标定去畸变，灰度复制为 BGR；运行时畸变设零，避免重复去畸变。消息往返转换的字段与载荷已做抽样一致性检查。

## 4. 标定与固定参数

- 雷达与 IMU、雷达与相机外参来自仓库上游 [NTU_VIRAL.yaml](../../FAST_LIVO2/config/upstream_ros1/NTU_VIRAL.yaml)，相机内参来自 [camera_NTU_VIRAL.yaml](../../FAST_LIVO2/config/upstream_ros1/camera_NTU_VIRAL.yaml)。没有使用 VBot 的相机或安装外参。
- 图像为 752 × 480；`fx=425.025856`、`fy=426.797626`、`cx=386.015187`、`cy=241.913034`。原始畸变系数见上述配置。
- 保留上游 NTU 雷达时间偏移 `−0.1 s`，IMU 和相机偏移为 0。该固定设备配置与“拟合真值时间偏移”不同，后者本次没有进行。
- 当前工程策略来自 [vbot_si_overrides.yaml](../../FAST_LIVO2/config/vbot_si_overrides.yaml)：`lidar_timed_fusion=true`、图像最大年龄 0.04 s、`img_point_cov=10000`、关闭曝光估计、IMU SI 单位倍率 1.0。文件名含 VBot，但本次仅复用其中的调度、单位和视觉权重设置；NTU 设备标定保留。
- 使用 16 线 Ouster 预处理；没有根据这两条序列的真值搜索参数。局部地图滑动沿用上游关闭设置，性能结果不代表长期运行内存上限。

首轮启动发现 ROS1 YAML 的整数写法被 ROS2 的浮点参数类型检查拒绝；仅修正浮点声明和外参数组类型，失败日志保留于 `runs/eee_03`。正式结果为 `runs/eee_03_r2` 和 `runs/nya_01`。协议、可执行文件、输入、配置和脚本哈希均有记录。

## 5. 评估协议与核验

采用 [NTU 原始评估教程](https://ntu-aris.github.io/ntu_viral_dataset/evaluation_tutorial.html) 的位置参考、棱镜补偿和刚体对齐原则，在本地实现同一对照协议：

1. 估计位姿从 IMU 中心变换到棱镜：`p_prism = p_imu + R_imu × [-0.293656, -0.012288, -0.273095] m`。
2. 只评估 LIO/LIVO 共有的输出时刻。真值线性插值，前后两个参考样本距待评估时刻均不超过 0.1 s；禁止外推和跨大缺口插值。
3. 两模式分别在相同样本上做一个全程 SE(3) 刚体最小二乘对齐，尺度固定为 1；不估计时间偏移，不使用逐段对齐。
4. 报告全量有效 ATE、覆盖率、误差分布及相对位移残差。Leica 不提供可用的姿态真值，因此不报告角度误差。

原始 MATLAB 教程取 CSV 第 1 列 `%time`；后来发布的官方 notebook 取第 3 列 `header.stamp`。本轮主结果按跑前确定的第 1 列计算。为了说明差异，下面补充只切换真值时间戳列、其余计算不变的敏感性检查；不声称这就是完整 notebook 的复现：

| 序列 | 模式 | 主口径第 1 列 ATE cm | 第 3 列敏感性 ATE cm |
|---|---|---:|---:|
| eee_03 | LIO | 6.8403 | 6.8477 |
| eee_03 | LIVO | 4.0560 | 4.0684 |
| nya_01 | LIO | 2.7651 | 2.7656 |
| nya_01 | LIVO | 2.7641 | 2.7646 |

核验包括：四项评估单元测试通过（旋转杆臂、禁止尺度拟合、拒绝无效真值插值、位移误差的平移不变性）；两模式配置差异检查；全部输入计数核对；SciPy 独立刚体拟合复核 ATE，差异小于 `1e-8 m`。`eee_03` 又完整重复运行一次，两个模式各 1,812 个最终位姿与首跑逐值完全相同。此重复验证表明当前离线流程可复现，不能代替跨设备或多场景统计。

## 6. 计算开销

测试机为 **AMD Ryzen 9 9950X，16 核 / 32 线程**，设置 `OMP_NUM_THREADS=3`、`OPENBLAS_NUM_THREADS=1`。下表为 native 估计进程处理完整输入的 wall time 和峰值 RSS，准备数据和图像去畸变耗时另计。

| 序列 | 模式 | native wall time s | 峰值 RSS MiB | 有点支持的视觉更新 / 总视觉更新 |
|---|---|---:|---:|---:|
| eee_03 | LIO | 14.15 | 1303 | 0 / 0 |
| eee_03 | LIVO | 22.74 | 1953 | 1168 / 1169 |
| nya_01 | LIO | 35.34 | 440 | 0 / 0 |
| nya_01 | LIVO | 46.18 | 1125 | 1247 / 1248 |

这是通过同步文件传输驱动同一原生估计器的桌面离线实验，未测 ROS2 DDS、实时传感器丢包、在线图像去畸变、语义模型并发或 S100 板载运行。不能根据此处快于数据时长就断言机器狗上可实时。

## 7. 复现与产物

工作目录：`build/benchmarks/ntu_viral_20261109/`。大体积原始数据和输入流保存在此目录，不放进源码提交。下载来源为 [NTU 官方主页链接的镜像](https://huggingface.co/datasets/Yuan-Shenghai/NTU-VIRAL-DATASET)，固定 revision 为 `4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd`。原始 ZIP 均通过源清单 MD5；SHA256 为本次实际下载文件计算值。

| 原始压缩包 | MD5 | SHA256 |
|---|---|---|
| eee_03.zip | 9bf10b5455ee6d2962324ef557f86433 | c842b269cbc9c31788b22095d49e49814499769ef22df9a0728c9e7f2a0f8b94 |
| nya_01.zip | 11ae1a0695c4af4d14398b28203159ce | 3eb0ede8f85148a41799fa10a29325975e3abf5ea58ec8d0f93df26baaa73646 |

小体积实验凭据另存于 [provenance 目录](../reports/public_slam_benchmark_2026-11-09/provenance/)，包括两组实际配置、输入审计、运行清单、下载校验和依赖版本；即使之后清理 build，大部分核验信息仍可查。原始 `nya_01.zip` 内另含一层 ZIP，解压流程已兼容并保留源文件。

冻结可执行文件 SHA256：`87db572e9db043ff727e023a9efb4f09e17ae2a2573953e2576f1fd10f4450e0`。

- `sources/SOURCE_MANIFEST.json`：原始下载来源、文件大小和 MD5。
- `protocol_before_results.json`：跑前固定协议；`data/*_download.json`：下载核验记录。
- `prepared/<sequence>/input_audit.json`：原始 bag 和输入流 SHA256、消息计数、时间戳与点格式检查。
- `runs/<run>/run.json`：主机信息、代码和可执行文件哈希、运行结果；同目录 `lio.yaml` / `livo.yaml` 为实际配置。
- `runs/<run>/<mode>/trajectory_final.tum`：每个时刻最终状态，姿态四元数顺序 `qx qy qz qw`；同目录保留控制台日志、时序统计和关键帧归档。
- [eee_03 数值](../reports/public_slam_benchmark_2026-11-09/eee_03/metrics.json)、[nya_01 数值](../reports/public_slam_benchmark_2026-11-09/nya_01/metrics.json)；各自同目录有对齐 CSV、误差图和 `audit.json`。

在仓库根目录、已有原始 bag 的环境中重新运行（每次使用新的输出目录）：

```bash
source /opt/ros/humble/setup.bash
source install/livo_integration/setup.bash
BENCH_ROOT="$PWD/build/benchmarks/ntu_viral_20261109"
BENCH_PY="$BENCH_ROOT/venv/bin/python"
BENCH_SEQ=eee_03  # 另一条为 nya_01
BENCH_REPEAT="$BENCH_ROOT/reproduce_$BENCH_SEQ"

# 首次准备可运行；已存在 prepared/<sequence> 时直接复用。
# "$BENCH_PY" FAST_LIVO2/scripts/prepare_ntu_benchmark.py \
#   --bag "$BENCH_ROOT/data/$BENCH_SEQ.bag" \
#   --output "$BENCH_ROOT/prepared/$BENCH_SEQ"

"$BENCH_PY" FAST_LIVO2/scripts/run_ntu_benchmark.py \
  --input "$BENCH_ROOT/prepared/$BENCH_SEQ" \
  --executable "$BENCH_ROOT/frozen/fastlivo_offline" \
  --output "$BENCH_REPEAT"
"$BENCH_PY" FAST_LIVO2/scripts/evaluate_ntu_benchmark.py \
  --run "$BENCH_REPEAT" \
  --ground-truth "$BENCH_ROOT/data/${BENCH_SEQ}_ground_truth.csv" \
  --output "$BENCH_REPEAT/evaluation"
"$BENCH_PY" FAST_LIVO2/scripts/audit_ntu_benchmark.py \
  --run "$BENCH_REPEAT" \
  --ground-truth "$BENCH_ROOT/data/${BENCH_SEQ}_ground_truth.csv" \
  --report "$BENCH_REPEAT/evaluation"
```

Python 环境通过 `venv --system-site-packages` 建立，额外安装 `rosbags==0.10.11`；本机 NumPy 1.26.4 与系统 SciPy 声明的支持范围有版本提示，数值测试和独立核验均通过。另存 `environment.txt` 记录实际依赖版本。后续 CI 应建立独立锁定环境。

NTU 数据及镜像标注 CC BY-NC-SA 4.0，保留原始归属。数据引用：Nguyen 等，*NTU VIRAL: A Visual-Inertial-Ranging-Lidar Dataset, From an Aerial Vehicle Viewpoint*，IJRR 41(3), 270–280, 2022。见 [数据集主页](https://ntu-aris.github.io/ntu_viral_dataset/)。

## 8. 下一步实验

先把这两条结果冻结成回归基线，再扩展场景。室内视觉贡献接近零，后续应增加图像调度接受率、特征支持数和视觉修正量诊断；若比较更强视觉权重，应单列开发实验，不回写本轮基线结果。

1. **NTU 扩充**：增加同设备其他场景，例如 `sbs_01` 和更有挑战的 `rtp` / `tnp` 序列。沿用本轮配置和评估口径，记录失败及覆盖率。若使用本轮调参，应把这两条标为开发集，新增序列作为独立测试。
2. **Hilti-Oxford 2022**：优先 `Exp14 Basement 2`（约 6 GB，有 6DoF 参考），随后长走廊、楼梯和多层场景，检查退化、俯仰变化和高度误差。该设备使用 Hesai 雷达，需要单独核对逐点时间、相机及 IMU 标定。官方序列、真值种类与下载见 [Hilti 数据页](https://hilti-challenge.com/dataset-2022)。本轮尚未下载或运行 Hilti。
3. **接回语义主线**：定位回归通过后，再检查关键帧对象在轨迹修正后的三维位置一致性、重复目标关联、地面分割和动态行人过滤。定位 ATE 不能代替对象检测 / 分割 / 三维框的标注评估。
4. **S100 验证**：在板载环境单独测构建、输入同步、长时间内存、频率和语义并发。本次桌面 benchmark 不算上机通过。
