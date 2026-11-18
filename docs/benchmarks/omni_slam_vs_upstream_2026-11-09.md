# omni_slam 与原版 FAST-LIO2 / FAST-LIVO2：公开基准对照

2026-11-09 · NTU VIRAL `eee_03`、`nya_01` 完整序列 · 原版 ROS1 Noetic 独立运行

## 结论

**omni_slam 目前没有表现出对原版的全面优势。停车场的主要退化集中在评估首段，需要优先排查；omni_slam 在礼堂中的视觉收益仍很小。** 礼堂序列中，omni_slam 的本次结果优于两个原版的观测值；与 FAST-LIO2 的差距较小。

以下为位置 ATE RMSE，单位 **cm**，越低越好。原版分别以 1×、2× 回放各跑一次，范围列出这两次观测值，**不是置信区间，也不是多次随机重复的统计分布**。omni_slam 使用上一轮冻结的原生离线轨迹，四种方法在相同真值时刻重新评估。

| 方法 | eee_03 停车场 | nya_01 礼堂 |
|---|---:|---:|
| FAST-LIO2 | 2.77 | 2.92 |
| FAST-LIVO2 | 2.89–3.23 | 3.83–3.84 |
| omni_slam (LIO) | 6.81 | 2.77 |
| omni_slam (LIVO) | 4.06 | 2.77 |

所有主对照完整跑完，输入观察器收到的 IMU、雷达和图像计数与原始 bag 审计一致。原版未添加内部队列计数埋点，因此“完整播放并收到”与“每条消息均完成估计更新”分别报告；输出覆盖见后表。

## 轨迹、首段问题与 Z 误差

下面两张图对应原版 **2× 回放主对照**，包含四种方法的轨迹、位置误差、Leica 坐标系 Z 残差和 ATE；1× 的复核数值另列。

![eee_03 四组比较](../reports/upstream_comparison_2026-11-09/eee_03/evaluation/comparison.png)

![nya_01 四组比较](../reports/upstream_comparison_2026-11-09/nya_01/evaluation/comparison.png)

`eee_03` 保留同一个全程刚体对齐，按评估区间开头 10 s 分段，得到以下位置 RMSE（cm）。这里没有删样本后重新对齐，正式成绩仍为全程结果。

| 方法 | 评估开头 10 s | 后续区间 |
|---|---:|---:|
| FAST-LIO2 | 2.92 | 2.76 |
| FAST-LIVO2 | 3.67 | 2.82 |
| omni_slam (LIO) | 23.96 | 3.02 |
| omni_slam (LIVO) | 12.02 | 2.72 |

omni_slam（LIVO）在后续区间已接近原版，首段是本轮最明确的改进目标。**评估起点约在原始输入开始后 29 s**，所以上述“开头 10 s”不是录包的最初 10 s。下一次诊断应覆盖原始输入开始的约 0–40 s，检查初始化、首段状态和融合调度；仅凭这条曲线，还不能确认某一个改动是根因。

下面是 2× 主对照的补充指标，误差单位均为 cm。独立关联数指该方法单独满足插值条件的真值样本数；实际精度统一使用四种方法的共同样本。

| 序列 | 方法 | 输出位姿数 | 独立关联数 | ATE 中位数 | ATE P95 | Z RMSE |
|---|---|---:|---:|---:|---:|---:|
| eee_03 | FAST-LIO2 | 1811 | 2850 | 2.27 | 4.86 | 0.83 |
| eee_03 | FAST-LIVO2 | 1809 | 2985 | 2.33 | 5.00 | 0.98 |
| eee_03 | omni_slam (LIO) | 1812 | 2952 | 2.70 | 6.46 | 1.32 |
| eee_03 | omni_slam (LIVO) | 1812 | 2952 | 2.33 | 5.91 | 1.09 |
| nya_01 | FAST-LIO2 | 3947 | 7413 | 2.79 | 4.28 | 1.50 |
| nya_01 | FAST-LIVO2 | 3945 | 7719 | 3.42 | 5.80 | 1.79 |
| nya_01 | omni_slam (LIO) | 3948 | 7269 | 2.50 | 4.23 | 1.14 |
| nya_01 | omni_slam (LIVO) | 3948 | 7269 | 2.50 | 4.24 | 1.13 |

Z 指全程刚体对齐后的 Leica 参考坐标系分量，不直接等于未经对齐的重力方向累计漂移。本轮不含回环、语义检测或地面分割评分。

## 原版的含义与版本

| 方法 | 实际来源与设置 |
|---|---|
| FAST-LIO2 | [hku-mars/FAST_LIO](https://github.com/hku-mars/FAST_LIO)，提交 `7cc4175de6f8ba2edf34bab02a42195b141027e9`，ikd-Tree 子模块固定为 `e2e3f4e9d3b95a9e66b1ba83dc98d4a05ed8a3c4`。 |
| FAST-LIVO2 | [hku-mars/FAST-LIVO2](https://github.com/hku-mars/FAST-LIVO2)，提交 `0d2c0346107b75b59934975adec9a6eeeb913c64`；与 omni_slam 最初移植时的上游提交相同。 |
| omni_slam（LIO） | 本工程基于 FAST-LIVO2 开发的定位前端，关闭视觉更新；不是原版 FAST-LIO2。 |
| omni_slam（LIVO） | 本工程视觉融合模式，包含完整扫描融合、SI 单位处理、保守视觉权重及此前工程修正。 |

两个官方仓库的估计器 `.cpp` / `.h` 和构建文件未修改。使用独立的 Ubuntu 20.04 / ROS1 Noetic 容器编译与运行；补齐依赖并先生成 ROS 消息头，解决构建依赖顺序。Livox 仅生成原始消息定义以满足编译，不运行硬件驱动；Sophus 使用已固定的旧版依赖及现代编译器复数 setter 兼容写法。相关依赖、辅助脚本和二进制哈希均保存。

FAST-LIO2 没有随上游提供本次所需的 NTU 专用配置，因此使用上游 Ouster 配置及 launch 默认策略，做设备适配：16 线、逐点时间单位 ns、盲区 1 m、NTU 雷达到 IMU 外参。其体素尺寸仍为 0.5 m、点抽样间隔 4、最大迭代 3；这些数值不是根据真值调出来的。FAST-LIVO2 采用上游 `NTU_VIRAL.yaml` 与 `camera_NTU_VIRAL.yaml`，保留视觉权重、曝光估计、初始化和图像时刻融合逻辑。

FAST-LIO2 通过输入转发将雷达 header 固定减去 0.1 s，使其扫描起点解释与 NTU 标定一致；FAST-LIVO2 和 omni_slam 使用上游配置中的同一固定偏移。没有拟合真值时间偏移，也没有套用 VBot 相机外参。

这是**当前版本、参数和运行栈的实际基线比较**，不等于各算法经过充分调参后的最优成绩，更不等于论文排行榜复现。omni_slam 与原版在初始化、IMU 处理、图像预处理、融合时序、视觉权重和输出时机上均有差异，不能把成绩差异直接归因于单个因素。

## 统一评估与回放复核

使用 [NTU VIRAL](https://ntu-aris.github.io/ntu_viral_dataset/) 的 Leica 位置真值。按 [官方评估教程](https://ntu-aris.github.io/ntu_viral_dataset/evaluation_tutorial.html) 补偿 IMU 到棱镜的旋转杆臂 `[-0.293656, -0.012288, -0.273095] m`，做尺度固定为 1 的全程 SE(3) 对齐。

本次为适应各方法不同的输出时刻，将估计棱镜位置插值到 **同一组原始 Leica CSV 第 1 列时间戳**。插值前后跨度分别不超过 0.1 s，边界允许 1 μs 浮点容差；禁止外推。四种方法取共同有效样本，再分别做一个全程刚体拟合。不使用姿态真值、尺度拟合、时间偏移拟合或分段重对齐。

原版 FAST-LIVO2 的 ROS 里程计消息使用发布时间，评估实际采用它自带的 `Log/result/<sequence>.txt`，该文件记录测量时刻。FAST-LIO2 使用其里程计的 `lidar_end_time`。这是避免将处理延迟误记为定位误差的必要步骤。

| 序列 | Leica 真值样本数 | 共同评估样本 / 覆盖率 | 评估跨度 s |
|---|---:|---:|---:|
| eee_03 | 2990 | 2807 (93.9%) | 150.71 |
| nya_01 | 7769 | 6925 (89.1%) | 391.97 |

上一轮 omni_slam 报告在两模式共有的估计时刻插值真值；本轮改为在真值时刻插值各方法的估计。因此 omni_slam 的 ATE 略有变化，这是采样口径变化，使用的原始轨迹没有变化。

原版回放复核在同一组主对照样本上计算：

| 序列 | 方法 | 1× ATE cm | 2× ATE cm | ATE 差值 mm |
|---|---|---:|---:|---:|
| eee_03 | FAST-LIO2 | 2.768 | 2.768 | 0.000 |
| eee_03 | FAST-LIVO2 | 3.233 | 2.885 | 3.479 |
| nya_01 | FAST-LIO2 | 2.921 | 2.921 | 0.000 |
| nya_01 | FAST-LIVO2 | 3.836 | 3.830 | 0.054 |

停车场 FAST-LIVO2 的两次运行首个输出时刻相差约 0.1 s，日志中的初始重力估计也不同，见 [启动诊断](../reports/upstream_comparison_2026-11-09/provenance/startup_diagnostic.json)。结果表明回放条件会影响上游表现；本次每种速度只跑一次，不能将速度因素与到达顺序、线程调度的影响完全分离。由于发现这一差异，又补跑了礼堂 1×，所有结果均保留，没有据此改参数。

验证包括：合成轨迹的共同采样、缺口排除与刚体对齐检查；四种方法 × 两条序列的 SciPy 独立 ATE 复核（差异小于 `1e-8 m`）；原版完整输入计数核对；运行前后源码哈希核对。omni_slam 的首轮评估测试和重复性记录见 [此前报告](public_slam_benchmark_2026-11-09.md)。

## 文件与复现

- [停车场输入轨迹与清单](../reports/upstream_comparison_2026-11-09/eee_03/manifest.json)、[停车场结果](../reports/upstream_comparison_2026-11-09/eee_03/evaluation/metrics.json)。
- [礼堂输入轨迹与清单](../reports/upstream_comparison_2026-11-09/nya_01/manifest.json)、[礼堂结果](../reports/upstream_comparison_2026-11-09/nya_01/evaluation/metrics.json)。
- [停车场两速复核](../reports/upstream_comparison_2026-11-09/eee_03_rate_validation.json)、[礼堂两速复核](../reports/upstream_comparison_2026-11-09/nya_01_rate_validation.json)。
- [配置、版本、容器依赖与辅助脚本快照](../reports/upstream_comparison_2026-11-09/provenance/)。每条序列 `inputs/` 保留原始 TUM 轨迹和真值 CSV，`evaluation/` 保留对齐 CSV、误差图与独立复核 JSON。
- 原版源码、完整控制台日志和可复用环境位于 `build/benchmarks/upstream_20261109/`；数据沿用 `build/benchmarks/ntu_viral_20261109/data/`。
- [统一评估脚本](../../FAST_LIVO2/scripts/compare_slam_benchmarks.py)。

在仓库根目录，可用新的输出目录重算（不需要 ROS1）：

```bash
build/benchmarks/ntu_viral_20261109/venv/bin/python \
  FAST_LIVO2/scripts/compare_slam_benchmarks.py \
  --manifest docs/reports/upstream_comparison_2026-11-09/eee_03/manifest.json \
  --output build/benchmarks/upstream_20261109/reproduce_eee03_evaluation
```

重新运行原版时，使用已构建镜像 `omni-benchmark/upstream-noetic:20261109`，将基准工作目录挂到 `/work`、原始 bag 目录只读挂到 `/datasets`，运行 `/work/tools/entry.sh --method fastlivo2 --sequence eee_03 --rate 1 --suffix _reproduce`。`--method` 可选 `fastlio2`，`--sequence` 可选 `nya_01`；每次使用新的 suffix。`provenance/tools/` 是脚本快照，执行前应放回原工作目录结构。

同一台 AMD Ryzen 9 9950X 上运行。原版 ROS1 播放、omni_slam 原生离线传输的调度和 I/O 不同，本轮不将 wall time 排成吞吐榜，也不推导 S100 板载性能。NTU 数据来源与 CC BY-NC-SA 4.0 归属保留；数据引用为 Nguyen 等，IJRR 2022 的 NTU VIRAL 论文。

## 下一步

1. **先定位停车场首段退化。** 对照原始 0–40 s 的初始化样本、重力、IMU 状态和首次融合更新，验证原因后再修改。
2. **做可解释的消融。** 增加原版 FAST-LIVO2 关闭视觉的基线，分别检查 IMU 单位策略、完整扫描融合和视觉权重；一次只改一个因素。本轮两条序列若用于调参，应标为开发集，再用未参与调参的公开序列验证。
3. **同时回归 VBot 六个包。** 公共数据改善后，还要检查原有墙面重影、地面错层、Z 误差和关键帧一致性，避免只适配 NTU。
4. **保留语义建图主线。** 定位修改通过回归后，再验证对象三维框、跨帧关联和地面分割随位姿更新的一致性，之后安排 S100 性能与实时输入测试。
