# 对象地图修订：方法与复现

论文中的 **Ours** 对应 `full_authorization`。它接收已有对象地图和多帧观测，决定哪些历史对象可以合并，以及新增实测点应分配给哪个对象。定位与前端模型在对比实验中保持固定。

## 方法与代码

| 步骤 | 实现 | 作用 |
| --- | --- | --- |
| 观测准备 | LIVO 或数据集适配器 | 提供位姿、实测点、图像掩码和语义特征 |
| 对象关联 | ConceptGraphs 风格关联，`history2` 配置 | 建立对象 ID，保存点集和原始观测 |
| 表面候选 | `mask_cover_revision.py`、`discovery_measured_input.py` | 根据掩码覆盖和源点索引建立候选表面 |
| 历史合并判断 | `revision_authorization.py` | 比较共享掩码与分离掩码的解释；移除任意单帧后仍需有正证据，组内对象须两两兼容 |
| 局部点分配 | `batch_stable_transport.py`、同名 `.cpp` | 计算对象种子到图节点的距离，确定旧点修改与新增点归属 |

旧点仅在合并获准且局部归属一致时修改标签；新增点需要掩码支持和局部身份支持。所有坐标来自实测点。

种子帧检查针对固定图和边权：移除一个种子帧组后，最近对象仍应保持不变。它不保证语义正确或位姿误差鲁棒性。现有消融显示历史身份保护和局部点分配有收益，额外种子帧检查相对最近种子没有独立 AP 增益。

## 复现层级

| 内容 | 仓库提供 | 所需外部输入 |
| --- | --- | --- |
| 核心演示 | 合成对象图、合并判断、图传播和 CPU 测试 | Python、NumPy、SciPy、g++ |
| ROS 语义建图 | 节点、消息、启动文件、配置和录包工具 | ROS 2 Humble、传感器标定、驱动、模型权重 |
| 数据集评测 | 数据适配、建图与评分脚本 | 数据集、模型、上游评测器及观测缓存 |
| 复现已有结果 | 实验脚本、结果表与协议记录 | 原运行的 `protocol.json`、seal、源点 / 掩码 / 描述子缓存及对应代码版本 |
| 论文编译 | LaTeX 与图表 | TeX Live / latexmk 或 Tectonic |

### 核心演示

按[根目录说明](../../README.zh-CN.md#快速开始)安装后执行：

```bash
python examples/revision_core_demo.py --output build/demo/revision_core.json
bash scripts/test_research_core.sh
```

示例包含可合并对象对、冲突合并和观测不足的节点。观测不足时输出 `-1`。输出目录可自选；已有文件不会覆盖。此入口不执行模型推理或数据集评分。

### 评测入口

从仓库根目录运行：

```bash
export PYTHONPATH="$PWD/omni_semantic_map:$PWD/scripts/benchmarks${PYTHONPATH:+:$PYTHONPATH}"
python scripts/benchmarks/run_revision_assessment.py --help
python scripts/benchmarks/run_revision_authorization.py --help
python scripts/benchmarks/run_revision_robot.py --help
```

| 脚本 | 实验 |
| --- | --- |
| `run_revision_authorization.py` | 两个开发窗口的方法与对照实验 |
| `run_revision_assessment.py` | 预先选定的 0009 / 0010 窗口 |
| `review_causal_revision.py` | 修改固定输出，分离历史保护与局部点分配的作用 |
| `run_reviewer_extension.py` | 0003–0006 空间留出窗口，含机制对照与外部核心适配 |
| `run_revision_robot.py` | 同一方法在 330 帧车库数据上的离线运行 |
| `evaluate_kitti360_development.py` | KITTI-360 预测投影与本地评分 |

脚本位于 `scripts/benchmarks/`。`--help` 可独立运行；执行阶段会读取原协议、缓存和输入哈希。部分默认路径指向原实验环境，需要按实际存储位置配置。

适配新数据或修改算法时，使用新的配置与输出目录，记录新的源码和输入哈希。保留旧协议、seal 与结果，以便区分复现和新实验。

### 数据与模型

| 入口 | 输入与模型 |
| --- | --- |
| KITTI-360 论文实验 | 图像、给定位姿、实测雷达点；SAM3、SigLIP、SegFormer |
| ROS 在线语义节点 | LIVO 语义帧；Mask R-CNN COCO，地面分割可选 SegFormer |
| 早期室内实验 | RGB-D、给定位姿；SAM2.1、SigLIP |

数据和权重按各自项目的许可获取，不包含在仓库中。GPU 推理依赖匹配驱动的 Torch / Torchvision；核心 CPU 环境只安装 `requirements/research-core.txt`。ROS 节点使用兼容 Humble 的 Python 3.10 环境，模型依赖应与系统 ROS Python 隔离。

DINO、VGGT 和学习式记忆实验见[历史分支索引](../../omni_semantic_map/RESEARCH_HISTORY.md)。

## 结果与来源

| 实验 | 范围 | 结果 | 详细记录 |
| --- | --- | --- | --- |
| 当前 Ours | KITTI-360 0009 / 0010 | AP 17.259 → 18.762，+1.503 | [主表](../paper/visibility_revision/latex/tables/current/primary.tex) |
| 输出干预 | 同一组固定预测 | 历史保护与局部归属有收益；额外种子帧检查 AP 增量为零 | [消融](../paper/visibility_revision/latex/sections/merged/causal_review.tex) |
| 空间留出扩展 | 已用序列 0003–0006 内新窗口 | AP +2.908；无条件修订均值高 0.080；0004 有退步 | [结果 JSON](results/reviewer_extension_20261117.json) |
| 车库集成 | 330 帧录包 | 四组对象地图与图像投影，无独立对象标注 | [补充材料](../paper/visibility_revision/latex/sections/merged/current_supplement.tex) |
| 早期 CG / OVO 对比 | Replica 8 个开发场景 | 40.85 / 33.56 PQ；三帧 OVO 为 38.59 | [开发报告](../benchmarks/shared_frontend_2026-11-10.md) |
| V1 几何修订 | SceneNN 052 / 054 / 057 / 061 | 平均 +0.842 PQ，略低于 SOR | [结果表](../paper/visibility_revision/latex/tables/v1/main.tex) |

KITTI-360 评价建筑和车辆，指标为本地适配 AP50:90 × 100，未提交官方测试服务器。两组窗口分别统计；空间留出窗口来自开发阶段用过的序列。不同前端和协议下的结果不合并排名。

其他录包的对象框、路线和地面图层见[录包实验档案](../paper/visibility_revision/latex/sections/merged/recording_atlas.tex)。

## 测试

`scripts/test_research_core.sh` 运行无需数据和模型的当前方法测试。完整研究测试还可能依赖 Torch、Open3D、OpenCV、Pillow、OmegaConf、ConceptGraphs / OVO 及 KITTI-360 评测器；缺少可选依赖的测试会跳过。

构建与测试记录见 [VALIDATION.md](VALIDATION.md)。数据集评测需要按上述入口单独执行。
