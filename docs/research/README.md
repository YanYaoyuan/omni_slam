# 对象级语义建图研究与复现

截至 2026-11-19，当前方法名为 **Ours / `full_authorization`**。定位支撑对象建图，论文评分单独评价对象地图。本文区分可独立运行的算法内核、依赖外部输入的完整系统，以及已封存的实验记录。

## 方法与代码

1. **实测几何与视觉观测**：机器人由 LIVO 给出关键帧位姿、图像和实测点。公开数据集使用给定位姿。SAM 掩码、SigLIP 描述及语义预测经可见性检查投影到三维。
2. **强基线对象图**：ConceptGraphs 风格关联与冻结 `history2` 配置维护对象 ID、三维表面和原始观测。当前 Ours 不把引入基础模型本身当作新贡献。
3. **表面候选**：`mask_cover_revision.py` 与实测输入适配器建立候选表面、掩码覆盖和源点索引。坐标保持来自真实测量。
4. **历史修改授权**：`revision_authorization.py` 比较共享与分离掩码的解释，要求移除任意单帧后仍有正证据。组内两两兼容防止冲突对象通过传递关系合并。
5. **局部身份与写入**：`batch_stable_transport.py/.cpp` 计算对象种子与观测帧组的图距离。旧点还需要授权与局部身份一致，新增点需要实际掩码见证和局部身份支持。不满足条件时保留旧身份或暂缓写入。

双帧图认证保证的是固定图上的种子帧组删除稳定性，不证明语义正确，也不自动抵抗位姿误差。当前消融支持历史保护与局部归属的作用，未显示额外认证相对最近种子的独立 AP 收益。

## 复现层级

| 层级 | 本仓库提供 | 另外需要 |
| --- | --- | --- |
| 核心机制 | 合成演示、身份授权与图传播实现、CPU 回归 | Python、NumPy、SciPy、g++ |
| ROS LIVO / 语义试跑 | 源码、消息、启动文件、示例配置、录包工具 | Humble、真实标定、驱动和本地模型 |
| 公开数据全流程 | 数据适配、前端/建图/评分脚本、方法与协议记录 | 数据集、权重、上游评测器、冻结输入与模型缓存 |
| 已发表述的精确实验重放 | 版本化执行脚本、论文结果与图表 | 对应 `protocol.json`、seal、源点/掩码/描述子缓存和源码哈希 |
| 论文编译 | LaTeX、正文、附录、固定图表与校验清单 | TeX Live / latexmk，或 Tectonic |

### 核心演示

按根 README 安装后执行：

```bash
python examples/revision_core_demo.py --output build/demo/revision_core.json
bash scripts/test_research_core.sh
```

演示应接受受多帧支持的对象对，拒绝通过冲突桥接的合并，并对缺少足够帧组支持的图节点输出 `-1`。输出目录可自选，已有结果拒绝覆盖。这里没有模型推理或 benchmark 跑分。

### 当前实验执行入口

所有命令从仓库根目录运行。设置：

```bash
export PYTHONPATH="$PWD/omni_semantic_map:$PWD/scripts/benchmarks${PYTHONPATH:+:$PYTHONPATH}"
python scripts/benchmarks/run_revision_assessment.py --help
python scripts/benchmarks/run_revision_authorization.py --help
python scripts/benchmarks/run_revision_robot.py --help
```

| 文件 | 定位 |
| --- | --- |
| `scripts/benchmarks/run_revision_authorization.py` | 两个已暴露开发窗口的机制形成与控制 |
| `scripts/benchmarks/run_revision_assessment.py` | 预先固定的 0009 / 0010 新窗口执行 |
| `scripts/benchmarks/review_causal_revision.py` | 固定输出干预、历史保护与局部归属消融 |
| `scripts/benchmarks/run_reviewer_extension.py` | 0003–0006 空间留出扩展，全部方法统一输出 |
| `scripts/benchmarks/run_revision_robot.py` | 330 帧车库中的同版本离线集成 |
| `scripts/benchmarks/evaluate_kitti360_development.py` | KITTI-360 本地协议的预测投影与评分 |

这些是**封存实验执行器**，会读取原协议和输入哈希。`--help` 可独立运行，执行 stage 必须先准备对应输入。代码中的本机存储目录保留历史来源含义，不是建议用户创建同名路径。重新适配数据时，应新建配置和结果根目录、记录新的输入/源码哈希；不要覆盖原始结果，也不要把修改路径后的运行冒充原封存重放。

### 模型与数据

- 室内共享前端：SAM2.1、SigLIP、RGB-D 和给定位姿。
- 当前室外对象图：SAM3、SigLIP、SegFormer、KITTI-360 图像及实测雷达点。
- ROS 在线试跑：Mask R-CNN COCO，地面可选 SegFormer。与 SAM3 离线论文实验区分。
- DINO、VGGT 与学习式记忆的脚本保留供追溯，见 [历史研究入口](../../omni_semantic_map/RESEARCH_HISTORY.md)。

模型权重与原始数据不随 Git 发布，按照各自官方项目条款自行获取。GPU 版 Torch / Torchvision 需按驱动与 GPU 配置匹配；核心 CPU 依赖不安装 GPU 推理栈。ROS 节点使用兼容 Humble 的 Python 3.10 环境，不在系统 ROS Python 中替换 NumPy。

## 结果与来源

| 实验版本 | 范围 | 主要结论 | 仓库内证据 |
| --- | --- | --- | --- |
| 共享前端 CG / OVO | Replica 8 个已查看场景 | 40.85 vs 33.56 PQ；三帧 OVO 38.59 | [开发报告](../benchmarks/shared_frontend_2026-11-10.md) |
| V1 可见性几何修订 | SceneNN 052 / 054 / 057 / 061 | 平均 +0.842 PQ，仍略低于 SOR | [固定表](../paper/visibility_revision/latex/tables/v1/main.tex) |
| 当前 Ours | KITTI-360 0009 / 0010 | 17.259 提升到 18.762 AP，+1.503 | [主表](../paper/visibility_revision/latex/tables/current/primary.tex) |
| 因果干预 | 同一预固定输出 | 历史保护与局部归属有收益，额外认证 AP 增量为零 | [消融正文](../paper/visibility_revision/latex/sections/merged/causal_review.tex) |
| 空间留出扩展 | 已用序列 0003–0006 内新窗口 | +2.908 AP，无条件均值更高 0.080，0004 有退步 | [全部方法摘要](results/reviewer_extension_20261117.json) |
| 真实机器人 | 车库与室外录包 | 对象地图、图像证据、三维框与地面展示 | [论文机器人档案](../paper/visibility_revision/latex/sections/merged/recording_atlas.tex) |

KITTI-360 报告 building/car、本地适配 AP50:90 × 100，未提交官方测试服务器。扩展窗口不是全新序列。不同前端、保留规则、评价域与指标不能直接合并排名。2026-11-19 全文复核已将四窗口扩展合入论文正文和附录 E.3，保持独立表格与协议说明。

## 测试范围

`scripts/test_research_core.sh` 只运行无需数据和模型的当前方法测试。完整历史测试还可能需要 Torch、Open3D、OpenCV、Pillow、OmegaConf、ConceptGraphs / OVO，以及 KITTI-360 官方评测器。依赖不存在而跳过的上游集成测试，不算完成全系统复现。

源码整理保持冻结算法文件内容，新增安装元数据、入口索引与演示。大规模 benchmark 不属于仓库整理时自动重跑的检查。
