# omni_slam：对象历史图在公共实例数据上的迁移验证

2026-11-11。完整重放 Replica 8 场景及 SceneNN 014、015、021，沿用固定输入、原始 RGB-D 参数和实例评价域。所有 11 个基线的几何、成员、类别及投影标签精确复现；每组新结果均在预测完成后才读取真值。

本轮主对照是此前声明的 **两次不同查询帧支持的历史合并**；一次支持作为消融。没有训练、DINO 新模型、按结果挑场景或事后扫阈值。

**两次支持版本在 Replica 上由 42.4435 提升至 42.8842（+0.4407 PQ×100），在 SceneNN 上由 19.1184 提升至 19.9377（+0.8193）。11 场景中 10 个上升、1 个回退，支持继续发展对象历史维护机制。** 这是有实例真值的开发评测进展；DINO 贡献尚未由本轮证明，也不能称为官方 benchmark 成绩或统计显著提升。

## 总体结果

数值为 class-agnostic observed-voxel instance PQ × 100；每个数据集内部按场景宏平均。两个数据集分别报告，不混成一个均值。“胜/平/负”均相对对应场景原底座。

| 数据集 | 方法 | PQ×100 | 变化 | 胜/平/负 | TP/FP/FN 合计 |
|---|---|---:|---:|---:|---:|
| Replica | baseline | 42.4435 | +0.0000 | 0/8/0 | 145/142/101 |
| Replica | native_bridge_1 | 42.5959 | +0.1524 | 4/0/4 | 144/137/102 |
| Replica | native_bridge_2 | 42.8842 | +0.4407 | 7/0/1 | 145/136/101 |
| SceneNN | baseline | 19.1184 | +0.0000 | 0/3/0 | 30/94/66 |
| SceneNN | native_bridge_1 | 20.0805 | +0.9620 | 3/0/0 | 31/89/65 |
| SceneNN | native_bridge_2 | 19.9377 | +0.8193 | 3/0/0 | 31/90/65 |

![两组数据集宏平均](../reports/evidence_retention_2026-11-11/public_history_v1/macro_pq.png)

## 所有场景，包含回退

最后四列为两次支持相对底座的变化；混合对象为既有错误诊断器的 `mixed_foreground` 数量，不等价于精确的错误合并事件数。

| 场景 | 底座 | 一次支持 | 两次支持 | 两次 ΔPQ×100 | ΔTP | ΔFP | Δ混合对象 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| replica_office_0 | 30.2672 | 30.9244 | 30.9227 | +0.6555 | +0 | -1 | +0 |
| replica_office_1 | 46.0832 | 44.8064 | 47.0190 | +0.9357 | +0 | -1 | +0 |
| replica_office_2 | 32.3091 | 32.7579 | 32.7579 | +0.4487 | +0 | -1 | +1 |
| replica_office_3 | 31.1166 | 31.7717 | 31.7717 | +0.6551 | +0 | -2 | +0 |
| replica_office_4 | 42.5080 | 43.3997 | 43.3579 | +0.8499 | +0 | -1 | +0 |
| replica_room_0 | 55.8296 | 55.7488 | 55.7819 | -0.0477 | +0 | +0 | +0 |
| replica_room_1 | 49.0101 | 48.9995 | 49.0157 | +0.0056 | +0 | +0 | +0 |
| replica_room_2 | 52.4243 | 52.3589 | 52.4469 | +0.0227 | +0 | +0 | +0 |
| scenenn_014 | 15.7244 | 15.8679 | 15.8679 | +0.1434 | +0 | -1 | -1 |
| scenenn_015 | 14.0783 | 14.4078 | 14.4078 | +0.3295 | +0 | -1 | +0 |
| scenenn_021 | 27.5525 | 29.9657 | 29.5374 | +1.9849 | +1 | -2 | +0 |

![逐场景变化](../reports/evidence_retention_2026-11-11/public_history_v1/per_scene_delta.png)

## 碎片、混合对象和匹配得失

使用此前冻结的 `instance_error_audit.py` 阈值，未按新实验调整。分类是最终几何的错误特征，不是检测或关联原因的直接真值。

| 数据集 | 方法 | 预测碎片特征 | 预测混合前景 | GT 碎片特征 | 新获 / 丢失匹配 GT |
|---|---|---:|---:|---:|---:|
| Replica | baseline | 30 | 12 | 8 | 0 / 0 |
| Replica | native_bridge_1 | 29 | 15 | 8 | 0 / 1 |
| Replica | native_bridge_2 | 29 | 13 | 8 | 0 / 0 |
| SceneNN | baseline | 10 | 6 | 3 | 0 / 0 |
| SceneNN | native_bridge_1 | 8 | 6 | 2 | 1 / 0 |
| SceneNN | native_bridge_2 | 8 | 5 | 2 | 1 / 0 |

[全部新增与丢失匹配 GT 的前后明细](../reports/evidence_retention_2026-11-11/public_history_v1/matched_transition_details.json)。匹配得失分开记录，避免净 TP 数掩盖同时发生的改善与退化。

两次支持版本在 Replica 保留全部 145 个原有匹配，FP 从 142 降到 136；并非新增了 6 个正确对象。SceneNN 则保留全部原有匹配，新增 1 个匹配，FP 从 94 降到 90。Replica 的混合前景特征仍从 12 增到 13，不能声称误合并已全面减少。

## 两个有真值依据的机制案例

案例在结果完成后选出：SceneNN 021 是主版本唯一新增匹配实例；office_1 是一次支持消融唯一丢失匹配实例。二者用于解释结果，不参与参数选择。

**正例：SceneNN 021，GT 实例 286。** 原对象历史 21 和 320 分别有 42、32 次观测，合并后成为 74 次观测的共同历史。该 GT 最佳实例 IoU 从 **0.3059 提升到 0.5317**，越过固定 0.5 匹配阈值。其余片段仍存在，不能说该实例已完全重建。

![新增实例匹配的空间证据](../reports/evidence_retention_2026-11-11/public_history_v1/scenenn021_gain.png)

这里合并对象的预测类别从 `bed` 变为 `pillow`。当前指标是**类别无关实例 PQ**，因此该例证明了实例覆盖/合并的改善，不证明类别语义识别改善。后续应单独评价类别与对象身份。[成员及类别明细](../reports/evidence_retention_2026-11-11/public_history_v1/scenenn021_gain.json)

**反例：Replica office_1，历史边 19 / 36。** 只有第 480 帧为这条边提供支持。一次支持版本将短片段 36 并入既有对象：GT 实例 2 的 29 个支持体素不变，混入 GT 实例 6 的体素从 **14 增至 43**，使 GT 2 IoU 从 **0.5577 降至 0.3580**。两次支持版本不接受该边，保留原来的匹配。

![歧义查询及两条历史的图像](../reports/evidence_retention_2026-11-11/public_history_v1/office1_failure_rgb.jpg)

![混入另一实例后的真值体素对照](../reports/evidence_retention_2026-11-11/public_history_v1/office1_failure_geometry.png)

图中只使用同一固定参考域的位置，共同 PCA 平面用于展示，不进行配准或几何修正。该图说明“两次支持”在此例有效，不代表两个相邻查询帧必然提供独立证据。[反例明细](../reports/evidence_retention_2026-11-11/public_history_v1/office1_failure.json)

## 合并边的 GT 一致性审查

对每个过滤前节点，用其原始点云寻找固定真值参考域中 5cm 内最近点；包含背景参与纯度分母，至少 5 个有效支持点且主导前景 GT 纯度 ≥0.7 时，才给出可判定的主导标签。距离外、void、背景主导或混合节点保留未判定。该步骤只在评估中执行。

| 数据集 | 方法 | 两端同一主导 GT | 两端不同主导 GT | 未判定 |
|---|---|---:|---:|---:|
| Replica | native_bridge_1 | 39 | 2 | 77 |
| Replica | native_bridge_2 | 27 | 1 | 44 |
| SceneNN | native_bridge_1 | 16 | 1 | 31 |
| SceneNN | native_bridge_2 | 11 | 0 | 18 |

该表不是合并精确率：节点 GT 对应受已重建几何和参考域影响，且间接连通的分量可能比单条边更复杂。不同主导 GT 的已合并边全部保留在[审查清单](../reports/evidence_retention_2026-11-11/public_history_v1/different_gt_edges.json)，不从均值中剔除。

例如 office_1 的 254 / 926 边在第 810、830 帧都有支持，仍连接了两个主导 GT 不同的节点；主版本保留了它。这说明只计时序票数还不够，也为后续 DINO 局部实例证据提供了具体待检验的问题。

## 下一步如何收敛

1. 保留原强底座及一次/两次支持消融，将两次支持历史图作为下一阶段开发对照。它在两组数据上都有正向均值，但不是所有场景最优；不为 SceneNN 单独换成一次支持以拼凑最高分。
2. 针对仍然混合的节点和重复出现的错误候选，检验“观测是否相互独立、局部外观是否对应同一实例”，而不只累加相邻帧票数。DINO 的下一轮评价应同时覆盖所有候选和这些困难案例，不能只挑同物高分图片。
3. 将实例身份/覆盖与类别语义分开报告，补类别感知指标及新的未参与开发场景。论文证据应包括固定强基线、历史图增量、DINO 增量、错误合并与碎片化对照；当前实证支持的是历史图增量。

## 方案与评价边界

- 固定 SAM2.1 / SigLIP 前端和数据集提供的位姿；CG 仍使用各场景原配置、2.5cm 重叠距离和 DBSCAN。未把车库稀疏 LiDAR 的 20cm 距离或关闭 DBSCAN 的配置带入 RGB-D 数据。
- 全部原生歧义匹配生成候选；同一帧只投一票，整组同帧冲突拒绝新图层合并。候选本身来自几何与 SigLIP，不是纯几何对照。
- 新图层合并发生在最终过滤之前，随后沿用原生去噪、过滤、来源消费保护和不同帧均值类别读出。后续原生合并不受新图层冲突规则约束，最终结果另行评价。
- Replica 使用已有观察到的 5cm 体素域；SceneNN 使用已纠正的 CVPR18 标注域与共享 void 掩码。全部为 assigned-points-only 1NN ≤5cm，严格 IoU>0.5。不是官方排行榜的完整复现或整场景召回。
- 这些公共场景的旧结果此前已被查看。本轮是在车库开发后冻结规则的跨数据集迁移验证，仍属于开发验证，不称作全新的盲测。
- 使用提供位姿和缓存前端，不能据此宣称 LIVO 定位改善、系统实时性或机器狗部署性能。没有新增 DINO 实验，所有本轮变化均不得归因于 DINO。

## 复现与产物

结果根目录：`build/benchmarks/evidence_retention_20261111/public_history_v1/`。`protocol.json` 固定所有场景、输入哈希、源码及评价方式；`runs/<scene>/` 保存原生歧义事件、过滤前节点、候选图、三组分类与点云；`evaluation/<scene>/` 保存逐点预测、PQ 和完整错误诊断。所有输出独立保存，旧基线不修改。

[完整结果 JSON](../reports/evidence_retention_2026-11-11/public_history_v1/summary.json) · [冻结协议](../reports/evidence_retention_2026-11-11/public_history_v1/protocol.json) · [逐场景表](../reports/evidence_retention_2026-11-11/public_history_v1/scene_table.json) · [上一轮车库历史图实验](livo_history_graph_2026-11-11.md)

```bash
# 新目录首次运行；不覆盖既有实验。
PYTHONPATH=omni_semantic_map python3 scripts/benchmarks/run_public_history_graph.py --freeze
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 build/benchmarks/expanded_semantic_20261109/cg_venv/bin/python scripts/benchmarks/run_public_history_graph.py --scene replica_office_0
OPENBLAS_NUM_THREADS=1 build/benchmarks/ovo_full_20261110/venv311/bin/python scripts/benchmarks/evaluate_public_history_graph.py --scene replica_office_0
# 对 protocol.json 中全部 11 个场景运行后汇总。
build/benchmarks/ovo_full_20261110/venv311/bin/python scripts/benchmarks/evaluate_public_history_graph.py --summarize
build/benchmarks/ovo_full_20261110/venv311/bin/python scripts/benchmarks/report_public_history_graph.py
build/benchmarks/ovo_full_20261110/venv311/bin/python scripts/benchmarks/illustrate_public_history_failure.py
build/benchmarks/ovo_full_20261110/venv311/bin/python scripts/benchmarks/illustrate_public_history_gain.py
```
