# omni_slam：DINO 历史验证与类别语义评测

2026-11-11。固定此前 11 场景及两次支持历史图，完成池化 DINO、局部 DINO 两个实际整场景关联分支；同时对原基线、历史图和 DINO 分支补充类别评价。所有特征使用冻结 DINOv2-small，没有训练或事后阈值搜索。

后续已完成覆盖匹配的第二轮对照及类别结论，优先阅读[六组完整结论](public_dino_scope_2026-11-11.md)。SceneNN 类别读取的执行修复见运行目录 `evaluation_repair.json`；原冻结协议和源文件保持不变。

## 四组对照

原底座仍为 SAM2.1 Hiera L + SigLIP384 + 原生 ConceptGraphs + 来源消费保护 + 不同帧均值读出。历史候选、点云、RGB-D 参数、提供位姿、最终处理及 SigLIP 类别读出全部固定。

| 数据集 | 方法 | 实例 PQ×100 | 相对历史图变化 | 类别感知 PQ×100 | 几何匹配中类别正确数 |
|---|---|---:|---:|---:|---:|
| Replica | baseline | 42.4435 | -0.4407 | 30.3463 | 101/145 |
| Replica | native_bridge_2 | 42.8842 | +0.0000 | 30.6823 | 101/145 |
| Replica | dino_pooled_2 | 42.7153 | -0.1689 | 30.5580 | 101/145 |
| Replica | dino_local_2 | 42.8842 | +0.0000 | 30.6823 | 101/145 |
| SceneNN | baseline | 19.1184 | -0.8193 | 7.1631 | 10/30 |
| SceneNN | native_bridge_2 | 19.9377 | +0.0000 | 7.2868 | 10/31 |
| SceneNN | dino_pooled_2 | 19.9503 | +0.0126 | 7.2951 | 10/31 |
| SceneNN | dino_local_2 | 19.9377 | +0.0000 | 7.2868 | 10/31 |

![实例与类别指标并列](../reports/evidence_retention_2026-11-11/public_dino_v1/comparison.png)

baseline 为原强基线，native_bridge_2 为此前两次支持历史图；另外两组仅增加 DINO 对同一候选边的支持票验证。类别感知指标采用实例汇总分母，不是官方逐类别宏平均 panoptic PQ。

## DINO 到底在哪里起作用

完整 2200 个语义帧提取 DINOv2-small 特征；图像宽缩放至 448，高取最接近比例的 14 倍数，不裁剪。按原实测支持点投影到 patch，保存支持计数加权池化向量及最多 24 个局部 token。它不替换 SAM2 或 SigLIP。

- 池化分支：查询与每条历史的池化余弦取历史最大值。
- 局部分支：对局部 token 做外观互近邻，至少 3 对才能形成平均余弦，再对历史取最大值。这里没有三维距离门限，不声称互近邻就是真实几何对应。
- 两条待连接历史的分数均须超过同帧其他有效观测中的最大竞争分数。历史或任一竞争者缺乏有效局部证据，保留原生支持票并标注回退，不用零分替代。
- 仍需两个不同查询帧支持，仍执行整组同帧冲突检查和共同候选排序。候选集合不因 DINO 扩张，删边后其他既有候选可能因连通结构改变而获准合并。

| 数据集 | 分支 | 候选提议 | 外观支持 | 外观否决 | 证据不足回退 |
|---|---|---:|---:|---:|---:|
| Replica | dino_pooled_2 | 6755 | 2514 | 4241 | 0 |
| Replica | dino_local_2 | 6755 | 0 | 8 | 6747 |
| SceneNN | dino_pooled_2 | 513 | 274 | 239 | 0 |
| SceneNN | dino_local_2 | 513 | 7 | 10 | 496 |

以下为相对历史图取消的合并边之 GT 主导标签诊断，未判定包括背景、混合和支持不足节点。该统计不是合并精确率。

| 数据集 | 分支 | 取消同 GT 边 | 取消异 GT 边 | 取消未判定边 | 新执行其他候选边 |
|---|---|---:|---:|---:|---:|
| Replica | dino_pooled_2 | 15 | 0 | 21 | 0 |
| Replica | dino_local_2 | 1 | 0 | 0 | 0 |
| SceneNN | dino_pooled_2 | 3 | 0 | 7 | 0 |
| SceneNN | dino_local_2 | 0 | 0 | 1 | 0 |

[全部图层变化](../reports/evidence_retention_2026-11-11/public_dino_v1/edge_changes.json) · [证据覆盖明细](../reports/evidence_retention_2026-11-11/public_dino_v1/coverage.json)

## 类别评测如何避免“实例涨分等于语义变好”

在固定类别评价域上先做严格 IoU>0.5 实例匹配，再要求预测类别与 GT 类别相同。错误类别的几何匹配同时计为类别 FP 和 FN。另报告“已经几何匹配的对象中，有多少类别正确”，以及前景体素类别准确率。

Replica 使用原 `info_semantic.json` 的实例 ID+1 → class_id，无额外类别转换。SceneNN 使用 CVPR18 唯一标注顶点与原实例 mesh 的精确坐标对应，对每个实例取主导 NYU 类别，要求纯度 ≥0.7；不明确的类别仅从类别评价域置 void，不改变原实例 PQ。

SceneNN 的固定预测词表使用官方类别表的直接名称匹配，仅预先增加 book/books、shelf/shelves、nightstand/night stand、tv-screen/television 四个名称别名；无法对应的预测计错，不能为提高分数丢弃。它是词表有限且跨版本的类别诊断，不能与 Replica 数字直接比较。官方名称来源：[SceneNN NYU 类别表](https://hkust-vgd.ust.hk/scenenn/contrib/nyu_class/nyu_color.xml)；另核对了 [ScanNet 官方类别文件](https://github.com/ScanNet/ScanNet/blob/master/Tasks/Benchmark/classes_SemVoxLabel-nyu40id.txt) 中共同类别的编号。

| 场景 | 可确定类别 GT / 原前景 GT | 类别评价有效体素 / 原有效体素 |
|---|---:|---:|
| replica_office_0 | 18/18 | 100.00% |
| replica_office_1 | 19/19 | 100.00% |
| replica_office_2 | 36/36 | 100.00% |
| replica_office_3 | 41/41 | 100.00% |
| replica_office_4 | 25/25 | 100.00% |
| replica_room_0 | 44/44 | 100.00% |
| replica_room_1 | 30/30 | 100.00% |
| replica_room_2 | 33/33 | 100.00% |
| scenenn_014 | 38/50 | 99.93% |
| scenenn_015 | 12/16 | 99.98% |
| scenenn_021 | 26/30 | 99.03% |

## 所有场景结果

| 场景 | 实例：底座 | 历史图 | 池化 DINO | 局部 DINO | 类别：底座 | 历史图 | 池化 DINO | 局部 DINO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| replica_office_0 | 30.2672 | 30.9227 | 30.2672 | 30.9227 | 22.9537 | 23.3736 | 22.9537 | 23.3736 |
| replica_office_1 | 46.0832 | 47.0190 | 46.9800 | 47.0190 | 21.8826 | 22.3689 | 22.3689 | 22.3689 |
| replica_office_2 | 32.3091 | 32.7579 | 32.7579 | 32.7579 | 25.3114 | 25.6630 | 25.6630 | 25.6630 |
| replica_office_3 | 31.1166 | 31.7717 | 31.1166 | 31.7717 | 26.7123 | 27.3059 | 26.7123 | 27.3059 |
| replica_office_4 | 42.5080 | 43.3579 | 43.3665 | 43.3579 | 42.5080 | 43.3579 | 43.3665 | 43.3579 |
| replica_room_0 | 55.8296 | 55.7819 | 55.8001 | 55.7819 | 30.0373 | 30.0018 | 30.0347 | 30.0018 |
| replica_room_1 | 49.0101 | 49.0157 | 49.0101 | 49.0157 | 40.1486 | 40.1486 | 40.1486 | 40.1486 |
| replica_room_2 | 52.4243 | 52.4469 | 52.4243 | 52.4469 | 33.2161 | 33.2388 | 33.2161 | 33.2388 |
| scenenn_014 | 15.7244 | 15.8679 | 15.8679 | 15.8679 | 4.5994 | 4.6463 | 4.6463 | 4.6463 |
| scenenn_015 | 14.0783 | 14.4078 | 14.4303 | 14.4078 | 7.7680 | 7.9588 | 7.9837 | 7.9588 |
| scenenn_021 | 27.5525 | 29.5374 | 29.5528 | 29.5374 | 9.1218 | 9.2553 | 9.2553 | 9.2553 |

## 复现与限制

原生事件、过滤前节点、原底座及历史图的分类和几何均精确复现。新增 4 项测试覆盖局部互近邻最少支持、不确定回退、双分支相对证据、几何正确但类别错误、未知预测类别及缺失 GT 类别；全部通过。

GT 仅用于结果完成后的评价，未用于提取特征或候选投票。所有场景此前已被查看，属于开发验证；使用提供位姿和缓存前端，不能据此宣称 LIVO 定位或实时部署改进。源代码、模型、RGB、缓存和输出哈希均保留，旧实验不修改。

运行目录：`build/benchmarks/evidence_retention_20261111/public_dino_v1/`。每场景的 `runs/` 保存新地图、全部支持票与图决策；`evaluation/` 保存实例/类别分数、类别参考依据和投影预测；`features/` 保存 200 帧 DINO 特征。

[完整结果](../reports/evidence_retention_2026-11-11/public_dino_v1/summary.json) · [冻结协议](../reports/evidence_retention_2026-11-11/public_dino_v1/protocol.json) · [匹配对象的类别得失](../reports/evidence_retention_2026-11-11/public_dino_v1/semantic_changes.json) · [前一轮历史图](public_history_graph_2026-11-11.md)
