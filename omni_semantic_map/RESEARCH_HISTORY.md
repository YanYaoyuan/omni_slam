# 历史语义研究入口

以下内容保留早期实验记录，不代表当前默认算法、当前论文结果或在线部署能力。当前入口见 [README](README.md)。

## 论文实验：对象身份评测

[依赖局部重算实验](../docs/benchmarks/dependency_local_fusion_2026-11-09.md)已完成 144 组主实验、432 次修正。新增 `DependencyReplayObjectMap`，按输入版本复用评分、融合和去噪节点。相对带缓存全量重放，局部修正几何平均加速 1.75×、28/32 组更快，所有主实验的身份与几何均保持参考结果；广域修正为 0.74×，两段真实广域更新也更慢。平均采样峰值 RSS 为 1045→1308 MiB，初始化成本也增加。66 项测试通过；连续真实位姿往返的缓存裁剪检查通过。该模块是有适用条件的研究实现，未替换在线默认关联。[方法说明](../docs/paper/dependency_local_fusion_method_2026-11-09.md)记录机制、对照和边界。

[融合初版 v0](../docs/benchmarks/replay_fusion_v0_2026-11-09.md)与[论文初稿](../docs/paper/omni_slam_v0_2026-11-09.md)已完成：新增 `ReplayableObjectMap`、原生 ConceptGraphs 核心适配器及独立实验入口。八个已查看开发场景的图框架池化 DINO/CLIP 均值为 15.078/15.187，对象融合为 18.712/19.628。初版与原生核心输出一致；全量位姿修正重放、几何恢复、往返恢复和历史输出保持在八场景均通过，58 项测试通过。这是可运行的研究基础，不是超过 ConceptGraphs 的新方法结论；合成恢复实验也不替代真实 LIVO 位姿评测。复现命令、对象点云/包围盒/证据输出与全部退步案例见报告。

[扩展场景及外部对照](../docs/benchmarks/expanded_semantic_2026-11-09.md)：当前版本固定后，累计六个未参与方法开发的 Replica 场景，最终观察体素 PQ×100 均值由旧版 16.079 提升至 16.834，三升一平两降。新增 ConceptGraphs 官方映射核心＋共享冻结观测对照，其六场景均值 22.178；它保留官方 CLIP、点云融合与后处理，但不等同于原版 GSA 前端端到端复现。新版 Flat 全帧回放和原版 Panoptic Mapping 因果记录对照也完成。51 项测试通过；当前尚不足以宣称超过强基线。

[可见表面归属改进](../docs/benchmarks/ownership_semantic_2026-11-09.md)：新增 `OwnershipGraph` 保守对照及 `CompetitiveGraph` 候选方法，以实测深度和预测掩码验证投影归属、重排图边、检查联合覆盖及分离反证。4 场景 × 11 方法已完成；新测试场景 office_1 持平，room_2 最终观察体素 PQ×100 从 22.434 提升至 23.559，开发 room_1 略退步。49 项测试及两段真实位姿更新的一致性检查通过。入口 `python -m omni_semantic_map.ownership_benchmark run --competitive --base <new_base> --scene <scene>`，该研究模块未替换在线节点默认关联。

[Replica 独立场景报告](../docs/benchmarks/replica_independent_2026-11-09.md)：office_0、room_1 各固定采样 200 帧，5 种方法共用预测掩码、提供位姿与 DINO 缓存。入口 `python -m omni_semantic_map.replica_benchmark` 分为 `extract / prepare / run / evaluate`，预测阶段不读取 GT；全部路径显式指定，已有结果拒绝覆盖。参数冻结后运行，主指标为观察表面体素 PQ，不是官方网格 AP；本轮尚未展示稳定整体优势。

[真实 LIVO 位姿修正报告](../docs/benchmarks/real_pose_revision_2026-11-09.md)：入口 `python -m omni_semantic_map.vbot_pose_revision`，验证位姿变化后的身份重划分与往返恢复。这是离线研究原型，未替换下文部署节点的关联默认行为；真实录包无对象 GT，结果不表示准确率提升。

Flat 公开数据的固定输入实验见 [Flat benchmark 报告](../docs/benchmarks/flat_semantic_benchmark_2026-11-09.md)。入口为
`python -m omni_semantic_map.flat_benchmark run --sequence <flat_dataset/run1> --output <new_run>`，
独立评价使用 `evaluate --sequence <flat_dataset/run1> --run <new_run> --output <new_metrics.json>`。
预测模式不读取真值分割；`--mode oracle_masks` 明确表示真值掩码诊断。
支持点按相机关键帧局部坐标保存于 `supports/*.npz`，保留像素索引和对象关联，便于后续重投影。
此处使用提供的 RGB-D 位姿，复用对象关联逻辑；它不是运行完整 LiDAR–IMU 定位系统。
`--method geometry_hungarian` 是内部几何对照，`omni_no_ambiguity` 是关闭歧义拒绝的实验变体。
生产关联默认门限保持不变。序列像素和观察表面体素的实例 PQ 均为诊断协议，不是官方 ScanNet AP。

冻结 DINOv2 与可撤销关联实验见 [DINO 首轮报告](../docs/benchmarks/flat_dino_2026-11-09.md)。
`python -m omni_semantic_map.dino_features` 提取带输入/权重哈希的冻结特征，支持显式 `--device cuda:0` / `cuda:1`；
`python -m omni_semantic_map.flat_dino_benchmark --sequence <sequence> --cache <cache> --method dino_local --output <new_run>`
复用固定检测与位姿。方法包括 `geometry`、`hsv`、`dino_mean`、`dino_local`、`dino_graph`。
图方法保存不可变观测和修正日志，`RevisableGraph.revise_poses()` 会重建候选关系并允许拆分旧融合；
逐帧因果输出与使用后续证据的 `final/` 回溯输出分别评价。当前是独立研究原型，未替换在线关联，
没有训练新的 Transformer，首轮结果尚未证明相对几何基线的稳定精度收益。

后续 [共视约束实验](../docs/benchmarks/flat_evidence_2026-11-09.md) 将同帧冲突细化为可撤销证据：
`python -m omni_semantic_map.flat_witness_benchmark --sequence <sequence> --cache <cache> --policy witness --output <new_run>`。
`hard` / `off` 是严格与无共视约束控制组，`--no-appearance` 隔离 DINO 代价。
只有表面接触及至少两个空间分离视图共同支持时，`WitnessGraph` 才撤销共视冲突；位姿变化后可重新拆分。
这是待独立场景验证的研究原型；[论文选题与实验锁定稿](../docs/paper/omni_slam_research_brief_2026-11-09.md)记录当前证据与未完成项。

`python -m omni_semantic_map.evaluation template objects.json --output annotations.json`
生成绑定源文件内容哈希的独立标注模板。逐观测填写 `valid_detection`，对有效目标填写跨帧一致的真实物体 `gt_instance_id`；不确定时保持 `null`，不要复制预测对象 ID 充当真值。

`python -m omni_semantic_map.evaluation evaluate objects.json --annotations annotations.json --output metrics.json`
输出关联 pair precision / recall / F1、拆分真值实例数、误合并预测记录数和标注覆盖率。缺少标注时指标为 `null`，不生成虚假的准确率。该评测仅覆盖已有深度观测，不能衡量检测召回率、完整物体三维框精度或端到端建图成功率。标注须核验全部场景/实例，部分标注只能用于诊断；输出文件已存在时拒绝覆盖。

