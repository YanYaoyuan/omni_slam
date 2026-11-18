# 主线进展：室外对象证据丢失定位

本轮是实际执行的逐阶段诊断和固定几何读出对照，不是新机制完成声明。

报告：[全部对象真实图像、阶段覆盖与对照](../../output/kitti360_retention_audit_v1/index.html)。

两个已暴露 KITTI-360 窗口，同一基线几何、对象身份、原选定观测和 19 类词表，使用已有 SigLIP 三通道缓存。全量报告原 learned 融合、仅 global、仅 masked、仅 bbox 和普通等权三通道五组；没有训练、GPU 推理或阈值搜索。

| 读出对照 | 合并完整窗口 AP50 × 100 |
|---|---:|
| 原 learned 融合 | 5.556 |
| 仅 global | 0.000 |
| 仅 masked | 24.074 |
| 仅 bbox | 20.370 |
| 等权三通道 | 12.963 |

masked 在 0002 新增 10 个正确车辆匹配，无原匹配丢失；0003 保留原匹配。所有建筑 AP50 仍为 0。分数由跨场景唯一命名的匹配记录共同重算，原 learned 控制完整复现旧成绩。不是官方榜单、独立测试、新 Ours 或重新建图结果。

此前怀疑的目标流失现在分解为：车辆在全部测量域有足够覆盖的 25/27，经掩码支持 22、连通支持 20、至少三帧历史组 14、最终对象观测 14、目标类别保留 3。这里的覆盖计数是假定完美身份/背景划分的必要条件诊断，不是实际召回。17/22 建筑在测量域就覆盖不足。原始预测、冻结代码和参考域均未改变。

0002 中 11 辆车在最终对象观测域仍有足够覆盖，却被非 car 语义类别排除。9 个对应对象的 masked 与 bbox 同时预测 car，而 learned 预测 sidewalk/road。全图特征多读作 sidewalk/road，支持上下文压制局部信息的机制解释；未直接记录当次融合权重，不能声称已经定位权重实现故障。

## 对下一新机制的约束

1. 主线仍为持续对象级语义地图，LIVO 提供位姿/几何。不得将目前零触发的删点规则等同主线。
2. 新机制保留对象局部证据与场景上下文的区别；若引入历史对象归属修订，应证明在更强读出基线上的独立收益。
3. masked-only、bbox-only、普通等权融合成为必需对照，不能只超越已知失配的 learned 读出便宣称创新。
4. 本轮两个窗口已暴露，后续只能作为开发数据；新增方法必须先冻结，再取得新的独立验证。不得把本轮事后恢复成绩写成预注册独立收益。
5. 机器人 330 帧 baseline/SOR/旧 Ours 的 14 个产物已重新校验，Ours 删除 0 点，SOR 删除 450 点。新机制的机器人对照未完成，独立语义标注按用户要求暂缓。对照产物存在不等于准确率提升。

三个目标仍保留：新机制实现和有效性验证、对应机器人对照、新增公开评测；目标未宣告完成。当前直接改变下一动作的证据，是局部语义信息仍然存在且可恢复，不能继续把全部失败归因于关联或几何过滤。

权威文件：

- `build/benchmarks/kitti360_retention_audit_v1/{protocol,summary}.json`
- `build/benchmarks/kitti360_semantic_channel_audit_v1/{protocol,readouts_complete,summary}.json`
- `output/kitti360_retention_audit_v1/pooled.json`
- `build/experiments/garage_visibility_20261113/run.json`

实现脚本：`audit_kitti360_object_retention.py`、`audit_kitti360_semantic_channels.py`、`report_kitti360_retention_audit.py`。初版阶段审计错误地用融合前原生节点代替 history2 组，其断言失败日志已存档；修正后按实际处理顺序核验支持集包含关系，不修改原建图算法。
