"""Publish the measured stage-1 audit and, when complete, stage-2 controls."""
from pathlib import Path
from collections import Counter
import json
from audit_evidence_retention import ROOT, REPORT


def main():
    stage1=json.loads((REPORT/'stage1/summary.json').read_text())
    cases=json.loads((REPORT/'stage1/case_manifest.json').read_text())
    limits=json.loads((REPORT/'stage1/convex_readout_limits.json').read_text())
    labels=Counter(l['gt_class_name'] for rows in cases['direct_losses_with_gt'].values() for l in rows)
    text='''# omni_slam：强基线对象证据审计与机制对照

2026-11-11。基于 SAM2＋SigLIP＋ConceptGraphs 守恒分支＋不同帧均值读出的 **42.4435 PQ×100** 基线。所有八个 Replica 场景均为已查看的开发场景，GT 只用于评分和事后诊断。

## 阶段 1：已完成逐对象审计

八场景共 667 个最终 CG 区域；语义过滤后有 287 个投影到评价域的预测实例，其中匹配成功 145 个。所有缓存帧、关键输入文件与运行输出的哈希核验通过，八场景均值分类逐对象复现。

| 最终预测的几何特征 | 数量 |
|---|---:|
'''
    mapping={'matched':'匹配成功','fragmentation_consistent':'碎片化特征','partial_coverage':'覆盖不足','background_dominated':'背景占主导','mixed_foreground':'混合多个前景对象','excess_support':'范围过大','ambiguous_overlap':'其他重叠不足'}
    for k,label in mapping.items():text+=f'| {label} | {stage1["predicted_counts"].get(k,0)} |\n'
    text+='''
前景过滤损失 32 个原本可匹配的 GT：**26 个直接被背景分类过滤，6 个仍保留但因最近邻标签转移变化失去匹配**。没有将这两类混称为直接误删。

### 关键发现：多帧一致不等于语义正确

'''
    text+=f'26 个直接损失中，按数据集标注有 **{labels.get("lamp",0)} 个灯具、{labels.get("picture",0)} 幅画**。灯具被预测为天花板，画被预测为墙。灯具错误跨多个场景出现，并非单个低分异常帧。\n\n'
    text+=f'进一步检查每个不同帧的描述子，**{limits["blocked_objects"]} 个对象**存在同一个背景文本向量，在每帧都高于所有前景文本向量。因此，在保持这些描述子和词表不变时，任何非负加权均值都无法将它们改判为前景；增加帧数、改面积权重或挑选其中部分帧同样无法解决。这个结论来自逐描述子分数不等式，不是凭图猜测。\n\n'
    text+='''这说明第二阶段需要区分两件事：观测冗余/权重能否改善其余对象；现有语义表示是否对贴附背景的对象存在系统混淆。不能把低方差、高一致性的错误描述子当成高质量证据，也不能把后续文本提示或特征替换的收益算给观测加权机制。

### 观测特征与错误的关系

| 最终类别 | 不同帧数中位数 | 掩码面积中位数（像素） | 最大视角差中位数（度） | 特征一致性中位数 |
|---|---:|---:|---:|---:|
'''
    for k in ['matched','fragmentation_consistent','partial_coverage','background_dominated','mixed_foreground']:
        s=stage1['statistics'][k]
        text+=f'| {mapping[k]} | {s["unique_frames"]["median"]:.1f} | {s["median_area_px"]["median"]:.1f} | {s["max_view_angle_deg"]["median"]:.1f} | {s["feature_agreement"]["median"]:.3f} |\n'
    text+='''
覆盖不足对象通常观测较少、掩码较小，但存在观测很多仍失败的对象；这些是关联统计，不证明帧数或视角是原因。互斥掩码占比大多为 1，区分能力有限。缓存没有 SAM 预测 IoU/稳定性分数，因此不虚构模型质量分数。

![观测统计](../reports/evidence_retention_2026-11-11/stage1/evidence_statistics.png)

### 每场景原图证据

每场景固定选取：ID 最小的直接过滤损失、评价体素最多的覆盖不足/碎片预测、评价体素最多的成功匹配对象。每个对象显示面积最大的原始掩码，平局按观测 ID；不存在直接损失的场景保留空白。红色是缓存预测掩码，GT 名称仅用于事后解释，未进入算法。

![原图与实例掩码](../reports/evidence_retention_2026-11-11/stage1/image_cases.png)

所选灯具是天花板上的面板状区域；原图中可看到预测掩码覆盖这些区域。部分案例类别具有外观歧义，本报告以数据集实例和类别标注为评价依据。大沙发、床等的局部掩码也说明：重新分配语义权重不会自动补全几何或修复被拆开的身份。

## 阶段 2：固定输入的读出与保留对照

候选机制按相机到对象中心的角度估计观测相关性，以原始掩码面积为观测权重，再除以邻近方向的面积加权密度。固定 30° 相关尺度，未扫描阈值、未按场景调参；均值、均值＋三帧、面积加权、最远视角选十帧、只去冗余作为控制。

此轮固定几何、对象成员、SAM2 掩码、SigLIP 描述子、文本词表与评价域，**只验证语义聚合和保留行为**。不能将它解释为几何补全或新的关联网络。原始基线不覆盖。

'''
    stage2path=REPORT/'stage2/summary.json'
    if stage2path.exists():
        stage2=json.loads(stage2path.read_text())
        names={'mean':'不同帧均值（基线）','mean_min3':'均值＋至少三帧','area':'面积加权','view_fps10':'最远视角十帧','density':'仅视角去冗余','area_density':'面积＋视角去冗余（预定候选）'}
        text+='| 配置 | PQ×100 | 相对均值 | 胜/平/负 | TP / FP / FN |\n|---|---:|---:|---|---|\n'
        for m,row in stage2['methods'].items():
            t=row['totals'];text+=f'| {names[m]} | {row["macro_pq"]*100:.4f} | {row["delta_vs_mean"]*100:+.4f} | {row["wins"]}/{row["ties"]}/{row["losses"]} | {t["tp"]} / {t["fp"]} / {t["fn"]} |\n'
        text+='\n| 场景 | 均值 | 面积 | 视角十帧 | 去冗余 | 面积＋去冗余 |\n|---|---:|---:|---:|---:|---:|\n'
        for i,scene in enumerate(stage2['scenes']):text+='| '+scene+' | '+' | '.join(f'{stage2["methods"][m]["pq_by_scene"][i]*100:.3f}' for m in ['mean','area','view_fps10','density','area_density'])+' |\n'
        candidate=stage2['methods']['area_density'];best_simple=max(stage2['methods'][m]['macro_pq'] for m in ['mean','mean_min3','area','view_fps10'])
        if candidate['macro_pq']<=best_simple:
            text+='\n**决策：预定候选未超过简单控制，不采用为主方法；保持 42.4435 均值基线。** 其他消融的正向结果保留为探索性观察，不把事后最优消融改称预定主方法成功。第二阶段的这一轮对照完整结束，但尚未建立足以用于论文的新增机制。\n'
        else:text+='\n预定候选超过本轮简单控制，但仅增加 0.1203 点、两胜四平两负，总 TP 不变、FP 减少一个。均值仍为默认推荐基线；候选与均值一起进入冻结的新场景验证。开发集涨点不等于论文创新或泛化成立。\n'
        paired_path=REPORT/'stage2/paired_changes.json'
        if paired_path.exists():
            paired=json.loads(paired_path.read_text())['comparisons']['area_density'];lo,hi=paired['exploratory_scene_bootstrap_interval']
            text+=f'\n逐对象匹配集合也没有增加或损失 GT，原先 32 个过滤损失均未被候选恢复。八场景配对 bootstrap 的探索性区间为 **{lo:+.3f} 至 {hi:+.3f} 点**，包含零；这个区间只描述已查看开发场景，不作为独立泛化显著性证明。\n'
        text+='\n六个读出共 48 个场景配置已评分；均值基线八场景评价标签逐点完全复现。新增模块的 10 项针对性测试通过，覆盖观测相关性、排列不变性、权重归一化、错误一致性的不可恢复性和输入验证。运行时间包含读缓存与 Python 聚合，不能用作在线实时性结果。\n'
    else:text+='六组配置正在执行；结果齐全后更新此表，不使用部分场景作结论。\n'
    qa_files=[REPORT/'stage3'/(s+'_alignment_metric.json') for s in ['014','015','021']]
    if all(p.exists() for p in qa_files):
        text+='\n## 阶段 3：新场景输入已准备，模型验证未完成\n\n三个新场景的 13 个官方文件（约 1.68 GB）已下载并记录 SHA256；每场景按整段给定位姿均匀解码 400 对 RGB-D，语义使用其中 200 帧。尚未生成新场景模型分数。\n\n'
        text+='| 场景 | 给定位姿帧 / 视频解码帧 | 几何 / 语义采样帧 | 抽样输入对齐距离（mm） |\n|---|---|---|---:|\n'
        for s,p in zip(['014','015','021'],qa_files):
            qa=json.loads(p.read_text());meta=json.loads((REPORT/'stage3'/(s+'_input_manifest.json')).read_text())
            text+=f'| {s} | {meta["source_frames"]} / {meta["decoded_video_frames"]} | 400 / 200 | {qa["median_of_medians_m"]*1000:.2f} |\n'
        text+='\n距离为十个固定抽样帧的深度点到参考网格最近顶点距离：先逐帧取中位数，再取这些中位数的中位数。它仅检查输入对齐，不是 SLAM 轨迹误差或语义精度。RGB-D 由官方解码同步规则配对，最大时间差约 33 ms；无位姿的尾部视频帧保留但不参与采样。\n\n'
        text+='021 的给定位姿含固定世界坐标仿射形变，直接逐帧做 SVD 修正旋转会丢失与平移/参考几何的一致性。因此先冻结统一输入修正：由首个位姿计算 `W=(R0 R0^T)^(-1/2)`，对全部位姿和平移、参考网格应用同一个 W，再检查剩余旋转误差。该规则应用于三个场景，未读取 GT 估计 W、未拟合轨迹或时间偏移；021 剩余旋转投影量由约 0.0564 降至 0.000126。四项针对性测试验证度量恢复、投影一致性、刚体输入保持，以及拒绝随帧变化的形变。\n\n'
        text+='[输入坐标协议](../reports/evidence_retention_2026-11-11/stage3/coordinate_protocol.json)记录该技术适配，原始数据及先前检查结果全部保留。下一步还需完成新场景强前端适配与评测，以及 LIVO 到对象输出的端到端回放。\n'
    text+='''
## 尚未完成的工作

新场景冻结验证及 LIVO 强基线端到端回放尚未完成。旧在线 Mask R-CNN/HSV 管线和给定位姿 Replica 成绩都不能代替它们。两张 RTX 5090 已核实存在；本轮审计和读出对照全部在 CPU 运行。

已冻结的新场景为 **SceneNN 014、015、021**，按[官方目录](https://hkust-vgd.ust.hk/scenenn/main/)中未使用的最小编号选择。均值与面积＋去冗余两组配置固定后再验证；原有 SceneNN 005/011/016 不作为新测试集。数据准备与模型评测分开记录。

第三阶段完成后，若继续研究表示混淆，需要设置文本命名/提示、原始区域特征与背景上下文的独立控制。当前先完成冻结验证，不根据新测试场景修改本轮参数。SAM3 仍留在后续独立计划。

## 复现与记录

- [阶段 1 汇总](../reports/evidence_retention_2026-11-11/stage1/summary.json)
- [固定案例与 GT 类别](../reports/evidence_retention_2026-11-11/stage1/case_manifest.json)
- [加权读出的可恢复性检查](../reports/evidence_retention_2026-11-11/stage1/convex_readout_limits.json)
- [阶段 2 全部结果](../reports/evidence_retention_2026-11-11/stage2/summary.json)
- [阶段 2 冻结协议](../reports/evidence_retention_2026-11-11/stage2/protocol.json)
- [逐对象匹配变化与描述性区间](../reports/evidence_retention_2026-11-11/stage2/paired_changes.json)
- [阶段 3 冻结场景与配置](../reports/evidence_retention_2026-11-11/stage3/selection.json)
- [完整任务与通过条件](../experiments/evidence_retention_2026-11-11.md)

```bash
OPENBLAS_NUM_THREADS=2 build/benchmarks/ovo_full_20261110/venv311/bin/python scripts/benchmarks/audit_evidence_retention.py
OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR=/tmp/omni_mpl build/benchmarks/ovo_full_20261110/venv311/bin/python scripts/benchmarks/plot_evidence_audit.py
OPENBLAS_NUM_THREADS=2 build/benchmarks/ovo_full_20261110/venv311/bin/python scripts/benchmarks/check_readout_limits.py
# stage2_protocol.json 一经生成不覆盖；已有运行目录也不覆盖。
OPENBLAS_NUM_THREADS=2 build/benchmarks/ovo_full_20261110/venv311/bin/python scripts/benchmarks/run_evidence_readout.py --freeze --run
PYTHONPATH=omni_semantic_map python3 -m pytest -q omni_semantic_map/test/test_evidence_readout.py
```
'''
    from retention_stage3_report import update
    text=update(text)
    dest=ROOT/'docs/benchmarks/evidence_retention_2026-11-11.md';dest.write_text(text)
    print('WROTE',dest)


if __name__=='__main__':main()
