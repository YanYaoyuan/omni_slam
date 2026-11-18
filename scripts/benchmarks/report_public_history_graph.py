"""Report every frozen public scene and both graph controls, including losses."""
from collections import Counter
import html
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/omni-public-history-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from audit_evidence_retention import ROOT, REPORT, dump
from run_public_history_graph import DEST, VARIANTS, read, verify_sources
from omni_semantic_map.flat_benchmark import sha256


def main():
    spec=read(DEST/'protocol.json');verify_sources(spec)
    summary=read(DEST/'summary.json');assert summary['complete']
    dest=REPORT/'public_history_v1';dest.mkdir(exist_ok=False)
    dump(dest/'summary.json',summary);dump(dest/'protocol.json',spec)
    rows=summary['scenes'];changed_gt=[];edge_cases=[];table=[]
    for scene in rows:
        key=scene['key'];run=DEST/'runs'/key;evaluation=DEST/'evaluation'/key
        state=read(run/'run.json');assert state['baseline_exact_geometry_memberships_classes']
        for name,h in scene['files_sha256'].items():
            assert sha256(evaluation/name)==h
        for variant in state['variants']:
            name=variant['variant']
            for filename,h in variant['hashes'].items():
                assert sha256(run/name/filename)==h
            if name=='baseline':continue
            before=read(evaluation/'baseline_audit.json');after=read(evaluation/(name+'_audit.json'))
            scores=scene['methods'][name]
            transition=scores['matched_gt_transition']
            for kind in ['gained_matched_gt','lost_matched_gt']:
                for gt in transition[kind]:
                    def relevant(audit):
                        return dict(gt=next(g for g in audit['ground_truth'] if g['gt_id']==gt),
                            predictions=[p for p in audit['predicted'] if p['dominant_gt']==gt or gt in p['mixed_gt_ids']])
                    changed_gt.append(dict(scene=key,variant=name,change=kind,gt=gt,before=relevant(before),after=relevant(after)))
            for edge in read(evaluation/(name+'_edges.json')):
                if edge['reason']=='joined' and edge['identity_signature']=='different_dominant_gt':
                    edge_cases.append(dict(scene=key,variant=name,**edge))
        a,b,c=[scene['methods'][name] for name,_ in VARIANTS]
        table.append(dict(scene=key,dataset=scene['dataset'],baseline=100*a['observed_voxel_instance_pq'],
            one=100*b['observed_voxel_instance_pq'],two=100*c['observed_voxel_instance_pq'],
            delta=100*(c['observed_voxel_instance_pq']-a['observed_voxel_instance_pq']),
            tp_delta=c['tp']-a['tp'],fp_delta=c['fp']-a['fp'],
            mixed_delta=c['predicted_error_counts'].get('mixed_foreground',0)-a['predicted_error_counts'].get('mixed_foreground',0),
            joined=next(v for v in state['variants'] if v['variant']=='native_bridge_2')['graph_reasons'].get('joined',0)))
    dump(dest/'matched_transition_details.json',changed_gt)
    dump(dest/'different_gt_edges.json',edge_cases)
    dump(dest/'scene_table.json',table)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.8),layout='constrained')
    colors=['#8d99a5','#9bb9ce','#246b98']
    for ax,dataset in zip(axes,['Replica','SceneNN']):
        ys=[summary['datasets'][dataset][name]['macro_pq100'] for name,_ in VARIANTS]
        bars=ax.bar(range(3),ys,color=colors,width=.65)
        ax.bar_label(bars,labels=[f'{x:.3f}' for x in ys],padding=5)
        ax.set_xticks(range(3),['Frozen baseline','Bridge: 1 frame','Bridge: 2 frames'],fontsize=9)
        ax.set_ylim(0,max(ys)*1.16);ax.set_ylabel('Macro instance PQ × 100')
        ax.set_title(dataset+(' · 8 scenes' if dataset=='Replica' else ' · 3 scenes'),loc='left',fontweight='bold')
        ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
    fig.suptitle('Public-data transfer of the frozen history graph',fontweight='bold',fontsize=14)
    fig.supxlabel('Provided poses · shared cached frontend · observed-voxel diagnostic · previously viewed scenes',fontsize=9)
    for ext in ['png','svg']:fig.savefig(dest/f'macro_pq.{ext}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,6.1),layout='constrained')
    delta=[r['delta'] for r in table];y=np.arange(len(table))
    ax.barh(y,delta,color=['#27769d' if x>=0 else '#bd6347' for x in delta])
    ax.set_yticks(y,[r['scene'] for r in table]);ax.invert_yaxis();ax.axvline(0,color='#697581',lw=1)
    ax.axhline(7.5,color='#9ba4ab',ls='--',lw=.8)
    extent=max(max(abs(x) for x in delta),.1)
    for i,x in enumerate(delta):
        ax.text(x+(.025*extent if x>=0 else -.025*extent),i,f'{x:+.3f}',va='center',ha='left' if x>=0 else 'right',fontsize=9)
    ax.set_xlim(min(0,min(delta))-.22*extent,max(0,max(delta))+.22*extent)
    ax.set_xlabel('PQ × 100 change relative to frozen baseline')
    ax.set_title('Two-frame history rule: all 11 scenes',loc='left',fontweight='bold')
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    for ext in ['png','svg']:fig.savefig(dest/f'per_scene_delta.{ext}',dpi=180)
    plt.close(fig)
    macro_lines=[]
    for dataset,methods in summary['datasets'].items():
        for name,_ in VARIANTS:
            m=methods[name];t=m['totals']
            macro_lines.append(f'| {dataset} | {name} | {m["macro_pq100"]:.4f} | {m["delta_pq100"]:+.4f} | {m["wins"]}/{m["ties"]}/{m["losses"]} | {t["tp"]}/{t["fp"]}/{t["fn"]} |')
    scene_lines=[f'| {r["scene"]} | {r["baseline"]:.4f} | {r["one"]:.4f} | {r["two"]:.4f} | {r["delta"]:+.4f} | {r["tp_delta"]:+d} | {r["fp_delta"]:+d} | {r["mixed_delta"]:+d} |' for r in table]
    edge_lines=[];error_lines=[]
    for dataset,methods in summary['datasets'].items():
        for name,_ in VARIANTS:
            m=methods[name];p=m['predicted_error_counts'];g=m['gt_error_counts'];e=m['joined_edge_signatures']
            error_lines.append(f'| {dataset} | {name} | {p.get("fragmentation_consistent",0)} | {p.get("mixed_foreground",0)} | {g.get("fragmentation_consistent",0)} | {m["gained_gt"]} / {m["lost_gt"]} |')
            if name!='baseline':
                edge_lines.append(f'| {dataset} | {name} | {e.get("same_dominant_gt",0)} | {e.get("different_dominant_gt",0)} | {e.get("unresolved",0)} |')
    report=ROOT/'docs/benchmarks/public_history_graph_2026-11-11.md'
    report.write_text('''# omni_slam：对象历史图在公共实例数据上的迁移验证

2026-11-11。完整重放 Replica 8 场景及 SceneNN 014、015、021，沿用固定输入、原始 RGB-D 参数和实例评价域。所有 11 个基线的几何、成员、类别及投影标签精确复现；每组新结果均在预测完成后才读取真值。

本轮主对照是此前声明的 **两次不同查询帧支持的历史合并**；一次支持作为消融。没有训练、DINO 新模型、按结果挑场景或事后扫阈值。

## 总体结果

数值为 class-agnostic observed-voxel instance PQ × 100；每个数据集内部按场景宏平均。两个数据集分别报告，不混成一个均值。“胜/平/负”均相对对应场景原底座。

| 数据集 | 方法 | PQ×100 | 变化 | 胜/平/负 | TP/FP/FN 合计 |
|---|---|---:|---:|---:|---:|
'''+ '\n'.join(macro_lines)+'''

![两组数据集宏平均](../reports/evidence_retention_2026-11-11/public_history_v1/macro_pq.png)

## 所有场景，包含回退

最后四列为两次支持相对底座的变化；混合对象为既有错误诊断器的 `mixed_foreground` 数量，不等价于精确的错误合并事件数。

| 场景 | 底座 | 一次支持 | 两次支持 | 两次 ΔPQ×100 | ΔTP | ΔFP | Δ混合对象 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(scene_lines)+'''

![逐场景变化](../reports/evidence_retention_2026-11-11/public_history_v1/per_scene_delta.png)

## 碎片、混合对象和匹配得失

使用此前冻结的 `instance_error_audit.py` 阈值，未按新实验调整。分类是最终几何的错误特征，不是检测或关联原因的直接真值。

| 数据集 | 方法 | 预测碎片特征 | 预测混合前景 | GT 碎片特征 | 新获 / 丢失匹配 GT |
|---|---|---:|---:|---:|---:|
'''+ '\n'.join(error_lines)+'''

[全部新增与丢失匹配 GT 的前后明细](../reports/evidence_retention_2026-11-11/public_history_v1/matched_transition_details.json)。匹配得失分开记录，避免净 TP 数掩盖同时发生的改善与退化。

## 合并边的 GT 一致性审查

对每个过滤前节点，用其原始点云寻找固定真值参考域中 5cm 内最近点；包含背景参与纯度分母，至少 5 个有效支持点且主导前景 GT 纯度 ≥0.7 时，才给出可判定的主导标签。距离外、void、背景主导或混合节点保留未判定。该步骤只在评估中执行。

| 数据集 | 方法 | 两端同一主导 GT | 两端不同主导 GT | 未判定 |
|---|---|---:|---:|---:|
'''+ '\n'.join(edge_lines)+'''

该表不是合并精确率：节点 GT 对应受已重建几何和参考域影响，且间接连通的分量可能比单条边更复杂。不同主导 GT 的已合并边全部保留在[审查清单](../reports/evidence_retention_2026-11-11/public_history_v1/different_gt_edges.json)，不从均值中剔除。

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
```
''')
    html_rows=''.join(f'<tr><td>{r["scene"]}</td><td>{r["baseline"]:.4f}</td><td>{r["one"]:.4f}</td><td>{r["two"]:.4f}</td><td class="{"up" if r["delta"]>=0 else "down"}">{r["delta"]:+.4f}</td></tr>' for r in table)
    (dest/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>公共数据历史图实验</title>
<style>body{font:16px/1.6 system-ui;max-width:1120px;margin:36px auto;padding:0 20px;color:#1b2934;background:#f5f7f9}section{background:white;padding:24px;margin:24px 0;border:1px solid #dae1e7;border-radius:8px}img{max-width:100%}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:9px 14px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}.up{color:#126c83}.down{color:#b15337}a{color:#155b9a}</style>
<h1>omni_slam · 公共实例数据历史图实验</h1><p>11 个场景，3 组方案。原基线逐场景精确复现，所有回退保留。固定输入、提供位姿、离线历史图；开发验证，非官方排行榜。</p>
<p><a href="../../../benchmarks/public_history_graph_2026-11-11.md">完整报告</a> · <a href="summary.json">结果 JSON</a> · <a href="different_gt_edges.json">不同 GT 合并边</a> · <a href="matched_transition_details.json">匹配得失</a></p>
<section><img src="macro_pq.png" alt="数据集宏平均 PQ"></section><section><h2>全部场景 PQ×100</h2><table><tr><th>场景</th><th>底座</th><th>一次支持</th><th>两次支持</th><th>两次变化</th></tr>'''+html_rows+'''</table></section><section><img src="per_scene_delta.png" alt="每个场景两次支持版本的变化"></section></html>''')
    print(report)


if __name__=='__main__':main()
