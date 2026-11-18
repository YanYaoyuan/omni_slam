"""Publish the one bounded pilot, including failures and its stop decision."""
from pathlib import Path
import html,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_object_repair_v2 import ROOT,OUT,KEYS,VARIANTS,read,sha,dump,verify

DEST=ROOT/'docs/reports/object_repair_v2_2026-11-12'
NAMES={'baseline':'CG','native_bridge_2':'v1 history','constraint_control':'CG + uniform veto','repair':'v2 repair','no_revoke':'v2 no-revoke'}


def report():
    verify();s=read(OUT/'summary.json');assert s['complete'];a=read(OUT/'audit.json');assert a['complete']
    rows=[read(OUT/'evaluation'/k/'metrics.json') for k in KEYS]
    for r in rows:assert sha(OUT/'evaluation'/r['key']/'metrics.json')==s['scene_results'][r['key']]
    DEST.mkdir(parents=True,exist_ok=True)
    table=['| 数据集 | 方法 | 实例 PQ×100 | TP / FP / FN |','|---|---|---:|---|']
    for dataset,group in s['datasets'].items():
        for method,v in group.items():
            table.append(f"| {dataset} | {NAMES[method]} | {v['pq100']:.4f} | {v['tp']}/{v['fp']}/{v['fn']} |" if v['complete'] else f'| {dataset} | {NAMES[method]} | N/A：运行未完成 | N/A |')
    scene_table=['| 场景 | 方法 | PQ×100 | Δ vs v1 | TP / FP / FN | 新增 / 丢失匹配 | 已分配有效体素比例 |','|---|---|---:|---:|---|---|---:|']
    costs=['| 场景 | 方法 | 帧更新中位数(ms) | P95(ms) | 离线导出(s) | 准备(s) | 进程峰值RSS(GiB) |','|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        baseline=r['methods']['native_bridge_2']['score']['observed_voxel_instance_pq']
        for method,v in r['methods'].items():
            if not v['complete']:
                scene_table.append(f"| {r['key']} | {NAMES[method]} | N/A | N/A | N/A | N/A | N/A |");continue
            x=v['score'];delta=100*(x['observed_voxel_instance_pq']-baseline)
            transition=v.get('transition');changes=(f"{len(transition['gained_matched_gt'])}/{len(transition['lost_matched_gt'])}" if transition else '—')
            scene_table.append(f"| {r['key']} | {NAMES[method]} | {100*x['observed_voxel_instance_pq']:.4f} | {delta:+.4f} | {x['tp']}/{x['fp']}/{x['fn']} | {changes} | {v['assigned_valid_fraction']:.2%} |")
            if method in VARIANTS:
                costs.append(f"| {r['key']} | {NAMES[method]} | {v['frame_median_s']*1000:.2f} | {v['frame_p95_s']*1000:.2f} | {v['finalization_s']:.2f} | {v['preparation_s']:.2f} | {v['cpu_peak_rss_bytes']/1024**3:.2f} |")
    gate_names={'mean_gain_each_dataset':'两个数据集各自均值 ΔPQ≥2.0','nondecreasing_scenes':'至少三场景不退步','tp_not_lower_each_dataset':'各数据集总TP不下降','positive_vs_constraint_each_dataset':'各数据集优于约束贯穿控制','positive_revocation_ablation_each_dataset':'各数据集撤销机制有正增量','new_matches_each_dataset':'各数据集均有新增正确匹配'}
    gates=['| 预定门槛 | 结果 |','|---|---|']
    for k,v in s['accuracy_gate_checks'].items():gates.append(f"| {gate_names[k]} | {'通过' if v else '未通过'} |")
    gates.append(f"| 成本：P95≤v1两倍 | {'通过' if s['cost_gate_passed'] else '未通过；Replica 历史 P95 缺失，SceneNN 可比记录另列'} |")
    if not s['all_maps_completed']:gates.append('| 所有 12 次建图成功 | 未通过，运行失败逐项保留 |')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    methods=list(NAMES)
    for ax,(dataset,group) in zip(axes,s['datasets'].items()):
        vals=[group[m].get('pq100',np.nan) for m in methods]
        bars=ax.barh(list(NAMES.values()),vals,color=['#8c9eaf','#53728f','#c28b43','#008577','#b75c49']);ax.invert_yaxis()
        ax.bar_label(bars,fmt='%.3f',padding=3);maximum=max([v for v in vals if np.isfinite(v)]+[1]);ax.set_xlim(0,maximum*1.2)
        ax.set_title(dataset+' · 2 development scenes',loc='left');ax.set_xlabel('Observed-domain instance PQ × 100')
    fig.suptitle('omni_slam v2 | One bounded pilot · identical cached frontend',fontweight='bold')
    fig.savefig(DEST/'comparison.png',dpi=180);fig.savefig(DEST/'comparison.svg');plt.close(fig)
    status='达到预定进入主实验门槛' if s['full_go_gate_passed'] else '未达到预定进入主实验门槛，本轮分支停止'
    joins=lambda lines:'\n'.join(lines)
    findings=[]
    for dataset,group in s['datasets'].items():
        if group['repair']['complete']:
            base=group['native_bridge_2'];new=group['repair']
            findings.append(f"{dataset}：v1→v2 的 PQ×100 为 {base['pq100']:.4f}→{new['pq100']:.4f}（{new['pq100']-base['pq100']:+.4f}），TP 从 {base['tp']} 变为 {new['tp']}。")
    if s['all_maps_completed']:
        gained=sum(len(r['methods']['repair']['transition']['gained_matched_gt']) for r in rows)
        lost=sum(len(r['methods']['repair']['transition']['lost_matched_gt']) for r in rows)
        extra_tp=sum(g['repair']['tp']-g['no_revoke']['tp'] for g in s['datasets'].values())
        extra_fp=sum(g['repair']['fp']-g['no_revoke']['fp'] for g in s['datasets'].values())
        findings.append(f'相对 v1，四场景合计新增 {gained} 个正确匹配、丢失 {lost} 个。相对 no-revoke，v2 的总 TP 变化 {extra_tp:+d}，FP 变化 {extra_fp:+d}；须同时考察，不能只报告新增匹配。')
        findings.append('三个新增方法都是预先封存的一次运行，没有按这些结果调参或切换主版本。')
    for r in rows:
        if r['historical_causal_p95_s'] is not None and r['methods']['repair']['complete']:
            ratio=r['methods']['repair']['frame_p95_s']/r['historical_causal_p95_s']
            findings.append(f"{r['key']} 的帧更新 P95 / 历史因果 P95 = {ratio:.3f}，两倍门槛{'通过' if ratio<=2 else '未通过'}。")
    finding_text='\n\n'.join(findings)
    md=f'''# omni_slam v2：可撤销对象关联的有限开发验证

2026-11-12。**{status}。** 四个已查看开发场景，三种新增方法，最多 12 次新建图；未增加检测、描述子网络或训练，未修改 v1 历史成绩。完成情况：{'12 次建图均成功' if s['all_maps_completed'] else '存在未完成项，逐项披露且不替换场景'}。

## 结论与失败分析

{finding_text}

这版固定网格和多帧边界票数能够执行拆分，却没有恢复足够多的正确实例。硬同帧约束控制也在 Replica 增加了碎片并损失匹配。支持点覆盖增加不能替代对象边界正确：新后端同时改变了支持归属和前景读出，不能把退化全部归因于某一个模块。v2/no-revoke 的直接消融表明，当前撤销策略的收益不足以抵消错误实例增加。

**这版原型不进入主线，继续保留 v1 为已验证参考。** 可复用其测试、支持点记账和诊断工具，但不把此次失败包装成性能贡献，也不据此排除所有可能的可撤销关联方法。

## 数据集内汇总

{joins(table)}

![比较](comparison.png)

## 预定门槛

{joins(gates)}

+2.0 是预先设定的原型可行性筛选值，不是论文录用标准。两场景/数据集不能支撑充分泛化或显著性结论。不能把新的开发数据涨点与 v1 独立测试混算；036/038/041 没有用于本轮算法评分或调参。

## 实现与归因边界

- CG 约束贯穿控制：在线关联、两帧历史图、原生最终合并均执行分量同帧否决，保留源消费保护。它单独揭示修复约束绕过的问题。
- v2：以固定 5cm 支持单元为节点；相邻单元同掩码为正证据，同时可见且各有至少 70% 唯一掩码支持的不同掩码为负证据。未知或遮挡不记负证据。负帧数至少 2 且不少于正帧数两倍时形成排斥边。
- 在被本帧观测触及的旧分量上做确定性的带排斥约束分区；单次重算最多 100000 单元，超限报失败。允许拆分旧关联，重新支持相同身份时也可合并；不把“同帧”单独作为永久否决。
- no-revoke 与 v2 使用完全相同的支持、边和票数、几何清理、过滤及类别读出，唯一差异是锁定既有分量、不许拆开。保存仍未解决的排斥冲突，不隐藏它们。
- 这是新的支持图映射后端，**不是只改 CG 一行代码的单因素实验**。v2/no-revoke 才是撤销开关的直接消融。没有直接从已融合的 CG 点云中凭空恢复信息，而是复用保存的原始观测支持重建归属。
- 观测掩码可以提供多个对象的不同支持点；守恒单位是 `(观测ID, 实测支持点ID)`，不是整张掩码。原生 2.5cm 下采样与 DBSCAN 清理复用原配置；至少三不同帧，类别用一帧一最大有效面积观测的均值。所有过滤、覆盖和匹配得失完整记录。

## 全部场景结果

{joins(scene_table)}

同一固定参考域，已分配点 1NN≤5cm，严格 IoU>0.5。CG/v1 的旧投影和指标逐项复核；新方法的投影和分数独立复算。数据集中位姿和前端缓存不变，Replica 沿用原 2000 几何帧/200 语义帧，SceneNN 沿用原 400/200 帧。

## 更新成本

{joins(costs)}

两路 CPU 工作进程、各单线程配置；未申请 GPU 推理。报告帧更新、离线导出、一次性单元准备与进程峰值RSS，不能称为机器人全系统实时延迟。旧 Replica 缺逐帧记录，P95 对照标为不可验证，不用总时长/帧数推算，也未为补计时扩展 12 次建图预算。SceneNN 旧时间来自已保存的因果回放记录；不同运行负载仍限制延迟比较。

## 保留的限制与停止决定

同帧不同掩码持续出现仍可能对应同一物体的不同部件；当前正负票比例是固定启发式，没有概率校准。固定网格不能分开落在同一单元内的相邻物体，也不能恢复从未进入观测支持的目标。拆出的局部片段沿用原掩码描述子，并没有获得新的局部视觉特征；几何拆分不保证类别随之纠正。局部更新在最坏情况下仍可能触及很大的旧分量，硬上限和 P95 必须一起看。

这一版验证固定给定位姿、缓存输入和原型分区；尚无实时传感器接入、位姿修订后的重算或完整对象 ID 生命周期保证。分区、拆分与类别解耦都已有相关研究，本轮不能仅凭实现完毕宣称新颖性。

决定：**{status}**。若未通过，不自动增加模型、场景或阈值搜索；所有失败与负结果归档。若通过，也只进入另行预注册的完整方法对比，不把这四个开发场景升格为独立测试。

[协议](protocol.json) · [汇总](summary.json) · [复算审计](audit.json) · [原始 v2 决策](../../paper/v2_decision_2026-11-12.md)
'''
    (DEST/'report.md').write_text(md)
    def table_html(lines):
        cells=[[x.strip() for x in line.strip('|').split('|')] for line in lines]
        return '<table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in cells[0])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in cells[2:])+'</tbody></table>'
    body=f'<h1>omni_slam v2 · 有限开发验证</h1><p class="note">{status}。四场景、三种新增方法、最多十二次建图。</p><p>{html.escape(finding_text)}</p><img src="comparison.png" alt="两数据集方法比较"><h2>汇总</h2>{table_html(table)}<h2>预定门槛</h2>{table_html(gates)}<h2>逐场景结果与匹配得失</h2>{table_html(scene_table)}<h2>成本</h2>{table_html(costs)}<p>这是已查看开发数据的 observed-domain 评测。旧 Replica P95 缺失，不宣称通过完整成本门槛或独立泛化。</p><p><a href="report.md">完整方法及限制</a> · <a href="summary.json">原始汇总</a> · <a href="audit.json">复算审计</a></p>'
    (DEST/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam v2 开发验证</title><style>body{max-width:1200px;margin:40px auto;padding:0 24px;font:16px/1.7 system-ui;color:#20364b}.note{padding:18px;background:#eef4f6;border-left:5px solid #008577}img{width:100%}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border-bottom:1px solid #dbe3e9;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#edf2f6}a{color:#008577}</style>'+body+'</html>')
    for name in ['protocol.json','summary.json','audit.json']:(DEST/name).write_bytes((OUT/name).read_bytes())
    for row in rows:
        p=DEST/'evaluation'/row['key'];p.mkdir(parents=True,exist_ok=True);(p/'metrics.json').write_bytes((OUT/'evaluation'/row['key']/'metrics.json').read_bytes())
    print('REPORT',DEST/'index.html',flush=True)


if __name__=='__main__':report()
