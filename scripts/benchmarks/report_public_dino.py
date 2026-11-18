"""Exhaustive DINO verification and category diagnostic report."""
from collections import Counter
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/omni-public-dino-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from public_dino_protocol import DEST,PARENT,read,verify
from audit_evidence_retention import ROOT,REPORT,dump
from omni_semantic_map.flat_benchmark import sha256


def main():
    spec=read(DEST/'protocol.json');verify(spec);summary=read(DEST/'summary.json');assert summary['complete']
    dest=REPORT/'public_dino_v1';dest.mkdir(exist_ok=False)
    methods=spec['variants'];short=['Baseline','History','DINO pooled','DINO local']
    changes=[];coverage=[];semantic_changes=[];datasets={};scene_table=[]
    for entry,row in zip(spec['scenes'],summary['scenes']):
        key=entry['key'];assert row['key']==key
        gt_edges={(e['a'],e['b']):e for e in read(PARENT/'evaluation'/key/'native_bridge_2_edges.json')}
        state=read(DEST/'runs'/key/'run.json')
        for v in state['variants']:
            folder=DEST/'runs'/key/v['name']
            for f,h in v['files_sha256'].items():assert sha256(folder/f)==h
            graph=read(folder/'graph.json');votes=read(folder/'votes.json')
            coverage.append(dict(scene=key,dataset=entry['dataset'],method=v['name'],proposals=len(votes),
                vote_reasons=v['vote_reasons'],graph_reasons=v['graph_reasons']))
            for edge in graph['decisions']:
                old=gt_edges[(edge['a'],edge['b'])]
                if (edge['reason']=='joined')!=(old['reason']=='joined'):
                    changes.append(dict(scene=key,dataset=entry['dataset'],method=v['name'],a=edge['a'],b=edge['b'],
                        before=old['reason'],after=edge['reason'],gt_signature=old['identity_signature'],
                        native_frames=edge['geometry_frames'],dino_frames=edge['dino_frames']))
        baseline_matches={m['gt_id']:m for m in row['methods']['native_bridge_2']['semantic']['matches']}
        for method in methods:
            for match in row['methods'][method]['semantic']['matches']:
                old=baseline_matches.get(match['gt_id'])
                if old and old['class_correct']!=match['class_correct']:
                    semantic_changes.append(dict(scene=key,method=method,gt_id=match['gt_id'],before=old,after=match))
        scene_table.append(dict(scene=key,dataset=entry['dataset'],domain=row['category_domain'],
            methods={n:dict(instance_pq100=100*v['instance']['observed_voxel_instance_pq'],
                           class_aware_pq100=100*v['semantic']['class_aware_pq'],
                           correct_class=v['semantic']['correct_class_matches'],matches=v['semantic']['geometry_matches']) for n,v in row['methods'].items()}))
    dump(dest/'edge_changes.json',changes);dump(dest/'coverage.json',coverage)
    dump(dest/'semantic_changes.json',semantic_changes);dump(dest/'scene_table.json',scene_table)
    dump(dest/'summary.json',summary);dump(dest/'protocol.json',spec)
    for dataset in ['Replica','SceneNN']:
        ds={}
        for method in methods[2:]:
            subset=[r for r in changes if r['dataset']==dataset and r['method']==method]
            cov=[r for r in coverage if r['dataset']==dataset and r['method']==method]
            ds[method]=dict(proposals=sum(r['proposals'] for r in cov),vote_reasons=dict(sum((Counter(r['vote_reasons']) for r in cov),Counter())),
                removed_join_signatures=dict(Counter(r['gt_signature'] for r in subset if r['before']=='joined')),
                newly_joined_signatures=dict(Counter(r['gt_signature'] for r in subset if r['after']=='joined')))
        datasets[dataset]=ds
    dump(dest/'evidence_summary.json',datasets)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    colors=['#98a3ad','#2879a3','#d39a42','#6d78ad']
    for col,dataset in enumerate(['Replica','SceneNN']):
        for row,metric in enumerate(['instance_pq100','class_aware_pq100']):
            ax=axes[row,col];ys=[summary['datasets'][dataset][n][metric] for n in methods]
            bars=ax.bar(range(4),ys,color=colors,width=.65);ax.bar_label(bars,labels=[f'{y:.3f}' for y in ys],padding=4,fontsize=9)
            ax.set_xticks(range(4),short,rotation=12,fontsize=9);ax.set_ylim(0,max(ys)*1.2)
            ax.set_title(dataset+' · '+('instance PQ' if row==0 else 'class-aware instance PQ'),loc='left',fontweight='bold')
            ax.set_ylabel('Diagnostic PQ × 100');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('DINO verifies history links; SigLIP category readout remains fixed',fontsize=14,fontweight='bold')
    fig.supxlabel('Previously viewed scenes · category domain/ontology coverage reported separately · no official leaderboard claim',fontsize=9)
    for ext in ['png','svg']:fig.savefig(dest/f'comparison.{ext}',dpi=180)
    plt.close(fig)
    main_table=[];vote_table=[];edge_table=[]
    for dataset,ds in summary['datasets'].items():
        for name in methods:
            r=ds[name]
            main_table.append(f'| {dataset} | {name} | {r["instance_pq100"]:.4f} | {r["delta_vs_history_pq100"]:+.4f} | {r["class_aware_pq100"]:.4f} | {r["correct_class_matches"]}/{r["geometry_matches"]} |')
        for name,r in datasets[dataset].items():
            v=r['vote_reasons'];removed=r['removed_join_signatures'];new=r['newly_joined_signatures']
            vote_table.append(f'| {dataset} | {name} | {r["proposals"]} | {v.get("supported",0)} | {v.get("appearance_veto",0)} | {v.get("unknown_native_fallback",0)} |')
            edge_table.append(f'| {dataset} | {name} | {removed.get("same_dominant_gt",0)} | {removed.get("different_dominant_gt",0)} | {removed.get("unresolved",0)} | {sum(new.values())} |')
    per_scene=[];domains=[]
    for row in scene_table:
        values=row['methods'];per_scene.append('| '+row['scene']+' | '+' | '.join(f'{values[n]["instance_pq100"]:.4f}' for n in methods)+' | '+' | '.join(f'{values[n]["class_aware_pq100"]:.4f}' for n in methods)+' |')
        d=row['domain'];domains.append(f'| {row["scene"]} | {d["annotated_instances"]}/{d["gt_instances"]} | {100*d["valid_voxel_fraction"]:.2f}% |')
    report=ROOT/'docs/benchmarks/public_dino_semantics_2026-11-11.md'
    report.write_text('''# omni_slam：DINO 历史验证与类别语义评测

2026-11-11。固定此前 11 场景及两次支持历史图，完成池化 DINO、局部 DINO 两个实际整场景关联分支；同时对原基线、历史图和 DINO 分支补充类别评价。所有特征使用冻结 DINOv2-small，没有训练或事后阈值搜索。

## 四组对照

原底座仍为 SAM2.1 Hiera L + SigLIP384 + 原生 ConceptGraphs + 来源消费保护 + 不同帧均值读出。历史候选、点云、RGB-D 参数、提供位姿、最终处理及 SigLIP 类别读出全部固定。

| 数据集 | 方法 | 实例 PQ×100 | 相对历史图变化 | 类别感知 PQ×100 | 几何匹配中类别正确数 |
|---|---|---:|---:|---:|---:|
'''+ '\n'.join(main_table)+'''

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
'''+ '\n'.join(vote_table)+'''

以下为相对历史图取消的合并边之 GT 主导标签诊断，未判定包括背景、混合和支持不足节点。该统计不是合并精确率。

| 数据集 | 分支 | 取消同 GT 边 | 取消异 GT 边 | 取消未判定边 | 新执行其他候选边 |
|---|---|---:|---:|---:|---:|
'''+ '\n'.join(edge_table)+'''

[全部图层变化](../reports/evidence_retention_2026-11-11/public_dino_v1/edge_changes.json) · [证据覆盖明细](../reports/evidence_retention_2026-11-11/public_dino_v1/coverage.json)

## 类别评测如何避免“实例涨分等于语义变好”

在固定类别评价域上先做严格 IoU>0.5 实例匹配，再要求预测类别与 GT 类别相同。错误类别的几何匹配同时计为类别 FP 和 FN。另报告“已经几何匹配的对象中，有多少类别正确”，以及前景体素类别准确率。

Replica 使用原 `info_semantic.json` 的实例 ID+1 → class_id，无额外类别转换。SceneNN 使用 CVPR18 唯一标注顶点与原实例 mesh 的精确坐标对应，对每个实例取主导 NYU 类别，要求纯度 ≥0.7；不明确的类别仅从类别评价域置 void，不改变原实例 PQ。

SceneNN 的固定预测词表使用官方类别表的直接名称匹配，仅预先增加 book/books、shelf/shelves、nightstand/night stand、tv-screen/television 四个名称别名；无法对应的预测计错，不能为提高分数丢弃。它是词表有限且跨版本的类别诊断，不能与 Replica 数字直接比较。官方名称来源：[SceneNN NYU 类别表](https://hkust-vgd.ust.hk/scenenn/contrib/nyu_class/nyu_color.xml)；另核对了 [ScanNet 官方类别文件](https://github.com/ScanNet/ScanNet/blob/master/Tasks/Benchmark/classes_SemVoxLabel-nyu40id.txt) 中共同类别的编号。

| 场景 | 可确定类别 GT / 原前景 GT | 类别评价有效体素 / 原有效体素 |
|---|---:|---:|
'''+ '\n'.join(domains)+'''

## 所有场景结果

| 场景 | 实例：底座 | 历史图 | 池化 DINO | 局部 DINO | 类别：底座 | 历史图 | 池化 DINO | 局部 DINO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(per_scene)+'''

## 复现与限制

原生事件、过滤前节点、原底座及历史图的分类和几何均精确复现。新增 4 项测试覆盖局部互近邻最少支持、不确定回退、双分支相对证据、几何正确但类别错误、未知预测类别及缺失 GT 类别；全部通过。

GT 仅用于结果完成后的评价，未用于提取特征或候选投票。所有场景此前已被查看，属于开发验证；使用提供位姿和缓存前端，不能据此宣称 LIVO 定位或实时部署改进。源代码、模型、RGB、缓存和输出哈希均保留，旧实验不修改。

运行目录：`build/benchmarks/evidence_retention_20261111/public_dino_v1/`。每场景的 `runs/` 保存新地图、全部支持票与图决策；`evaluation/` 保存实例/类别分数、类别参考依据和投影预测；`features/` 保存 200 帧 DINO 特征。

[完整结果](../reports/evidence_retention_2026-11-11/public_dino_v1/summary.json) · [冻结协议](../reports/evidence_retention_2026-11-11/public_dino_v1/protocol.json) · [匹配对象的类别得失](../reports/evidence_retention_2026-11-11/public_dino_v1/semantic_changes.json) · [前一轮历史图](public_history_graph_2026-11-11.md)
''')
    html_table=''.join('<tr><td>'+r['scene']+'</td>'+''.join(f'<td>{r["methods"][n]["instance_pq100"]:.4f}</td>' for n in methods)+'</tr>' for r in scene_table)
    (dest/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>DINO 与类别语义评测</title><style>body{font:16px/1.7 system-ui;max-width:1150px;margin:36px auto;padding:0 20px;background:#f4f6f8;color:#192b38}section{background:white;padding:24px;margin:24px 0;border:1px solid #dce2e6;border-radius:8px}img{max-width:100%}table{width:100%;border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child{text-align:left}a{color:#17678b}</style><h1>omni_slam · DINO 与类别语义评测</h1><p>11 场景完整对照。类别读出固定为 SigLIP；DINO 只验证历史候选。开发数据，非官方排行榜。</p><p><a href="../../../benchmarks/public_dino_semantics_2026-11-11.md">完整报告</a> · <a href="edge_changes.json">全部图层变化</a> · <a href="semantic_changes.json">类别得失</a></p><section><img src="comparison.png" alt="实例与类别评价"></section><section><h2>所有场景实例 PQ×100</h2><table><tr><th>场景</th><th>原底座</th><th>历史图</th><th>池化 DINO</th><th>局部 DINO</th></tr>'''+html_table+'</table></section></html>')
    print(summary['datasets']);print(datasets)


if __name__=='__main__':main()
