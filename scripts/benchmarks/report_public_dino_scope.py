"""Report all public DINO variants, including the unsuccessful ones."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/omni-public-dino-mpl')
from collections import Counter
import html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from public_dino_protocol import read,verify,DEST as V1,PARENT
from public_dino_scope import DEST as V2,NAMES
from audit_evidence_retention import ROOT,REPORT,dump
from omni_semantic_map.flat_benchmark import sha256


def main():
    dest=REPORT/'public_dino_scope_v2';dest.mkdir(exist_ok=False)
    all_methods=['baseline','native_bridge_2','dino_pooled_2','dino_local_2']+NAMES
    labels=['Strong baseline','History (2 frames)','Pooled / full scope','Local / strict fallback','Pooled / matched scope','Local / matched scope']
    summaries=[read(d/'summary.json') for d in [V1,V2]]
    for d in [V1,V2]:
        verify(read(d/'protocol.json'));repair=read(d/'evaluation_repair.json')
        assert sha256(repair['source'])==repair['sha256']
        dump(dest/(d.name+'_protocol.json'),read(d/'protocol.json'))
        dump(dest/(d.name+'_evaluation_repair.json'),repair)
    data={ds:{**summaries[0]['datasets'][ds],**summaries[1]['datasets'][ds]} for ds in ['Replica','SceneNN']}
    coverage=[];scope_rows=[];edge_changes=[];scenes=[]
    for entry in read(V2/'protocol.json')['scenes']:
        key=entry['key'];ds=entry['dataset'];row={}
        for d in [V1,V2]:
            metrics=read(d/'evaluation'/key/'metrics.json');state=read(d/'runs'/key/'run.json')
            assert metrics['run_sha256']==sha256(d/'runs'/key/'run.json')
            for f,h in metrics['files_sha256'].items():assert sha256(d/'evaluation'/key/f)==h
            row.update(metrics['methods'])
            for v in state['variants']:
                for f,h in v['files_sha256'].items():assert sha256(d/'runs'/key/v['name']/f)==h
                coverage.append(dict(scene=key,dataset=ds,method=v['name'],**v['vote_reasons']))
                old={(e['a'],e['b']):e for e in read(PARENT/'evaluation'/key/'native_bridge_2_edges.json')}
                for e in read(d/'runs'/key/v['name']/'graph.json')['decisions']:
                    b=old[e['a'],e['b']]
                    if (e['reason']=='joined')!=(b['reason']=='joined'):
                        edge_changes.append(dict(scene=key,method=v['name'],before=b['reason'],after=e['reason'],gt_signature=b['identity_signature'],a=e['a'],b=e['b']))
        run=V2/'runs'/key;state=read(run/'run.json');assert state['shared_scope_sha256']==sha256(run/'scopes.json')
        scopes=read(run/'scopes.json');votes=[read(run/n/'votes.json') for n in NAMES]
        assert len(votes[0])==len(votes[1])==len(scopes)
        for i,(a,b) in enumerate(zip(*votes)):
            assert a['scope_index']==b['scope_index']==i
            assert (a['evidence']['reason']=='unknown_native_fallback')==(b['evidence']['reason']=='unknown_native_fallback')
        original=np.array([s['original_counts'] for s in scopes]);eligible=np.array([[len(g) for g in s['eligible']] for s in scopes])
        scope_rows.append(dict(scene=key,dataset=ds,proposals=len(scopes),original_counts=original.sum(axis=0).tolist(),eligible_counts=eligible.sum(axis=0).tolist(),same_scope_verified=True))
        scenes.append(dict(scene=key,dataset=ds,methods=row,category_domain=metrics['category_domain']))
    combined=dict(datasets=data,scenes=scenes,coverage=coverage,scope=scope_rows,edge_changes=edge_changes)
    dump(dest/'results.json',combined)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    colors=['#89959f','#287fa4','#ce9d47','#8993b9','#b36c44','#6957a6']
    for j,ds in enumerate(data):
        for i,metric in enumerate(['instance_pq100','class_aware_pq100']):
            ax=axes[i,j];ys=[data[ds][m][metric] for m in all_methods]
            bars=ax.barh(np.arange(6),ys,color=colors);ax.invert_yaxis();ax.set_yticks(np.arange(6),labels,fontsize=9)
            ax.bar_label(bars,labels=[f'{y:.3f}' for y in ys],padding=4,fontsize=9);ax.set_xlim(0,max(ys)*1.18)
            ax.set_title(ds+' | '+('instance PQ' if i==0 else 'class-aware PQ'),loc='left',fontweight='bold')
            ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);ax.set_xlabel('Diagnostic PQ × 100')
    fig.suptitle('Frozen frontend + native history: does DINO add value?',fontsize=15,fontweight='bold')
    fig.supxlabel('11 previously viewed scenes · provided poses · pooled-instance category diagnostic, not official benchmark PQ',fontsize=9)
    for ext in ['png','svg']:fig.savefig(dest/f'comparison.{ext}',dpi=180)
    plt.close(fig)
    delta=np.array([[100*(s['methods'][m]['instance']['observed_voxel_instance_pq']-s['methods']['native_bridge_2']['instance']['observed_voxel_instance_pq']) for m in all_methods[2:]] for s in scenes])
    fig,ax=plt.subplots(figsize=(10,7),layout='constrained');bound=max(float(abs(delta).max()),.01)
    im=ax.imshow(delta,cmap='RdYlGn',vmin=-bound,vmax=bound,aspect='auto')
    ax.set_xticks(range(4),labels[2:],rotation=12,ha='right');ax.set_yticks(range(len(scenes)),[s['scene'] for s in scenes])
    for i in range(len(scenes)):
        for j in range(4):ax.text(j,i,f'{delta[i,j]:+.3f}',ha='center',va='center',fontsize=9)
    fig.colorbar(im,ax=ax,label='Instance PQ × 100 delta vs history');ax.set_title('All scenes: changes relative to the fixed history graph',loc='left')
    for ext in ['png','svg']:fig.savefig(dest/f'deltas.{ext}',dpi=180)
    plt.close(fig)
    table=[];covtable=[];pairtable=[]
    for ds in data:
        for m in all_methods:
            r=data[ds][m];base=data[ds]['baseline']['instance_pq100']
            table.append(f'| {ds} | {m} | {r["instance_pq100"]:.4f} | {r["instance_pq100"]-base:+.4f} | {r["delta_vs_history_pq100"]:+.4f} | {r["class_aware_pq100"]:.4f} | {r["correct_class_matches"]}/{r["geometry_matches"]} |')
        for m in all_methods[2:]:
            c=sum((Counter({k:r.get(k,0) for k in ['supported','appearance_veto','unknown_native_fallback']}) for r in coverage if r['dataset']==ds and r['method']==m),Counter());total=sum(c.values())
            covtable.append(f'| {ds} | {m} | {total} | {c["supported"]} | {c["appearance_veto"]} | {c["unknown_native_fallback"]} | {100*(1-c["unknown_native_fallback"]/total):.2f}% |')
        subset=[s for s in scenes if s['dataset']==ds]
        d=np.array([100*(s['methods'][NAMES[1]]['instance']['observed_voxel_instance_pq']-s['methods'][NAMES[0]]['instance']['observed_voxel_instance_pq']) for s in subset])
        pairtable.append(f'| {ds} | {d.mean():+.4f} | {(d>1e-8).sum()} / {(abs(d)<=1e-8).sum()} / {(d< -1e-8).sum()} |')
    domain_table=[]
    for s in scenes:
        d=s['category_domain'];unknown=s['methods']['native_bridge_2']['semantic']['unknown_predicted_classes']
        domain_table.append(f'| {s["scene"]} | {d["annotated_instances"]}/{d["gt_instances"]} | {100*d["valid_voxel_fraction"]:.2f}% | {unknown} |')
    report=ROOT/'docs/benchmarks/public_dino_scope_2026-11-11.md'
    text='''# omni_slam：DINO 公共场景对照与语义诊断

2026-11-11。完成 Replica 8 + SceneNN 3 的全部 11 场景。两轮共生成 44 张新增 DINO 分支地图；复用两组固定参考，共比较 6 种方法。2200 个语义帧提取冻结 DINOv2-small 特征，无训练、无按场景选阈值。

## 保留的强基线和方法边界

强基线是 SAM2.1 Hiera L 分割 + SigLIP384 特征 + 原生 ConceptGraphs + 来源消费保护 + 不同帧均值类别读出。历史图增加原生歧义候选的跨帧支持与整组同帧冲突检查，要求两次独立查询帧支持。DINO 在这次实验中仅验证历史关联票，不生成候选、不替换类别读出。

公开 RGB-D 实验保持原生 DBSCAN 开启、重叠距离 0.025m，使用数据集提供位姿。不能混用此前 LIVO 工程候选的 DBSCAN 关闭、0.2m 配置，也不能将这些结果算成 LIVO 定位提升。

## 完整结果（含不涨点分支）

| 数据集 | 方法 | 实例 PQ×100 | 比原强基线 | 比历史图 | 类别感知 PQ×100 | 匹配中类别正确 |
|---|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(table)+'''

![六组完整对照](../reports/evidence_retention_2026-11-11/public_dino_scope_v2/comparison.png)

表中类别指标使用实例汇总分母，非官方逐类别宏平均 panoptic PQ；类别评价域与原实例评价域可能不同，见后文覆盖表。

## 为什么增加第二轮对照

第一轮局部验证规定：任一同帧竞争者无法形成至少 3 对外观互近邻，就整票退回原生判断。小对象和局部 token 不足使该分支几乎不参与，地图评分持平不能解释成 DINO 验证有效。

第二轮在发现上述覆盖问题后单独冻结协议，再运行全部场景。对每条查询，只保留局部匹配可计算的历史成员和同帧竞争者，池化与局部两组使用**完全相同的成员和竞争者集合**。任一历史分支或竞争集合为空时，两组共同回退；其余缺失竞争者被排除并记录。这只是部分竞争集合上的验证，不声称覆盖完整负样本。未进行阈值搜索；属于开发扩展，不是独立盲测或最初预注册实验。

| 数据集 | 方法 | 候选票 | 外观支持 | 外观否决 | 回退 | 实际参与率 |
|---|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(covtable)+'''

固定相同局部有效范围后，局部相对池化的直接配对比较：

| 数据集 | 局部减池化：平均实例 PQ×100 | 场景胜 / 平 / 负 |
|---|---:|---:|
'''+ '\n'.join(pairtable)+'''

![全部场景差值](../reports/evidence_retention_2026-11-11/public_dino_scope_v2/deltas.png)

## 类别瓶颈：床恢复出来了，但被叫作枕头

SceneNN021 的 GT 实例 286，在 CVPR18 标注中有 58,960 个唯一标注顶点，主导类别为 bed（NYU 4），纯度 100%。历史图将碎片合并后得到有效几何匹配，但固定 SigLIP 均值读出预测 pillow（NYU 18），因此在类别指标中仍是错误。原实例评价域上的 IoU 与类别评价域上的 IoU 不应混写；完整数值保存在各自指标和匹配记录中。

这一例说明，跨帧证据累积能改善对象几何，并不保证类别证据正确。下一阶段应分别研究“同一对象的证据如何保留”和“哪些观测适合决定类别”，避免把二者用一个平均相似度混合。

## 类别评价覆盖

Replica 直接对应原实例类别。SceneNN 通过 CVPR18 唯一坐标精确匹配原 mesh 实例，主导类别纯度至少 0.7 才参与类别评价；模糊 GT 仅在类别评价中置 void。固定预测词表仅用直接名称和 4 组预先声明的别名对应 NYU 类别。未知预测类别计错，不删除。SceneNN 的类别得分同时受词表覆盖、标注版本和分类质量影响，不能全部归因于分类器。

| 场景 | 有类别 GT / 原 GT | 类别有效体素比例 | 历史图未知类别预测实例数 |
|---|---:|---:|---:|
'''+ '\n'.join(domain_table)+'''

## 可复现性与限制

- 所有 11 场景的快照恢复，都先通过原强基线和历史图的分类 JSON、几何数组逐项精确一致检查，再运行新分支。恢复器只重建有序观测的原生递归特征，RGB 置零仅供不使用颜色的几何评价导出，不适用于彩色渲染。
- 两个对照分支共享同一份候选范围文件，并逐票校验共享范围与回退条件；保留所有否决、回退和最终图决策。GT 仅在地图完成后用于评价。
- 原协议及源文件未改写。类别读取修复单独记录为 `evaluation_repair.json`：加入已有 h5py 的加载路径，并补上 NumPy unique 的 return_inverse 参数。修复前 SceneNN 类别评分未产出，修复不改变标签规则、阈值或指标。
- 4 项功能测试通过。全部场景此前均已被查看，本次为开发验证；没有独立测试集显著性结论、实时速度结论或官方排行榜排名。

[机器可读完整结果与候选覆盖](../reports/evidence_retention_2026-11-11/public_dino_scope_v2/results.json) · [第一轮完整报告](public_dino_semantics_2026-11-11.md) · [此前历史图结果](public_history_graph_2026-11-11.md)

运行目录：`build/benchmarks/evidence_retention_20261111/public_dino_v1/` 与 `public_dino_scope_v2/`。新增源文件、冻结协议、评测修复、特征和输出均有哈希记录。
'''
    report.write_text(text)
    htmlrows=''.join('<tr>'+''.join('<td>'+html.escape(x.strip())+'</td>' for x in r.strip('|').split('|'))+'</tr>' for r in table)
    (dest/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>omni_slam · DINO 公共场景对照</title><style>body{font:16px/1.7 system-ui;max-width:1200px;padding:28px;margin:auto;background:#f3f5f8;color:#182c3c}section{background:white;padding:24px;margin:22px 0;border-radius:10px}img{width:100%}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border-bottom:1px solid #dde3e8;text-align:right}a{color:#176b9b}</style><h1>omni_slam · DINO 公共场景对照</h1><p>11 场景 · 44 张新增地图 · 6 组比较。固定强基线，实例与类别分开评测。</p><p><a href="../../../benchmarks/public_dino_scope_2026-11-11.md">完整报告</a> · <a href="results.json">原始汇总与覆盖</a></p><section><img src="comparison.png" alt="实例与类别指标"></section><section><h2>完整结果</h2><table><tr><th>数据集</th><th>方法</th><th>实例PQ</th><th>比底座</th><th>比历史图</th><th>类别PQ</th><th>类别正确</th></tr>'''+htmlrows+'''</table></section><section><img src="deltas.png" alt="全部场景相对历史图的差值"></section><section><h2>几何成功不等于类别成功</h2><p>SceneNN021 的床合并后跨过几何匹配门限，但类别仍读成枕头。下一步需要单独改善类别证据，而不能把实例涨点直接当成语义识别改善。</p><p>结果来自此前已查看的开发场景及提供位姿。类别指标是诊断指标，不代表官方排行榜。</p></section></html>''')
    print(data)


if __name__=='__main__':main()
