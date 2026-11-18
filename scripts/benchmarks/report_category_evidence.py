"""Render exhaustive category results, evidence images, and deployment candidate."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/omni-category-mpl')
import html
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_evidence_retention import ROOT,OUT,REPORT,dump
from public_dino_protocol import read,verify
from omni_semantic_map.flat_benchmark import sha256

DEST=OUT/'public_category_evidence_v2'
METHOD='consensus_sqrt_area'


def main():
    spec=read(DEST/'protocol.json');verify(spec);summary=read(DEST/'summary.json');assert summary['complete']
    dest=REPORT/'category_evidence_v2';dest.mkdir(exist_ok=False)
    manifest=read(OUT/'consensus_category_candidate_v1/manifest.json');assert manifest['complete']
    for p,h in manifest['outputs'].items():assert sha256(OUT/'consensus_category_candidate_v1'/p)==h
    # Repeated v1 controls must reproduce exactly; v2 is an ablation, not a silently revised implementation.
    v1=read(OUT/'public_category_evidence_v1/summary.json')
    for ds in v1['datasets']:
        for m,v in v1['datasets'][ds]['methods'].items():assert v==summary['datasets'][ds]['methods'][m]
    for e in spec['scenes']:
        run=read(DEST/'runs'/e['key']/'run.json')
        for f,h in run['files_sha256'].items():assert sha256(DEST/'runs'/e['key']/f)==h
    dump(dest/'summary.json',summary);dump(dest/'protocol.json',spec);dump(dest/'candidate_manifest.json',manifest)
    methods=spec['methods'];labels=['Mean','Area','Exclusive quality','Quality + SigLIP','Quality + DINO','Score median','Consensus / uniform','Consensus / sqrt area','Consensus / area']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(14,7),layout='constrained')
    for j,ds in enumerate(['Replica','SceneNN']):
        values=summary['datasets'][ds]['methods'];ys=[values[m]['fixed_class_pq100'] for m in methods]
        bars=axes[j].barh(range(len(methods)),ys,color=['#248e78' if m==METHOD else '#557b98' if m=='mean' else '#aab6c2' for m in methods])
        axes[j].invert_yaxis();axes[j].set_yticks(range(len(methods)),labels,fontsize=9)
        axes[j].bar_label(bars,labels=[f'{y:.3f}  ({values[m]["fixed_correct"]}/{values[m]["fixed_matches"]})' for m,y in zip(methods,ys)],padding=4,fontsize=9)
        axes[j].set_xlim(0,max(ys)*1.42);axes[j].set_xlabel('Class-aware diagnostic PQ × 100')
        axes[j].set_title(ds+' | fixed objects and matches',loc='left',fontweight='bold');axes[j].grid(axis='x',alpha=.15);axes[j].set_axisbelow(True)
    fig.suptitle('Category evidence: real corrections without deleting difficult objects',fontsize=15,fontweight='bold')
    fig.supxlabel('Parenthesized counts = correct categories / fixed geometric matches. All scenes previously viewed; no official PQ claim.',fontsize=9)
    for ext in ['png','svg']:fig.savefig(dest/f'comparison.{ext}',dpi=150,bbox_inches='tight')
    plt.close(fig)
    # Candidate improves only categories: show this explicitly next to refiltering losses.
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for j,ds in enumerate(['Replica','SceneNN']):
        values=summary['datasets'][ds]['methods'];base=values['mean']
        m=['area','quality_siglip','quality_dino','score_median',METHOD]
        y=[values[x]['fixed_class_pq100']-base['fixed_class_pq100'] for x in m]
        z=[values[x]['refiltered_instance_pq100']-base['refiltered_instance_pq100'] for x in m]
        x=np.arange(len(m));axes[j].bar(x-.18,y,.36,label='Category gain, fixed foreground',color='#248e78')
        axes[j].bar(x+.18,z,.36,label='Instance change if re-filtered',color='#cf7960')
        axes[j].axhline(0,color='#333',lw=.8);axes[j].set_xticks(x,['Area','Quality\nSigLIP','Quality\nDINO','Score\nmedian','Consensus\nsqrt area'],fontsize=9)
        axes[j].set_title(ds,loc='left',fontweight='bold');axes[j].set_ylabel('PQ × 100 change vs history mean');axes[j].legend(fontsize=8)
    fig.suptitle('Keep object existence separate from category updates',fontweight='bold')
    for ext in ['png','svg']:fig.savefig(dest/f'retention.{ext}',dpi=150,bbox_inches='tight')
    plt.close(fig)
    entries={e['key']:e for e in spec['scenes']};scenes={s['key']:s for s in summary['scenes']}
    changes=[];table=[];scene_table=[]
    for key,s in scenes.items():
        objects={o['object_id']:o for o in read(DEST/'runs'/key/'predictions.json')['objects']}
        for c in s['changes']:
            if c['method']==METHOD:
                row=objects[c['object_id']];changes.append(dict(scene=key,gt_id=c['gt_id'],object_id=c['object_id'],
                    before=row['methods']['mean']['name'],after=row['methods'][METHOD]['name'],correct=c['after']['class_correct']))
        b=s['methods']['mean']['fixed_foreground'];c=s['methods'][METHOD]['fixed_foreground']
        scene_table.append(f'| {key} | {100*b["class_aware_pq"]:.4f} | {100*c["class_aware_pq"]:.4f} | {100*(c["class_aware_pq"]-b["class_aware_pq"]):+.4f} | {b["correct_class_matches"]} → {c["correct_class_matches"]} / {b["geometry_matches"]} |')
    dump(dest/'category_changes.json',changes)
    for ds in ['Replica','SceneNN']:
        for m in methods:
            r=summary['datasets'][ds]['methods'][m]
            table.append(f'| {ds} | {m} | {r["fixed_class_pq100"]:.4f} | {r["delta_fixed_class_pq100"]:+.4f} | {r["fixed_correct"]}/{r["fixed_matches"]} | {r["class_gains"]}/{r["class_losses"]} | {r["refiltered_instance_pq100"]:.4f} |')
    # Case selection is explicitly post-hoc: bed, chair and nightstand illustrate mechanisms.
    cases=[('scenenn_021',6),('replica_office_1',5),('scenenn_014',74)]
    fig,axes=plt.subplots(3,3,figsize=(15,11),layout='constrained');case_records=[]
    for row_index,(key,oid) in enumerate(cases):
        entry=entries[key];cache=Path(entry['cache']);cm=read(cache/'manifest.json')
        obj=next(o for o in read(DEST/'runs'/key/'predictions.json')['objects'] if o['object_id']==oid)
        data=np.load(DEST/'runs'/key/'view_scores.npz');ids=np.array(obj['selected_observation_ids']);weights=np.array(obj['methods'][METHOD]['weights'])
        vocab=read(cache/'vocabulary.json');old_name=obj['methods']['mean']['name'];new_name=obj['methods'][METHOD]['name']
        first=int(np.argmax(weights));zero=np.flatnonzero(weights==0)
        second=int(zero[np.argmax(data['areas'][ids[zero]])]) if len(zero) else int(np.argmin(weights))
        evidence=[]
        for col,local in enumerate([first,second]):
            obs=int(ids[local]);record=next(r for r in cm['records'] if r['observation_offset']<=obs<r['observation_offset']+r['masks'])
            rgb=next(r for r in entry['images'] if r['frame_id']==record['frame_id']);assert sha256(rgb['rgb'])==rgb['sha256']
            im=np.array(Image.open(rgb['rgb']).convert('RGB'));f=np.load(cache/'frames'/record['file'])
            mask=np.unpackbits(f['masks_packed'][obs-record['observation_offset']],axis=1)[:,:im.shape[1]].astype(bool)
            yy,xx=np.where(mask);x0,x1=xx.min(),xx.max()+1;y0,y1=yy.min(),yy.max()+1
            padx=max(20,int((x1-x0)*.2));pady=max(20,int((y1-y0)*.2));x0=max(0,x0-padx);x1=min(im.shape[1],x1+padx);y0=max(0,y0-pady);y1=min(im.shape[0],y1+pady)
            overlay=im.astype(float);overlay[mask]=.75*overlay[mask]+.25*np.array([255,90,50])
            ax=axes[row_index,col];ax.imshow(overlay[y0:y1,x0:x1].astype(np.uint8));ax.axis('off')
            ax.set_title(('Highest new weight' if col==0 else 'Excluded view (largest mask)' if len(zero) else 'Lowest new weight')+f'\nframe {record["frame_id"]}, weight {weights[local]:.3f}',fontsize=10)
            evidence.append(dict(observation_id=obs,frame=record['frame_id'],rgb=rgb['rgb'],weight=float(weights[local]),crop=[int(x0),int(y0),int(x1),int(y1)]))
        score=data['scores'][ids];margin=score[:,vocab['names'].index(new_name)]-score[:,vocab['names'].index(old_name)]
        order=np.argsort(data['frames'][ids]);ax=axes[row_index,2]
        ax.axhline(0,color='#444',lw=.8);ax.scatter(data['frames'][ids][order],margin[order],c=np.where(weights[order]>0,'#248e78','#b4bbc2'),s=18)
        ax.set_xlabel('Frame index');ax.set_ylabel(f'{new_name} − {old_name}\ntext similarity');ax.set_title(f'{key} / object {oid}\n{old_name} → {new_name}',fontsize=10,fontweight='bold')
        ax.grid(alpha=.15)
        case_records.append(dict(scene=key,object_id=oid,before=old_name,after=new_name,evidence=evidence))
    fig.suptitle('Same object, revised category: cached masks and per-view evidence',fontsize=15,fontweight='bold')
    fig.supxlabel('Post-hoc examples, not case selection for inference. Green = retained evidence; gray = excluded evidence. Similarities are not probabilities.',fontsize=9)
    for ext in ['png','svg']:fig.savefig(dest/f'cases.{ext}',dpi=150,bbox_inches='tight')
    plt.close(fig);dump(dest/'cases.json',case_records)
    change_table=['| '+c['scene']+' | '+str(c['gt_id'])+' | '+str(c['object_id'])+' | '+c['before']+' → '+c['after']+' | '+('修正' if c['correct'] else '退步')+' |' for c in changes]
    report=ROOT/'docs/benchmarks/category_evidence_2026-11-11.md'
    report.write_text('''# omni_slam：固定对象的跨帧类别优化

2026-11-11。完成 11 个公共场景的 6 组预先冻结对照，以及看过结果后单独冻结的 3 组消融。共比较 9 种读出；原强基线、历史图、DINO 关联实验和训练计划均保留。

## 本轮结果与选定候选

**类别正确对象增加 8 个，原先正确类别未损失；地图几何和对象保留集合不变。** 开发候选使用 SigLIP 跨帧一致性筛选＋面积平方根加权，不需要新增 DINO 推理。它是在已经查看过的开发场景上选出的候选，尚不是独立测试结论。

| 数据集 | 原类别 PQ×100 | 新类别 PQ×100 | 提升 | 类别正确 / 固定匹配对象 | 实例 PQ×100 |
|---|---:|---:|---:|---|---:|
| Replica 8 | 30.6823 | **32.1679** | **+1.4855** | **101 → 106 / 145** | 42.8842，不变 |
| SceneNN 3 | 7.2868 | **9.0069** | **+1.7201** | **10 → 13 / 31** | 19.9377，不变 |

分数为固定观测域上的类别感知实例 PQ，不是官方逐类别宏平均 panoptic PQ。固定对象匹配集合与分母，避免通过删除难对象提高条件准确率。Replica 3 胜 5 平，SceneNN 2 胜 1 平；本轮候选没有场景下降，但样本少且均已查看，不宣称显著泛化。

## 算法与当前架构

```mermaid
flowchart LR
 A[RGB + depth / LiDAR + poses] --> B[SAM2.1 masks + SigLIP384]
 B --> C[Native CG + source ownership guard]
 C --> D[Two-frame supported history graph]
 D --> E[Persistent geometry and observation IDs]
 E --> F[Original mean-based foreground retention]
 E --> G[One largest mask per unique frame]
 G --> H[SigLIP median cosine centrality]
 H --> I[Keep centrality at or above median]
 I --> J[Sqrt area weighted descriptor]
 J --> K[Fixed vocabulary category update]
 F --> L[Object map with updated labels]
 K --> L
```

1. 继续使用 SAM2.1 Hiera L + SigLIP384 学习前端 + 原生 ConceptGraphs + 来源消费保护 + 两次查询帧支持历史图，保留对象身份和实测几何。
2. 每个对象每帧只取原始掩码面积最大的一个观测，平局用固定观测 ID。对 SigLIP 描述子归一化，计算两两余弦，去掉自匹配后取中位数作为一致性分数。
3. 保留一致性不小于其中位数的观测，包含全部平局；用原掩码面积平方根加权原 SigLIP 描述子，再归一化并对原词表读出。少于三帧或一致性特征缺失时退回面积平方根加权。
4. 类别更新与对象是否存在分开。新类别不直接删除已经保留的对象。本轮保留策略仍沿用原均值的前景决定，因此也保留了其既有漏检和误保留，不能宣称对象存在性已经解决。

该机制是当前可复现的工程改进；中位数筛选、面积权重本身不能被包装成已有充分证据的独立论文创新。

## 消融支持什么，不支持什么

- 首轮质量权重为 sqrt(area) × exclusive_area/area，再比较 SigLIP 或 DINO 的一致性筛选。exclusive_area 仅反映同帧掩码争用，不是直接测得的遮挡率。
- 第二轮去掉独占比例，使用均匀、面积平方根、面积三种权重。平方根版本在本轮指标上复现原质量＋SigLIP 结果，因此选用更简单的版本；没有证据证明独占比例带来额外收益。
- DINO 版本在 Replica 只有 104 个类别正确对象，SigLIP 为 106；SceneNN 均为 13。本轮不把 DINO 写成收益来源。
- 分数中位数在 Replica 类别 PQ 略高，但修正 6 个同时损失 1 个；SceneNN 只修正 2 个。面积一致性版本也出现 1 个类别退步。候选选择依据包含稳定性，不能只报告单一最高分。

![全部对照](../reports/evidence_retention_2026-11-11/category_evidence_v2/comparison.png)

| 数据集 | 方法 | 固定对象类别 PQ×100 | 相对均值 | 类别正确 / 匹配 | 修正 / 退步 | 若按新类别重新过滤：实例 PQ×100 |
|---|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(table)+'''

重新按类别过滤时，选定候选在 Replica 实例 PQ 从 42.8842 降到 42.5407。因此本轮导出采用**类别更新不触发对象删除**。这项解耦是预先声明的主评价方式，不是看到下降后更换分母；重新过滤的结果作为次评价完整保留。

![类别更新和对象过滤](../reports/evidence_retention_2026-11-11/category_evidence_v2/retention.png)

## 修正了哪些对象

| 场景 | GT ID | 地图对象 ID | 类别变化 | 结果 |
|---|---:|---:|---|---|
'''+ '\n'.join(change_table)+'''

SceneNN021 的床（GT286、地图对象6）从 pillow 修正为 bed。该对象有 74 个不同帧观测，其中 38 帧的单帧最高类别原本就是 bed；这次是利用已有正确证据，而非新增标注或更换图像模型。面积加权本身也能修正这张床，因此不能单独以此例证明一致性筛选的必要性。

下图事后选择床、椅子、床头柜三个例子。左图为候选权重最高的缓存掩码，中图为被排除观测中面积最大的掩码，右图显示全部不同帧上的“新类别减旧类别”文本相似度。红色半透明区域是原缓存预测掩码；绿点为入选证据，灰点为排除证据。例子未参与规则设计或推理选帧。

![原图和跨帧证据](../reports/evidence_retention_2026-11-11/category_evidence_v2/cases.png)

## 所有场景

| 场景 | 原类别 PQ×100 | 候选类别 PQ×100 | 变化 | 类别正确数 / 固定匹配 |
|---|---:|---:|---:|---|
'''+ '\n'.join(scene_table)+'''

## 仍然存在的表示与词表限制

原均值的 Replica 44 个类别错误中，17 个存在一个共同竞争类别，在每个观测里都高于 GT 文本向量；任何非负帧权重都不能纠正这类错误。24 个错误对象至少有一帧的最高类别正确，说明证据选择确有改进空间，但该统计不是可实现方法上界或实际增益。

SceneNN 原有 21 个类别错误，其中 10 个 GT 类别无法由当前固定映射词表表达，3 个另有共同竞争类别阻挡，7 个存在单帧正确机会。词表未覆盖和表示错误需单独处理，不能继续靠调权重解决。SceneNN 类别评测沿用前轮固定 NYU 名称映射和跨版本 GT 域，不更改词表来追求当前分数。

## 交付、验证与下一步

- 模块：`omni_semantic_map/omni_semantic_map/consensus_category.py`。独立返回类别及证据权重，不返回删除对象的决定。
- 全部 837 个最终区域通过与消融实验的类别、分数和权重一致性校验；原前景地图共 414 个对象，11 个场景已导出新标签、原几何引用和实测 AABB。AABB 不是补全物体范围或训练得到的三维检测框。
- 导出目录：`build/benchmarks/evidence_retention_20261111/consensus_category_candidate_v1/`，每场景含 `classification.json`、`object_map.json`。类别候选尚未激活到 ROS 在线节点。
- 10 项相关测试通过。两轮共同方法的全数据集汇总完全一致；原均值类别和分数复现、固定匹配检查、缓存及输出哈希验证均通过。没有新增训练或 GPU 抢占。
- 下一阶段先固定该候选及原均值作双基线，用新增独立场景检查是否保持类别收益；再把只更新类别、不删除对象的接口接入 LIVO 关键帧链路。对共同竞争类别阻挡的样本，另立完整对象图像上下文/表示试验；词表扩展独立计分。既定训练和 SAM3 计划保持独立。

本轮全部场景此前已经查看，候选选择属于开发过程；不能把开发集零退步写成真实机器人零风险或论文泛化保证。

[完整结果](../reports/evidence_retention_2026-11-11/category_evidence_v2/summary.json) · [全部候选类别变化](../reports/evidence_retention_2026-11-11/category_evidence_v2/category_changes.json) · [冻结协议](../reports/evidence_retention_2026-11-11/category_evidence_v2/protocol.json) · [上一轮 DINO 关联对照](public_dino_scope_2026-11-11.md)
''')
    htmlrows=''.join('<tr>'+''.join('<td>'+html.escape(x.strip())+'</td>' for x in r.strip('|').split('|'))+'</tr>' for r in change_table)
    (dest/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>omni_slam · 跨帧类别优化</title><style>body{font:16px/1.7 system-ui;max-width:1250px;padding:28px;margin:auto;background:#f3f5f8;color:#183142}section{background:white;padding:24px;margin:22px 0;border-radius:10px}img{width:100%}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border-bottom:1px solid #dde3e8;text-align:left}a{color:#176b9b}.metric{font-size:25px;font-weight:700;color:#167b67}</style><h1>omni_slam · 跨帧类别优化</h1><p>11 场景 · 9 组读出对照 · 固定对象身份、几何和前景集合。</p><section><p class="metric">类别正确对象 +8 · 原先正确类别损失 0</p><p>Replica：101 → 106 / 145，类别 PQ +1.486。SceneNN：10 → 13 / 31，类别 PQ +1.720。</p><p>候选：SigLIP 一致性筛选＋面积平方根加权。类别更新不删除对象，原实例 PQ 保持不变。所有场景已经查看，结果属于开发验证。</p><a href="../../../benchmarks/category_evidence_2026-11-11.md">完整报告、消融和限制</a> · <a href="summary.json">完整数据</a></section><section><img src="comparison.png" alt="九组对照"></section><section><h2>实际类别修正</h2><table><tr><th>场景</th><th>GT ID</th><th>对象 ID</th><th>变化</th><th>结果</th></tr>'''+htmlrows+'''</table><img src="cases.png" alt="床、椅子和床头柜的图像与跨帧证据"></section><section><h2>类别更新和对象存在性分开</h2><img src="retention.png" alt="重新过滤会损失部分几何指标"><p>837 个区域的模块输出已核验，11 个场景导出类别地图；尚未启用到 ROS 在线节点。此处类别 PQ 是项目诊断指标，非官方排行榜。</p></section></html>''')
    print('REPORT',report,flush=True)


if __name__=='__main__':main()
