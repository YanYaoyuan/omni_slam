"""Completion report for frozen unseen-scene validation, including weak results."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/omni-holdout-report')
from pathlib import Path
import html
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_evidence_retention import ROOT,OUT,REPORT,dump
from public_dino_protocol import read,verify
from omni_semantic_map.flat_benchmark import sha256
from omni_semantic_map.shared_observations import observation_table

DEST=OUT/'category_holdout_v1'


def main():
    selection=read(DEST/'selection.json');verify(selection)
    summary=read(DEST/'summary.json');audit=read(DEST/'completion_audit/audit.json')
    assert summary['complete'] and audit['complete'] and read(DEST/'queue_status.json')['complete']
    assert sha256(DEST/'summary.json')==audit['summary_sha256']
    dest=REPORT/'category_holdout_v1';dest.mkdir(exist_ok=False)
    for name,data in [('summary',summary),('audit',audit),('selection',selection),('annotation_amendment',read(DEST/'annotation_amendment.json'))]:dump(dest/(name+'.json'),data)
    methods=['mean','area','consensus_sqrt_area'];labels=['Unique-frame mean','Area weighted','Consensus + sqrt area']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained')
    for j,s in enumerate(summary['scenes']):
        values=[100*s['methods'][m]['fixed']['class_aware_pq'] for m in methods]
        bars=axes[j].bar(range(3),values,color=['#8398a8','#c9a45d','#268b76'])
        axes[j].bar_label(bars,labels=[f'{v:.3f}\n{s["methods"][m]["fixed"]["correct_class_matches"]}/{s["methods"][m]["fixed"]["geometry_matches"]}' for m,v in zip(methods,values)],padding=4)
        axes[j].set_xticks(range(3),['Mean','Area','Consensus'],rotation=12);axes[j].set_ylim(0,max(values)*1.4)
        axes[j].set_title('SceneNN '+s['scene']+' | category PQ',loc='left',fontweight='bold');axes[j].set_ylabel('Diagnostic PQ × 100')
    names=['baseline','native_bridge_1','native_bridge_2'];values=[audit['macro_geometry'][m]['instance_pq100'] for m in names]
    bars=axes[2].bar(range(3),values,color=['#8398a8','#aeb7c0','#557eaa']);axes[2].bar_label(bars,labels=[f'{v:.3f}' for v in values],padding=4)
    axes[2].set_xticks(range(3),['Native CG','History / 1','History / 2'],rotation=12);axes[2].set_ylim(0,max(values)*1.3)
    axes[2].set_title('025 + 032 | instance PQ',loc='left',fontweight='bold');axes[2].set_ylabel('Diagnostic PQ × 100')
    for ax in axes:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Unseen-sequence validation: a gain over mean, a tie with area weighting',fontsize=14,fontweight='bold')
    fig.supxlabel('030 has no compatible category annotation and is not scored. Geometry controls are a post-hoc audit; all configurations unchanged.',fontsize=9)
    for ext in ['png','svg']:fig.savefig(dest/f'comparison.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)
    # There is exactly one category correction; show it rather than selecting a flattering subset.
    scene='032';pred=read(DEST/'predictions/032.json');obj=next(r for r in pred['objects'] if r['object_id']==27)
    cache=DEST/'frontend/032/cache';cm=read(cache/'manifest.json');job=read(DEST/'jobs/032.json')
    f,frames,areas=observation_table(cache);text=np.load(cache/'text_features.npy');text/=np.linalg.norm(text,axis=1,keepdims=True);vocab=read(cache/'vocabulary.json')
    candidate=obj['methods']['consensus_sqrt_area'];ids=np.array(candidate['selected_observation_ids']);weights=np.array(candidate['weights'])
    chosen=int(ids[weights.argmax()]);record=next(r for r in cm['records'] if r['observation_offset']<=chosen<r['observation_offset']+r['masks'])
    image_record=next(r for r in job['frames'] if r['frame_id']==record['frame_id']);assert sha256(image_record['rgb'])==job['input_sha256'][image_record['rgb']]
    im=np.array(Image.open(image_record['rgb']).convert('RGB'));data=np.load(cache/'frames'/record['file'])
    mask=np.unpackbits(data['masks_packed'][chosen-record['observation_offset']],axis=1)[:,:im.shape[1]].astype(bool)
    y,x=np.where(mask);pad=40;x0=max(0,int(x.min())-pad);x1=min(im.shape[1],int(x.max())+pad);y0=max(0,int(y.min())-pad);y1=min(im.shape[0],int(y.max())+pad)
    overlay=im.astype(float);overlay[mask]=overlay[mask]*.75+np.array([255,105,40])*.25
    scores=f[ids].astype(float)@text.T;margin=scores[:,vocab['names'].index('desk')]-scores[:,vocab['names'].index('table')]
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    axes[0].imshow(overlay[y0:y1,x0:x1].astype(np.uint8));axes[0].axis('off')
    axes[0].set_title(f'032 / object 27 / frame {record["frame_id"]}\nCached predicted mask, highest candidate weight',fontsize=11)
    axes[1].scatter(frames[ids],margin,c=np.where(weights>0,'#268b76','#b8bdc5'),s=30);axes[1].axhline(0,color='#444',lw=.8)
    axes[1].set_xlabel('Frame index');axes[1].set_ylabel('desk − table text similarity');axes[1].grid(alpha=.15)
    axes[1].set_title('Mean: table → area and candidate: desk\nSame object, same category-domain IoU = 0.5688',fontsize=11)
    fig.suptitle('The only new category-correct object',fontsize=14,fontweight='bold')
    fig.supxlabel('GT NYU class 14 (desk), instance 1313150, annotation purity 96.90%. Green = selected evidence; gray = excluded. Post-hoc illustration.',fontsize=9)
    for ext in ['png','svg']:fig.savefig(dest/f'desk_case.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)
    dump(dest/'desk_case.json',dict(scene=scene,object_id=27,gt_id=1313150,frame=record['frame_id'],rgb=image_record['rgb'],
        observation_id=chosen,crop=[x0,y0,x1,y1],predictions=obj['methods'],note='Only observed category correction; area weighting also fixes it.'))
    macro=audit['macro_geometry'];rows=[]
    for name,label in [('baseline','原强基线＋均值'),('native_bridge_2','两帧历史图＋均值')]:
        r=macro[name];correct=sum(s['methods'][name]['semantic']['correct_class_matches'] for s in audit['scene_geometry'])
        matches=sum(s['methods'][name]['semantic']['geometry_matches'] for s in audit['scene_geometry'])
        rows.append(f'| {label} | {r["instance_pq100"]:.4f} | {r["category_pq100"]:.4f} | {correct}/{matches} |')
    for name,label in [('area','两帧历史图＋面积加权'),('consensus_sqrt_area','两帧历史图＋一致性/平方根面积')]:
        r=summary['methods'][name];rows.append(f'| {label} | {macro["native_bridge_2"]["instance_pq100"]:.4f} | {r["class_pq100"]:.4f} | {r["correct"]}/{r["matches"]} |')
    scenes=[];domains=[];secondary=[]
    for s in summary['scenes']:
        methods=s['methods'];scenes.append('| '+s['scene']+' | '+' | '.join(f'{100*methods[m]["fixed"]["class_aware_pq"]:.4f}' for m in ['mean','area','consensus_sqrt_area'])+f' | {methods["mean"]["fixed"]["correct_class_matches"]} → {methods["consensus_sqrt_area"]["fixed"]["correct_class_matches"]} / {methods["mean"]["fixed"]["geometry_matches"]} |')
        d=s['domain'];domains.append(f'| {s["scene"]} | {100*d["within_mesh_5cm"]:.2f}% | {100*d["valid_voxel_fraction"]:.2f}% | {100*d["category_valid_voxel_fraction"]:.2f}% | {d["category_gt_instances"]}/{d["gt_instances"]} |')
        for m,v in methods.items():secondary.append(f'| {s["scene"]} | {m} | {100*v["refiltered_instance"]["observed_voxel_instance_pq"]:.4f} | {100*v["refiltered_semantic"]["class_aware_pq"]:.4f} |')
    report=ROOT/'docs/benchmarks/category_holdout_2026-11-11.md'
    report.write_text('''# omni_slam：新场景类别验证完成

本轮预先固定 SceneNN 025、030、032，保持原模型、词表、参数和采样方式。三个场景的强前端和建图已全部完成；025、032 完成同口径评分，030 缺官方 CVPR18 类别标注，保留建图但不评分。没有替换场景，也没有根据新分数调整参数。

## 结论

**候选在新场景上优于均值，但尚未证明比简单面积加权更好。** 两个可评分场景的类别 PQ×100 从 6.7434 提升到 7.7412（+0.9978）；固定类别评价域中，类别正确对象由 7/15 增至 8/15，增加 1 个、损失 0 个。025 持平，032 改善。面积加权获得完全相同的主指标和同一个类别修正。

补充核查显示，两帧历史图相对原强基线的实例 PQ×100 从 11.6630 提升到 12.6519（+0.9889），原实例评价域内有效匹配由 13 增至 14，未丢失原匹配。历史图和类别更新两部分的收益应分别报告。

| 方案 | 实例 PQ×100 | 类别 PQ×100 | 类别正确 / 类别域匹配 |
|---|---:|---:|---:|
'''+ '\n'.join(rows)+'''

“原强基线”仍为 SAM2.1 Hiera L＋SigLIP384 学习前端＋原生 ConceptGraphs＋来源消费保护＋不同帧均值。后两行固定历史图的对象保留与几何，只更新类别。比较为场景宏平均，属于项目观测域诊断，不是官方逐类别宏平均 panoptic PQ 或官方排行榜结果。

类别评价仅包含可确定 GT 类别的区域，因此其几何匹配分母是 15；原实例评价有效匹配为 14。评价域缩小会改变 IoU，两个数量不能混用。

![新场景与几何对照](../reports/evidence_retention_2026-11-11/category_holdout_v1/comparison.png)

## 每个选定场景

| 场景 | 均值类别 PQ×100 | 面积类别 PQ×100 | 候选类别 PQ×100 | 类别正确变化 / 固定类别域匹配 |
|---|---:|---:|---:|---|
'''+ '\n'.join(scenes)+'''
| 030 | 不可评分 | 不可评分 | 不可评分 | 缺同口径官方类别标注；建图已完成 |

总计 1,200 个几何帧、600 个语义帧。025/030/032 的最终历史图区域分别为 56/103/125，原前景保留为 39/75/78；预测和原图缓存均保留。

## 唯一新增类别修正：table → desk

032 的地图对象 27 对应 GT 实例 1313150。官方类别为 NYU 14（desk），该实例主导标注纯度为 96.90%。均值预测 table，面积加权与候选均预测 desk。三个读出的对象点集、匹配 ID 和类别评价域 IoU=0.56876296 完全一致，因此这是类别修正，不是几何变化。

desk/table 是数据集中的细分类别区别。下图用于检查这个分数来自什么变化，不能把单个细粒度纠正放大为全面识别改善。

![对象原图及跨帧证据](../reports/evidence_retention_2026-11-11/category_holdout_v1/desk_case.png)

## 评价覆盖和低分的含义

| 场景 | 实测点距 GT mesh≤5cm | 有类别标注的原评价体素比例 | 类别有效域 / 原有效域 | 可确定类别 GT / 原前景 GT |
|---|---:|---:|---:|---:|
'''+ '\n'.join(domains)+'''

标注覆盖并非整张地图：025、032 的原有效体素分别约为共同观测域的 67.38%、48.54%。类别评价再排除 GT 类别不确定部分。所有分支共用这些固定域，不能据此声称整图对象召回。

两帧历史图在原实例域的 62 个可见 GT 中仅匹配 14 个。事后错误签名为：25 个 IoU 不足、15 个预测覆盖很低、5 个涉及混合预测、3 个符合碎片化特征。它们描述结果形态，尚不能单凭该统计确定是掩码、投影、关联还是过滤导致。当前限制仍主要在完整对象几何与覆盖，类别权重只能解决其中一部分。

## 补充对照完整保留

原生已有的一帧支持历史图在本次事后几何核查中得到实例 PQ×100=12.9018，高于冻结两帧分支的 12.6519；两者都新增 GT556 的几何匹配。该检查在主要类别结果完成后追加，属于次级诊断，不改变已冻结的两帧候选，也不构成新的独立选型成绩。

如果按新类别重新过滤对象，得到以下次评价。主评价继续固定原保留对象；不能根据这次有利的过滤结果切换口径。

| 场景 | 重新过滤方法 | 实例 PQ×100 | 类别 PQ×100 |
|---|---|---:|---:|
'''+ '\n'.join(secondary)+'''

## 验证与下一步

源文件、输入图像、模型阶段输出、缓存、映射输出和预测哈希验证通过；历史图主分支投影标签逐点复现，三种类别读出的主指标精确复现。全部场景预测完成后才读取 GT 评分。未重新训练，也未在新场景上扫参数。

当前保留“历史图＋固定对象类别更新”为候选链路，后续对比持续包含均值和面积加权两个简单控制。新增序列只有两个、类别域匹配仅 15 个；先扩大独立数据覆盖，再判断一致性筛选是否有稳定增益。025、032 此后已成为已查看数据，不能再次当成未见测试集调参后复报。

工程侧已完成真实 LIVO 缓存证据的 ROS 类别消费者验证（331 次更新、5 类故障注入）。实时 SAM2＋SigLIP 特征生产者与机器狗全链路性能仍待接入，不能由本次 GPU 离线处理速度推断。

[原始类别结果](../reports/evidence_retention_2026-11-11/category_holdout_v1/summary.json) · [完整性与几何审计](../reports/evidence_retention_2026-11-11/category_holdout_v1/audit.json) · [选型冻结记录](../reports/evidence_retention_2026-11-11/category_holdout_v1/selection.json) · [030 缺标注记录](../reports/evidence_retention_2026-11-11/category_holdout_v1/annotation_amendment.json) · [LIVO/ROS 接口说明](../deployment/category_stream_2026-11-11.md)

运行目录：`build/benchmarks/evidence_retention_20261111/category_holdout_v1/`。队列 `queue_status.json` 的 frontend、mapping、evaluation 三阶段均已完成。
''')
    htmlrows=''.join('<tr>'+''.join('<td>'+html.escape(x.strip())+'</td>' for x in r.strip('|').split('|'))+'</tr>' for r in rows)
    (dest/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>omni_slam 新场景验证</title><style>body{font:16px/1.7 system-ui;max-width:1200px;margin:auto;padding:28px;background:#f3f5f8;color:#183142}section{background:white;padding:24px;margin:24px 0;border-radius:10px}img{width:100%}table{width:100%;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}a{color:#176b9b}</style><h1>omni_slam · 新场景验证完成</h1><p>025、030、032 均完成建图；025、032 可按固定类别口径评分，030 缺标注。</p><section><h2>历史图与类别更新均有改善</h2><p>类别 PQ：6.743 → 7.741；类别正确：7/15 → 8/15。简单面积加权与候选持平，额外一致性收益尚未建立。</p><p>两帧历史图相对原强基线，实例 PQ：11.663 → 12.652。下表后两行固定对象点集与保留决定。</p><table><tr><th>方案</th><th>实例PQ</th><th>类别PQ</th><th>类别正确</th></tr>'''+htmlrows+'''</table><p><a href="../../../benchmarks/category_holdout_2026-11-11.md">完整报告与限制</a> · <a href="audit.json">完整性/几何审计</a></p></section><section><img src="comparison.png" alt="逐场景类别对照和几何对照"></section><section><h2>唯一新增类别修正：table → desk</h2><img src="desk_case.png" alt="原图掩码与跨帧类别证据"><p>实例、点集和 IoU 不变；属于数据集细分类别修正。该例面积加权也能修复。</p></section><section><h2>结果边界</h2><p>两个可评分的新序列，固定类别域内仅 15 个几何匹配。原有效体素覆盖为 67.38% 和 48.54%。这是项目观测域诊断，不是官方排行榜或机器人实时测试。</p></section></html>''')
    print('REPORT COMPLETE',report,flush=True)


if __name__=='__main__':main()
