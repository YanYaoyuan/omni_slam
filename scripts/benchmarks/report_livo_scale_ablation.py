"""Development report and complete evidence gallery for sparse-scale controls."""
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_evidence_retention import ROOT, OUT, REPORT


def main():
    root=OUT/'scale_ablation_v1';dest=REPORT/'scale_ablation_v1'
    data=json.loads((dest/'summary.json').read_text())
    review=json.loads((dest/'visual_review.json').read_text())
    job=json.loads((OUT/'jobs/livo_full_v2.json').read_text())
    xyz=np.load(OUT/'livo_full_strong_v2/cache/geometry.npz')['xyz']
    sample=xyz[::max(1,int(np.ceil(len(xyz)/80000)))]
    trajectory=np.array([np.array(f['pose'])[:3,3] for f in job['frames']])
    fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained')
    for ax,variant,title in zip(axes,['native','both'],['Connected support + native CG','Connected support + scale adaptation']):
        path=root/'cache_supportcc'/variant;d=json.loads((path/'diagnostics.json').read_text())
        g=np.load(path/'geometry.npz');objects=[o for o in d['final_objects'] if o['retained']]
        ax.scatter(sample[:,0],sample[:,1],s=.2,c='#c7cdd3',rasterized=True)
        ax.plot(trajectory[:,0],trajectory[:,1],c='#303a44',lw=1,label='LIVO trajectory')
        for o in objects:
            p=g['xyz'][g['instance']==o['evaluation_id']]
            color='#d05632' if o['alias'] in [6,3075] else '#147eac'
            ax.scatter(p[:,0],p[:,1],s=2,c=color,rasterized=True)
            low=p.min(0);extent=p.max(0)-low
            ax.add_patch(plt.Rectangle(low[:2],extent[0],extent[1],fill=False,color=color,lw=.7))
        ax.set(xlabel='X (m)',ylabel='Y (m)',title=f'{title}\n{len(objects)} foreground predictions (not GT object count)',
               xlim=(xyz[:,0].min()-1,xyz[:,0].max()+1),ylim=(xyz[:,1].min()-1,xyz[:,1].max()+1))
        ax.set_aspect('equal');ax.grid(alpha=.15)
    fig.suptitle('Same full recording, SAM2 masks and SigLIP descriptors\nNo DBSCAN; overlap radius tied to 0.20 m LiDAR sampling; voxel downsampling remains 0.025 m',fontsize=12)
    fig.savefig(dest/'map_comparison.png',dpi=170);fig.savefig(dest/'map_comparison.pdf');plt.close(fig)
    display={'native':'原生参数','no_dbscan':'仅关闭 DBSCAN','sensor_overlap':'仅匹配距离 0.20m','both':'两项结合'}
    text='''# omni_slam：稀疏尺度适配的八组整包对照

2026-11-11。**有明确的工程进展：定位并分离了密集点云默认参数导致的关联不足与几何塌缩。** 当前候选是在已有连通支持上关闭原生 DBSCAN，并将重叠匹配距离独立设为 LiDAR 的 0.20m 采样尺度。SAM2、SigLIP、词表、均值读出和相似度阈值不变。

这是在已经查看的 rosbag2245 上做的开发实验，不是新的独立测试，也不是 DINO 或新网络的收益。八组都跑完；两种输入各自的原生分支均精确复现旧几何与分类。旧实验文件与默认配置没有修改。

## 原因与固定对照

原 CG 将 `downsample_voxel_size=0.025` 同时用于降采样和点间重叠距离；LIVO 输入的 `filter_size_surf=0.2`。另一个独立问题是原生 DBSCAN 的 0.1m 半径、10 点密度条件，会把某些累积车辆表面变成微小密集簇。

本次在运行前冻结 2×2 对照，分别使用原始支持与既有最大连通支持，合计八组。0.20m 只作用于因果关联和最终合并的点间距离查询；输出降采样仍为 0.025m，空间包围盒预筛选、1.2 匹配阈值及其他门限保持原值。没有扫描参数或改 GT 评价。

| 输入支持 | 配置 | 单帧对象 | ≥3 帧对象 | 最终前景预测 | 前景少于16点 | 起点/返回观测同一最终对象 |
|---|---|---:|---:|---:|---:|---|
'''
    for r in data['rows']:
        text+=f'| {"原始" if r["cache"]=="cache" else "连通支持"} | {display[r["variant"]]} | {r["single_view_objects"]} | {r["multiview_objects"]} | {r["retained_predictions"]} | {r["retained_below_16_points"]} | {"是" if r["start_return_same_output"] else "否"} |\n'
    text+='''
单帧/多帧对象为最终后处理之前的记录；前景数量为最终语义过滤之后。它们衡量保留与关联行为，**不是召回率、PQ、独立物体数量或身份准确率**。

![相同空间范围下的地图对照](../reports/evidence_retention_2026-11-11/scale_ablation_v1/map_comparison.png)

## 可以确认的改善

1. **几何塌缩可以独立消除。** 连通支持上仅关闭 DBSCAN，返回车辆从最终 11 点恢复到 377 点，观测表面范围约 0.90×3.60×1.17m；对象数和关联成员保持该对照的原有结果，因此这项收益不能归为身份改善。
2. **关联尺度确实是瓶颈之一。** 连通支持上调匹配距离后，达到至少三帧的对象由 7 增至 110，最终前景预测由 3 增至 41。起点 alias 6 与返回 alias 3075 的观测进入同一最终对象；图像支持它们来自同一黑车，但没有独立身份 GT。
3. **两项结合才能同时保留该案例的身份与表面。** 联合候选黑车对象包含 18 个不同帧、1,138 点，表面范围约 1.37×4.36×1.74m；仅调整距离而保留 DBSCAN 时，该合并对象只剩 61 点。白色 SUV 联合候选为 10 帧、630 点，范围约 2.51×4.45×1.48m。

![起点与返回黑车的同一最终对象；红点显示残余投影不一致](../reports/evidence_retention_2026-11-11/stage3/scale_v1_supportcc_both/object_000006.jpg)

![白色 SUV 的跨帧表面支持](../reports/evidence_retention_2026-11-11/stage3/scale_v1_supportcc_both/object_002599.jpg)

## 全部输出审核与剩余问题

已查看联合候选全部 41 个保留预测的最早、最大掩码、最晚证据图。其中 39 个类别有可见图像支持，1 个被遮盖的车辆形状和 1 个顶部长条结构类别不确定。**这是助手对类别的定性审核，不能写成 39/41 精度；未证明全部实例身份唯一或三维位置准确。** [全部 41 个对象画廊](../reports/evidence_retention_2026-11-11/scale_ablation_v1/gallery.html)与[审核记录](../reports/evidence_retention_2026-11-11/scale_ablation_v1/visual_review.json)保留所有输出。

- 联合黑车的最终点投影到起始帧时，部分点在车顶上方；投影到返回帧时，部分点在车体下方。身份连回不等于多帧表面对齐已解决。
- 原始支持上直接放宽距离会累积背景点，车辆表面框可膨胀到十余米。因此不能仅凭关联数增加推广这项设置。
- 联合候选最终 7 个区域含同帧额外掩码，共 10 次，均为被过滤的 wall 区域；保留前景没有此类计数冲突。但这不是“前景无误合并”的证明。
- 先前 alias 2652 的固定观测仍被连通支持过滤。新地图可能通过其他观测表示同一物理车，不能用单个观测缺失直接推断整车漏检；需要独立对象关联标注。
- 多张图中明确有行人，最终没有 person 输出，不能解释成场景无行人。动态对象策略尚未验证。
- 门的多扇结构、车辆局部遮挡、跨远时间间隔的匹配仍缺独立身份标注；当前 41 个输出不能作为场景真实对象总数。

![新增门类预测；同一画面可见行人但未输出 person](../reports/evidence_retention_2026-11-11/stage3/scale_v1_supportcc_both/object_000281.jpg)

## 当前决策与下一步

保留“连通支持＋关闭 DBSCAN＋独立采样尺度匹配”作为 **LIVO 工程候选 v1**，不覆盖公开数据强基线或宣称论文 benchmark 涨点。它已解决这个开发案例中的微小几何塌缩，并将起点/返回观测归入同一对象，但仍需验证全场景身份、稀疏支持损失及位姿误差下的表面一致性。

下一步固定该候选，先补对象身份及跨视角几何评测，再在同样输入和 SigLIP 读出下单独加入 DINO 局部对应。工程尺度修正与 DINO 的方法收益分别做消融，避免把底座参数问题算成新模型贡献。

## 复现

所有运行使用已缓存的 330 个 LIVO 原子帧，不重新训练或占用 GPU。八组后端运行均为 CPU，计时不包含此前前端推理，不能换算成在线端到端时延。

```bash
# 首次执行冻结协议；已有 protocol.json 时不要再次 --freeze。
OPENBLAS_NUM_THREADS=2 build/benchmarks/expanded_semantic_20261109/cg_venv/bin/python scripts/benchmarks/run_livo_scale_ablation.py --freeze
# 每个新输出目录只运行一次；输入支持和 variant 依协议枚举。
OPENBLAS_NUM_THREADS=2 build/benchmarks/expanded_semantic_20261109/cg_venv/bin/python scripts/benchmarks/run_livo_scale_ablation.py --cache cache_supportcc --variant both
OPENBLAS_NUM_THREADS=2 build/benchmarks/ovo_full_20261110/venv311/bin/python scripts/benchmarks/summarize_livo_scale_ablation.py
```

[冻结协议](../reports/evidence_retention_2026-11-11/scale_ablation_v1/protocol.json) · [全部八组数据](../reports/evidence_retention_2026-11-11/scale_ablation_v1/summary.json) · [上一轮三阶段报告](evidence_retention_2026-11-11.md)
'''
    doc=ROOT/'docs/benchmarks/livo_scale_2026-11-11.md';doc.write_text(text)
    gallery=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>omni_slam 对象证据</title><style>body{font:16px/1.6 system-ui;max-width:1500px;margin:32px auto;padding:0 20px;color:#1b2936}article{margin:30px 0;border-top:1px solid #ccc;padding-top:12px}img{width:100%;height:auto}small{color:#506070}</style><h1>稀疏尺度候选：全部 41 个前景预测</h1><p>相同录包、掩码和特征；每行依次为最早、最大掩码和最晚帧。绿色为最终点投影落在成员掩码内，红色为掩码外。类别视觉审核不是独立 GT 精度，三维框不是完整物体真值。</p>']
    for obj in review['objects']:
        relative='../stage3/scale_v1_supportcc_both/'+Path(obj['image']).name
        gallery.append(f'<article><h2>{obj["alias"]} · {html.escape(obj["name"])} · {html.escape(obj["semantic_verdict"])}</h2><img loading="lazy" src="{relative}" alt="对象 {obj["alias"]} 的三个证据帧"><small>{html.escape(obj["note"])}</small></article>')
    gallery.append('</html>');(dest/'gallery.html').write_text('\n'.join(gallery))
    print('WROTE',doc)


if __name__=='__main__':main()
