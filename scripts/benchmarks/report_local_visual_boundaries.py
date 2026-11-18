"""Actual image/patch evidence and complete conditional boundary diagnostics."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/omni_local_boundary_mpl')
import html
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from paper_final_common import ROOT, read, dump, sha
from audit_local_visual_boundaries import B, OUT, KEYS

DEST = ROOT/'output/local_visual_boundary_audit_v1'


def image_example(key):
    m = read(B/'discovery_measured_inputs_v1'/key/'manifest.json')
    xyz = np.load(m['geometry'])['xyz']
    mapping = np.load(B/'full_lidar_surface_inputs_v1'/key/'old_to_full.npy')
    with np.load(B/'batch_stable_transport_v1/fields'/key/'full/field.npz') as f:
        cell, centers = f['source_to_cell'][mapping], f['centers']
    with np.load(B/'surface_continuation_probe_v1/features'/key/'edges.npz') as f:
        left, right, kind = f['left'], f['right'], f['pair_kind']
    distance = np.linalg.norm(xyz-centers[cell], axis=1)
    # First chronological image containing any co-visible differing FG owners.
    # No GT, model score or outcome is used to pick the example.
    for rec in m['records']:
        with np.load(rec['file']) as f: ids, pixels = f['visible_source_ids'], f['pixel_xy']
        order = np.lexsort((ids, distance[ids], cell[ids])); _, first = np.unique(cell[ids][order], return_index=True)
        sel = order[first]; nodes, xy = cell[ids[sel]], pixels[sel]
        lookup = np.full(len(centers), -1, int); lookup[nodes] = np.arange(len(nodes))
        a, b = lookup[left], lookup[right]; visible = (a>=0)&(b>=0)
        crossings = np.flatnonzero(visible & (kind==2))
        if len(crossings): break
    else: raise AssertionError('No illustrative co-visible pair')
    edge = int(crossings[0]); midpoint = (xy[a[edge]]+xy[b[edge]])/2
    image = np.asarray(Image.open(rec['rgb']).convert('RGB')); h,w = image.shape[:2]
    x0 = float(np.clip(midpoint[0]-140, 0, w-280)); y0 = float(np.clip(midpoint[1]-110, 0, h-220))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.7), gridspec_kw={'width_ratios':[2.4, 1]}, layout='constrained')
    for ax in axes:
        ax.imshow(image)
        for x in np.arange(1,37)*w/37-.5: ax.axvline(x, color='white', alpha=.55, lw=.55)
        for y in np.arange(1,10)*h/10-.5: ax.axhline(y, color='white', alpha=.55, lw=.55)
        ax.set_axis_off()
    axes[0].add_patch(Rectangle((x0,y0),280,220,fill=False,edgecolor='#ffda46',lw=2))
    axes[0].set_title(f'{key} | original RGB {rec["frame_id"]} | 10 x 37 patch grid',loc='left')
    axes[1].set_xlim(x0,x0+280); axes[1].set_ylim(y0+220,y0)
    region = visible.copy(); idx = np.flatnonzero(visible)
    mid = (xy[a[idx]]+xy[b[idx]])/2
    region[idx] &= (mid[:,0]>=x0)&(mid[:,0]<x0+280)&(mid[:,1]>=y0)&(mid[:,1]<y0+220)
    displayed={}
    for pair, color in [(1,'#00e6ce'),(2,'#ff3c79')]:
        ids_edge = np.flatnonzero(region & (kind==pair))
        if len(ids_edge)>80: ids_edge = ids_edge[np.linspace(0,len(ids_edge)-1,80).astype(int)]
        displayed[str(pair)] = ids_edge.tolist()
        seg = np.stack([xy[a[ids_edge]],xy[b[ids_edge]]],axis=1)
        axes[1].add_collection(LineCollection(seg, colors=color, linewidths=1.3,alpha=.85))
    points = xy[[a[edge], b[edge]]]
    axes[1].scatter(points[:,0],points[:,1],s=36,facecolor='none',edgecolor='yellow',lw=1.5)
    axes[1].set_title('Measured cell proxies\ncyan: same owner / pink: different owners')
    fig.suptitle('Frozen predicted ownership, not GT boundaries or segmentation accuracy',fontsize=12,color='#555')
    path=DEST/f'{key}_patches.png';fig.savefig(path,dpi=150);plt.close(fig)
    return dict(key=key,frame_id=rec['frame_id'],rgb=rec['rgb'],rgb_sha256=sha(rec['rgb']),
                selection='First chronological co-visible differing-foreground edge, first edge in fixed graph order',
                highlighted_edge=edge,displayed_edge_ids=displayed,crop=[x0,y0,280,220],figure=path.name)


def main():
    d=read(OUT/'summary.json'); full=read(B/'full_source_visual_visibility_audit_v1/summary.json')
    assert d['complete'] and full['complete'];DEST.mkdir(parents=True,exist_ok=False)
    for s in [d,full]:
        for f,h in s['files'].items():assert sha(f)==h,f
    examples=[image_example(k) for k in KEYS]
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    labels=['Same FG vs other FG','Same FG vs FG/BG boundary']
    names=['foreground_vs_foreground','foreground_vs_background_boundary']
    controls=[('visual_cosine','VGGT cosine','#008e82'),('negative_pixel_distance','Pixel distance','#6c7e90'),('negative_edge_length','3D edge length','#d59a40')]
    for ax,key in zip(axes,KEYS):
        for j,(control,label,color) in enumerate(controls):
            values=[d['results'][key]['rankings'][n]['auroc'][control] for n in names]
            ax.bar(np.arange(2)+(j-1)*.24,values,.22,label=label,color=color)
        ax.axhline(.5,color='#666',ls='--',lw=.8);ax.set_ylim(0,1);ax.set_xticks([0,1],labels,fontsize=9)
        ax.set_title(key);ax.set_ylabel('Conditional pseudo-label AUROC');ax.legend(fontsize=8)
    fig.savefig(DEST/'rankings.png',dpi=150);plt.close(fig)
    rows=[];coverage=[]
    for key in KEYS:
        s=d['results'][key];v=full['results'][key]
        for n in names:
            r=s['rankings'][n]
            rows.append(f'<tr><td>{key}</td><td>{n}</td><td>{r["positive_edges"]:,} / {r["negative_edges"]:,}</td>'+''.join(f'<td>{r["auroc"][c]:.4f}</td>' for c,_,_ in controls)+'</tr>')
        coverage.append(f'<tr><td>{key}</td><td>{s["denied_cells"]:,}</td><td>{s["denied_cells_with_observed_incident_edge"]:,}</td><td>{v["newly_observed_sources"]:,}</td><td>{v["additional_observed_cells"]:,}</td><td>{v["denied_cells_newly_observed"]}</td></tr>')
    page=f'''<!doctype html><meta charset="utf-8"><title>omni_slam 局部视觉证据核查</title>
<style>body{{max-width:1200px;margin:36px auto;padding:0 24px;color:#20364a;font:16px/1.65 system-ui;background:#f5f7f9}}section{{padding:24px;background:white;margin:20px 0;border-radius:12px}}img{{width:100%}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{text-align:left;padding:9px;border-bottom:1px solid #ddd}}a{{color:#008e82}}</style>
<h1>omni_slam：局部视觉信息能否支持对象边界？</h1>
<p>2026-11-14 · 两个已暴露室外开发窗口 · 复用108帧缓存 · 0新训练／0新模型推理／0新地图。</p>
<section><h2>研究决定</h2><p>当前缓存 VGGT 局部余弦读出没有提供稳定的相邻实例区分优势，因此不启动“把余弦当边代价”的完整地图分支。它与前一轮纯几何边学习失败分别保留；不据此否定 VGGT、DINO、所有层或训练后的分割模型。观测覆盖有限，单靠重新组合旧特征不能保证补齐缺失证据。</p>
<p>主线仍是可持续更新的对象级语义地图。此前公开与机器人对照已经完成，候选尚未稳定超过强控制；本核查不产生新的AP，也不宣布总目标完成。</p></section>
<section><h2>相同边上的区分能力</h2><p>用既有预测锚点给纯单归属单元标记；冲突／未知单元不作负例。数值只是条件伪标签排序诊断。不同前景对象的可观测边仅49／129条，不能外推成真实实例准确率。每条边对同帧有效观测取均值，并与像素距离、三维边长作相同样本对照。</p>
<img src="rankings.png"><table><tr><th>窗口</th><th>条件</th><th>正／负边数</th><th>视觉余弦</th><th>像素距离</th><th>三维边长</th></tr>{''.join(rows)}</table></section>
<section><h2>原始图像与特征分辨率</h2><p>白格为10×37 patch区域，彩线为实际可见测量点对应的相邻单元。双线性读出可以区分同格内像素，因此落在同一格不等于描述符完全相同。示例按固定顺序选取最早有跨预测对象边的帧，而非按得分或视觉效果挑选；完整统计使用全部108帧。</p>{''.join('<img src="'+e['figure']+'">' for e in examples)}</section>
<section><h2>排除旧输入账本遗漏这一解释</h2><p>对完整雷达来源重新执行相同相机、位姿、当前实测深度及5厘米可见性检查。108帧旧可见点ID和像素全部精确重现。新增可见来源多为旧单元内的其他测量：新增150／62个可见单元，但没有让此前完全不可见的拒绝单元获得观测。此处只考察既有图像和严格门限；不是所有可能视角的上界。</p>
<table><tr><th>窗口</th><th>来源帧证书拒绝单元</th><th>有同帧视觉边</th><th>新增可见来源</th><th>新增可见单元</th><th>首次可见的拒绝单元</th></tr>{''.join(coverage)}</table></section>
<section><h2>与已有工作的区别及下一步边界</h2><p><a href="https://zju3dv.github.io/sam_graph/">SAM-guided Graph Cut</a>已使用多视图分割图及伪标签训练，简单换成图网络不能单独算新贡献。其当前公开入口的最终图分割调用仍标为TODO，尚未在本项目复现。<a href="https://github.com/gong208/SAM-V">SAM-V</a>则发布了训练后的SAM/VGGT融合分割器和权重入口；它是可考虑的外部强对照，不能把接入它写成Ours创新。本次只核查公开代码说明，未下载、适配或评测。</p>
<p>下一候选须作用于完整对象地图的实际错误，并明确所需新观测或分割证据从哪里来。先比较候选与共享同一新增证据的简单控制，再决定是否扩大评测；不自动追加旧特征网络、阈值或分辨率搜索。</p></section>
<section><h2>可复核产物</h2><p><a href="../../build/benchmarks/local_visual_boundary_audit_v1/summary.json">完整局部读出</a> · <a href="../../build/benchmarks/local_visual_boundary_audit_v1/protocol.json">读取前协议</a> · <a href="../../build/benchmarks/full_source_visual_visibility_audit_v1/summary.json">完整来源可见性重放</a> · <a href="../batch_transport_assessment_v1/index.html">已有公开地图对比</a> · <a href="../batch_transport_robot_v1/index.html">已有车库对比</a></p></section>'''
    (DEST/'index.html').write_text(page)
    dump(DEST/'examples.json',examples)
    dump(DEST/'manifest.json',dict(complete=True,files={str(p):sha(p) for p in DEST.iterdir() if p.name!='manifest.json'},
        source_sha256=sha(__file__),readout_sha256=sha(OUT/'summary.json'),full_visibility_sha256=sha(B/'full_source_visual_visibility_audit_v1/summary.json'),goal_complete=False))
    print('LOCAL VISUAL REPORT',DEST/'index.html',flush=True)


if __name__=='__main__':main()
