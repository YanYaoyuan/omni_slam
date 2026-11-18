"""Complete-map results, actual RGB and every reference object's outcome."""
from pathlib import Path
import runpy
import html
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from run_object_mask_cover import OUT,PARENT,METHODS,verify
from evaluate_kitti360_development import transfer,read_static_ply
from omni_semantic_map.kitti360_reference_identity import encode_reference

DEST=ROOT/'output/object_mask_cover_v1'
NAMES=dict(baseline='Frozen strong baseline',geometry_history='Geometry + history',single_history='Single-mask explanation',
    cover_current='Union explanation / current only',cover_history='Ours candidate / union + history')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    p=verify();s=read(OUT/'summary.json');audit=read(OUT/'delivery_audit.json');seal=read(OUT/'maps_complete.json')
    assert s['complete'] and audit['complete'] and not DEST.exists()
    for f,h in seal['files'].items():assert sha(f)==h,f
    DEST.mkdir(parents=True);parts=[];details={};visuals={}
    taxonomy=runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['id2label']
    for c in p['contexts']:
        key=c['key'];ref=dict(np.load(c['reference']));raw=read_static_ply(c['raw_reference'])
        physical,_=encode_reference(raw['semantic'],raw['instance'],taxonomy,'target_only');tables={};maps={}
        for mode,truth in [('primary',ref['train_instance']),('raw_targets',physical)]:
            mass=np.bincount(truth,weights=ref['confidence']);gt=np.flatnonzero(np.isin(np.arange(len(mass))//1000,[2,13])&(mass>=100))
            tables[mode]={int(g):dict(gt_id=int(g),class_name='building' if g//1000==2 else 'car',methods={}) for g in gt}
        for method in METHODS:
            path=Path(c['ours']) if method=='baseline' else OUT/'maps'/key/method
            maps[method]=dict(np.load(path/'geometry.npz'));records=read(path/'classification.json')['objects']
            classes={r['object_id']:r['predictions']['mean']['train_id'] for r in records}
            pred=np.load(OUT/'evaluation'/key/method/'reference_prediction.npz')['instance']
            pm=np.bincount(pred,weights=ref['confidence'],minlength=max(classes)+1)
            for mode,truth in [('primary',ref['train_instance']),('raw_targets',physical)]:
                mass=np.bincount(truth,weights=ref['confidence'])
                for gid,row in tables[mode].items():
                    take=truth==gid;inter=np.bincount(pred[take],weights=ref['confidence'][take],minlength=len(pm))
                    ids=np.array([o for o,cls in classes.items() if cls==gid//1000]);values=inter[ids]/np.maximum(1e-12,mass[gid]+pm[ids]-inter[ids])
                    best=int(np.argmax(values));row['methods'][method]=dict(best_iou=float(values[best]),object_id=int(ids[best]))
        details[key]={mode:list(rows.values()) for mode,rows in tables.items()}
        show=['baseline','geometry_history','single_history','cover_history']
        bounds=np.stack([np.min([g['xyz'][:,:2].min(0) for g in maps.values()],axis=0),np.max([g['xyz'][:,:2].max(0) for g in maps.values()],axis=0)])
        fig,axes=plt.subplots(1,4,figsize=(18,5),constrained_layout=True)
        for ax,method in zip(axes,show):
            g=maps[method];ii=np.arange(0,len(g['xyz']),max(1,len(g['xyz'])//100000))
            ax.scatter(g['xyz'][ii,0],g['xyz'][ii,1],c=g['instance'][ii]%20,cmap='tab20',s=.3,vmin=0,vmax=19,rasterized=True)
            ax.set_xlim(bounds[:,0]);ax.set_ylim(bounds[:,1]);ax.set_aspect('equal');ax.set_title(NAMES[method],fontsize=10);ax.set_xlabel('World x / m');ax.set_ylabel('World y / m')
        fig.suptitle(key+' / complete measured maps / shared bounds / repeated palette; new IDs not aligned across methods')
        fig.savefig(DEST/(key+'_maps.png'),dpi=160);plt.close(fig)
        views=read(PARENT/'prepared'/key/'views.json');selected=views[len(views)//2];v=np.load(selected['file'])
        frame=next(f for f in read(c['job'])['frames'] if f['frame_id']==selected['frame_id']);rgb=np.asarray(Image.open(frame['rgb']).convert('RGB'))
        full=np.load(Path(c['full'])/'geometry.npz')['xyz'];source=full[v['source_ids']];xy=v['pixel_xy']
        fig,axes=plt.subplots(1,3,figsize=(18,3),constrained_layout=True)
        for ax,method in zip(axes,['baseline','geometry_history','cover_history']):
            g=maps[method];label=transfer(g['xyz'],g['instance'],source,.05);take=label>0
            ax.imshow(rgb);ax.scatter(xy[take,0],xy[take,1],c=label[take]%20,cmap='tab20',s=1,vmin=0,vmax=19,alpha=.75)
            ax.set_title(NAMES[method]);ax.axis('off')
        fig.suptitle(f'{key} / actual RGB {selected["frame_id"]}, fixed median input frame / projected measured-source labels, not GT or dense masks')
        fig.savefig(DEST/(key+'_rgb.png'),dpi=160);plt.close(fig)
        visuals[key]=dict(frame_id=selected['frame_id'],selection='fixed median original frame, no GT selection',display_radius_m=.05,evaluation_radius_m=.2)
        rows=''
        for method in METHODS:
            r=s['results'][key][method];stat=seal['results'][key].get(method,{})
            rows+='<tr><td>'+NAMES[method]+'</td>'+''.join(f'<td>{r[m]["metrics"]["allAp"]*100:.5f}</td>' for m in ['primary','raw_targets','uniform_scores'])
            rows+='<td>'+html.escape(str(r['raw_targets']['changes_vs_baseline']))+'</td><td>'+str(stat.get('new_identities',0))+'</td><td>'+str(stat.get('old_points_changed',0))+'</td></tr>'
        parts.append(f'<h2>{key}</h2><img src="{key}_maps.png"><img src="{key}_rgb.png"><table><tr><th>Method</th><th>Primary AP %</th><th>Raw-target diagnostic</th><th>Uniform score</th><th>Raw AP50 gain/loss IDs</th><th>New IDs</th><th>Old labels changed</th></tr>{rows}</table>')
        for mode,table in tables.items():
            rows=''
            for gid,r in table.items():
                rows+=f'<tr><td>{gid}</td><td>{r["class_name"]}</td>'+''.join(f'<td>{r["methods"][m]["best_iou"]:.5f}</td>' for m in METHODS)+'</tr>'
            parts.append('<details><summary>Every '+mode+' object: best same-class IoU (not AP matching)</summary><table><tr><th>GT ID</th><th>Class</th>'+''.join('<th>'+NAMES[m]+'</th>' for m in METHODS)+'</tr>'+rows+'</table></details>')
    mean=''.join('<tr><td>'+NAMES[m]+'</td>'+''.join(f'<td>{s["mean_ap"][mode][m]*100:.6f}</td>' for mode in ['primary','raw_targets','uniform_scores'])+'</tr>' for m in METHODS)
    same=all(v['single_vs_cover_same_geometry'] and v['single_vs_cover_same_labels'] and v['single_vs_cover_same_classes_scores'] for v in audit['results'].values())
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam：完整表面与多掩码解释</title>
<style>body{max-width:1500px;margin:40px auto;padding:0 24px;font:16px/1.6 sans-serif;color:#203748}img{width:100%}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:7px;border-bottom:1px solid #d8e3e6;text-align:right}td:first-child,th:first-child{text-align:left}.note{background:#fff1db;padding:18px}details{margin:20px 0;overflow:auto}</style>
<h1>omni_slam：完整表面、多局部掩码解释与历史修订</h1><p class="note">冻结开发门槛：'''+str(s['development_gate'])+'''。单掩码与多掩码地图的实际坐标、标签、类别和置信分数完全相同：'''+str(same)+'''。不得将共同收益归因于多掩码机制。当前强基线保留；两个窗口均已用于开发，不是独立场景或官方榜单。</p>
<p>固定三维局部连通表面。每个视图允许多个部分掩码联合解释一个表面；用并集内的支持质量减去并集外泄漏和固定查询代价，重复覆盖只计一次。以同输入的纯几何、单掩码及禁止历史改写对照检验机制。所有历史组只允许直接通过接纳的表面、且传播证据一致的旧点改归属；未解释区域保留原标签。</p>
<p>多掩码在58个区域-视图案例中提高解释效用，但两个窗口分别与单掩码共同接纳55/39个表面，无新增通过者。完整地图是否获益取决于下表；效用改善本身不算AP提升。八份新地图输出中，单/多历史版为相同预测，不是八个不同方法证据。</p>
<table><tr><th>Method</th><th>Primary mean AP50:90 %</th><th>Raw-target diagnostic</th><th>Uniform-score diagnostic</th></tr>'''+mean+'</table>'+''.join(parts)+'''
<p>原始GT身份诊断将garage别名置void，主指标保持原上游兼容编码。所有实际坐标来自测量，完整输出重新评分；8份地图已重放，原始掩码解释已独立解包核对，6次实际删帧验证已认证标签保持不变。核验不等于语义准确率或统计独立性。本阶段无新训练/模型推理/机器人运行/独立评测。</p>
<p>完整三项目标仍未完成。若冻结门槛失败，不搜索覆盖成本、半径、支持帧数或评分权重来挽救本版。</p></html>'''
    (DEST/'index.html').write_text(page)
    dump(OUT/'per_object.json',dict(complete=True,results=details,scope='All eligible primary/raw-target reference objects; bestIoU independent diagnostic'))
    dump(OUT/'report_audit.json',dict(complete=True,source_sha256=sha(__file__),summary_sha256=sha(OUT/'summary.json'),
        visuals=visuals,single_cover_identical_predictions=same,files={str(f):sha(f) for f in DEST.iterdir()},per_object_sha256=sha(OUT/'per_object.json')))
    print('COVER REPORT',DEST/'index.html',flush=True)


if __name__=='__main__':main()
