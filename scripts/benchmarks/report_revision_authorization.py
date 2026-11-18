"""Authoritative full-map comparison with observed identity evidence."""
from pathlib import Path
import runpy
import html
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from run_revision_authorization import OUT,PARENT,OBS,RAW,METHODS,verify
from evaluate_kitti360_development import transfer,read_static_ply
from omni_semantic_map.kitti360_reference_identity import encode_reference

DEST=ROOT/'output/revision_authorization_v1'
NAMES=dict(baseline='Frozen strong baseline',unconditional='Parent: unconditional revision',identity_only='Identity evidence only',
    write_only='Write witness only',current_only='Full new points / old IDs locked',ours='Ours candidate / authorized revision')


def verify_witnesses(contexts):
    rawseal=read(RAW);counts={}
    for c in contexts:
        key=c['key'];a=np.load(OBS/'prepared'/key/'atoms.npz');part=np.load(PARENT/'prepared'/key/'components.npz')
        d=np.load(OUT/'prepared'/key/'field.npz');parent=np.load(PARENT/'transport'/key/'cover/field.npz')
        cellatom=d['cell_atom'];rawby={r['frame_id']:r for r in rawseal['results'][key]};records=read(PARENT/'prepared'/key/'explanations.json');rows=[]
        for rank,rec in enumerate(read(OBS/'prepared'/key/'views.json')):
            v=np.load(rec['file']);px=v['pixel_xy'];raw=np.load(rawby[rec['frame_id']]['file']);node=cellatom[v['cells']]
            for r in records:
                co=r['component']
                if r['frame_id']!=rec['frame_id'] or not part['cover_accepted'][co] or parent['root_owner'][part['roots'][co]]<=0:continue
                mask=np.zeros(len(px),bool)
                for q in r['cover']['native_rows']:
                    # Independent packed-byte readout (preparation used unpackbits).
                    mask |= ((raw['masks_packed'][q,px[:,1],px[:,0]//8]>>(7-px[:,0]%8))&1).astype(bool)
                ii=np.flatnonzero(mask&(node>=0));nn=np.unique(node[ii][part['component'][node[ii]]==co])
                rows.extend((int(t),int(co),rank) for t in nn)
        expected=np.unique(np.asarray(rows,np.int64).reshape(-1,3),axis=0)
        np.testing.assert_array_equal(expected,d['witnesses']);w=np.zeros(len(a['cell_ids']),bool);w[expected[:,0]]=True
        np.testing.assert_array_equal(w,d['witness_node']);counts[key]=len(expected)
    return counts


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    p=verify();s=read(OUT/'summary.json');audit=read(OUT/'delivery_audit.json');seal=read(OUT/'maps_complete.json')
    assert s['complete'] and audit['complete'] and not DEST.exists()
    for f,h in seal['files'].items():assert sha(f)==h,f
    witness_counts=verify_witnesses(p['contexts']);DEST.mkdir(parents=True)
    taxonomy=runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['id2label'];details={};parts=[];visuals={}
    provenance=read(ROOT/'build/benchmarks/object_revision_provenance_v1/summary.json')
    for c in p['contexts']:
        key=c['key'];ref=dict(np.load(c['reference']));raw=read_static_ply(c['raw_reference']);physical,_=encode_reference(raw['semantic'],raw['instance'],taxonomy,'target_only')
        tables={};maps={}
        for mode,truth in [('primary',ref['train_instance']),('raw_targets',physical)]:
            mass=np.bincount(truth,weights=ref['confidence']);gt=np.flatnonzero(np.isin(np.arange(len(mass))//1000,[2,13])&(mass>=100))
            tables[mode]={int(g):dict(gt_id=int(g),class_name='building' if g//1000==2 else 'car',methods={}) for g in gt}
        for method in METHODS:
            path=Path(c['ours']) if method=='baseline' else PARENT/'maps'/key/'cover_history' if method=='unconditional' else OUT/'maps'/key/method
            maps[method]=dict(np.load(path/'geometry.npz'));classes={r['object_id']:r['predictions']['mean']['train_id'] for r in read(path/'classification.json')['objects']}
            pred=np.load(OUT/'evaluation'/key/method/'reference_prediction.npz')['instance'];pm=np.bincount(pred,weights=ref['confidence'],minlength=max(classes)+1)
            for mode,truth in [('primary',ref['train_instance']),('raw_targets',physical)]:
                mass=np.bincount(truth,weights=ref['confidence'])
                for gid,row in tables[mode].items():
                    take=truth==gid;inter=np.bincount(pred[take],weights=ref['confidence'][take],minlength=len(pm))
                    ids=np.array([o for o,cls in classes.items() if cls==gid//1000]);values=inter[ids]/np.maximum(1e-12,mass[gid]+pm[ids]-inter[ids]);best=int(np.argmax(values))
                    row['methods'][method]=dict(best_iou=float(values[best]),object_id=int(ids[best]))
        details[key]={m:list(v.values()) for m,v in tables.items()}
        show=['baseline','unconditional','identity_only','ours'];bounds=np.stack([np.min([g['xyz'][:,:2].min(0) for g in maps.values()],axis=0),np.max([g['xyz'][:,:2].max(0) for g in maps.values()],axis=0)])
        fig,axes=plt.subplots(1,4,figsize=(18,5),constrained_layout=True)
        for ax,m in zip(axes,show):
            g=maps[m];ii=np.arange(0,len(g['xyz']),max(1,len(g['xyz'])//100000))
            ax.scatter(g['xyz'][ii,0],g['xyz'][ii,1],c=g['instance'][ii]%20,cmap='tab20',s=.3,vmin=0,vmax=19,rasterized=True)
            ax.set_xlim(bounds[:,0]);ax.set_ylim(bounds[:,1]);ax.set_aspect('equal');ax.set_title(NAMES[m],fontsize=10);ax.set_xlabel('World x / m');ax.set_ylabel('World y / m')
        fig.suptitle(key+' / actual complete maps / common bounds / repeated palette, no cross-method new-ID alignment')
        fig.savefig(DEST/(key+'_maps.png'),dpi=160);plt.close(fig)
        views=read(OBS/'prepared'/key/'views.json');rec=views[len(views)//2];v=np.load(rec['file']);frame=next(f for f in read(c['job'])['frames'] if f['frame_id']==rec['frame_id'])
        rgb=np.asarray(Image.open(frame['rgb']).convert('RGB'));full=np.load(Path(c['full'])/'geometry.npz')['xyz'];sources=full[v['source_ids']];px=v['pixel_xy']
        fig,axes=plt.subplots(1,3,figsize=(18,3),constrained_layout=True)
        for ax,m in zip(axes,['baseline','unconditional','ours']):
            g=maps[m];labels=transfer(g['xyz'],g['instance'],sources,.05);keep=labels>0
            ax.imshow(rgb);ax.scatter(px[keep,0],px[keep,1],c=labels[keep]%20,cmap='tab20',s=1,vmin=0,vmax=19,alpha=.75);ax.set_title(NAMES[m]);ax.axis('off')
        fig.suptitle(f'{key} / real RGB {rec["frame_id"]}, fixed median input frame / measured source labels, not GT or dense masks')
        fig.savefig(DEST/(key+'_rgb.png'),dpi=160);plt.close(fig)
        visuals[key]=dict(frame_id=rec['frame_id'],selection='fixed median original frame',display_radius_m=.05,benchmark_radius_m=.2)
        rows=''
        for m in METHODS:
            r=s['results'][key][m];stat=seal['results'][key].get(m,{})
            rows+='<tr><td>'+NAMES[m]+'</td>'+''.join(f'<td>{r[mode]["metrics"]["allAp"]*100:.5f}</td>' for mode in ['primary','raw_targets','uniform_scores'])
            rows+='<td>'+html.escape(str(r['raw_targets']['changes_vs_baseline']))+'</td><td>'+str(stat.get('old_points_changed','—'))+'</td></tr>'
        parts.append(f'<h2>{key}</h2><img src="{key}_maps.png"><img src="{key}_rgb.png"><table><tr><th>Method</th><th>Primary AP %</th><th>Raw target</th><th>Uniform score</th><th>Raw AP50 gained/lost</th><th>Old labels changed</th></tr>{rows}</table>')
        for mode,table in tables.items():
            body=''
            for gid,r in table.items():body+=f'<tr><td>{gid}</td><td>{r["class_name"]}</td>'+''.join(f'<td>{r["methods"][m]["best_iou"]:.5f}</td>' for m in METHODS)+'</tr>'
            parts.append('<details><summary>Every '+mode+' object: independent best same-class IoU, not AP matching</summary><table><tr><th>GT ID</th><th>Class</th>'+''.join('<th>'+NAMES[m]+'</th>' for m in METHODS)+'</tr>'+body+'</table></details>')
    evidence=read(OUT/'prepared/kitti360_0000/evidence.json');fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,pair,title in zip(axes,['3_4','25_28'],['Legacy vehicles 3 / 4','Legacy buildings 25 / 28']):
        r=evidence[pair];obs=r['observations'];x=np.arange(len(obs))
        ax.plot(x,[v['shared'] for v in obs],'o-',label='Shared explanation',c='#159b84');ax.plot(x,[v['separate'] for v in obs],'s-',label='Separate explanation',c='#d77652')
        ax.set_xticks(x,[str(v['frame_id']) for v in obs],rotation=60,fontsize=8);ax.set_xlabel('Actual jointly observed frame');ax.set_ylim(0,1);ax.legend(fontsize=8)
        ax.set_title(title+f' / approved: {r["approved"]}\nWorst deletion margin: {r["worst_after_one_deletion"]:.3f}')
    fig.suptitle('Targeted provenance cases identified before this method / native scores are not calibrated identity probabilities')
    fig.savefig(DEST/'identity_evidence.png',dpi=160);plt.close(fig)
    mean=''.join('<tr><td>'+NAMES[m]+'</td>'+''.join(f'<td>{s["mean_ap"][mode][m]*100:.6f}</td>' for mode in ['primary','raw_targets','uniform_scores'])+'</tr>' for m in METHODS)
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam：观测证据授权的身份与地图写入</title>
<style>body{max-width:1500px;margin:40px auto;padding:0 24px;font:16px/1.6 sans-serif;color:#203748}img{width:100%}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:7px;border-bottom:1px solid #d8e3e6;text-align:right}td:first-child,th:first-child{text-align:left}.note{background:#fff1db;padding:18px}details{margin:20px 0;overflow:auto}</style>
<h1>omni_slam：观测证据授权的身份与地图写入</h1><p class="note">冻结开发门槛：'''+str(s['development_gate'])+'''。全部对照先导出再评分。两窗口已用于开发；不是独立测试或官方榜单。局部身份修复、点数变化、稳定性证明都不能替代完整地图结果。</p>
<p>旧身份只有在共享解释优于分别解释、且删除任意一个共同视图后仍成立时，才允许局部归属合并；同一接收组的每一对都需兼容，防止通过第三个对象强行传递合并。几何候选内部保留未授权合并的旧身份，并以实际掩码见证的旧身份单元为锚点竞争。新点仅写入有掩码见证的候选单元，并需两个不同观测帧的局部图认证。隐藏和未解释不是背景。原坐标、类别、分数均保持。</p>
<img src="identity_evidence.png"><p>原逐源点追溯中，176,696/150,186个新增点只有25,448/34,187个被缓存视图选中并得到自身掩码支持，重复支持的精确源点仅961/648。缓存的选择/遮挡处理不等于物理不可见，所以本版在三维单元保留观测来源，而不要求同一个稠密源点被重复选中。这是设计依据，不是一个成功精度结论。</p>
<table><tr><th>Method</th><th>Primary mean AP50:90 %</th><th>Raw-target diagnostic</th><th>Uniform-score diagnostic</th></tr>'''+mean+'</table>'+''.join(parts)+'''
<p>身份处理消融在候选内使用竞争归属，候选外保留旧提案；写入消融只检查直接单元见证而保留旧提案身份；禁止历史改写版与完整版本的新几何和标签完全相同。两者在0002均没有通过的旧身份合并，因此会出现真实相同预测，报告保留，不将同分解释为独立收益。</p>
<p>核验包括：全部局部图节点的最近对象组距离经独立SciPy计算；实际撤除4个完整图像帧种子批次；所有候选身份对逐个删视图重算最差差值；8份地图导出重放；全部实际单元-帧掩码见证用原始位掩码重新生成。对应保证限于固定图和既有证据，不代表统计独立或语义正确。本阶段未新增机器人或独立室外评测，总目标尚未完成。</p></html>'''
    (DEST/'index.html').write_text(page)
    dump(OUT/'per_object.json',dict(complete=True,results=details))
    dump(OUT/'report_audit.json',dict(complete=True,source_sha256=sha(__file__),summary_sha256=sha(OUT/'summary.json'),
        raw_witness_cell_frames_replayed=witness_counts,visuals=visuals,files={str(f):sha(f) for f in DEST.iterdir()},per_object_sha256=sha(OUT/'per_object.json')))
    print('AUTHORIZED REVISION REPORT',DEST/'index.html','raw witnesses',witness_counts,flush=True)


if __name__=='__main__':main()
