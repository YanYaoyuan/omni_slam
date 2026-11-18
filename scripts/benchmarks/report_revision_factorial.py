"""Controlled mechanism contrasts, real imagery, all objects and readout QA."""
from pathlib import Path
import html
import runpy
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from run_revision_factorial import OUT,PUBLIC,NEW,verify
from evaluate_kitti360_development import read_static_ply,transfer
from omni_semantic_map.kitti360_reference_identity import encode_reference
from omni_semantic_map.discovery_measured_input import robot_measured_visibility

DEST=ROOT/'output/revision_factorial_v1'
METHODS=['baseline','unconditional','authorized_history_only','identity_only','write_only','authorized_history_write','current_only','ours']
NAMES=dict(baseline='Frozen strong baseline',unconditional='Unconditional history',
           authorized_history_only='Authorized history / original new labels',identity_only='Authorized history + local new labels',
           write_only='Witness / unconditional history',authorized_history_write='Witness / authorized history / original new labels',
           current_only='Witness + local labels / old IDs locked',ours='Previous Ours: witness + authorized history + local labels')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    p=verify();s=read(OUT/'summary.json');old=read(PUBLIC/'summary.json')
    seal=read(OUT/'maps_complete.json');assert s['complete']
    if DEST.exists():
        for f,h in read(OUT/'report_audit.json')['files'].items():assert sha(f)==h,f
    assert read(OUT/'prediction_audit.json')['complete']
    for f,h in seal['files'].items():assert sha(f)==h,f
    DEST.mkdir(parents=True,exist_ok=True);results={};objects={};images=[];brute=0
    tax=runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['id2label']
    public=[c for c in p['contexts'] if c['kind']=='outdoor']
    for c in public:
        k=c['key'];results[k]={**old['results'][k],**s['results'][k]}
        ref=dict(np.load(c['reference']));raw=read_static_ply(c['raw_reference'])
        truth,_=encode_reference(raw['semantic'],raw['instance'],tax,'target_only')
        modes={'primary':ref['train_instance'],'raw_targets':truth};tables={}
        for mode,gt in modes.items():
            mass=np.bincount(gt,weights=ref['confidence'])
            ids=np.flatnonzero(np.isin(np.arange(len(mass))//1000,[2,13])&(mass>=100))
            tables[mode]={int(i):dict(gt_id=int(i),methods={}) for i in ids}
        for method in METHODS:
            if method in NEW:
                path=OUT/'maps'/k/method;evalpath=OUT/'evaluation'/k/method
            else:
                path=Path(c['ours']) if method=='baseline' else Path(c['parent'])/'maps'/k/'cover_history' if method=='unconditional' else Path(c['authorization'])/'maps'/k/method
                evalpath=PUBLIC/'evaluation'/k/method
            g=dict(np.load(path/'geometry.npz'))
            pred=np.load(evalpath/'reference_prediction.npz')['instance']
            records=read(path/'classification.json')['objects']
            classes={r['object_id']:r['predictions']['mean']['train_id'] for r in records}
            if method in NEW:
                # Fixed indices independent of GT/outcome, exact brute-force XYZ.
                selected=np.unique(np.linspace(0,len(ref['xyz'])-1,128,dtype=np.int64))
                for i in selected:
                    d2=np.sum((g['xyz']-ref['xyz'][i])**2,axis=1);best=int(d2.argmin())
                    value=int(g['instance'][best]) if d2[best]<.2**2 else 0
                    assert pred[i]==value,(k,method,int(i),int(pred[i]),value)
                    brute+=1
            pm=np.bincount(pred,weights=ref['confidence'],minlength=max(classes)+1)
            for mode,gt in modes.items():
                mass=np.bincount(gt,weights=ref['confidence'])
                for gid,row in tables[mode].items():
                    ids=np.array([o for o,t in classes.items() if t==gid//1000]);take=gt==gid
                    inter=np.bincount(pred[take],weights=ref['confidence'][take],minlength=len(pm))
                    values=inter[ids]/np.maximum(1e-12,mass[gid]+pm[ids]-inter[ids]);best=int(np.argmax(values))
                    row['methods'][method]=dict(object_id=int(ids[best]),best_iou=float(values[best]))
        objects[k]={mode:list(rows.values()) for mode,rows in tables.items()}
    # Mechanism effect at each window/support, not just a cancelling mean.
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    contrasts={}
    for col,c in enumerate(public):
        k=c['key'];contrasts[k]={}
        for row,(support,methods) in enumerate([
            ('All parent additions',['unconditional','authorized_history_only','identity_only']),
            ('Witnessed additions',['write_only','authorized_history_write','ours'])]):
            ax=axes[row,col];values=[results[k][m]['primary']['metrics']['allAp']*100 for m in methods]
            bars=ax.bar(np.arange(3),values,color=['#98a6b3','#3377a7','#11977e'])
            ax.axhline(results[k]['baseline']['primary']['metrics']['allAp']*100,c='#39424b',ls='--',lw=1,label='Strong baseline')
            ax.set_xticks(np.arange(3),['Unconditional\nhistory','Authorized history\noriginal new labels','Authorized history\nlocal new labels'])
            ax.set_ylim(0,max(values)+8);ax.set_ylabel('AP50:90 (%)');ax.set_title(k+' / '+support)
            ax.bar_label(bars,fmt='%.4f',fontsize=9);ax.legend(fontsize=8)
            contrasts[k][support]=dict(history_authorization_pp=values[1]-values[0],
                local_new_assignment_pp=values[2]-values[1])
    fig.suptitle('Separate legacy-ID authorization from new-region ownership / same frozen proposals')
    fig.savefig(DEST/'mechanism_contrasts.png',dpi=160);plt.close(fig);images.append('mechanism_contrasts.png')
    # Every context: same fixed median view, shared RGB, exact map projections.
    visual_meta=[]
    for c in p['contexts']:
        k=c['key'];job=read(c['job']);frames=job['frames'];f=frames[len(frames)//2]
        if c['kind']=='outdoor':
            obs=ROOT/'build/benchmarks/censored_object_partition_v1/prepared'/k
            views=read(obs/'views.json');rec=views[len(views)//2]
            f=next(v for v in frames if v['frame_id']==rec['frame_id']);v=dict(np.load(rec['file']))
            full=np.load(Path(c['full'])/'geometry.npz')['xyz'];source=full[v['source_ids']];xy=v['pixel_xy']
        rgb=np.asarray(Image.open(f['rgb']).convert('RGB'));fig,axes=plt.subplots(1,3,figsize=(16,2.6 if c['kind']=='outdoor' else 3.6),constrained_layout=True)
        for ax,m in zip(axes,['authorized_history_only','authorized_history_write','ours']):
            path=OUT/'maps'/k/m if m in NEW else Path(c['authorization'])/'maps'/k/'ours'
            g=dict(np.load(path/'geometry.npz'));ax.imshow(rgb)
            if c['kind']=='outdoor':
                labels=transfer(g['xyz'],g['instance'],source,.05);known=labels>0;px=xy[known];lab=labels[known]
            else:
                ids,px,_=robot_measured_visibility(g['xyz'],np.load(f['cloud'])['points'],f);lab=g['instance'][ids]
            ax.scatter(px[:,0],px[:,1],c=lab%20,cmap='tab20',vmin=0,vmax=19,s=2,alpha=.8)
            ax.set_title({'authorized_history_only':'Authorized history / original new labels',
                          'authorized_history_write':'Same + cell witness','ours':'Previous Ours + local new labels'}[m],fontsize=9);ax.axis('off')
        fig.suptitle(f'{k} / actual median RGB {f["frame_id"]} / source projection, not segmentation GT')
        name=k+'_rgb.png';fig.savefig(DEST/name,dpi=150);plt.close(fig);images.append(name)
        visual_meta.append(dict(key=k,frame_id=f['frame_id'],selection='fixed median cached frame',
            public_display_transfer_m=.05 if c['kind']=='outdoor' else None,public_score_transfer_m=.2 if c['kind']=='outdoor' else None))
    mean={mode:{m:float(np.mean([r[m][mode]['metrics']['allAp'] for r in results.values()])) for m in METHODS}
          for mode in ['primary','raw_targets','uniform_scores']}
    body=''
    for m in METHODS:
        body+='<tr><td>'+NAMES[m]+'</td>'+''.join(f'<td>{results[c["key"]][m]["primary"]["metrics"]["allAp"]*100:.6f}</td>' for c in public)
        body+=''.join(f'<td>{mean[mode][m]*100:.6f}</td>' for mode in ['primary','raw_targets','uniform_scores'])+'</tr>'
    sections=''
    for k,rows in objects.items():
        for mode,values in rows.items():
            tbody=''.join(f'<tr><td>{r["gt_id"]}</td>'+''.join(f'<td>{r["methods"][m]["best_iou"]:.5f}</td>' for m in METHODS)+'</tr>' for r in values)
            sections+='<details><summary>'+k+' / '+mode+' / every eligible object (best same-class IoU, not AP matching)</summary><table><tr><th>GT ID</th>'+''.join('<th>'+NAMES[m]+'</th>' for m in METHODS)+'</tr>'+tbody+'</table></details>'
    outcomes=''
    for c in public:
        k=c['key']
        for m in NEW:
            r=results[k][m]['raw_targets']['changes_vs_baseline']
            outcomes+='<tr><td>'+k+'</td><td>'+NAMES[m]+'</td><td>'+html.escape(str(r['gained']))+'</td><td>'+html.escape(str(r['lost']))+'</td></tr>'
    text='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam：分离旧身份保护与新增区域归属</title>
<style>body{font:16px/1.7 sans-serif;max-width:1500px;margin:36px auto;padding:0 24px;color:#203748}img{width:100%}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:7px;border-bottom:1px solid #d3dce2;text-align:right}td:first-child,th:first-child{text-align:left}.note{background:#fff2da;padding:18px}details{overflow:auto;margin:18px 0}pre{white-space:pre-wrap}</style>
<h1>omni_slam：分离旧身份保护与新增区域归属</h1>
<p class="note">固定的6份机制对照已导出并核验，其中4份公共窗口预测完成12次评分，2份为同版本车库对照。没有新模型、训练、参数搜索或未使用场景。本轮是事后机制对照，不作为新颖性声明或通过验收的正式Ours；原失败门槛保持不变。</p>
<p>只改变旧点历史标签时，完整新增坐标、来源ID、标签和分类分数逐项不变。与此前完整候选相比，只改变新增区域重分配/认证时，全部旧点标签保持不变。后者包括未认证点弃权，因此不是严格只改标签的对照。</p>
<img src="mechanism_contrasts.png"><table><tr><th>方法</th><th>0000 AP%</th><th>0002 AP%</th><th>主均值</th><th>原始类别诊断</th><th>统一分数诊断</th></tr>'''+body+'''</table>
<p>两种新增对照均未满足冻结的各窗口AP/AP50不退步、平均至少提升1个百分点且不丢原匹配的要求。只授权历史版恢复3处原始类别建筑匹配但仍损失1辆车；带写入见证版新增2、损失1。不能用恢复的局部例子或微小均值涨幅改变验收结论。</p>
<p>在相同写入范围下，局部新增归属对0000和0002作用相反，平均数几乎抵消。这解释了“均值看起来一样”但实际决策不同的情况。车库只授权历史版与已有cover_current完全相同，因为全部旧身份合并均被拒绝；报告如实保留，未假称独立收益。</p>
<table><tr><th>窗口</th><th>对照</th><th>新增原始类别匹配</th><th>丢失匹配</th></tr>'''+outcomes+'</table>'+''.join('<img src="'+f+'">' for f in images[1:])+sections+'''
<p>真实RGB按每场景缓存中位帧选取，彩点是实测来源投影，颜色循环使用。公共展示最近点半径0.05米，评分为0.2米。车库使用当前点云深度门控，没有独立GT，不报告准确率。</p>
<p>下一机制需要区分：旧身份是否应修订，以及新区域应归入旧对象、保留新对象假设还是保持未知。此处只完成缺失的控制实验，没有据GT挑选对象拼接赢家，也未自动启动另一个变体。</p></html>'''
    (DEST/'index.html').write_text(text)
    dump(OUT/'per_object.json',dict(complete=True,results=objects))
    dump(OUT/'report_audit.json',dict(complete=True,source_sha256=sha(__file__),
         brute_force_reference_checks=brute,contrasts=contrasts,mean_ap=mean,visuals=visual_meta,
         summary_sha256=sha(OUT/'summary.json'),per_object_sha256=sha(OUT/'per_object.json'),
         files={str(f):sha(f) for f in DEST.iterdir()},goal_complete=False))
    print('FACTORIAL REPORT COMPLETE',DEST/'index.html','brute checks',brute,'contrasts',contrasts,flush=True)


if __name__=='__main__':
    main()
