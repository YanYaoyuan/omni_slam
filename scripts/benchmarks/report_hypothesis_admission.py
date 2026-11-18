"""Full map results and all new Ours objects, including ignored predictions."""
import numpy as np
from PIL import Image
from paper_final_common import ROOT,read,dump,sha
from run_hypothesis_admission import OUT,BASE,INPUT,EVIDENCE,METHODS,verify,sealed
from capture_prethreshold_candidates import OUT as RAW
from evaluate_kitti360_development import transfer

REPORT=ROOT/'output/hypothesis_admission_v1'
NAMES=dict(baseline='冻结强基线：不变几何',native01='原生分数 >0.1',three_frame='简单三帧确认',seed_quality='单帧质量判别',ours_evidence='Ours：跨视角证据质量判别',shuffled_evidence='打乱历史证据')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=verify();seal=sealed('predictions_complete');summary=read(OUT/'summary.json');assert summary['prediction_seal_sha256']==sha(OUT/'predictions_complete.json')
    REPORT.mkdir(parents=True,exist_ok=True);table='';figures='';newrows='';details={};keys=[s['key'] for s in p['scenes']]
    for method in METHODS:
        values=[summary['results'][k]['methods'][method]['metrics']['allAp']*100 for k in keys];mean=summary['mean_primary_ap'][method]*100
        table+=f'<tr><td>{NAMES[method]}</td>'+''.join(f'<td>{v:.3f}</td>' for v in values+[mean])+f'<td>{mean-summary["mean_primary_ap"]["baseline"]*100:+.3f}</td></tr>'
    captures=read(RAW/'predictions_complete.json')
    for s in p['scenes']:
        key=s['key'];folder=OUT/'maps'/key/'ours_evidence';base=dict(np.load(OUT/'maps'/key/'baseline/geometry.npz'));ours=dict(np.load(folder/'geometry.npz'));objects=read(folder/'classification.json')['objects'];decisions=read(folder/'decisions.json');reference=dict(np.load(s['reference']));projected=transfer(ours['xyz'],ours['instance'],reference['xyz'],.2)
        masses=np.bincount(projected,weights=reference['confidence'],minlength=max(o['object_id'] for o in objects)+1);accepted=[o for o in sorted(objects,key=lambda o:o['object_id']) if o['predictions']['mean']['semantic_id'] in [11,26] and masses[o['object_id']]>=100];pred_ids={o['object_id']:i for i,o in enumerate(accepted)};matches=read(folder/'matches.json')['window']['prediction'];by_pred={r['pred_id']:r for rows in matches.values() for r in rows};labels=dict(np.load(OUT/'training_labels'/key/'TRAINING_ONLY.npz'));meta=read(EVIDENCE/'index'/key/'hypotheses.json')
        manifest=read(INPUT/key/'manifest.json');frames={r['frame_id']:r for r in manifest['records']};raws={r['frame_id']:r for r in captures['results'][key]};source=np.load(manifest['geometry'])['xyz'];scene_details=[]
        fig,axes=plt.subplots(1,2,figsize=(13,5),sharex=True,sharey=True)
        for ax,title,g in zip(axes,['Frozen intrinsic baseline','Ours: actual map additions'],[base,ours]):
            bg=source[::max(1,len(source)//30000)];old=base['xyz'][::max(1,len(base['xyz'])//25000)];ax.scatter(bg[:,0],bg[:,1],s=.15,c='#ccd3d9');ax.scatter(old[:,0],old[:,1],s=.4,c='#2486a5')
            if title.startswith('Ours'):
                add=g['xyz'][len(base['xyz']):];ax.scatter(add[:,0],add[:,1],s=1,c='#e35b3e')
            ax.set_title(title);ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.set_aspect('equal');ax.grid(alpha=.2)
        fig.suptitle(key+' | measured map XY view | blue: frozen map, orange: additions');fig.tight_layout();mapname=key+'_maps.png';fig.savefig(REPORT/mapname,dpi=160);plt.close(fig)
        figures+=f'<h3>{key} · 完整地图 XY 投影</h3><img src="{mapname}"><p>共同灰色背景为实际测量；蓝色为冻结旧图，橙色为 Ours 新增点。颜色表示来源，不表示正确性。两面板采用相同坐标范围，没有 GT 裁剪。</p>'
        for oid,j in zip(decisions['new_object_ids'],decisions['selected_hypotheses']):
            m=meta[j];pred=by_pred.get(pred_ids.get(oid,-1));best=0.;gid=None;ignored=None;reason='reference mass <100; skipped'
            if pred is not None:
                record=next(o for o in objects if o['object_id']==oid);assert pred['confidence']==record['predictions']['mean']['score']
                options=[(g['intersection']/(g['vert_count']+pred['vert_count']-g['intersection']),g['instance_id']) for g in pred['matchedGt']]
                if options: best,gid=max(options)
                ignored=pred['voidIntersection']+sum(g['intersection'] for g in pred['matchedGt'] if g['instance_id']<1000 or g['vert_count']<100)
                ignored/=pred['vert_count'];reason='AP50 match' if best>.5 else ('AP50 ignored by evaluator' if ignored>.5 else 'no AP50 match')
            count=int((ours['instance']==oid).sum());r=dict(object_id=oid,hypothesis=j,seed_frame=m['frame_id'],semantic_id=m['semantic_id'],map_points=count,standalone_training_proxy_iou=float(labels['best_iou'][j]),final_map_best_iou=float(best),best_gt=gid,reference_mass=float(masses[oid]),ignored_fraction=ignored,evaluation_status=reason);scene_details.append(r)
            newrows+='<tr>'+''.join(f'<td>{v}</td>' for v in [key,oid,'building' if m['semantic_id']==11 else 'car',count,f'{r["standalone_training_proxy_iou"]:.3f}',f'{best:.3f}',f'{masses[oid]:.1f}',reason])+'</tr>'
            frame=frames[m['frame_id']];data=np.load(raws[m['frame_id']]['file']);rgb=np.asarray(Image.open(frame['rgb']).convert('RGB'));mask=np.unpackbits(data['masks_packed'][m['query_id']],axis=1)[:,:rgb.shape[1]].astype(bool);view=np.load(frame['file']);added_ids=ours['added_source_ids'][ours['instance'][len(base['xyz']):]==oid];pixels=view['pixel_xy'][np.isin(view['visible_source_ids'],added_ids)]
            fig,axes=plt.subplots(1,2,figsize=(13,2.5))
            for ax in axes:ax.imshow(rgb);ax.axis('off')
            overlay=np.zeros((*mask.shape,4));overlay[mask]=[1.,.55,.05,.4];axes[1].imshow(overlay);axes[1].scatter(pixels[:,0],pixels[:,1],s=1,c='#00cc96');axes[0].set_title('Actual RGB');axes[1].set_title('Original seed mask + actual added measurements')
            fig.suptitle(f'{key} | new object {oid} | frame {m["frame_id"]} | final IoU {best:.3f} | {reason}',fontsize=10);fig.tight_layout();name=f'{key}_object{oid}.png';fig.savefig(REPORT/name,dpi=160,bbox_inches='tight');plt.close(fig)
            figures+=f'<h4>新增对象 {oid} · {reason}</h4><img src="{name}">'
        details[key]=scene_details
    gate=summary['advance_gate']
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam · 对象假设接纳对照</title><style>body{{font:16px/1.7 system-ui;max-width:1450px;margin:32px auto;padding:0 24px;color:#193349}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border:1px solid #d5dfe8;text-align:right}}td:first-child{{text-align:left}}.note{{padding:20px;background:#edf5f7;border-left:5px solid #078e82}}.scroll{{overflow:auto}}</style>
<h1>对象假设接纳：完整地图对照与消融</h1><p class="note"><b>Ours 与冻结强基线的主指标相同，未通过继续门槛。</b>本轮完成6次交叉拟合、10份新地图及2份原图对照。地图并非相同缓存：Ours 在0000/0002分别新增4/2个对象，但没有新增 AP50 正确匹配。两个窗口均为已暴露开发数据，是原始相机/雷达建图任务的本地 KITTI-360 适配，不是官方榜单或独立泛化成绩。</p>
<h2>主要结果：AP50:90 ×100</h2><table><tr><th>方法</th><th>0000</th><th>0002</th><th>均值</th><th>相对强基线</th></tr>{table}</table>
<p>底图为此前冻结的不变几何版，非较弱原始图。所有新方法共享17,082个原始假设、旧对象源点保护、候选可用条件、同类抑制和源点独占。全部地图封存后才统一评分；原图分数精确复现。固定通过门槛为 Ours 超过最强对照至少2点、每窗口主指标/AP50不退步、至少新增一个正确匹配且不丢失旧匹配；实际通过状态：{gate}。</p>
<h2>为什么指标相同</h2><p>新增的建筑片段没有达到完整对象 IoU&gt;0.5；部分新车辆预测按现有评测的非目标类/忽略域规则被忽略。不能把“AP不变”理解成新增对象正确或无害。下表列出全部六个新增对象及实际地图投影后的结果，独立候选代理 IoU 与最终地图 IoU 分开。</p>
<div class="scroll"><table><tr><th>窗口</th><th>对象</th><th>类别</th><th>新增点</th><th>独立候选代理 IoU</th><th>最终地图最佳 IoU</th><th>参考权重面积</th><th>评测状态</th></tr>{newrows}</table></div>
<p>0000 新增建筑 ID 41 和 43 的最佳参考对象都是 2001，最终地图 IoU 分别约为 0.240 和 0.203。这说明接纳了多个建筑片段后，跨视角身份仍没有合并；共享的几何去重并不能代替对象关联。另一些新增车辆在真实图像中确有对应目标（如货车），但当前 car/building 评测的忽略规则使其不贡献主指标，不能据图像表现宣称 benchmark 涨点。</p><h2>表示与训练覆盖限制</h2><p>0000的所有单视角建筑候选，独立几何代理 IoU 最高为0.475，没有建筑正例；0002有7个建筑正例。故训练0000→评估0002的质量头没有建筑正例监督。本轮只学习接纳固定的单视角组件，不生成跨视角完整几何；即使识别到了正确片段，接纳模块也不能补出缺失部分。这是本轮方法的限制，不能用它证明其他完整建图机制无效。</p>
<p>单帧头9维、跨视角头23维，共享固定正则化逻辑回归；另有同类别内打乱历史的消融。特征及训练标签用原始组件，独立于已学习底图的占用状态，后者只用于最终地图分配。每个质量头只使用另一窗口的标签训练，标准化也只用训练行；这是开发交叉拟合，不代表独立测试。</p>
<h2>实际图像与地图：保留全部新增对象</h2>{figures}
<h2>决定与主线</h2><p>本版封存，保留强基线，不调整阈值再追分，也不扩展本版机器人或独立测试。当前证据要求区分“找回其他视角的有效掩码”与“形成完整对象几何”；后续方法若继续，必须解决后者并用完整地图证明收益。新机制的稳定优势、对应机器人和新独立评测仍未整体完成。</p>
<p><a href="summary.json">全部指标与逐对象得失</a> · <a href="new_objects.json">全部新增对象明细</a> · <a href="../../build/benchmarks/hypothesis_admission_v1/protocol.json">冻结协议</a> · <a href="../query_hypothesis_evidence_v1/index.html">上轮证据读取</a></p></html>'''
    (REPORT/'index.html').write_text(page);dump(REPORT/'summary.json',summary);dump(REPORT/'new_objects.json',details)
    print('REPORT',REPORT/'index.html','new objects',sum(map(len,details.values())),flush=True)


if __name__=='__main__':main()
