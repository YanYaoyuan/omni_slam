"""Report the frozen-identity intervention without promoting it to a method."""
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from run_fixed_identity_recovery_control import OUT,PARENT,METHODS,verify
from run_discovery_query_benchmark import INPUT

REPORT=ROOT/'output/fixed_identity_recovery_control_v1'
NAMES=dict(largest='Original support',all='All raw supports',pointwise='Pointwise recovery',component='Component recovery')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=verify(True);s=read(OUT/'summary.json');seal=read(OUT/'predictions_complete.json');parent=read(PARENT/'summary.json')
    assert sha(OUT/'predictions_complete.json')==s['prediction_seal_sha256']
    for f,h in seal['files'].items():assert sha(f)==h,f
    REPORT.mkdir(parents=True,exist_ok=True);table='';body='';means={m:[] for m in METHODS};adaptive={m:[] for m in METHODS};exact={}
    for scene in p['scenes']:
        key=scene['key'];rows=s['results'][key]['methods'];base=rows['largest']['run']
        assert base['baseline_bitwise_equal'];exact[key]=base['native_geometry_checks']
        for method in METHODS:
            r=rows[method]
            if 'metrics' not in r:continue
            a=r['adaptive_metrics'];f=r['metrics'];means[method].append(f['allAp']);adaptive[method].append(a['allAp'])
            table+=f'<tr><td>{key}</td><td>{NAMES[method]}</td><td>{100*a["allAp"]:.3f}</td><td>{100*f["allAp"]:.3f}</td><td>{100*a["allAp50%"]:.3f}</td><td>{100*f["allAp50%"]:.3f}</td><td>{len(r["changes_vs_original"]["gained"])}</td><td>{len(r["changes_vs_original"]["lost"])}</td></tr>'
        manifest=read(INPUT/key/'manifest.json');source=np.load(manifest['geometry'])['xyz'];sample=np.linspace(0,len(source)-1,min(60000,len(source)),dtype=int)
        choices=[(PARENT/'maps'/key/'largest','Original strong baseline'),(PARENT/'maps'/key/'component','Recovery + adaptive identity'),(OUT/'maps'/key/'component','Same recovery + frozen identity')]
        fig,axes=plt.subplots(1,3,figsize=(15,5),sharex=True,sharey=True)
        for ax,(folder,name) in zip(axes,choices):
            g=np.load(folder/'geometry.npz');xyz=g['xyz'];ids=g['instance'];cls=read(folder/'classification.json')['objects'];cars=[r['object_id'] for r in cls if r['predictions']['mean']['semantic_id']==26]
            ix=np.linspace(0,len(xyz)-1,min(60000,len(xyz)),dtype=int) if len(xyz) else np.empty(0,int)
            ax.scatter(source[sample,0],source[sample,1],s=.4,c='#dfe4e9',rasterized=True)
            ax.scatter(xyz[ix,0],xyz[ix,1],s=1,c=np.where(np.isin(ids[ix],cars),'#008eb5','#df992b'),rasterized=True)
            ax.set_aspect('equal');ax.set_title(name+f'\n{len(cls)} predicted objects',fontsize=10);ax.set_xlabel('x / m')
        axes[0].set_ylabel('y / m');fig.suptitle(key+' | Actual final maps: blue car, orange building, grey measured context',fontsize=11)
        fig.tight_layout(rect=[0,0,1,.91]);fig.savefig(REPORT/(key+'_maps.png'),dpi=160,bbox_inches='tight');plt.close(fig)
        body+=f'<h2>{key}</h2><img src="{key}_maps.png"><p>真实点云，统一坐标和比例，显示最多每组 60,000 点。原基线在 {base["native_geometry_checks"]} 次原生合并/去噪的点云哈希检查及最终坐标逐元素检查中复现。固定身份的对象数量相同是对照设计要求，不是性能收益。</p>'
        if scene['reference']:
            gt={}
            for method in METHODS:
                gt[method]={}
                for category,objects in read(OUT/'maps'/key/method/'matches.json')['window']['groundTruth'].items():
                    for obj in objects:
                        if obj['instance_id']<1000 or obj['vert_count']<100:continue
                        gt[method][obj['instance_id']]=(category,max((x['intersection']/(obj['vert_count']+x['vert_count']-x['intersection']) for x in obj['matched_pred']),default=0.))
            lines=''.join(f'<tr><td>{gid}</td><td>{gt["largest"][gid][0]}</td>'+''.join(f'<td>{gt[m][gid][1]:.4f}</td>' for m in METHODS)+'</tr>' for gid in sorted(gt['largest']))
            body+='<details><summary>固定身份路径的全部逐目标最佳同类 IoU</summary><table><tr><th>GT</th><th>类别</th>'+''.join(f'<th>{NAMES[m]}</th>' for m in METHODS)+f'</tr>{lines}</table></details>'
        else:body+='<p>车库无独立准确率标签。固定对象身份后仅检验几何更新链路与显示，不把对象数或视觉覆盖当作正确率。</p>'
    mean={m:float(np.mean(v)) for m,v in means.items()};am={m:float(np.mean(v)) for m,v in adaptive.items()}
    decision='固定身份后仍低于原强基线；单独冻结身份不能解决支持恢复问题。' if mean['component']<mean['largest'] else '固定身份的几何恢复达到或超过原基线均值；仍需检查每个场景和对象得失，不能视作新方法验收。'
    dump(REPORT/'summary.json',s);dump(REPORT/'audit.json',dict(mean_fixed_ap=mean,mean_adaptive_ap=am,exact_geometry_checks=exact,decision=decision))
    (REPORT/'index.html').write_text(f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam · 固定身份的支持恢复因果对照</title><style>body{{font:16px/1.7 system-ui;max-width:1450px;margin:32px auto;padding:0 24px;color:#193349}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border:1px solid #d5dee5;text-align:right}}td:first-child,td:nth-child(2){{text-align:left}}img{{width:100%}}.note{{padding:20px;background:#edf5f7;border-left:5px solid #058e85}}code{{overflow-wrap:anywhere}}</style><h1>固定对象身份后，恢复支持是否仍造成退步？</h1><p class="note"><b>{decision}</b><br>本轮是已有失败候选的因果诊断，不是新方法或独立测试。没有训练、提示搜索、阈值修改或新增模型推理。</p><p>完整记录原基线的观测接纳、合并、定期/最终去噪顺序；复用同一几何内核与随机种子，分别重放原支持、全部支持、逐点确认和连通域确认。对象归属、拒绝记录、最终对象保留和类别/置信度全部固定；新增可接纳但原来无归属的观测被忽略。这是一项完整身份路径干预，不能宣称只改变了一个关联因素，也不能发现原基线没有的对象。</p><h2>相同支持，身份路径自适应与固定</h2><table><tr><th>窗口</th><th>支持策略</th><th>自适应 AP50:90</th><th>固定 AP50:90</th><th>自适应 AP50</th><th>固定 AP50</th><th>固定路径相对原基线新增 GT</th><th>丢失 GT</th></tr>{table}</table><p>数值均乘 100。旧的两个开发窗口，本地相机/雷达建图适配评测，不是官方榜单。原强基线两窗口 AP50:90 均值 {100*mean["largest"]:.3f}；恢复候选自适应 {100*am["component"]:.3f}，固定身份 {100*mean["component"]:.3f}。</p>{body}<h2>可复核产物</h2><p>协议：<code>{OUT/'protocol.json'}</code><br>预测封存：<code>{OUT/'predictions_complete.json'}</code><br><a href="summary.json">完整分数与得失</a> · <a href="audit.json">对照核验</a></p></html>''')
    print('REPORT',REPORT/'index.html',decision,mean,flush=True)


if __name__=='__main__':main()
