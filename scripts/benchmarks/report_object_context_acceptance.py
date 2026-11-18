"""Report the fixed trial, including actual imagery, all objects and score ties."""
import html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from paper_final_common import ROOT,read,dump,sha
from run_object_context_acceptance import OUT,CONTEXT,INPUT,METHODS,KEYS,verify,sealed

DEST=ROOT/'output/object_context_acceptance_v1'
NAMES=dict(baseline='强基线',expanded_geometry='扩大候选＋纯几何',simple_context='简单类别排除',category_context='类别条件化',ours_object_context='Ours：对象条件化')


def main():
    p=verify();seal=sealed('predictions_complete');s=read(OUT/'summary.json');audit=read(OUT/'artifact_audit.json');details=read(OUT/'object_diagnostics.json')['results']
    assert audit['prediction_seal_sha256']==s['prediction_seal_sha256']==sha(OUT/'predictions_complete.json')
    DEST.mkdir(parents=True,exist_ok=True);plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    score_rows='';change_rows='';quality_rows='';figures='';images=[];ties={}
    for m in METHODS:
        values=[s['results'][k]['methods'][m]['metrics']['allAp']*100 for k in KEYS]+[s['mean_primary_ap'][m]*100,(s['mean_primary_ap'][m]-s['mean_primary_ap']['baseline'])*100]
        score_rows+='<tr'+(' class="ours"' if m=='ours_object_context' else '')+'><td>'+NAMES[m]+'</td>'+''.join(f'<td>{v:.3f}</td>' for v in values)+'</tr>'
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    english=['Baseline','Geometry','Simple context','Category context','Ours: object context']
    for ax,key in zip(axes,KEYS):
        values=[100*s['results'][key]['methods'][m]['metrics']['allAp'] for m in METHODS]
        bars=ax.bar(np.arange(5),values,color=['#526a7d','#8ba2ad','#d5b96e','#7599c7','#15958a']);ax.bar_label(bars,fmt='%.3f',fontsize=8)
        ax.set(xticks=np.arange(5),xticklabels=english,ylim=(0,max(values)*1.15),ylabel='Full-map AP50:90 x100',title=key+' / exposed development')
        ax.tick_params(axis='x',labelrotation=23)
    fig.savefig(DEST/'full_map_ap.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3.5),layout='constrained')
    for target in KEYS:
        for method in ['category_context','ours_object_context']:
            tr=read(OUT/'models'/target/method/'training.json');ax.plot(np.arange(1,41),tr['loss'],label=target[-4:]+'/'+method.replace('_context',''))
    ax.set(xlabel='Fixed training epoch',ylabel='Training BCE',title='Four fixed fits; no checkpoint selection / not validation accuracy');ax.legend(fontsize=8)
    fig.savefig(DEST/'training.png',dpi=150);plt.close(fig)
    for key in KEYS:
        ties[key]={}
        for a,b in [('expanded_geometry','simple_context'),('category_context','ours_object_context')]:
            aa={(o['object_id'],q['gt_id']):q['iou'] for o in details[key][a] for q in o['overlaps'] if o['reference_mass']>=100}
            bb={(o['object_id'],q['gt_id']):q['iou'] for o in details[key][b] for q in o['overlaps'] if o['reference_mass']>=100}
            pairs=set(aa)|set(bb)
            ties[key][a+'/'+b]=dict(changed_ious=sum(abs(aa.get(v,0)-bb.get(v,0))>1e-12 for v in pairs),
                strict_iou_threshold_flips={str(round(float(t),2)):int(sum((aa.get(v,0)>t)!=(bb.get(v,0)>t) for v in pairs)) for t in np.arange(.5,.95,.05)})
        for method in METHODS:
            row=s['results'][key]['methods'][method];delta=row['changes_vs_baseline']
            change_rows+='<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [key,NAMES[method],row['run']['added_points'],delta['gained'],delta['lost']])+'</tr>'
            if method!='baseline':
                q=audit['source_quality'][key][method]
                quality_rows+='<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [key,NAMES[method],q['known_added'],q['unknown_added'],f"{100*q['weighted_precision']:.2f}%",f"{100*q['candidate_positive_recall']:.2f}%"])+'</tr>'
        source=np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz'];base=dict(np.load(OUT/'maps'/key/'baseline/geometry.npz'));n=len(base['xyz'])
        fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
        limits=[(float(source[:,d].min()),float(source[:,d].max())) for d in [0,1]]
        for ax,method in zip(axes,['baseline','category_context','ours_object_context']):
            g=dict(np.load(OUT/'maps'/key/method/'geometry.npz'));ax.scatter(source[::10,0],source[::10,1],c='#dce3e7',s=.15,rasterized=True)
            ax.scatter(g['xyz'][:n,0],g['xyz'][:n,1],c='#526a7d',s=.25,rasterized=True)
            ax.scatter(g['xyz'][n:,0],g['xyz'][n:,1],c='#00a88e',s=1,rasterized=True)
            ax.set(xlim=limits[0],ylim=limits[1],xlabel='x (m)',ylabel='y (m)',title=method);ax.set_aspect('equal')
        fig.suptitle(key+': same full extent; gray=measured scene, blue=old objects, green=additions (NOT GT)')
        name=key+'_maps.png';fig.savefig(DEST/name,dpi=150);plt.close(fig);figures+=f'<img src="{name}">'
        # First frame with16common proposal sources, independent of predictions/GT.
        candidates=np.load(OUT/'features'/key/'candidates.npz')['source_ids'];manifest=read(INPUT/key/'manifest.json');records=read(CONTEXT/'predictions_complete.json')['results'][key]
        category=dict(np.load(OUT/'maps'/key/'category_context/decisions.npz'));ours=dict(np.load(OUT/'maps'/key/'ours_object_context/decisions.npz'))
        for rec in records:
            d=dict(np.load(rec['file']));take=np.isin(d['source_ids'],candidates)
            if take.sum()<16:continue
            r=next(v for v in manifest['records'] if v['frame_id']==rec['frame_id']);rgb=np.asarray(Image.open(r['rgb']).convert('RGB'));xy=d['pixel_xy'][take]
            ix=np.searchsorted(candidates,d['source_ids'][take]);a=category['owners'][ix];b=ours['owners'][ix];score=d['point_scores'][take];native=d['semantic_ids'][score.argmax(1)];known=score.max(1)>0
            colors=np.where(~known,'#e6c34e',np.where(native==26,'#20a5d0',np.where(native==11,'#e79b47','#d7d7d7')))
            result=np.where((a>0)&(b==a),'#19bba2',np.where(b>0,'#368bec',np.where(a>0,'#eb8b44','#aeb7be')))
            fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
            for ax in axes:ax.imshow(rgb);ax.axis('off')
            axes[0].set_title('Actual RGB');axes[1].scatter(xy[:,0],xy[:,1],s=2,c=colors,linewidths=0);axes[1].set_title('Native category winner\nCar blue | building orange | context gray',fontsize=10)
            axes[2].scatter(xy[:,0],xy[:,1],s=2,c=result,linewidths=0);axes[2].set_title('Admission comparison\nBoth green | Ours differs blue | category-only orange',fontsize=10)
            fig.suptitle(f'{key} frame {rec["frame_id"]}; common candidate sources, no GT shown')
            name=key+'_actual.png';fig.savefig(DEST/name,dpi=150);plt.close(fig);figures+=f'<img src="{name}">'
            images.append(dict(key=key,frame_id=rec['frame_id'],rgb=r['rgb'],rgb_sha256=sha(r['rgb']),selection='first frame with>=16common candidate sources, no predictions/GT selection',file=name));break
    dump(DEST/'threshold_audit.json',dict(complete=True,results=ties,source_sha256=sha(__file__)))
    # Every original object gets a row; this is not a cherry-picked failure list.
    object_html=''
    for key in KEYS:
        keyed={m:{o['object_id']:o for o in details[key][m]} for m in METHODS}
        rows=''
        for oid,base in keyed['baseline'].items():
            gid=base['overlaps'][0]['gt_id'] if base['overlaps'] else None
            vals=[]
            for m in METHODS:
                q=next((v for v in keyed[m][oid]['overlaps'] if v['gt_id']==gid),None);vals.append(q['iou'] if q else 0.)
            rows+=f'<tr><td>{oid}</td><td>{gid}</td>'+''.join(f'<td>{v:.4f}</td>' for v in vals)+'</tr>'
        object_html+=f'<h2>{key}</h2><table><tr><th>对象ID</th><th>原最佳GT</th>'+''.join('<th>'+NAMES[m]+'</th>' for m in METHODS)+f'</tr>{rows}</table>'
    style='<style>body{font:16px/1.7 system-ui;color:#1f3543;max-width:1400px;margin:30px auto;padding:0 24px}img{width:100%}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #d5dfe5;padding:7px}.ours{background:#e7f5f1;font-weight:bold}.note{padding:18px;background:#fff4dc}a{color:#176b89}</style>'
    (DEST/'objects.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>所有对象IoU</title>'+style+'<h1>全部对象：相对原最佳GT的完整参考IoU</h1><p>诊断表保留全部ID；原最佳GT并非一定正确匹配，也未用于新预测或排序。正式AP使用原评测器。</p>'+object_html+'</html>')
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam 对象条件化接纳</title>{style}
<h1>omni_slam：对象条件化环境证据接纳</h1><p class="note"><b>未通过冻结验收，封存本版，不替换强基线。</b>19类证据、四次固定拟合、十份完整地图均完成。Ours均值21.843，强基线23.134；没有新增AP50正确实例，0000丢失车辆13024。没有扩展机器人或新增独立场景，也没有重新调参。</p>
<h2>完整地图主指标 AP50:90 ×100</h2><table><tr><th>方法</th><th>0000</th><th>0002</th><th>均值</th><th>相对强基线</th></tr>{score_rows}</table><img src="full_map_ap.png">
<p>两窗口均已反复用于开发。旧基线继承两窗口交叉训练监督，因此这不是无泄漏的嵌套交叉验证。新拟合仅直接读取另一窗口标签，但不能据此宣称独立泛化。采用统一的本地全参考KITTI适配协议，不是官方榜单。</p>
<h2>五组是否真的不同</h2><table><tr><th>窗口</th><th>方法</th><th>新增点</th><th>新增AP50 GT</th><th>丢失AP50 GT</th></tr>{change_rows}</table>
<p>纯几何与简单类别排除分别改变1388/5357个源点决策；类别与对象条件化改变1909/2680个。地图哈希均不同。前两者在两个窗口各有12个预测—GT IoU发生变化，但均未跨越0.50至0.90的评测阈值，所以主AP同分。0000的类别/对象条件化有35个IoU变化，同样没有阈值跨越，故同分；0002存在跨越，AP确实不同。没有复用同一预测冒充消融。</p>
<h2>哪里改善，为什么仍未验收</h2><p>0000已知参考候选中，Ours新增点的置信度加权正确率92.24%、候选正点召回81.60%，高于纯几何的90.08%/38.28%，也高于类别条件化90.98%/71.46%。但这是有已知源点/对象监督的子集，不是所有新增点精度，更不是完整地图AP。</p>
<table><tr><th>窗口</th><th>方法</th><th>已知新增点</th><th>未知新增点</th><th>已知点加权正确率</th><th>候选正点召回</th></tr>{quality_rows}</table>
<p>0000车辆13024的最佳对象仍为ID2：原IoU0.5135→Ours0.4966。正确参考交集只增加17，而预测参考面积增加437，越过0.5门槛丢失匹配。0002车辆13002的对象ID27从0.9040降至0.5487。局部总体点正确率改善，不能掩盖个别对象被明显污染；0002已知子集也退化，且6325个新增点不在可靠监督范围内。</p>
<h2>输入、训练与真实结果</h2><p>19个标准类别，108次图像编码、2052次解码；car/building全部旧查询分数、框与全图二值掩码精确复现。两学习模型使用相同50维输入结构、32/32隐藏层、40轮和随机种子；仅将对象核心改为同类平均核心作为条件化消融。6项针对性测试通过，全部旧XYZ/ID/分类记录、源点独占追加和固定模型推理已逐项复核。</p><img src="training.png">{figures}
<h2>全部对象与复核文件</h2><p><a href="objects.html">全部83个对象的IoU</a> · <a href="summary.json">完整指标</a> · <a href="artifact_audit.json">推理和导出审计</a> · <a href="threshold_audit.json">同分阈值审计</a> · <a href="../../build/benchmarks/object_context_acceptance_v1/protocol.json">冻结协议</a></p>
<p>新机制及其机器人、独立场景有效性仍未完成。此结果没有支持把对象条件化接纳作为论文核心贡献，也不自动开启另一个参数或特征变体。</p></html>'''
    (DEST/'index.html').write_text(body)
    for name in ['summary.json','artifact_audit.json','object_diagnostics.json']:dump(DEST/name,read(OUT/name))
    dump(DEST/'images.json',images)
    dump(DEST/'manifest.json',dict(generator_sha256=sha(__file__),files={f.name:sha(f) for f in DEST.iterdir() if f.is_file() and f.name!='manifest.json'}))
    print('REPORT',DEST/'index.html',flush=True)


if __name__=='__main__':main()
