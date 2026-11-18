"""Complete crossfit result, mechanism controls and actual robot imagery."""
import numpy as np
from PIL import Image
from paper_final_common import ROOT,read,dump,sha
from run_surface_ownership import OUT,PRIOR,INPUT,METHODS,verify,sealed

REPORT=ROOT/'output/surface_ownership_v1'
NAMES=dict(baseline='Strong baseline',nearest_core='Nearest core',learned_geometry='Learned geometry',independent='Independent pairs',ours_competitive='Ours candidate: competitive ownership')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=verify(True);sealed('predictions_complete');s=read(OUT/'summary.json');training=sealed('trained');fs=sealed('features_complete')
    assert sha(OUT/'predictions_complete.json')==s['prediction_seal_sha256'];REPORT.mkdir(parents=True,exist_ok=True)
    table='';body='';images={}
    fig,axes=plt.subplots(1,3,figsize=(14,3.5))
    for ax,scene in zip(axes,p['scenes']):
        key=scene['key']
        for method in METHODS[2:]:ax.plot(np.arange(1,41),training['stats'][key+'/'+method]['loss'],label=NAMES[method])
        ax.set_title('Target: '+key);ax.set_xlabel('Epoch');ax.set_ylabel('Training loss');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(REPORT/'training.png',dpi=160);plt.close(fig)
    for scene in p['scenes']:
        key=scene['key'];rows=s['results'][key]['methods'];manifest=read(INPUT/key/'manifest.json');source=np.load(manifest['geometry'])['xyz']
        core=np.load(OUT/'maps'/key/'baseline'/'geometry.npz');n=len(core['xyz']);dec=np.load(OUT/'maps'/key/'ours_competitive'/'decisions.npz');accepted=dec['source_ids'][dec['owners']>0]
        for method in METHODS:
            row=rows[method];m=row.get('metrics');changes=row.get('changes_vs_baseline',{})
            ap=f'{100*m["allAp"]:.3f}' if m else '—';ap50=f'{100*m["allAp50%"]:.3f}' if m else '—'
            table+=f'<tr><td>{key}</td><td>{NAMES[method]}</td><td>{ap}</td><td>{ap50}</td><td>{row["run"]["added_points"]}</td><td>{len(changes.get("gained",[])) if m else "—"}</td><td>{len(changes.get("lost",[])) if m else "—"}</td></tr>'
        sample=np.linspace(0,len(source)-1,min(50000,len(source)),dtype=int);ix=np.linspace(0,n-1,min(50000,n),dtype=int)
        fig,axes=plt.subplots(1,3,figsize=(15,5),sharex=True,sharey=True)
        for ax,method in zip(axes,['baseline','nearest_core','ours_competitive']):
            g=np.load(OUT/'maps'/key/method/'geometry.npz');ax.scatter(source[sample,0],source[sample,1],s=.3,c='#e1e6ec');ax.scatter(core['xyz'][ix,0],core['xyz'][ix,1],s=.7,c='#137eab')
            if len(g['xyz'])>n:ax.scatter(g['xyz'][n:,0],g['xyz'][n:,1],s=1.1,c='#ec943c')
            ax.set_title(NAMES[method]+f'\n{len(g["xyz"])-n} added measured points',fontsize=9);ax.set_aspect('equal');ax.set_xlabel('x / m')
        axes[0].set_ylabel('y / m');fig.suptitle(key+' | Blue: immutable core; orange: accepted measured additions',fontsize=10);fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(REPORT/(key+'_maps.png'),dpi=160,bbox_inches='tight');plt.close(fig)
        # Select by prediction visibility, never by reference labels or metric.
        best=None
        for r in manifest['records']:
            measured=np.load(r['file']);use=np.isin(measured['visible_source_ids'],accepted);count=int(use.sum())
            if best is None or count>best[0]:best=(count,r,measured['pixel_xy'][use])
        count,r,px=best;rgb=np.asarray(Image.open(r['rgb']).convert('RGB'))
        fig,axes=plt.subplots(1,2,figsize=(14,4))
        for ax in axes:ax.imshow(rgb);ax.axis('off')
        axes[0].set_title('Actual RGB | frame '+str(r['frame_id']));axes[1].set_title(f'Accepted measured additions visible here: {count}')
        if len(px):axes[1].scatter(px[:,0],px[:,1],s=3,c='#ff952b',alpha=.75)
        fig.tight_layout();fig.savefig(REPORT/(key+'_rgb.png'),dpi=150,bbox_inches='tight');plt.close(fig)
        images[key]=dict(frame_id=r['frame_id'],rgb=r['rgb'],rgb_sha256=sha(r['rgb']),visible_accepted=count,selection='Most visible accepted source IDs; earliest tie. Prediction-only illustrative selection, not independent accuracy.')
        feature=fs['stats'][key]
        body+=f'<h2>{key}</h2><p>候选源点 {feature["candidate_sources"]}；被多个对象同时提议 {feature["multi_owner_sources"]}。蓝色旧地图完整不变，橙色为接受的新实测点，不表示 GT 正确性。</p><img src="{key}_maps.png"><img src="{key}_rgb.png"><p>相机帧按候选接受点的可见数量选择，未用 GT 选图；橙色投影只表示实测点可见，不代表当前掩码或独立正确标注。</p>'
        if scene['reference']:
            gt={}
            for method in METHODS:
                gt[method]={}
                for category,objects in read(OUT/'maps'/key/method/'matches.json')['window']['groundTruth'].items():
                    for obj in objects:
                        if obj['instance_id']<1000 or obj['vert_count']<100:continue
                        gt[method][obj['instance_id']]=(category,max((x['intersection']/(obj['vert_count']+x['vert_count']-x['intersection']) for x in obj['matched_pred']),default=0.))
            lines=''.join(f'<tr><td>{gid}</td><td>{gt["baseline"][gid][0]}</td>'+''.join(f'<td>{gt[m][gid][1]:.4f}</td>' for m in METHODS)+'</tr>' for gid in sorted(gt['baseline']))
            body+='<details><summary>全部 GT：最佳同类对象 IoU</summary><table><tr><th>GT</th><th>类别</th>'+''.join(f'<th>{NAMES[m]}</th>' for m in METHODS)+f'</tr>{lines}</table></details>'
        else:body+='<p><b>车库无独立标注：只报告执行、真实点云与输入图像，不报告准确率提高。</b></p>'
    decision='通过预设开发门槛；还需独立验证。' if s['advance_gate'] else '未通过预设开发门槛；保留强基线，不扩大到独立窗口或追加调参。'
    means='；'.join(f'{NAMES[m]} {100*s["mean_primary_ap"][m]:.3f}' for m in METHODS)
    (REPORT/'index.html').write_text(f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam · 实测点对象归属交叉训练</title><style>body{{font:16px/1.7 system-ui;max-width:1450px;margin:32px auto;padding:0 24px;color:#193349}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border:1px solid #d5dee5;text-align:right}}td:first-child,td:nth-child(2){{text-align:left}}img{{max-width:100%}}.note{{padding:20px;background:#edf5f7;border-left:5px solid #058e85}}code{{overflow-wrap:anywhere}}</style><h1>实测点对象归属：学习、竞争与拒绝分配</h1><p class="note"><b>{decision}</b><br>主线为持续对象级语义建图。候选仅补充既有对象的实测表面，不改变原核心点、身份或语义读出，也不发现新对象。没有在线部署或方法新颖性结论。</p><p>两个已暴露室外窗口互换训练与评测；车库模型在两窗口上训练。每个目标使用自己的其他场景模型，标准化统计也只来自训练场景。三种模型各固定 40 轮，没有验证集选轮数或参数搜索。已有开发设计接触过这两个窗口，因此交叉训练不等于独立测试。</p><h2>完整地图结果</h2><table><tr><th>目标窗口</th><th>方法</th><th>AP50:90 ×100</th><th>AP50 ×100</th><th>新增点</th><th>新增匹配</th><th>丢失匹配</th></tr>{table}</table><p>两窗口 AP50:90 均值：{means}。采用同输入、全参考域的本地 KITTI 适配评测，不是官方榜单。先封存所有预测再评分。原地图前缀逐元素一致，追加点严格来自实测源，单源最多归属一个对象。所有组均无追加最终滤波。</p><h2>必要对照与监督边界</h2><p>Nearest core 是 0.4 m 内最近候选对象核心点；Learned geometry 只用空间特征；Independent pairs 独立二分类；Ours candidate 联合比较所有候选对象与 null，要求单一归属后验超过 0.5。对象编号与 GT 不作为特征。训练时区分原参考有效路面标签与忽略标签，未知标注不作为负样本；对象核心归属不可靠的训练组排除。</p><img src="training.png"><p>曲线仅为训练损失。BCE 与集合似然目标不同，损失绝对值不能横比，也不能替代完整地图分数。</p>{body}<h2>产物与复现</h2><p><code>{OUT/'protocol.json'}</code><br><a href="summary.json">完整评分</a> · <a href="images.json">图像来源</a> · <a href="feature_stats.json">候选证据规模</a></p></html>''')
    dump(REPORT/'summary.json',s);dump(REPORT/'images.json',images);dump(REPORT/'feature_stats.json',fs['stats']);print('REPORT',REPORT/'index.html',decision,flush=True)


if __name__=='__main__':main()
