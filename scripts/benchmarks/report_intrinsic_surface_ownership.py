"""Actual maps and crossfit evidence for the coordinate-representation fix."""
import numpy as np
from PIL import Image
from paper_final_common import ROOT,read,dump,sha
from run_intrinsic_surface_ownership import OUT,PARENT,INPUT,METHODS,TRAINED,verify,sealed

REPORT=ROOT/'output/intrinsic_surface_ownership_v1'
NAMES=dict(baseline='Strong baseline',nearest_core='Nearest core',axis_geometry='Axis geometry / BCE',axis_evidence='Axis geometry + evidence',intrinsic_geometry='Intrinsic geometry / BCE',ours_intrinsic='Ours candidate: intrinsic + evidence')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=verify(True);s=read(OUT/'summary.json');seal=sealed('predictions_complete');train=sealed('trained');features=sealed('features_complete');REPORT.mkdir(parents=True,exist_ok=True)
    assert sha(OUT/'predictions_complete.json')==s['prediction_seal_sha256'];table='';body='';provenance={}
    fig,axes=plt.subplots(1,3,figsize=(14,3.5))
    for ax,scene in zip(axes,p['scenes']):
        for method in TRAINED:ax.plot(np.arange(1,41),train['stats'][scene['key']+'/'+method]['loss'],label=NAMES[method])
        ax.set_title('Target: '+scene['key']);ax.set_xlabel('Epoch');ax.set_ylabel('Train BCE');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(REPORT/'training.png',dpi=150);plt.close(fig)
    rigid_table=''
    old_audit=read(PARENT/'coordinate_audit.json')['results']
    for scene in p['scenes']:
        key=scene['key'];rows=s['results'][key]['methods'];manifest=read(INPUT/key/'manifest.json');source=np.load(manifest['geometry'])['xyz'];base=np.load(OUT/'maps'/key/'baseline'/'geometry.npz');n=len(base['xyz'])
        for method in METHODS:
            r=rows[method];metrics=r.get('metrics');changes=r.get('changes_vs_baseline',{})
            table+=f'<tr><td>{key}</td><td>{NAMES[method]}</td><td>{100*metrics["allAp"]:.3f}</td><td>{100*metrics["allAp50%"]:.3f}</td><td>{r["run"]["added_points"]}</td><td>{len(changes["gained"])}</td><td>{len(changes["lost"])}</td></tr>' if metrics else f'<tr><td>{key}</td><td>{NAMES[method]}</td><td>—</td><td>—</td><td>{r["run"]["added_points"]}</td><td>—</td><td>—</td></tr>'
        rigid_table+=f'<tr><td>{key}</td><td>Old axis + evidence</td><td>yaw90</td><td>{old_audit[key]["independent"]["changed_decisions"]}</td><td>—</td></tr>'
        for method in ['intrinsic_geometry','ours_intrinsic']:
            for transform,r in seal['rigid_invariance'][key+'/'+method].items():rigid_table+=f'<tr><td>{key}</td><td>{NAMES[method]}</td><td>{transform}</td><td>{r["changed_decisions"]}</td><td>{r["max_probability_absolute_error"]:.2e}</td></tr>'
        sample=np.linspace(0,len(source)-1,min(len(source),50000),dtype=int);ix=np.linspace(0,n-1,min(n,50000),dtype=int)
        fig,axes=plt.subplots(1,3,figsize=(15,5),sharex=True,sharey=True)
        for ax,method in zip(axes,['baseline','axis_evidence','ours_intrinsic']):
            g=np.load(OUT/'maps'/key/method/'geometry.npz');ax.scatter(source[sample,0],source[sample,1],s=.3,c='#e1e6ec');ax.scatter(base['xyz'][ix,0],base['xyz'][ix,1],s=.7,c='#137eab')
            if len(g['xyz'])>n:ax.scatter(g['xyz'][n:,0],g['xyz'][n:,1],s=1.1,c='#ec943c')
            ax.set_title(NAMES[method]+f'\n{len(g["xyz"])-n} added measurements',fontsize=9);ax.set_aspect('equal');ax.set_xlabel('x / m')
        axes[0].set_ylabel('y / m');fig.suptitle(key+' | Blue: immutable core; orange: measured additions',fontsize=10);fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(REPORT/(key+'_maps.png'),dpi=160,bbox_inches='tight');plt.close(fig)
        decisions=np.load(OUT/'maps'/key/'ours_intrinsic/decisions.npz');accepted=decisions['source_ids'][decisions['owners']>0];best=None
        for r in manifest['records']:
            measured=np.load(r['file']);use=np.isin(measured['visible_source_ids'],accepted);count=int(use.sum())
            if best is None or count>best[0]:best=(count,r,measured['pixel_xy'][use])
        count,r,px=best;rgb=np.asarray(Image.open(r['rgb']).convert('RGB'));fig,axes=plt.subplots(1,2,figsize=(14,4))
        for ax in axes:ax.imshow(rgb);ax.axis('off')
        axes[0].set_title('Actual RGB | frame '+str(r['frame_id']));axes[1].set_title(f'Accepted measurements visible here: {count}')
        if len(px):axes[1].scatter(px[:,0],px[:,1],s=3,c='#ff952b',alpha=.75)
        fig.tight_layout();fig.savefig(REPORT/(key+'_rgb.png'),dpi=150,bbox_inches='tight');plt.close(fig)
        provenance[key]=dict(rgb=r['rgb'],rgb_sha256=sha(r['rgb']),frame=r['frame_id'],selection='Largest visible accepted-source count, earliest tie; no GT used for image selection.')
        body+=f'<h2>{key}</h2><img src="{key}_maps.png"><img src="{key}_rgb.png"><p>相机帧按接受源点可见数量选择，未使用 GT；橙色表示实际接受的测量，不代表独立正确标注。三幅地图共享坐标与比例，灰色是实测环境，蓝色旧对象点不变。</p>'
        if scene['reference']:
            pergt={}
            for method in METHODS:
                pergt[method]={}
                for category,objects in read(OUT/'maps'/key/method/'matches.json')['window']['groundTruth'].items():
                    for obj in objects:
                        if obj['instance_id']<1000 or obj['vert_count']<100:continue
                        pergt[method][obj['instance_id']]=(category,max((x['intersection']/(obj['vert_count']+x['vert_count']-x['intersection']) for x in obj['matched_pred']),default=0.))
            lines=''.join(f'<tr><td>{gid}</td><td>{pergt["baseline"][gid][0]}</td>'+''.join(f'<td>{pergt[m][gid][1]:.4f}</td>' for m in METHODS)+'</tr>' for gid in sorted(pergt['baseline']))
            body+='<details><summary>全部参考对象：最佳同类 IoU</summary><table><tr><th>GT</th><th>类别</th>'+''.join(f'<th>{NAMES[m]}</th>' for m in METHODS)+f'</tr>{lines}</table></details>'
        else:body+='<p><b>车库无独立标注：只验证执行和可视化，不报告机器人准确率。</b></p>'
    decision='通过冻结开发门槛，仍须独立评测。' if s['advance_gate'] else '未通过全部冻结开发门槛；保留原强基线，不自动扩展到保留窗口。'
    mean='；'.join(f'{NAMES[m]} {100*s["mean_primary_ap"][m]:.3f}' for m in METHODS)
    (REPORT/'index.html').write_text(f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam · 坐标不变的实测点归属</title><style>body{{font:16px/1.7 system-ui;max-width:1450px;margin:32px auto;padding:0 24px;color:#193349}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border:1px solid #d5dee5;text-align:right}}td:first-child,td:nth-child(2){{text-align:left}}img{{max-width:100%}}.note{{padding:20px;background:#edf5f7;border-left:5px solid #058e85}}code{{overflow-wrap:anywhere}}</style><h1>对象归属的坐标不变表示：完整地图与机器人对照</h1><p class="note"><b>{decision}</b><br>主线为持续对象级语义建图。替换依赖世界轴的归属特征，保留相同观测、模型、训练预算和原对象地图。坐标不变性是必要性质，不能单独当作论文创新或准确率收益。</p><h2>完整地图比较</h2><table><tr><th>场景</th><th>方法</th><th>AP50:90 ×100</th><th>AP50 ×100</th><th>新增实测点</th><th>新增匹配</th><th>丢失匹配</th></tr>{table}</table><p>两窗口主指标均值：{mean}。已暴露窗口的场景交叉训练；不是独立测试或官方榜单。全部预测封存后评分，三个复制对照逐字节相同且重算分数复现。</p><h2>机制与公平对照</h2><p>用点到核心/其他核心的距离、相对质心距离、正则化 Mahalanobis 距离、径向与垂向方差、协方差特征值描述对象形状。没有选择 PCA 主轴，避免重根和主轴符号不稳定。几何对照两种表示均为 12 维，完整模型均为 20 维；统一独立 BCE、两层 32 维 MLP、种子 0、40 轮、训练场景标准化。原跨帧证据、候选源点、训练监督和拒绝门限不变。旧联合竞争不作为候选贡献继续扩展。</p><p>对象核心、身份、语义分数固定，只追加原始掩码实际包含且尚未归属的测量点。不能新增对象身份或修正已有核心错误。目标窗口标签不参与自己的模型拟合；机器人仅用两个公共训练窗口。</p><h2>实际几何的刚体坐标变换核验</h2><table><tr><th>场景</th><th>模型</th><th>坐标变换</th><th>决策变化</th><th>最大概率差</th></tr>{rigid_table}</table><p>对全部实际核心点、候选点和其他对象点重算几何特征。yaw90 为 90° 世界偏航；rigid_xyz 为三轴旋转加平移。冻结物理支持/空间单元成员，检验归属模块的坐标性质，未重跑上游分割、关联或网格构造。不是物理视角变化、噪声鲁棒性或整体 SLAM 坐标不变性声明。变换未用来选择更好输出，也没有额外地图评分。</p><h2>训练记录</h2><img src="training.png"><p>训练损失不能替代完整地图指标；固定 40 轮，没有按目标成绩选模型。</p>{body}<h2>复核资料</h2><p><code>{OUT/'protocol.json'}</code><br><a href="summary.json">完整评分</a> · <a href="invariance.json">刚体不变性核验</a> · <a href="images.json">真实图像来源</a></p></html>''')
    dump(REPORT/'summary.json',s);dump(REPORT/'invariance.json',dict(feature_checks=features['stats'],model_checks=seal['rigid_invariance']));dump(REPORT/'images.json',provenance);print('REPORT',REPORT/'index.html',decision,flush=True)


if __name__=='__main__':main()
