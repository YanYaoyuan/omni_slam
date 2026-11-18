"""Independent windows, developer-selected candidate, and the matching robot arm."""
import numpy as np
from PIL import Image
from paper_final_common import ROOT,read,dump,sha
from run_intrinsic_public_holdout import OUT,INPUT,DEV,METHODS,verify,sealed

REPORT=ROOT/'output/intrinsic_public_holdout_v1'
NAMES=dict(baseline='Strong baseline',nearest_core='Nearest core',axis_geometry='Axis geometry',axis_evidence='Axis + temporal evidence',ours_intrinsic_geometry='Ours: intrinsic geometry',intrinsic_evidence='Intrinsic + temporal evidence')


def figure(key,map_root,input_root,choices,output_prefix):
    import matplotlib.pyplot as plt
    manifest=read(input_root/key/'manifest.json');source=np.load(manifest['geometry'])['xyz'];base=np.load(map_root/key/'baseline/geometry.npz');n=len(base['xyz']);sample=np.linspace(0,len(source)-1,min(50000,len(source)),dtype=int);ix=np.linspace(0,n-1,min(n,50000),dtype=int)
    span=np.ptp(source[:,:2],axis=0);height=float(np.clip(4.3*span[1]/max(span[0],1e-6)+1.,3.,6.))
    fig,axes=plt.subplots(1,3,figsize=(15,height),sharex=True,sharey=True)
    for ax,(method,name) in zip(axes,choices):
        g=np.load(map_root/key/method/'geometry.npz');ax.scatter(source[sample,0],source[sample,1],s=.3,c='#e1e6ec');ax.scatter(base['xyz'][ix,0],base['xyz'][ix,1],s=.7,c='#137eab')
        if len(g['xyz'])>n:ax.scatter(g['xyz'][n:,0],g['xyz'][n:,1],s=1.1,c='#ec943c')
        ax.set_title(name+f'\n{len(g["xyz"])-n} added measurements',fontsize=9);ax.set_aspect('equal');ax.set_xlabel('x / m')
    axes[0].set_ylabel('y / m');fig.suptitle(key+' | Blue: original object map; orange: accepted measured additions',fontsize=10);fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(REPORT/(output_prefix+'_maps.png'),dpi=160,bbox_inches='tight');plt.close(fig)
    primary=choices[-1][0];decision=np.load(map_root/key/primary/'decisions.npz');accepted=decision['source_ids'][decision['owners']>0];best=None
    for r in manifest['records']:
        z=np.load(r['file']);use=np.isin(z['visible_source_ids'],accepted);count=int(use.sum())
        if best is None or count>best[0]:best=(count,r,z['pixel_xy'][use])
    count,r,px=best;image=np.asarray(Image.open(r['rgb']).convert('RGB'));fig,axes=plt.subplots(1,2,figsize=(14,4))
    for ax in axes:ax.imshow(image);ax.axis('off')
    axes[0].set_title('Actual RGB | frame '+str(r['frame_id']));axes[1].set_title(f'Ours accepted measurements visible here: {count}')
    if len(px):axes[1].scatter(px[:,0],px[:,1],s=3,c='#ff952b',alpha=.75)
    fig.tight_layout();fig.savefig(REPORT/(output_prefix+'_rgb.png'),dpi=150,bbox_inches='tight');plt.close(fig)
    return dict(frame_id=r['frame_id'],rgb=r['rgb'],sha256=sha(r['rgb']),accepted_visible=count,selection='Largest accepted-point visibility count; earliest tie. No GT used for image selection.')


def main():
    import matplotlib
    matplotlib.use('Agg')
    p=verify(True);s=read(OUT/'summary.json');seal=sealed('predictions_complete');assert sha(OUT/'predictions_complete.json')==s['prediction_seal_sha256'];REPORT.mkdir(parents=True,exist_ok=True)
    table='';body='';images={};per_objects={};class_table=''
    for scene in p['scenes']:
        key=scene['key'];rows=s['results'][key]['methods'];per_objects[key]={}
        for method in METHODS:
            row=rows[method];m=row['metrics'];change=row['changes_vs_baseline'];table+=f'<tr><td>{key}</td><td>{NAMES[method]}</td><td>{100*m["allAp"]:.3f}</td><td>{100*m["allAp50%"]:.3f}</td><td>{row["run"]["added_points"]}</td><td>{len(change["gained"])}</td><td>{len(change["lost"])}</td></tr>'
            for category,v in m.get('classes',{}).items():class_table+=f'<tr><td>{key}</td><td>{NAMES[method]}</td><td>{category}</td><td>{100*v["ap"]:.3f}</td><td>{100*v["ap50%"]:.3f}</td></tr>'
            for category,objects in read(OUT/'maps'/key/method/'matches.json')['window']['groundTruth'].items():
                for obj in objects:
                    if obj['instance_id']<1000 or obj['vert_count']<100:continue
                    entry=per_objects[key].setdefault(obj['instance_id'],dict(category=category));entry[method]=max((r['intersection']/(obj['vert_count']+r['vert_count']-r['intersection']) for r in obj['matched_pred']),default=0.)
        images[key]=figure(key,OUT/'maps',INPUT,[(m,NAMES[m]) for m in ['baseline','nearest_core','ours_intrinsic_geometry']],key)
        body+=f'<h2>{key}：独立窗口</h2><img src="{key}_maps.png"><img src="{key}_rgb.png"><p>共享相同输入与旧对象核心。橙色表示新增实测点，不表示 GT 正确性；真实图像按接受点可见数量选取，未按分数或参考标签选图。</p>'
        lines=''.join(f'<tr><td>{gid}</td><td>{r["category"]}</td>'+''.join(f'<td>{r[m]:.4f}</td>' for m in METHODS)+'</tr>' for gid,r in sorted(per_objects[key].items()))
        body+='<details><summary>全部参考对象的最佳同类 IoU</summary><table><tr><th>GT</th><th>类别</th>'+''.join(f'<th>{NAMES[m]}</th>' for m in METHODS)+f'</tr>{lines}</table></details>'
    robot='rosbag2245';measured=ROOT/'build/benchmarks/discovery_measured_inputs_v1';images[robot]=figure(robot,DEV/'maps',measured,[('baseline','Strong baseline'),('nearest_core','Nearest core'),('intrinsic_geometry','Ours: intrinsic geometry')],robot)
    robot_run=read(DEV/'maps'/robot/'intrinsic_geometry/run.json');body+=f'<h2>真实机器狗车库：同一简化模型</h2><img src="{robot}_maps.png"><img src="{robot}_rgb.png"><p>330 帧真实 LIVO 输入；保持原 {robot_run["objects"]} 个预测对象，追加 {robot_run["added_points"]} 个实测源点。这里展示与独立评测同一已冻结模型的已有机器人执行结果，没有重新选模型。无独立机器人标签，不报告准确率提高或 S100 实时性能。</p>'
    means=''.join(f'<tr><td>{NAMES[m]}</td><td>{100*s["mean_primary_ap"][m]:.3f}</td></tr>' for m in METHODS)
    gain=100*(s['mean_primary_ap']['ours_intrinsic_geometry']-s['mean_primary_ap']['baseline']);decision='通过预登记的独立确认门槛。' if s['independent_gate'] else '未通过预登记的独立确认门槛。'
    (REPORT/'index.html').write_text(f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam · 新室外窗口独立确认</title><style>body{{font:16px/1.7 system-ui;max-width:1450px;margin:32px auto;padding:0 24px;color:#193349}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border:1px solid #d5dee5;text-align:right}}td:first-child,td:nth-child(2){{text-align:left}}img{{max-width:100%}}.note{{padding:20px;background:#edf5f7;border-left:5px solid #058e85}}code{{overflow-wrap:anywhere}}</style><h1>omni_slam：不变几何归属的独立场景确认</h1><p class="note"><b>{decision}</b><br>Ours 为在开发消融后选定的几何简化版；相对基线的独立两窗口 AP50:90 均值变化 {gain:+.3f} 点，新增 {s["gained_gt"]} 个正确匹配，丢失 {s["lost_gt"]} 个。主候选身份、阈值和权重在读取新参考之前固定，结果不用于换方法。</p><h2>选择与独立性</h2><p>原完整跨帧候选未通过开发门槛，失败保留。简化版在开发窗口主指标提高 2.750 点，但有 3 增 / 1 失；依据这项开发证据，另行冻结本次确认。它不是预先指定的开发赢家，也不把旧失败改判成功。两新窗口来自此前按划分/位姿元数据保留的 0004 与 0005，共 46 + 49 帧。所有十二份预测封存后才解码两份参考。无新拟合、阈值搜索、提示变化或替换窗口。</p><p>训练监督来自已暴露的 0000/0002 窗口，本次属于自定义场景划分下的独立确认。不是官方训练/验证协议或服务器榜单，也不是整段行驶序列全量评测。输入是相机与原始雷达的采样建图，不能直接横比官方累积 XYZRGB 实例分割榜单。</p><h2>全部方法：完整参考域</h2><table><tr><th>新窗口</th><th>方法</th><th>AP50:90 ×100</th><th>AP50 ×100</th><th>新增点</th><th>新增匹配</th><th>丢失匹配</th></tr>{table}</table><h2>两个独立窗口均值</h2><table><tr><th>方法</th><th>AP50:90 ×100</th></tr>{means}</table><details><summary>按类别分项</summary><table><tr><th>窗口</th><th>方法</th><th>类别</th><th>AP50:90</th><th>AP50</th></tr>{class_table}</table></details><p>相同前端 SAM3、SigLIP 描述子、原 CG/history2、原语义置信度；仅条件归属模块不同。Ours 使用坐标不变的 12 维几何表示与原固定小型 MLP。实测源点只能分给原掩码支持的对象，允许拒绝；旧核心、身份和类别不变。几何不变性是必要性质，不单独作为创新声明。</p>{body}<h2>复核与下一步边界</h2><p>两种固定刚体变换的归属检查：{s["rigid_invariance_pass"]}。仅检验归属模块，冻结物理支持与空间单元成员，没有重跑上游分割/关联。新结果全部保留，不做新窗口上的方法选择或参数迭代。</p><p><code>{OUT/'protocol.json'}</code><br><a href="summary.json">完整结果与门槛</a> · <a href="per_object.json">全部逐对象 IoU</a> · <a href="images.json">真实图像来源</a> · <a href="../intrinsic_surface_ownership_v1/index.html">开发对照（独立于本表）</a></p></html>''')
    dump(REPORT/'summary.json',s);dump(REPORT/'per_object.json',per_objects);dump(REPORT/'images.json',images);print('REPORT',REPORT/'index.html',decision,flush=True)


if __name__=='__main__':main()
