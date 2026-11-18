"""Complete development results, explicit density diagnostic and actual images."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/omni_batch_transport_mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from paper_final_common import ROOT,read,dump,sha
from run_batch_stable_transport import OUT,FULL,OLD,METHODS,verify,seal
from control_batch_transport_density import OUT as DENSITY

DEST=ROOT/'output/batch_stable_transport_v1'
NAMES=dict(baseline='Frozen baseline',full_nearest='Full / nearest seed',full_two_cells='Full / two seed cells',camera_frame='Camera / frame certificate',ours_frame='Ours / frame certificate',same_cell='Same-cell lift (posthoc)')
ALL=METHODS+['same_cell']


def folder(key,method):return DENSITY/'maps'/key if method=='same_cell' else OUT/'maps'/key/method


def run():
    p=verify();seal('prepared');seal('predictions_complete');s=read(OUT/'summary.json');den=read(DENSITY/'summary.json');assert den['complete']
    assert read(OUT/'artifact_audit.json')['complete'];DEST.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10});all_objects={};images=[];cases=[]
    means={**s['mean_primary_ap'],'same_cell':den['mean_primary_ap']}
    fig,ax=plt.subplots(figsize=(11,4.5),layout='constrained');values=[means[m]*100 for m in ALL];bars=ax.barh(np.arange(len(ALL)),values,color=['#8393a2','#687da8','#739fba','#b3bec6','#009c93','#b9b2a0'])
    for bar,m,v in zip(bars,ALL,values):ax.text(v+.25,bar.get_y()+bar.get_height()/2,f'{v:.3f}  ({v-values[0]:+.3f})',va='center')
    ax.set_yticks(np.arange(len(ALL)),[NAMES[m] for m in ALL]);ax.invert_yaxis();ax.set_xlim(0,max(values)+8);ax.set_xlabel('Mean AP50:90 × 100; Δ versus frozen baseline');ax.grid(axis='x',alpha=.15);ax.set_title('Two exposed KITTI-360 windows | local adapted evaluation')
    file=DEST/'metrics.png';fig.savefig(file,dpi=160);plt.close(fig);images.append(file.name)
    for scene in p['scenes']:
        key=scene['key'];ref=np.load(scene['reference']);truth=ref['train_instance'];weight=ref['confidence'].astype(float)
        mass=np.bincount(truth,weights=weight);gids=np.flatnonzero((mass>=100)&np.isin(np.arange(len(mass))//1000,[2,13]));rows={int(g):dict(gt=int(g),category='building' if g//1000==2 else 'car',methods={}) for g in gids}
        for method in ALL:
            f=folder(key,method);pred=np.load(f/'reference_prediction.npz')['instance'];objects=read(f/'classification.json')['objects'];pmass=np.bincount(pred,weights=weight)
            joint=np.bincount(pred.astype(np.int64)*len(mass)+truth,weights=weight,minlength=len(pmass)*len(mass)).reshape(-1,len(mass))
            for gid in gids:
                best=dict(iou=0.,predicted_id=None)
                for obj in objects:
                    oid=obj['object_id']
                    if oid>=len(pmass) or obj['predictions']['mean']['train_id']!=gid//1000:continue
                    inter=joint[oid,gid];iou=float(inter/max(1e-12,mass[gid]+pmass[oid]-inter))
                    if iou>best['iou']:best=dict(iou=iou,predicted_id=oid)
                rows[int(gid)]['methods'][method]=best
        all_objects[key]=list(rows.values());manifest=read(OLD/key/'manifest.json');record=manifest['records'][len(manifest['records'])//2]
        source=np.load(FULL/key/'geometry.npz')['xyz'];low=np.quantile(source[:,:2],.005,axis=0);high=np.quantile(source[:,:2],.995,axis=0)
        fig=plt.figure(figsize=(14,9),layout='constrained');grid=fig.add_gridspec(2,3,height_ratios=[.8,1.5]);ax=fig.add_subplot(grid[0,:]);ax.imshow(Image.open(record['rgb']));ax.axis('off');ax.set_title(f'{key} | real RGB frame {record["frame_id"]}',loc='left',weight='bold')
        for j,method in enumerate(['baseline','full_two_cells','ours_frame']):
            ax=fig.add_subplot(grid[1,j]);g=np.load(folder(key,method)/'geometry.npz');xyz=g['xyz'];ids=g['instance'];stride=max(1,(len(xyz)+99999)//100000)
            ax.scatter(xyz[::stride,0],xyz[::stride,1],c=ids[::stride]%20,cmap='tab20',vmin=0,vmax=19,s=.4)
            ax.set_xlim(low[0],high[0]);ax.set_ylim(low[1],high[1]);ax.set_aspect('equal');ax.grid(alpha=.15);ax.set_xlabel('x / m');ax.set_ylabel('y / m')
            ap=s['results'][key]['methods'][method]['metrics']['allAp']*100;ax.set_title(f'{NAMES[method]}\nAP50:90 {ap:.3f}')
        file=DEST/(key+'.png');fig.savefig(file,dpi=160);plt.close(fig);images.append(file.name)
        # Both extrema, not only improved objects; selection is explicitly posthoc.
        ordered=sorted(rows.values(),key=lambda r:(r['methods']['ours_frame']['iou']-r['methods']['baseline']['iou'],r['gt']))
        smallest=ordered[0]['methods']['ours_frame']['iou']-ordered[0]['methods']['baseline']['iou']
        first_case='largest_decrease' if smallest<0 else 'no_decrease_min_change'
        for direction,row in [(first_case,ordered[0]),('largest_increase',ordered[-1])]:
            gid=row['gt'];q=ref['xyz'][truth==gid];lo=q[:,:2].min(0)-.5;hi=q[:,:2].max(0)+.5
            fig,axes=plt.subplots(1,4,figsize=(15,4.7),layout='constrained')
            for ax,method in zip(axes,['baseline','same_cell','full_two_cells','ours_frame']):
                stride=max(1,len(q)//20000);ax.scatter(q[::stride,0],q[::stride,1],s=.6,c='#c4c9cf')
                g=np.load(folder(key,method)/'geometry.npz');oid=row['methods'][method]['predicted_id'];xyz=g['xyz'][g['instance']==oid] if oid is not None else np.empty((0,3))
                inside=(xyz[:,0]>=lo[0])&(xyz[:,0]<=hi[0])&(xyz[:,1]>=lo[1])&(xyz[:,1]<=hi[1]);xyz=xyz[inside];stride=max(1,len(xyz)//15000)
                short=dict(baseline='Baseline',same_cell='Same-cell lift',full_two_cells='Two-cell control',ours_frame='Ours')[method]
                ax.scatter(xyz[::stride,0],xyz[::stride,1],s=1.4,c='#009c93');ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal');ax.grid(alpha=.15);ax.set_xlabel('x / m');ax.set_ylabel('y / m');ax.set_title(f'{short}\nIoU {row["methods"][method]["iou"]:.4f}',fontsize=9)
            fig.suptitle(f'{key} | GT {gid} | {direction.replace("_"," ")} in best same-class IoU\nGray: reference surface; green: predicted geometry; XY crop for display only',fontsize=11)
            file=DEST/f'{key}_{direction}_{gid}.png';fig.savefig(file,dpi=160);plt.close(fig);images.append(file.name);cases.append(dict(key=key,selection=direction,image=file.name,**row))
    assert sum(len(v) for v in all_objects.values())==94
    table=[]
    for method in ALL:
        vals=[(den['results'][scene['key']] if method=='same_cell' else s['results'][scene['key']]['methods'][method])['metrics']['allAp']*100 for scene in p['scenes']]
        table.append('<tr><td>'+NAMES[method]+'</td>'+''.join(f'<td>{v:.3f}</td>' for v in vals)+f'<td>{means[method]*100:.3f}</td><td>{(means[method]-means["baseline"])*100:+.3f}</td></tr>')
    detail=[]
    for key,rows in all_objects.items():
        for row in rows:detail.append(f'<tr><td>{key}</td><td>{row["gt"]}</td>'+''.join(f'<td>{row["methods"][m]["iou"]:.4f}</td>' for m in ALL)+'</tr>')
    candidate_delta=(means['ours_frame']-means['baseline'])*100;robust_delta=(means['ours_frame']-means['full_two_cells'])*100;density_delta=(means['ours_frame']-means['same_cell'])*100
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam 来源帧稳定的对象表面补全</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;color:#17354a;background:#f4f7fa;max-width:1200px;margin:32px auto;padding:0 24px}}section{{background:white;padding:24px;margin:20px 0;border-radius:10px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #dce5ec;text-align:left}}img{{width:100%;height:auto}}.status{{padding:16px;border-left:5px solid #009c93;background:#edf8f6}}</style>
<h1>omni_slam：来源帧稳定的对象表面补全</h1><p>2026-11-14 · 固定十份开发地图＋事后两份密度对照 · 无新训练</p>
<p class="status">Ours均值{means['ours_frame']*100:.3f}，较原强基线提高{candidate_delta:.3f}个百分点，较两个位置锚点对照提高{robust_delta:.3f}个百分点。零新增、零丢失AP50匹配；原预设“至少新增一个匹配”的门槛未通过，原结论保持。已有完整地图收益，尚未证明独立泛化或机器人语义准确率。</p>
<section><h2>机制及边界</h2><p>同一份完整雷达来源构成20厘米单元图，单元代表点是实际返回；局部0.4米内16近邻的对称并集提供共同图，边权为实际欧氏距离。旧预测对象来源作为身份锚点，其他已观测语义类别作为背景锚点；未知保持未知。没有使用GT选择锚点。</p>
<p>令D<sub>o,f</sub>(v)为单元v到对象o、来源采集帧f锚点的最短路距离。d<sub>1</sub>和d<sub>2</sub>分别为该对象最近、第二近的不同来源帧距离，c为最近其他对象或背景的距离。Ours只在d<sub>2</sub>有限且d<sub>2</sub>&lt;c−10<sup>−10</sup>时补入该对象的实测源点。删除任何一批对象锚点后至少仍有另一批胜过全部原竞争锚点，因此得到固定图上的来源批次删除保证。</p>
<p>两条路径可以共享边；这不是不相交路径保证。采集帧不同不代表统计独立，跨帧共同错误仍可通过。源点采集帧也不是给它赋予对象标签的图像观测帧，因此不能声称已证明对任意单帧分割错误稳健。证书只删除锚点标签，不删除几何节点、重跑前端或改变图。它不证明语义正确、没有姿态误差或具备在线运行速度。</p>
<p>旧地图坐标、ID、类别和排序分数不动，新增点均从同一真实来源集合中取出，没有新对象出生或历史ID重写。普通图传播已有先例，例如<a href="https://openaccess.thecvf.com/content/ICCV2023/html/Dong_Collaborative_Propagation_on_Multiple_Instance_Graphs_for_3D_Instance_Segmentation_ICCV_2023_paper.html">RWSeg</a>；图与最短路本身不作为新颖性证明。需要单独验证来源批次判据对完整对象几何的贡献。</p></section>
<section><h2>完整地图结果</h2><p>主指标：本地适配KITTI-360 building/car置信度加权AP50:90×100，两个已暴露开发窗口。不是官方榜单或独立测试。输入、类别分数与最终处理对所有完整输入条件一致。</p><img src="metrics.png" alt="All conditions AP comparison"><table><tr><th>条件</th><th>0000</th><th>0002</th><th>均值</th><th>比原基线</th></tr>{''.join(table)}</table>
<p>同输入的两个位置锚点控制，把删除批次从来源帧改为空间单元；Ours相对它在两窗口分别提高0.147／0.312个百分点。Ours在0000低于最近锚点传播约0.248个百分点，不能声称逐场景优于全部方法。相对旧输入的同一帧判据，提升包含完整测量域改变的作用，不能全部归因于新机制。</p></section>
<section><h2>排除单纯加密解释</h2><p>十份地图评分后，发现大量新增点的传播距离为零，于是明确追加两份事后控制：只补入已有唯一前景锚点的同一单元，不走图、不新增模型或搜索半径。原十份预测、门槛和失败条件均未修改。</p><p>该控制均值{means['same_cell']*100:.3f}，Ours仍高{density_delta:.3f}个百分点。说明当前结果不能仅由同单元点密度变化解释；它仍不等于对所有其他密度方法的排除，也没有把来源帧判据的独立贡献从{robust_delta:.3f}放大成{candidate_delta:.3f}。</p></section>
<section><h2>真实输入与完整地图</h2>{''.join(f'<img src="{scene["key"]}.png" alt="Actual input and map {scene["key"]}">' for scene in p['scenes'])}<p>相同实例配色与坐标范围，按完整测量0.5%–99.5%分位显示，并作确定性抽样；评分使用全部点。颜色代表预测实例，非相机RGB。</p></section>
<section><h2>对象变化的两端</h2>{''.join(f'<img src="{c["image"]}" alt="{c["selection"]}">' for c in cases)}<p>每窗口按全域最佳同类IoU变化选择最大和最小变化各一个；若最小变化为零，标注未下降，不能称为退步。均为事后解释，没有按案例选择参数。全部94个GT同时列在下方。最佳IoU不等同于官方置信度排序匹配或AP。</p></section>
<section><h2>验证与去留</h2><p>六项核心测试通过，包括随机图逐组最短路核查及实际删除每个来源批次。真实图的全部首距离用独立SciPy求解器核对；每窗口最多8个固定节点穷尽全部来源组距离，另实际撤除4个固定来源帧，检查全部认证前景单元。十份地图所有新增坐标、归属、旧图前缀、分数和来源均重放通过。原基线评分精确复现。</p><p>本次有完整地图质量收益，但未实现预设的新增AP50对象恢复。保留候选与全部控制，不改门槛、不追调参数、不把它直接写成已通过的创新。若另行开展独立几何质量验证，必须明确它是基于开发结果提出的新假设、先冻结协议，再使用尚未评分的新窗口；不能倒称本轮通过。</p>
<p><a href="../../build/benchmarks/batch_stable_transport_v1/protocol.json">原冻结协议</a> · <a href="../../build/benchmarks/batch_transport_density_control_v1/protocol.json">事后密度对照协议</a> · <a href="../../build/benchmarks/batch_stable_transport_v1/artifact_audit.json">独立复核</a> · <a href="summary.json">全部主实验结果</a> · <a href="density_summary.json">密度结果</a> · <a href="all_objects.json">全部对象IoU</a></p></section>
<section><h2>全部94个合格GT</h2><div style="overflow:auto"><table><tr><th>窗口</th><th>GT</th>{''.join(f'<th>{NAMES[m]}</th>' for m in ALL)}</tr>{''.join(detail)}</table></div></section>
<section><h2>总目标仍在进行</h2><p>目标仍是有效的新机制、同一版本机器人对照和新增评测。这次完成开发机制与完整地图对照，机器人和新窗口尚未运行该候选；原始逐点射线恢复也不是此图传播模块的前置条件。不能把实现、数据准备或本报告替代后两项验证。</p><p>已另行<a href="../../build/benchmarks/batch_transport_assessment_v1/protocol.json">预注册未使用窗口的几何质量验证</a>，明确为观察开发结果后的新假设，原发现对象门槛保持未通过。锁定当前代码、参数及全部强对照，在0006／0007全部预测封存后才解码标注；若通过，再执行同版本330帧机器人对照。当前仅注册与输入准备，不冒称验证已完成。</p></section></html>'''
    (DEST/'index.html').write_text(page)
    for name,obj in [('summary.json',s),('density_summary.json',den),('all_objects.json',all_objects),('cases.json',cases),('images.json',images)]:dump(DEST/name,obj)
    for old_image in DEST.glob('*.png'):
        if old_image.name not in images:old_image.unlink()
    dump(DEST/'manifest.json',dict(complete=True,files={str(f):sha(f) for f in DEST.iterdir() if f.name!='manifest.json'},source_sha256=sha(__file__),original_advance_gate=s['advance_gate'],goal_complete=False))
    print('BATCH TRANSPORT REPORT',DEST/'index.html',candidate_delta,robust_delta,density_delta,flush=True)


if __name__=='__main__':run()
