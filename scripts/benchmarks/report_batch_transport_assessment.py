"""Independent assessment: every scene/control/object, with real RGB and maps."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/omni_assessment_mpl')
import html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from paper_final_common import ROOT,read,dump,sha
from run_batch_transport_assessment import OUT,FULL,INPUT,METHODS,seal

DEST=ROOT/'output/batch_transport_assessment_v1'
NAMES=dict(baseline='Strong baseline',same_cell='Same-cell lift',full_nearest='Nearest anchor',full_two_cells='Two-cell control',ours_frame='Ours / frame certificate')


def percentage(value):
    return 'N/A' if value is None else f'{value*100:.3f}'


def main():
    summary=read(OUT/'summary.json');assert summary['complete']
    audit=read(OUT/'prediction_audit.json');assert audit['complete']
    seal(OUT/'predictions_complete.json');p=read(OUT/'protocol.json');DEST.mkdir(parents=True,exist_ok=True)
    scenes=[s['key'] for s in p['scenes']];rows=[];objects={};images=[]
    for method in METHODS:
        values=[summary['results'][key]['methods'][method]['metrics']['allAp'] for key in scenes]
        mean=summary['mean_primary_ap'][method];base=summary['mean_primary_ap']['baseline']
        delta='N/A' if mean is None or base is None else f'{(mean-base)*100:+.3f}'
        rows.append('<tr><td>'+NAMES[method]+'</td>'+''.join(f'<td>{percentage(v)}</td>' for v in values)+f'<td>{percentage(mean)}</td><td>{delta}</td></tr>')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    for key in scenes:
        reference=np.load(OUT/'evaluation'/key/'reference.npz');truth=reference['train_instance'];weight=reference['confidence'];mass=np.bincount(truth,weights=weight)
        gids=np.flatnonzero((mass>=100)&np.isin(np.arange(len(mass))//1000,[2,13]));table={int(i):dict(gt=int(i),category='building' if i//1000==2 else 'car',methods={}) for i in gids}
        for method in METHODS:
            folder=OUT/'final_maps'/key/method;pred=np.load(folder/'reference_prediction.npz')['instance'];cls=read(folder/'classification.json')['objects'];pmass=np.bincount(pred,weights=weight)
            joint=np.bincount(pred.astype(np.int64)*len(mass)+truth,weights=weight,minlength=len(pmass)*len(mass)).reshape(-1,len(mass))
            for gid in gids:
                best=dict(iou=0.,predicted_id=None)
                for obj in cls:
                    oid=obj['object_id']
                    if oid>=len(pmass) or obj['predictions']['mean']['train_id']!=gid//1000:continue
                    inter=joint[oid,gid];iou=float(inter/max(1e-12,mass[gid]+pmass[oid]-inter))
                    if iou>best['iou']:best=dict(iou=iou,predicted_id=oid)
                table[int(gid)]['methods'][method]=best
        objects[key]=list(table.values())
        manifest=read(INPUT/key/'manifest.json');record=manifest['records'][len(manifest['records'])//2]
        source=np.load(FULL/key/'geometry.npz')['xyz'];lo=np.quantile(source[:,:2],.005,axis=0);hi=np.quantile(source[:,:2],.995,axis=0)
        fig,axes=plt.subplots(2,3,figsize=(16,10),layout='constrained');axes=axes.ravel();axes[0].imshow(Image.open(record['rgb']));axes[0].axis('off');axes[0].set_title(f'{key} | actual RGB {record["frame_id"]}')
        for ax,method in zip(axes[1:],METHODS):
            g=np.load(OUT/'final_maps'/key/method/'geometry.npz');xyz=g['xyz'];labels=g['instance'];step=max(1,(len(xyz)+79999)//80000)
            backstep=max(1,(len(source)+99999)//100000)
            ax.scatter(source[::backstep,0],source[::backstep,1],s=.1,color='#cdd3d9',alpha=.4,rasterized=True)
            ax.scatter(xyz[::step,0],xyz[::step,1],s=.7,c=labels[::step]%20,cmap='tab20',vmin=0,vmax=19,rasterized=True)
            ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal');ax.set_xlabel('x / m');ax.set_ylabel('y / m');ax.grid(alpha=.12)
            score=summary['results'][key]['methods'][method]['metrics']['allAp']
            ax.set_title(f'{NAMES[method]} | AP {percentage(score)}\n{len(xyz):,} points / {len(np.unique(labels))} objects',color='#008a7d' if method=='ours_frame' else '#20364a')
        file=DEST/(key+'.png');fig.savefig(file,dpi=150);plt.close(fig);images.append(file.name)
    details=[];counts={}
    for key,items in objects.items():
        delta=np.array([r['methods']['ours_frame']['iou']-r['methods']['baseline']['iou'] for r in items]);counts[key]=dict(improved=int((delta>1e-12).sum()),decreased=int((delta < -1e-12).sum()),unchanged=int((np.abs(delta)<=1e-12).sum()))
        for row in items:details.append(f'<tr><td>{key}</td><td>{row["gt"]}</td>'+''.join(f'<td>{row["methods"][m]["iou"]:.4f}</td>' for m in METHODS)+'</tr>')
    diffs=[]
    for key,row in audit['results'].items():
        for pair,change in row['comparisons'].items():diffs.append(f'<tr><td>{key}</td><td>{html.escape(pair)}</td><td>{change["different_source_assignments"]:,}</td><td>{change["identical_export"]}</td></tr>')
    gate='通过' if summary['independent_gate'] else '未通过'
    robot=ROOT/'build/benchmarks/batch_transport_robot_v1/summary.json'
    robot_status='机器人同版本对照尚未完成。'
    if robot.exists() and read(robot)['complete']:
        robot_status='<a href="../batch_transport_robot_v1/index.html">330帧、五组同版本车库对照已完成</a>；公开验收未通过，机器人仅作诊断，不改变验收结论。'
    page=f'''<!doctype html><meta charset="utf-8"><title>omni_slam 独立窗口评测</title>
<style>body{{max-width:1320px;margin:36px auto;padding:0 24px;background:#f5f7f9;color:#20364a;font:16px/1.65 system-ui}}section{{background:white;padding:24px;margin:18px 0;border-radius:12px}}h1{{font-size:30px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #dce4e9;text-align:right}}td:first-child,th:first-child{{text-align:left}}img{{width:100%;background:white}}a{{color:#008a7d}}.ours{{color:#008a7d}}</style>
<h1>omni_slam：完整对象地图的独立窗口评测</h1><p>固定 KITTI-360 0006／0007；本地适配 building/car AP50:90，非官方榜单。两窗口全部十份地图封存后才解码参考标注。候选选择来自旧开发结果，新窗口未用于调参。</p>
<section><h2>预设验收：{gate}</h2><p>要求：Ours 均值比原强基线至少高2个百分点，超过三个同输入控制；每窗口主AP和AP50不低于基线。新增匹配 {summary['gained']}，丢失匹配 {summary['lost']}。不以平均分掩盖失败场景。</p>
<table><tr><th>方法</th>{''.join('<th>'+k+'</th>' for k in scenes)}<th>均值</th><th>Δ基线</th></tr>{''.join(rows)}</table><p>0007 的完整参考中没有建筑或汽车，目标AP未定义。保留 N/A，不改成零，不删除此窗口后重新宣布双窗口门槛通过。0006 有55个合格目标GT；Ours 比基线高3.251个百分点，但比两个位置锚点控制低0.385个百分点。因此当前不能声称独立机制优势。<a href="../../build/benchmarks/batch_transport_assessment_v1/reference_domain_audit.json">参考域核查</a>。</p></section>
<section><h2>真实图像与完整地图</h2><p>灰色为完整实测来源，彩色为对象点；各方法使用相同视域和对象配色。显示范围截取坐标0.5%至99.5%分位，计分使用全部静态参考点。空地图及其GT保留。</p>{''.join('<img src="'+f+'">' for f in images)}</section>
<section><h2>“结果一样”的逐点核查</h2><p>比较每个实际来源点的新增对象归属；零差异明确列出，不把同一导出重复称作不同效果。全部地图检查原坐标／ID／分数前缀和真实测量来源。</p><table><tr><th>窗口</th><th>比较</th><th>归属不同的来源点</th><th>导出相同</th></tr>{''.join(diffs)}</table></section>
<section><h2>全部合格参考对象</h2><p>统计每个GT的最佳同类置信度加权IoU，用于几何得失诊断，不替代一对一匹配AP。{html.escape(str(counts))}</p><table><tr><th>窗口</th><th>GT</th>{''.join('<th>'+NAMES[m]+'</th>' for m in METHODS)}</tr>{''.join(details)}</table></section>
<section><h2>证据及范围</h2><p><a href="../../build/benchmarks/batch_transport_assessment_v1/summary.json">完整计分</a> · <a href="../../build/benchmarks/batch_transport_assessment_v1/prediction_audit.json">逐点导出核验</a> · <a href="all_objects.json">全部对象IoU</a> · <a href="../../build/benchmarks/batch_transport_assessment_v1/protocol.json">预注册协议</a> · <a href="../../build/benchmarks/batch_transport_assessment_v1/mechanism_comparison.json">Ours与同输入控制的逐对象差异</a></p><p>不确定语义背景及空输入错误以独立修正记录封存；旧地图精确重放，未改变模型、阈值、窗口、验收规则。{robot_status}本报告不声称机器人准确率或总目标完成。</p></section>'''
    (DEST/'index.html').write_text(page);dump(DEST/'all_objects.json',objects);dump(DEST/'object_change_counts.json',counts)
    dump(DEST/'manifest.json',dict(complete=True,files={str(f):sha(f) for f in DEST.iterdir() if f.name!='manifest.json'},source_sha256=sha(__file__),independent_gate=summary['independent_gate'],goal_complete=False))
    print('INDEPENDENT REPORT',DEST/'index.html',counts,flush=True)


if __name__=='__main__':main()
