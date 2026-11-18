"""Report every evaluated condition and GT object from sealed complete maps."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/omni_relative_mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from paper_final_common import ROOT, read, dump, sha
from run_object_relative_evidence import OUT, METHODS, verify, seal, parent

DEST = ROOT / 'output/object_relative_evidence_v1'
NAMES = dict(baseline='Strong baseline', frozen_raw='Frozen raw-memory model',
             shape_drop='Drop shape features', ours_append_only='Ours / append only',
             ours_relative='Ours / object-relative evidence')


def run():
    p = verify(); seal('prepared'); seal('trained'); seal('predictions_complete')
    summary = read(OUT/'summary.json'); assert read(OUT/'map_audit.json')['complete']
    DEST.mkdir(parents=True, exist_ok=True); plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    all_objects = {}; changes = []; figures = []
    for scene in p['scenes']:
        key = scene['key']; ref = np.load(scene['reference']); truth = ref['train_instance']; weight = ref['confidence'].astype(float)
        mass = np.bincount(truth, weights=weight); eligible = np.flatnonzero((mass>=100)&np.isin(np.arange(len(mass))//1000,[2,13]))
        rows = {int(g):dict(gt=int(g), methods={}) for g in eligible}
        for method in METHODS:
            folder = OUT/'maps'/key/method; prediction = np.load(folder/'reference_prediction.npz')['instance']
            objects = read(folder/'classification.json')['objects']; pmass = np.bincount(prediction, weights=weight)
            joint = np.bincount(prediction.astype(np.int64)*len(mass)+truth, weights=weight, minlength=len(pmass)*len(mass)).reshape(-1,len(mass))
            for gid in eligible:
                best = dict(iou=0., predicted_id=None)
                for obj in objects:
                    oid = obj['object_id']
                    if oid>=len(pmass) or obj['predictions']['mean']['train_id']!=gid//1000: continue
                    inter = joint[oid,gid]; iou = float(inter/max(1e-12,mass[gid]+pmass[oid]-inter))
                    if iou>best['iou']: best = dict(iou=iou,predicted_id=oid)
                rows[int(gid)]['methods'][method] = best
        all_objects[key] = list(rows.values())
        delta = summary['results'][key]['methods']['ours_relative']['changes_vs_baseline']
        for sign in ['gained','lost']:
            for gid in delta[sign]: changes.append(dict(key=key,change=sign,**rows[gid]))
        manifest = read(parent.INPUT/key/'manifest.json'); rec = manifest['records'][len(manifest['records'])//2]
        source = np.load(manifest['geometry'])['xyz']; low = np.quantile(source[:,:2],.005,axis=0); high = np.quantile(source[:,:2],.995,axis=0)
        fig = plt.figure(figsize=(14,9),layout='constrained'); grid = fig.add_gridspec(2,3,height_ratios=[.8,1.5])
        ax = fig.add_subplot(grid[0,:]); ax.imshow(Image.open(rec['rgb'])); ax.axis('off'); ax.set_title(f'{key} | real RGB frame {rec["frame_id"]}',loc='left',weight='bold')
        for j,method in enumerate(['baseline','shape_drop','ours_relative']):
            ax = fig.add_subplot(grid[1,j]); g = np.load(OUT/'maps'/key/method/'geometry.npz'); xyz=g['xyz']; ids=g['instance']; stride=max(1,(len(xyz)+89999)//90000)
            ax.scatter(xyz[::stride,0],xyz[::stride,1],c=ids[::stride]%20,cmap='tab20',vmin=0,vmax=19,s=.4)
            ax.set_xlim(low[0],high[0]); ax.set_ylim(low[1],high[1]); ax.set_aspect('equal'); ax.grid(alpha=.15); ax.set_xlabel('x / m'); ax.set_ylabel('y / m')
            ap=summary['results'][key]['methods'][method]['metrics']['allAp']*100
            ax.set_title(f'{NAMES[method]}\nAP50:90 {ap:.3f}',fontsize=10)
        file=DEST/f'{key}.png';fig.savefig(file,dpi=150);plt.close(fig);figures.append(dict(file=file.name,rgb=rec['rgb'],rgb_sha256=sha(rec['rgb'])))
    # Exhaustive changed AP50 matches; no handpicked positive cases.
    for event in changes:
        key=event['key'];gid=event['gt'];scene=next(s for s in p['scenes'] if s['key']==key);ref=np.load(scene['reference'])
        xyz=ref['xyz'][ref['train_instance']==gid];low=xyz[:,:2].min(0)-1;high=xyz[:,:2].max(0)+1
        fig,axes=plt.subplots(1,3,figsize=(12,4.5),layout='constrained')
        for ax,method in zip(axes,['baseline','shape_drop','ours_relative']):
            ax.scatter(xyz[::max(1,len(xyz)//15000),0],xyz[::max(1,len(xyz)//15000),1],c='#cdd2d8',s=.5,label='GT surface')
            g=np.load(OUT/'maps'/key/method/'geometry.npz');oid=event['methods'][method]['predicted_id'];q=g['xyz'][g['instance']==oid] if oid is not None else np.empty((0,3))
            inside=(q[:,0]>=low[0])&(q[:,0]<=high[0])&(q[:,1]>=low[1])&(q[:,1]<=high[1]);q=q[inside]
            ax.scatter(q[:,0],q[:,1],c='#008a89',s=1,label='Predicted object');ax.set_xlim(low[0],high[0]);ax.set_ylim(low[1],high[1]);ax.set_aspect('equal');ax.grid(alpha=.15);ax.set_xlabel('x / m');ax.set_ylabel('y / m')
            ax.set_title(f'{NAMES[method]}\nIoU {event["methods"][method]["iou"]:.4f}',fontsize=9)
        axes[0].legend(fontsize=8);fig.suptitle(f'{key} | GT {gid} | {event["change"]} AP50 match')
        event['image']=f'{key}_gt{gid}.png';fig.savefig(DEST/event['image'],dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained');trained=read(OUT/'trained.json')
    for ax,key in zip(axes,parent.KEYS):
        for name,row in trained['stats'].items():
            if name.startswith(key+'/'):ax.plot(np.arange(1,41),row['loss'],label=NAMES[name.split('/')[1]])
        ax.set_title(f'Target {key}; trained on other window');ax.set_xlabel('Epoch');ax.set_ylabel('Weighted grouped NLL');ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.savefig(DEST/'training.png',dpi=150);plt.close(fig)
    table=[]
    for method in METHODS:
        values=[summary['results'][k]['methods'][method]['metrics']['allAp']*100 for k in parent.KEYS]
        table.append('<tr><td>'+NAMES[method]+'</td>'+''.join(f'<td>{v:.3f}</td>' for v in values)+f'<td>{summary["mean_primary_ap"][method]*100:.3f}</td></tr>')
    detail=[]
    for key,rows in all_objects.items():
        for r in rows:detail.append(f'<tr><td>{key}</td><td>{r["gt"]}</td>'+''.join(f'<td>{r["methods"][m]["iou"]:.4f}</td>' for m in METHODS)+'</tr>')
    status='通过开发指标门槛，仍需独立与机器人验证' if summary['advance_gate'] else '未通过开发门槛，本版本封存，保留原强基线'
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam 对象自身参照实验</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;color:#17354a;background:#f4f7fa;max-width:1200px;margin:32px auto;padding:0 24px}}section{{background:white;padding:24px;margin:20px 0;border-radius:10px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #dce5ec;text-align:left}}img{{width:100%;height:auto}}.status{{padding:16px;border-left:5px solid #078a8a;background:#edf8f6}}</style>
<h1>omni_slam：对象自身参照与历史几何修订</h1><p>2026-11-14 · 四次固定训练 · 十份完整室外地图</p><p class="status">{status}。Ours 为预指定候选名称，不代表已证实优势。</p>
<section><h2>依据与单一改动</h2><p>前一版被误删的车辆13009有655个点，其训练目标均要求保留，图像类别证据也支持车辆。固定权重，仅替换三个整体形状统计量，可让642个恢复保留。建筑2011也对这组特征敏感，但替换会同时放回错误点。这是人工输入干预，说明模型敏感性，不是可部署改进或唯一物理原因。</p>
<p>本轮将点的十维几何特征改为其在同一预测对象原始几何中的经验分位，判断相对异常程度。每20厘米单元取一个真实代表点，并排除查询所在整个单元，避免直接自我参照；不改其余图像、记忆、类别或候选输入。对象整体常量形状特征因此归零。直接去掉三个形状特征作为必要强对照。</p>
<p>参照来自预测，可能含错误；分位值不代表正确概率。该经验归一化操作本身不是创新证明，也不解决漏检对象出生。方法仍为固定身份、窗口末端修订；不是在线SLAM新算法。</p></section>
<section><h2>完整地图结果</h2><p>本地适配KITTI-360 building/car AP50:90 ×100。两个窗口均已暴露，继承父级监督，不是独立测试、严格嵌套交叉验证或官方榜单。</p>
<table><tr><th>方法</th><th>0000</th><th>0002</th><th>均值</th></tr>{''.join(table)}</table><p>Ours 相对强基线新增{summary['gained']}、丢失{summary['lost']}个AP50正确匹配。仅追加对照使用完全相同权重与决策，只保留全部旧几何。</p></section>
<section><h2>实际场景与地图</h2>{''.join(f'<img src="{r["file"]}" alt="{r["file"]}">' for r in figures)}<p>相同实例配色及坐标范围；显示范围按共同实测源点0.5%–99.5%分位裁剪，评分使用完整地图。</p></section>
<section><h2>全部新增／丢失匹配</h2>{''.join(f'<img src="{e["image"]}" alt="{e["key"]} GT {e["gt"]}">' for e in changes) or '<p>没有新增或丢失AP50匹配；仍需比较完整AP及全部对象IoU。</p>'}<p>图中IoU为完整参考域的同类最佳预测，非裁剪图IoU；不是额外的一对一匹配分数。</p></section>
<section><h2>训练与核验</h2><img src="training.png" alt="Four fixed training curves"><p>四项针对性测试通过。输入抽查4096个候选的全部十个几何分位，采用独立直接计数法；其余通道及候选可用性保持不变。十份地图的来源、旧点修改、身份与类别重放通过；原强基线与前版输出精确复现。四次训练均使用预定40轮最终权重，没有按结果选轮数、阈值或种子。</p>
<p>评分后核查确认一个表示缺陷：两窗口全部31974／31175个有效新增候选，其到对象核心的距离分位都为1，实际距离却覆盖约0.039–68.172米／0.041–65.602米。分位替换丢掉了参照最大值之外的尺度差异，不能作为完整几何证据。这不是增加训练轮数能恢复的输入信息；也不能据此唯一解释全部AP变化。<a href="../../build/benchmarks/object_relative_evidence_v1/rank_information_diagnostic.json">距离信息核查</a>。</p>
<p>训练前发现视觉隔离合格点不足以支撑四个对象的几何参照，改用全部原有对象几何。原协议与代码、修订原因均保留；未改变查询视觉条件，没有据结果调参。</p>
<p><a href="../../build/benchmarks/object_relative_evidence_v1/protocol.json">冻结协议</a> · <a href="../../build/benchmarks/object_relative_evidence_v1/pretraining_revision.json">训练前修订</a> · <a href="../../build/benchmarks/object_relative_evidence_v1/input_audit.json">输入核查</a> · <a href="../../build/benchmarks/object_relative_evidence_v1/map_audit.json">地图核查</a> · <a href="summary.json">完整指标</a> · <a href="all_objects.json">全部对象IoU</a></p></section>
<section><h2>全部94个合格GT对象</h2><div style="overflow:auto"><table><tr><th>窗口</th><th>GT</th>{''.join(f'<th>{NAMES[m]}</th>' for m in METHODS)}</tr>{''.join(detail)}</table></div></section>
<section><h2>主线状态</h2><p>目标仍为可验证的对象级语义地图收益及论文证据链。有效新机制、同版本机器人对照与新增独立评测，没有由这次局部机制实现或开发评测替代。未通过的版本不继续追加参数搜索。</p></section></html>'''
    (DEST/'index.html').write_text(page);dump(DEST/'summary.json',summary);dump(DEST/'all_objects.json',all_objects);dump(DEST/'changed_matches.json',changes);dump(DEST/'images.json',figures)
    dump(DEST/'manifest.json',dict(complete=True,files={str(f):sha(f) for f in DEST.iterdir() if f.is_file() and f.name!='manifest.json'},source_sha256=sha(__file__),advance_gate=summary['advance_gate']))
    print('RELATIVE REPORT',DEST/'index.html','GTs',sum(map(len,all_objects.values())),flush=True)
    for e in changes:print('RELATIVE CHANGED',e['key'],e['gt'],e['change'],{m:r['iou'] for m,r in e['methods'].items()},flush=True)


if __name__ == '__main__':run()
