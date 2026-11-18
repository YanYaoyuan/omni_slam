"""Actual stream imagery, complete-map comparisons, and all-object accounting."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/omni_memory_mpl')
from pathlib import Path
import html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from paper_final_common import ROOT,read,dump,sha
from run_persistent_source_memory import OUT,INPUT,METHODS,verify,seal

DEST=ROOT/'output/persistent_source_memory_v1'
NAMES=dict(baseline='Frozen strong baseline',current_commit='Current-source commit',
           no_geometry='Trained without ray relations',ours_history='Ours candidate: history revision')


def run():
    p=verify();seal('trained');seal('predictions_complete');summary=read(OUT/'summary.json')
    assert read(OUT/'input_audit.json')['complete'] and read(OUT/'map_audit.json')['complete']
    DEST.mkdir(parents=True,exist_ok=True);plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    all_objects={};images=[]
    for scene in p['scenes']:
        key=scene['key'];manifest=read(INPUT/key/'manifest.json');records=manifest['records'];ref=np.load(scene['reference'])
        truth=ref['train_instance'];weight=ref['confidence'].astype(float);mass=np.bincount(truth,weights=weight)
        gt=np.flatnonzero((mass>=100)&np.isin(np.arange(len(mass))//1000,[2,13]))
        per_gt={int(g):dict(gt=int(g),category='building' if g//1000==2 else 'car',confidence_mass=float(mass[g]),methods={}) for g in gt}
        for method in METHODS:
            folder=OUT/'maps'/key/method;pred=np.load(folder/'reference_prediction.npz')['instance'];objects=read(folder/'classification.json')['objects']
            pmass=np.bincount(pred,weights=weight)
            joint=np.bincount(pred.astype(np.int64)*len(mass)+truth,weights=weight,
                minlength=len(pmass)*len(mass)).reshape(-1,len(mass))
            for g in gt:
                best=0.;oid=None
                for obj in objects:
                    i=obj['object_id']
                    if obj['predictions']['mean']['train_id']!=g//1000 or i>=len(pmass):continue
                    inter=joint[i,g];value=float(inter/max(mass[g]+pmass[i]-inter,1e-12))
                    if value>best:best=value;oid=i
                per_gt[int(g)]['methods'][method]=dict(best_iou=best,predicted_id=oid)
        all_objects[key]=list(per_gt.values())
        rgb=records[len(records)//2]['rgb'];xyz=np.load(manifest['geometry'])['xyz']
        low=np.quantile(xyz[:,:2],.005,axis=0);high=np.quantile(xyz[:,:2],.995,axis=0)
        fig=plt.figure(figsize=(14,9),layout='constrained');grid=fig.add_gridspec(2,3,height_ratios=[.8,1.5])
        ax=fig.add_subplot(grid[0,:]);ax.imshow(Image.open(rgb));ax.set_axis_off()
        ax.set_title(f'{key} | actual input RGB, frame {records[len(records)//2]["frame_id"]}',loc='left',weight='bold')
        for j,method in enumerate(['baseline','current_commit','ours_history']):
            ax=fig.add_subplot(grid[1,j]);g=np.load(OUT/'maps'/key/method/'geometry.npz');points=g['xyz'];ids=g['instance']
            stride=max(1,(len(points)+89999)//90000)
            if len(points):ax.scatter(points[::stride,0],points[::stride,1],c=ids[::stride]%20,cmap='tab20',vmin=0,vmax=19,s=.4,rasterized=True)
            else:ax.text(.5,.5,'No retained objects',ha='center',va='center',transform=ax.transAxes)
            ap=summary['results'][key]['methods'][method]['metrics']['allAp']*100
            ax.set_title(f'{NAMES[method]}\nAP50:90 {ap:.3f} | {len(points):,} points',fontsize=10)
            ax.set_xlim(low[0],high[0]);ax.set_ylim(low[1],high[1]);ax.set_aspect('equal');ax.set_xlabel('x / m');ax.set_ylabel('y / m');ax.grid(alpha=.15)
        fig.suptitle('Measured geometry only · instance colors are local to each method',fontsize=13)
        file=DEST/f'{key}.png';fig.savefig(file,dpi=160);plt.close(fig)
        images.append(dict(file=file.name,source_rgb=rgb,source_rgb_sha256=sha(rgb),points_sampled_for_display_only=True))
    dump(DEST/'all_objects.json',all_objects)
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    trained=read(OUT/'trained.json')
    for name,row in trained['stats'].items():ax.plot(np.arange(1,41),row['loss'],label=name)
    ax.set_xlabel('Epoch');ax.set_ylabel('Training loss (ownership + existence)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    ax.set_title('All four fixed-budget fits · training loss is not map accuracy')
    fig.savefig(DEST/'training.png',dpi=160);plt.close(fig)
    rows=[]
    for method in METHODS:
        values=[summary['results'][k]['methods'][method]['metrics']['allAp']*100 for k in all_objects]
        changes=[summary['results'][k]['methods'][method]['changes_vs_baseline'] for k in all_objects]
        rows.append('<tr><td>'+NAMES[method]+'</td>'+''.join(f'<td>{v:.3f}</td>' for v in values)+
            f'<td>{summary["mean_primary_ap"][method]*100:.3f}</td><td>{sum(len(r["gained"]) for r in changes)} / {sum(len(r["lost"]) for r in changes)}</td></tr>')
    object_rows=[]
    for key,objects in all_objects.items():
        for row in objects:
            values=[row['methods'][m]['best_iou'] for m in METHODS];delta=values[-1]-values[0]
            object_rows.append(f'<tr><td>{key}</td><td>{row["gt"]}</td><td>{row["category"]}</td>'+''.join(f'<td>{v:.4f}</td>' for v in values)+f'<td>{delta:+.4f}</td></tr>')
    audit=read(OUT/'map_audit.json');inputaudit=read(OUT/'input_audit.json')
    diagnosis=read(OUT/'state_diagnosis.json')
    state_rows=[]
    for name,row in diagnosis['results'].items():
        last=row['final']
        state_rows.append(f'<tr><td>{name}</td><td>{last["slot_cosine_min"]:.9f}</td><td>{last["mean_foreground_logit_spread"]:.2e}</td><td>{last["first_singular_energy_fraction"]:.9f}</td></tr>')
    numerical='；'.join(f'{k} 末帧 {r["revision"]["final_frame_changed"]:,} 个源点的离散ID改变' for k,r in audit['results'].items())
    numerical_interpretation=('末帧同输入、同状态的复算ID保持一致；本次没有观察到该检查中的批次ID翻转。已证实的是槽位与实例分数近乎重合，不能进一步把数值风险写成已测得的翻转。历史归属确实改变，但没有恢复正确实例。'
        if audit['all_final_frame_invariance_checks_passed'] else
        '末帧输入和记忆状态没有变化，仍出现离散ID差异，说明数值/批次敏感性；这些变化不能归因于新证据纠错。连续输出数值接近不保证近相同分数的argmax稳定。')
    revisions='；'.join(f'{k}：{r["revision"]["raw_changed_sources"]:,} 个原始源点归属改变' for k,r in audit['results'].items())
    status='通过预设开发门槛；仍须独立测试和同版本机器人验证' if summary['advance_gate'] else '未通过预设开发门槛；本固定版本封存，不扩大机器人或独立评测'
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam — 持续对象记忆实验</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;color:#183247;background:#f5f8fb;max-width:1180px;margin:32px auto;padding:0 24px}}h1,h2{{line-height:1.25}}section{{background:white;padding:24px;margin:20px 0;border-radius:10px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px;border-bottom:1px solid #dce4ec;text-align:left}}img{{width:100%;height:auto}}.status{{border-left:5px solid #087f8c;padding:12px;background:#e9f6f5}}small{{color:#567}}code{{font-size:14px}}</style>
<h1>omni_slam：持续对象记忆与历史源点归属修订</h1><p>2026-11-14 · 室外 KITTI-360 开发实验 · 四次训练，八份完整地图</p>
<p class="status">{status}。新模型不是冻结强基线的替代版本。</p>
<section><h2>检验的机制</h2><p>每次收到图像，以当前及前三帧的冻结 VGGT 特征读取真实 LiDAR 源点。对象槽位跨帧保持；再次观测的实测点用于取回历史 token，几何关系作为可学习偏置。训练同时解码当前与历史源点。窗口结束后重新解码所有历史源点，允许改归属、回到背景或形成不同实例。</p>
<p>当前提交与 Ours 使用完全相同的权重、上下文、记忆状态和计算；当前提交对照丢弃最后一次历史解码，保留每个点最后可见时的归属。无几何对照独立训练，容量、数据、种子和预算相同。比较中的 Ours 是候选标签，不代表已胜出。</p>
<p>128 个固定槽位不回收，背景独立。没有推理期 GT 槽位绑定，没有旧地图回填，没有预测深度生成点，也没有 SOR 或连通域后处理。该版本是有限窗口历史修订，不宣称无限时长建图；完整源点账本仍随数据增长。</p></section>
<section><h2>完整地图成绩</h2><p>AP50:90 ×100，building/car 两类，本地适配的完整静态窗口评测；不是官方排行榜。两个窗口均已用于开发。两方向分别只用另一窗口训练任务头；基线继承旧开发过程，因此不能称严格独立泛化。</p>
<table><thead><tr><th>方法</th><th>0000</th><th>0002</th><th>均值</th><th>新增 / 丢失 AP50 GT</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p>{revisions}。归属改变不等于纠错；以下完整地图与逐对象 IoU 给出实际得失。</p></section>
<section><h2>已定位的机制故障：实例槽位坍缩</h2><p>四个最终权重的状态重放表明：不同对象槽位在序列末尾几乎成为同一个向量，128槽状态的第一奇异值几乎解释全部能量。前景实例分数差已接近浮点误差量级。这些是固定权重的实际诊断，没有重新训练或按评分修改解码。</p>
<table><thead><tr><th>预测窗口 / 模型</th><th>最小槽位余弦相似度</th><th>平均前景logit极差</th><th>首奇异值能量占比</th></tr></thead><tbody>{''.join(state_rows)}</tbody></table>
<p>{numerical}。{numerical_interpretation}</p>
<p>这次未能隔离出可靠的历史修订收益：当前随机槽位、全场景读取的训练实现没有学成稳定实例表示。它不证明所有对象记忆或学习方法无效，也不支持直接延长训练。原强基线保留，不以本版替换。<a href="../../build/benchmarks/persistent_source_memory_v1/state_diagnosis.json">逐帧状态诊断</a></p></section>
<section><h2>真实图像与地图</h2>{''.join(f'<img src="{r["file"]}" alt="{r["file"]}">' for r in images)}
<p>地图采用相同坐标范围，范围为原实测源点 XY 的0.5%–99.5%分位，仅用于显示；评分不裁剪。颜色是每种方法自身的实例编号，不表示跨方法相同对象。没有将车库旧图冒充这个新模块的机器人验证。</p></section>
<section><h2>训练、预检与审计</h2><img src="training.png" alt="All training curves">
<p>四次训练均使用最终第40轮权重，无测试选权重、早停或搜索。当前 11 项核心测试通过；108 帧的图像历史范围、约206万个源点的注册、在线类别累计，以及109556条训练查询外观通过独立重放。</p>
<p>训练前有一次公开记录的输入修正：均匀抽历史点只产生38/29对几何关系，无法充分检验几何消融；改为最多128对精确再次观测源点加其他历史样本。旧代码、协议、输入包全部保留。修正先于标签生成、优化和评分，未增加模型调用、训练次数或地图预算。</p>
<p>实际重复源点对：{inputaudit['results']['kitti360_0000']['exact_repeat_pairs']:,} / {inputaudit['results']['kitti360_0002']['exact_repeat_pairs']:,}。这只是输入机制覆盖，不是准确率证据。深度冲突仅为软关系，不能解释为不同对象真值。</p>
<p><a href="../../build/benchmarks/persistent_source_memory_v1/protocol.json">冻结协议</a> · <a href="../../build/benchmarks/persistent_source_memory_v1/pretraining_revision.json">训练前修正记录</a> · <a href="../../build/benchmarks/persistent_source_memory_v1/input_audit.json">输入审计</a> · <a href="../../build/benchmarks/persistent_source_memory_v1/map_audit.json">地图审计</a> · <a href="summary.json">全部指标</a> · <a href="all_objects.json">逐对象数据</a></p></section>
<section><h2>全部合格对象的几何得失</h2><p>每个 GT 与同类预测的最大置信度加权 IoU，允许不同 GT 各自选最佳预测，仅用于诊断；不是一对一匹配成绩，也不是新的 AP。未预测对象保留为0。</p>
<table><thead><tr><th>窗口</th><th>GT</th><th>类别</th><th>强基线</th><th>当前提交</th><th>无几何</th><th>Ours</th><th>Δ基线</th></tr></thead><tbody>{''.join(object_rows)}</tbody></table></section>
<section><h2>主线状态</h2><p>主线仍为 omni_slam 对象级语义建图，LIVO 提供机器人定位与实测几何。训练完成或使用 Transformer 均不能替代完整地图收益；有效新机制、同版本机器人与未参与开发的新增评测，必须分别有证据。</p></section></html>'''
    (DEST/'index.html').write_text(body)
    dump(DEST/'summary.json',summary);dump(DEST/'images.json',images)
    dump(DEST/'manifest.json',dict(files={str(f):sha(f) for f in DEST.iterdir() if f.is_file() and f.name!='manifest.json'},
        source_sha256=sha(__file__),complete=True,advance_gate=summary['advance_gate']))
    print('MEMORY REPORT',DEST/'index.html','all GTs',sum(map(len,all_objects.values())),flush=True)


if __name__=='__main__':run()
