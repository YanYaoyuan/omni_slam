"""Full-map results and loss-domain/object-level diagnostic from sealed outputs."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/omni_provenance_mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from paper_final_common import ROOT,read,dump,sha
from run_provenance_object_memory import OUT,INPUT,BASE,METHODS,verify,seal

DEST=ROOT/'output/provenance_object_memory_v1'
NAMES=dict(baseline='Strong baseline',mean_disjoint='Mean / disjoint',retrieval_disjoint='Retrieval / disjoint',
    attention_all='Attention / all views',ours_append_only='Ours / append only',ours_revision='Ours / history revision')


def agreement(chosen,owners,positive,eligible,weight):
    correct=(((chosen[:,None]==owners)&positive).any(1))|((chosen==0)&~positive.any(1))
    use=eligible&(weight>0)
    return dict(queries=int(use.sum()),confidence_mass=float(weight[use].sum()),
        weighted_target_agreement=float(np.sum(weight[use]*correct[use])/weight[use].sum()) if use.any() else None,
        correct_queries=int(correct[use].sum()))


def run():
    p=verify();seal('trained');seal('features_complete');seal('predictions_complete')
    summary=read(OUT/'summary.json');assert read(OUT/'map_audit.json')['complete'] and read(OUT/'input_audit.json')['complete']
    DEST.mkdir(parents=True,exist_ok=True);plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    objects={};point_agreement={};figures=[];changes=[]
    for scene in p['scenes']:
        key=scene['key'];ref=np.load(scene['reference']);truth=ref['train_instance'];weight=ref['confidence'].astype(float)
        mass=np.bincount(truth,weights=weight);gt=np.flatnonzero((mass>=100)&np.isin(np.arange(len(mass))//1000,[2,13]))
        rows={int(g):dict(gt=int(g),category='car' if g//1000==13 else 'building',methods={}) for g in gt}
        for method in METHODS:
            folder=OUT/'maps'/key/method;pred=np.load(folder/'reference_prediction.npz')['instance'];cls=read(folder/'classification.json')['objects']
            pmass=np.bincount(pred,weights=weight);joint=np.bincount(pred.astype(np.int64)*len(mass)+truth,weights=weight,
                minlength=len(pmass)*len(mass)).reshape(-1,len(mass))
            for g in gt:
                best=dict(iou=0.,predicted_id=None,intersection=0.,false_mass=0.)
                for o in cls:
                    i=o['object_id']
                    if o['predictions']['mean']['train_id']!=g//1000 or i>=len(pmass):continue
                    inter=joint[i,g];iou=float(inter/max(1e-12,mass[g]+pmass[i]-inter))
                    if iou>best['iou']:best=dict(iou=iou,predicted_id=i,intersection=float(inter),false_mass=float(pmass[i]-inter))
                rows[int(g)]['methods'][method]=best
        objects[key]=list(rows.values())
        d=np.load(OUT/'features'/key/'candidates.npz');y=np.load(OUT/'training_only'/key/'labels.npz')
        base=np.load(BASE/key/'geometry_graph/geometry.npz');n_old=len(d['old_indices'])
        baseline=np.r_[base['instance'][d['old_indices']],np.zeros(len(d['new_source_ids']),np.int64)]
        point_agreement[key]={}
        for method in METHODS:
            choice=baseline.copy() if method=='baseline' else np.load(OUT/'maps'/key/method/'decisions.npz')['owners'].copy()
            if method=='ours_append_only':choice[:n_old]=baseline[:n_old]
            groups={'all':np.ones(len(choice),bool),'old':np.arange(len(choice))<n_old,'new':np.arange(len(choice))>=n_old}
            point_agreement[key][method]={name:agreement(choice,d['owners'],y['positive'],y['eligible']&group,y['weight']) for name,group in groups.items()}
        delta=summary['results'][key]['methods']['ours_revision']['changes_vs_baseline']
        for sign in ['gained','lost']:
            for gid in delta[sign]:changes.append(dict(key=key,change=sign,**rows[gid]))
        m=read(INPUT/key/'manifest.json');rec=m['records'][len(m['records'])//2];source=np.load(m['geometry'])['xyz']
        low=np.quantile(source[:,:2],.005,axis=0);high=np.quantile(source[:,:2],.995,axis=0)
        fig=plt.figure(figsize=(14,9),layout='constrained');grid=fig.add_gridspec(2,3,height_ratios=[.8,1.5])
        ax=fig.add_subplot(grid[0,:]);ax.imshow(Image.open(rec['rgb']));ax.axis('off');ax.set_title(f'{key} | real RGB frame {rec["frame_id"]}',loc='left',weight='bold')
        for j,method in enumerate(['baseline','ours_append_only','ours_revision']):
            ax=fig.add_subplot(grid[1,j]);g=np.load(OUT/'maps'/key/method/'geometry.npz');xyz=g['xyz'];ids=g['instance'];stride=max(1,(len(xyz)+89999)//90000)
            ax.scatter(xyz[::stride,0],xyz[::stride,1],c=ids[::stride]%20,cmap='tab20',vmin=0,vmax=19,s=.4)
            ax.set_xlim(low[0],high[0]);ax.set_ylim(low[1],high[1]);ax.set_aspect('equal');ax.grid(alpha=.15);ax.set_xlabel('x / m');ax.set_ylabel('y / m')
            ap=summary['results'][key]['methods'][method]['metrics']['allAp']*100
            ax.set_title(f'{NAMES[method]}\nAP50:90 {ap:.3f}',fontsize=10)
        fig.suptitle('Same object IDs and colors · exact original / measured coordinates')
        file=DEST/f'{key}.png';fig.savefig(file,dpi=150);plt.close(fig);figures.append(dict(file=file.name,rgb=rec['rgb'],rgb_sha256=sha(rec['rgb'])))
    dump(DEST/'all_objects.json',objects);dump(DEST/'changed_matches.json',changes)
    dump(DEST/'point_target_agreement.json',dict(results=point_agreement,
        scope='Post-scoring diagnosis against the fixed training-target construction for each exposed window. This is candidate-set target agreement, not semantic ground-truth accuracy, AP or independent validation. A known point without a GT-matched candidate has null as its target. Same common eligible queries and reference-confidence weights across methods.'))
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained');trained=read(OUT/'trained.json')
    for ax,key in zip(axes,objects):
        for name,row in trained['stats'].items():
            if name.startswith(key+'/'):ax.plot(np.arange(1,41),row['loss'],label=NAMES[name.split('/')[1]])
        ax.set_title(f'Target {key} (trained on the other window)');ax.set_xlabel('Epoch');ax.set_ylabel('Confidence-weighted grouped NLL');ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.savefig(DEST/'training.png',dpi=150);plt.close(fig)
    # Show every changed AP50 match; the selection is exhaustive, not a best-case montage.
    for event in changes:
        key=event['key'];gid=event['gt'];scene=next(s for s in p['scenes'] if s['key']==key);ref=np.load(scene['reference'])
        truth_xyz=ref['xyz'][ref['train_instance']==gid];low=truth_xyz[:,:2].min(0)-1;high=truth_xyz[:,:2].max(0)+1
        fig,axes=plt.subplots(1,3,figsize=(12,4.5),layout='constrained')
        for ax,method in zip(axes,['baseline','ours_append_only','ours_revision']):
            ax.scatter(truth_xyz[::max(1,len(truth_xyz)//15000),0],truth_xyz[::max(1,len(truth_xyz)//15000),1],c='#cdd2d8',s=.5,label='GT surface')
            g=np.load(OUT/'maps'/key/method/'geometry.npz');oid=event['methods'][method]['predicted_id'];take=g['instance']==oid if oid is not None else np.zeros(len(g['xyz']),bool)
            points=g['xyz'][take];within=(points[:,0]>=low[0])&(points[:,0]<=high[0])&(points[:,1]>=low[1])&(points[:,1]<=high[1]);points=points[within]
            ax.scatter(points[:,0],points[:,1],c='#008a89',s=1,label='Predicted object')
            ax.set_xlim(low[0],high[0]);ax.set_ylim(low[1],high[1]);ax.set_aspect('equal');ax.set_xlabel('x / m');ax.set_ylabel('y / m');ax.grid(alpha=.15)
            ax.set_title(f'{NAMES[method]}\nIoU {event["methods"][method]["iou"]:.4f}',fontsize=9)
        axes[0].legend(fontsize=8);fig.suptitle(f'{key} | GT {gid} | {event["change"]} AP50 match (all changed matches shown)')
        file=DEST/f'{key}_gt{gid}.png';fig.savefig(file,dpi=150);plt.close(fig);event['image']=file.name
    dump(DEST/'changed_matches.json',changes)
    table=[]
    for method in METHODS:
        values=[summary['results'][k]['methods'][method]['metrics']['allAp']*100 for k in objects]
        table.append('<tr><td>'+NAMES[method]+'</td>'+''.join(f'<td>{v:.3f}</td>' for v in values)+f'<td>{summary["mean_primary_ap"][method]*100:.3f}</td></tr>')
    point_rows=[]
    for key,row in point_agreement.items():
        for method in METHODS:point_rows.append(f'<tr><td>{key}</td><td>{NAMES[method]}</td>'+''.join(f'<td>{row[method][g]["weighted_target_agreement"]*100:.2f}%</td>' for g in ['all','old','new'])+'</tr>')
    per_object=[]
    for key,rows in objects.items():
        for r in rows:per_object.append(f'<tr><td>{key}</td><td>{r["gt"]}</td><td>{r["category"]}</td>'+''.join(f'<td>{r["methods"][m]["iou"]:.4f}</td>' for m in METHODS)+'</tr>')
    verdict='通过开发门槛，仍需独立测试与同版本机器人对照' if summary['advance_gate'] else '未通过开发门槛；本版封存，原强基线保留'
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam — 有来源的对象记忆实验</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;color:#17354a;background:#f4f7fa;max-width:1200px;margin:32px auto;padding:0 24px}}h1,h2{{line-height:1.25}}section{{background:white;padding:24px;margin:20px 0;border-radius:10px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #dce5ec;text-align:left}}img{{width:100%;height:auto}}.status{{padding:16px;border-left:5px solid #078a8a;background:#edf8f6}}small{{color:#536b7d}}</style>
<h1>omni_slam：有来源的对象记忆与历史归属修订</h1><p>2026-11-14 · 八次固定训练 · 十二份完整室外地图</p><p class="status">{verdict}。Ours 是本轮预指定候选，不表示已证明优于对照。</p>
<section><h2>检验什么</h2><p>保留强基线已有对象ID，把精确实测源点的逐视角外观存入对应对象记忆。对象未被观察时不更新；没有随机槽位或全场景混合更新。查询点读取该对象的视角证据，学习保留、移出或重新分配旧地图点，并接纳候选实测补点。</p>
<p>候选的关键假设：排除与查询共享VGGT图像上下文的视角，再做条件化读取，是否比平均、最高描述子相似度检索、全部视角注意力更可靠。第四个对照使用完全相同权重和预测，只补点，不改旧点。平均对照平均学习后的逐视角证据，检索对照按原始描述子余弦选择视角；四种训练模型架构和参数量相同。</p>
<p>这是窗口结束后的固定预测身份修订，没有新增对象ID，不解决全部漏检实例。对象身份来自已有全窗口预测，因此不能称因果在线发现。“输入不重叠”只指显式VGGT图像缓存，不是统计独立；上游对象关联和几何仍共享信息。保留身份和来源本身不是创新证明。</p></section>
<section><h2>完整地图成绩</h2><p>building/car，AP50:90 ×100，本地适配完整静态窗口；不是官方排行榜，也不是独立测试。0000/0002均已用于开发，父级地图继承此前交叉训练，不能称严格嵌套泛化。</p>
<table><thead><tr><th>条件</th><th>0000</th><th>0002</th><th>均值</th></tr></thead><tbody>{''.join(table)}</tbody></table>
<p>Ours 新增 {summary['gained']}、丢失 {summary['lost']} 个 AP50 GT 匹配。相同对象数、低训练损失或局部涨点不能替代全局门槛。</p></section>
<section><h2>真实图像与完整地图</h2>{''.join(f'<img src="{r["file"]}" alt="{r["file"]}">' for r in figures)}
<p>使用同一坐标范围与实例配色。显示范围按原实测源点XY的0.5%–99.5%分位统一裁剪，评分不裁剪；显示仅做确定性抽样。所有坐标来自原地图或实际测量。</p></section>
<section><h2>全部新增和丢失匹配</h2>{''.join(f'<img src="{e["image"]}" alt="GT {e["gt"]}">' for e in changes)}
<p>这里穷举本轮Ours改变的AP50匹配，没有只挑正例。每图显示GT边界外扩1米的XY范围；IoU按完整参考域计算，不按裁剪图计算。灰色是真实参考表面，绿色是同类最大IoU预测。</p></section>
<section><h2>训练及其目标与地图指标的差别</h2><img src="training.png" alt="Eight fixed training curves">
<table><thead><tr><th>窗口</th><th>条件</th><th>全部查询</th><th>旧点查询</th><th>新点查询</th></tr></thead><tbody>{''.join(point_rows)}</tbody></table>
<p>上表是评分后按原训练目标定义核查的置信度加权一致率，使用所有共同合格查询。不是语义GT准确率或官方指标：没有GT匹配候选时，训练目标就是null；没有候选的真实对象不能因此算作已恢复。它帮助判断点级优化与完整对象地图收益是否一致，不能取代AP或证明因果。</p></section>
<section><h2>审计与边界</h2><p>5项核心测试通过。818条对象视角记录、245300条源点观测的来源及描述子全部重放；所有查询的图像隔离、候选有效性和最后观测语义值核对通过。对旧点新增竞争对象的0.4米半径逐项核对；视觉余弦抽样核验11220项，不能称全部余弦逐项重算。</p>
<p>十二份预测先封存后评分，强基线分数精确复现；未知/缺测旧点保持原归属，所有新增点有唯一实测来源，现有对象类别和排序不变。没有新基础模型调用，没有参数搜索或按结果重新挑选Ours。原全局随机槽位失败版本未更改。</p>
<p><a href="../../build/benchmarks/provenance_object_memory_v1/protocol.json">冻结协议</a> · <a href="../../build/benchmarks/provenance_object_memory_v1/input_audit.json">输入核验</a> · <a href="../../build/benchmarks/provenance_object_memory_v1/map_audit.json">地图核验</a> · <a href="summary.json">完整指标</a> · <a href="point_target_agreement.json">点级目标核查</a> · <a href="all_objects.json">全部94个对象</a></p></section>
<section><h2>全部对象IoU</h2><p>各GT与同类预测的最大加权IoU，仅作几何诊断，不是额外的一对一匹配成绩；未预测对象仍记录为0。</p><div style="overflow:auto"><table><thead><tr><th>窗口</th><th>GT</th><th>类别</th>{''.join(f'<th>{NAMES[m]}</th>' for m in METHODS)}</tr></thead><tbody>{''.join(per_object)}</tbody></table></div></section>
<section><h2>主线状态</h2><p>omni_slam对象级语义建图仍是主线，LIVO为定位与实测几何支撑。本轮落实了稳定身份的对象记忆和真实历史归属修订，但稳定增益仍需证据；机器人对照和新增独立评测没有被旧链路或本轮开发窗口替代。</p></section></html>'''
    (DEST/'index.html').write_text(page);dump(DEST/'summary.json',summary);dump(DEST/'images.json',figures)
    dump(DEST/'manifest.json',dict(complete=True,files={str(f):sha(f) for f in DEST.iterdir() if f.is_file() and f.name!='manifest.json'},
        source_sha256=sha(__file__),advance_gate=summary['advance_gate']))
    print('PROVENANCE REPORT',DEST/'index.html','all GTs',sum(map(len,objects.values())),flush=True)
    for key,rows in point_agreement.items():print('POINT TARGET AGREEMENT',key,{m:r['all']['weighted_target_agreement'] for m,r in rows.items()},flush=True)
    for e in changes:print('CHANGED OBJECT',e['key'],e['gt'],e['change'],{m:r['iou'] for m,r in e['methods'].items()},flush=True)


if __name__=='__main__':run()
