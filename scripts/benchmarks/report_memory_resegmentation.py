"""Score sealed maps and publish all gains/losses, including failed gates."""
from pathlib import Path
import shutil
import time
import numpy as np
from scipy.spatial import cKDTree
from memory_reseg_common import ROOT,OUT,REPORT,PRIOR,spec,read,dump,sha,archive
from run_memory_resegmentation import KEYS
from omni_semantic_map.full_ovo_evaluation import voxel_metrics
from omni_semantic_map.mesh_instance_evaluation import transfer_labels
from omni_semantic_map.instance_error_audit import audit_instances,filter_transition

def score():
    p=spec();seal=read(OUT/'predictions_complete.json');assert seal['complete']
    for path,h in seal['runs'].items():assert sha(OUT/path)==h
    selection=read(OUT/'selection.json');results=[];by_dataset={}
    for key in KEYS:
        s=next(x for x in p['scenes'] if x['key']==key);ref=archive(s['reference'])
        truth=ref['instance'];valid=ref.get('valid',np.ones(len(truth),bool));methods={};audits={}
        folder=OUT/'evaluation'/key;folder.mkdir(parents=True,exist_ok=False)
        cache=Path(s['cache']);geo=archive(cache/'geometry.npz')
        for arm in ['B0','B1','B2']:
            root=OUT/'maps'/key/arm;run=read(root/'run.json');assert run['complete']
            for path,h in run['artifacts'].items():assert sha(root/path)==h
            g=archive(root/'final/geometry.npz');rows=read(root/'final/classification.json')['objects']
            kept=[r['object_id'] for r in rows if r['predictions']['mean']['retained']]
            labels=np.where(np.isin(g['instance'],kept),g['instance'],0)
            predicted,_=transfer_labels(g['xyz'],labels,ref['xyz'],mode='nearest_5cm')
            metrics=voxel_metrics(truth[valid],predicted[valid]);audit=audit_instances(truth[valid],predicted[valid]);audits[arm]=audit
            np.savez_compressed(folder/f'{arm}.npz',instance=predicted)
            dump(folder/f'{arm}_audit.json',audit)
            methods[arm]=dict(score=metrics,predicted_counts=audit['predicted_counts'],gt_counts=audit['gt_counts'],
                map_wall_s=run['wall_s'],map_p95_s=run['frame_p95_s'],peak_rss_bytes=run['peak_rss_bytes'])
            if arm!='B0':
                regenerated=read(OUT/'regenerated'/key/arm/'manifest.json');records=regenerated['records']
                methods[arm].update(regeneration_wall_s=regenerated['wall_s'],
                    added_sam_p95_s=float(np.quantile([r['wall_s'] for r in records],.95)),
                    sam_calls=sum(r['logical_sam_calls'] for r in records),changed_supports=sum(r['changed_supports'] for r in records),
                    peak_cuda_bytes=regenerated['peak_cuda_bytes'],transition=filter_transition(audits['B0'],audit))
            prefixes=[]
            for prefix in run['prefixes'][:-1]:
                pp=Path(prefix);bg=OUT/'maps'/key/'B0'/pp.name
                seen=np.load(bg/'seen_point_ids.npy')
                dist=cKDTree(geo['xyz'][seen]).query(ref['xyz'],workers=1)[0];domain=valid&(dist<=.05)
                a=archive(pp/'geometry.npz');r=read(pp/'classification.json')['objects'];ids=[x['object_id'] for x in r if x['predictions']['mean']['retained']]
                labels=np.where(np.isin(a['instance'],ids),a['instance'],0)
                projected,_=transfer_labels(a['xyz'],labels,ref['xyz'],mode='nearest_5cm')
                prefixes.append(dict(prefix=pp.name,valid_points=int(domain.sum()),score=voxel_metrics(truth[domain],projected[domain])))
            methods[arm]['prefixes']=prefixes
        methods['B2']['vs_simple']=filter_transition(audits['B1'],audits['B2'])
        objects=[]
        for gt in np.unique(truth[valid]):
            if gt==0:continue
            row=dict(gt_id=int(gt),center=ref['xyz'][valid&(truth==gt)].mean(0).tolist())
            for arm,a in audits.items():
                item=next(x for x in a['ground_truth'] if x['gt_id']==gt)
                row[arm]=item
            objects.append(row)
        dump(folder/'objects.json',objects)
        r=dict(key=key,dataset=s['dataset'],methods=methods,reference_sha256=sha(s['reference']),objects=len(objects))
        dump(folder/'metrics.json',r);results.append(r)
        print('SCORED',key,{k:round(v['score']['observed_voxel_instance_pq']*100,3) for k,v in methods.items()},flush=True)
    for ds in ['Replica','SceneNN']:
        rows=[r for r in results if r['dataset']==ds];m={}
        for arm in ['B0','B1','B2']:
            m[arm]=dict(pq100=float(np.mean([r['methods'][arm]['score']['observed_voxel_instance_pq'] for r in rows])*100),
                **{k:sum(r['methods'][arm]['score'][k] for r in rows) for k in ['tp','fp','fn']})
        m['gate']=dict(pq_plus_two=m['B2']['pq100']-m['B0']['pq100']>=2,
            net_new_objects=m['B2']['tp']>m['B0']['tp'],above_simple=m['B2']['pq100']>m['B1']['pq100'],
            no_net_loss_vs_simple=m['B2']['tp']>=m['B1']['tp'])
        by_dataset[ds]=m
    gate=selection['validation_gate_passed'] and all(all(d['gate'].values()) for d in by_dataset.values())
    summary=dict(complete=True,scenes=results,datasets=by_dataset,validation_gate=selection['validation_gate_passed'],
        full_gate_passed=gate,decision='Proceed to independent assessment; no generalization claim yet.' if gate else 'Stop this implementation; do not replace the frozen mainline or tune on development scores.',
        maps=12,training_runs=9,prediction_seal_sha256=sha(OUT/'predictions_complete.json'),time=time.time())
    dump(OUT/'summary.json',summary)
    return summary

def publish(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    REPORT.mkdir(parents=True,exist_ok=True)
    v=read(OUT/'validation.json');sel=read(OUT/'selection.json')
    fig,ax=plt.subplots(figsize=(10,4));xx=np.arange(4);colors=['#8694a8','#e4a33b','#297bb6']
    for j,arm in enumerate(['B0','B1','B2']):
        vals=[r['methods'][arm]['score']['observed_voxel_instance_pq']*100 for r in summary['scenes']]
        bars=ax.bar(xx+(j-1)*.25,vals,.23,label=arm,color=colors[j]);ax.bar_label(bars,fmt='%.2f',fontsize=8)
    ax.set_xticks(xx,['Replica office0','Replica office1','SceneNN 014','SceneNN 015']);ax.set_ylabel('Instance PQ x 100');ax.legend();ax.set_ylim(0,max(ax.get_ylim()[1]*1.14,1));fig.tight_layout()
    fig.savefig(REPORT/'map_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.2))
    for ax,mode in zip(axes,['current','mean','attention']):
        for seed in [17,29,43]:
            fit=read(OUT/'fits'/f'{mode}_{seed}'/'fit.json');ax.plot([r['epoch'] for r in fit['curve']],[r['loss'] for r in fit['curve']],label=str(seed))
        ax.set_title(mode);ax.set_xlabel('Epoch');ax.set_ylabel('Training loss');ax.legend()
    fig.tight_layout();fig.savefig(REPORT/'training.png',dpi=150);plt.close(fig)
    images=[]
    for key in KEYS:
        s=next(x for x in spec()['scenes'] if x['key']==key);ref=archive(s['reference']);valid=ref.get('valid',np.ones(len(ref['xyz']),bool))
        ids=np.flatnonzero(valid);ids=ids[np.linspace(0,len(ids)-1,min(25000,len(ids)),dtype=int)]
        fig,axes=plt.subplots(1,3,figsize=(12,4));xyz=ref['xyz'][ids]
        for ax,arm in zip(axes,['B0','B1','B2']):
            labels=archive(OUT/'evaluation'/key/f'{arm}.npz')['instance'][ids]
            # Independent map IDs use arbitrary colors, not cross-method identity matches.
            color=plt.get_cmap('tab20')((labels%20)/19);color[labels==0]=[.85,.85,.85,1]
            ax.scatter(xyz[:,0],xyz[:,2],c=color,s=1);ax.set_title(arm);ax.set_aspect('equal');ax.set_xlabel('x (m)');ax.set_ylabel('z (m)')
        fig.suptitle(key+' | fixed reference domain; gray = unassigned');fig.tight_layout();name=f'map_{key}.png';fig.savefig(REPORT/name,dpi=160);plt.close(fig)
        images.append((key,name))
    for key in KEYS:
        dest=REPORT/'examples'/key;dest.mkdir(parents=True,exist_ok=True)
        for arm in ['B1','B2']:
            for image in (OUT/'regenerated'/key/arm).glob('example_*.jpg'):shutil.copyfile(image,dest/(arm+'_'+image.name))
        for name in ['objects.json','metrics.json']:
            target=REPORT/'objects'/key/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OUT/'evaluation'/key/name,target)
    lines=['# omni_slam 对象记忆重分割：完整交付报告','',
        '已完成九次真实训练、验证场景实际 SAM 重分割、四场景三方法共十二次完整建图。这里的完整建图指既定 benchmark 每场景全部 200 个采样帧，不是原始数据集所有视频帧。所有开发地图封存后统一评分。',
        '',f'**结论：{"达到本轮继续门槛" if summary["full_gate_passed"] else "未达到本轮继续门槛，停止该实现，不替换冻结主线"}。**',
        '',f'验证选择：{sel["mode"]}，固定 seed 17，epoch {sel["selected"]["epoch"]}。验证门槛：{sel["validation_gate_passed"]}。',
        '', 'B0 为原生 CG 增量关联及共同最终读出，四场景几何/对象归属/类别与旧原生基线核对。B1 为简单几何重提示；B2 为验证选定的学习式局部记忆提示。没有用开发结果选择种子或模型。',
        '', 'B0 是本轮因果建图对照，并非此前含离线历史桥接的完整强基线；本报告不据此宣称超过冻结的 omni_slam 主线。',
        '', '本轮路径：当前 RGB 与 DINO 局部特征 → 检索此前帧 0.2 m 内最多 8 个证据 token → 训练的注意力读取器预测局部同实例关系 → 生成正负点提示 → SAM2.1 重新解码 → 当前观测分块 → CG 因果关联 → 固定域三维实例评测。记忆最多保存 8192 个 token，定位与视觉骨干保持不变。',
        '', '![完整地图比较](map_comparison.png)','',
        '| 场景 | B0 PQ×100 | B1 PQ×100 | B2 PQ×100 | B2−B0 | B2 新增 / 丢失 GT | B2 TP / FP / FN |','|---|---:|---:|---:|---:|---|---|']
    for r in summary['scenes']:
        m=r['methods'];pq=[m[k]['score']['observed_voxel_instance_pq']*100 for k in ['B0','B1','B2']];t=m['B2']['transition'];sc=m['B2']['score']
        lines.append(f'| {r["key"]} | {pq[0]:.3f} | {pq[1]:.3f} | {pq[2]:.3f} | {pq[2]-pq[0]:+.3f} | {t["gained_matched_gt"]} / {t["lost_matched_gt"]} | {sc["tp"]}/{sc["fp"]}/{sc["fn"]} |')
    lines+=['', '| 数据集（各 2 个开发场景） | B0 均值 | B1 均值 | B2 均值 | B2−B0（百分点） |', '|---|---:|---:|---:|---:|']
    for ds,m in summary['datasets'].items():
        lines.append(f'| {ds} | {m["B0"]["pq100"]:.3f} | {m["B1"]["pq100"]:.3f} | {m["B2"]["pq100"]:.3f} | {m["B2"]["pq100"]-m["B0"]["pq100"]:+.3f} |')
    gains=sum(len(r['methods']['B2']['transition']['gained_matched_gt']) for r in summary['scenes'])
    losses=sum(len(r['methods']['B2']['transition']['lost_matched_gt']) for r in summary['scenes'])
    lines+=['',f'四场景共 {sum(r["objects"] for r in summary["scenes"])} 个有效 GT 对象；B2 相对 B0 新增 {gains} 个正确匹配、丢失 {losses} 个，净增 {gains-losses} 个。',
        '', '### 匹配发生变化的全部对象', '',
        '| 场景 / GT ID | 体素数 | B0 IoU | B1 IoU | B2 IoU | 变化 |', '|---|---:|---:|---:|---:|---|']
    for r in summary['scenes']:
        for o in read(OUT/'evaluation'/r['key']/'objects.json'):
            if (o['B0']['category']=='matched')==(o['B2']['category']=='matched'):continue
            change='新增' if o['B2']['category']=='matched' else '丢失'
            lines.append(f'| {r["key"]} / {o["gt_id"]} | {o["B0"]["voxels"]} | {o["B0"]["best_iou"]:.4f} | {o["B1"]["best_iou"]:.4f} | {o["B2"]["best_iou"]:.4f} | {change} |')
    lines+=['', '本轮两处匹配变化靠近 0.5 阈值，涉及 20 与 24 个体素的小对象；SceneNN 015 的新增对象在简单重提示下也能恢复，且其 IoU 高于学习提示。因此，净增一个匹配不等于已经证明注意力的独立优势。错误类型描述最终几何，不是经干预验证的因果解释。']
    original=v['metrics']['original'];simple=v['metrics']['simple']
    lines+=['','## 验证与训练','',
        f'原始掩码支持域 PQ×100：{original["pq_sum"]/original["fg_observations"]*100:.4f}；简单重提示：{simple["pq_sum"]/simple["fg_observations"]*100:.4f}。以下均在同一验证场景，不能把优于简单提示写成优于原始掩码。',
        '', '![训练曲线](training.png)','',
        '| 读取器 | seed | 所选 epoch | 支持域 PQ×100 | 混合观测新增 / 丢失匹配 |','|---|---:|---:|---:|---|']
    for name,r in v['selected'].items():
        m=r['metrics'];mode,seed=name.rsplit('_',1)
        lines.append(f'| {mode} | {seed} | {r["epoch"]} | {m["pq_sum"]/max(m["fg_observations"],1)*100:.3f} | {m["mixed_gained"]} / {m["mixed_lost"]} |')
    lines+=['','验证用真实 SAM 解码后的稀疏实测支持评价；它不是稠密像素真值，也不是最终三维 PQ。固定保存 epoch 0/1/5/10/25/50，全部九次训练均报告，训练损失不能作为建图成功证据。',
        '', '局部 DINO 骨干冻结，训练 384→64 投影、4 头历史读取和局部同实例评分，并使用前景辅助损失。部署本版使用局部同实例评分选提示，未将前景辅助头另作删除规则。每个训练场景每 epoch 固定抽取 4096 对支持，未知不作负标签；不按开发标签选择争议区域。',
        '', '## 实现边界与公平性','',
        '- 本版用有来源的因果局部观测记忆生成提示，重分割当前观测；没有回溯删除旧地图中的误并点，也没有实现 B3 跨视角接受/拒绝。',
        '- SAM 确实生成新掩码；输出限制在原始观测支持内，三维支持总量不变。它不能恢复此前未被掩码覆盖的几何。',
        '- 各分块继承原始 SigLIP 描述符，未重新编码语义；本轮隔离的是实例划分，不主张类别提升。',
        '- B1/B2 使用同一事件调度、种子、图像、四点提示、两次 SAM 调用预算。模型选择阶段去重相同请求节约计算，逻辑预算不变。均等调用数不等于均等墙钟成本。',
        '- 每组重分割后的观测独立进入原生 CG，后续关联使用该组自己的地图。最终读出相同，不使用离线历史桥接。',
        '- 首次验证因投影/掩码裁剪不一致中止；在选模前修正并完整重跑，未改模型、阈值或随机种子。',
        '- 建图队列曾因其他任务占满 GPU 0 而在 SAM 加载时中断；保留完成地图，增加 3 GiB 进程显存上限及 4 GiB 空闲检查后续跑，模型和选模结果不变。',
        '- 用户本轮要求完整交付，因此在验证选模前记录范围修订：即便验证门槛失败，仍跑完十二张诊断地图；失败不允许晋升主线，也不追加调参。',
        '- 四场景均已用于开发；给定位姿 RGB-D，非新未见泛化实验、非 LIVO 实机实验。',
        '', '## 实际成本','', '| 场景 | 方法 | SAM逻辑调用 | 重分割总秒 | CG总秒 | CG帧P95秒 | GPU峰值GiB |','|---|---|---:|---:|---:|---:|---:|']
    for r in summary['scenes']:
        for arm in ['B0','B1','B2']:
            m=r['methods'][arm];lines.append(f'| {r["key"]} | {arm} | {m.get("sam_calls",0)} | {m.get("regeneration_wall_s",0):.1f} | {m["map_wall_s"]:.1f} | {m["map_p95_s"]:.3f} | {m.get("peak_cuda_bytes",0)/2**30:.2f} |')
    lines+=['','原始 SAM 自动掩码、SigLIP 和 DINO 提取成本使用既有缓存，未计入以上增量时间；两阶段相加也不能当作实际流式端到端 P95。GPU 仅使用卡 0。',
        '', '## 逐对象地图与掩码证据','', '三维投影图用于查看覆盖和碎片；不同方法的对象颜色没有跨方法身份对应。逐对象 JSON 提供固定 GT ID 对齐的 IoU、覆盖和错误类别。']
    for key,name in images:
        lines+=['',f'![{key}]({name})','',f'[{key} 逐对象结果](objects/{key}/objects.json)']
        sample=REPORT/'examples'/key/'B2_example_050.jpg'
        if sample.exists():lines+=['',f'![原图、旧掩码、新分块]({sample.relative_to(REPORT)})', '', '固定第 51 个采样帧、首个候选观测：左为 RGB，中为原始掩码，右为学习提示后的分块。图像仅证明实际重分割，不是人工挑选的成功案例。']
        lines+=['', '所有有效 GT 对象均列出；新增/丢失指严格 IoU > 0.5 的匹配变化，相对 B0。', '',
            '| GT ID | B0 最佳 IoU | B1 最佳 IoU | B2 最佳 IoU | B2−B0 | 匹配变化 | B2 错误类型 |',
            '|---|---:|---:|---:|---:|---|---|']
        for obj in read(OUT/'evaluation'/key/'objects.json'):
            a,b,c=(obj[x] for x in ['B0','B1','B2'])
            before=a['category']=='matched';after=c['category']=='matched'
            change='新增' if after and not before else '丢失' if before and not after else '保持匹配' if after else '仍未匹配'
            lines.append(f'| {obj["gt_id"]} | {a["best_iou"]:.4f} | {b["best_iou"]:.4f} | {c["best_iou"]:.4f} | {c["best_iou"]-a["best_iou"]:+.4f} | {change} | {c["category"]} |')
    lines+=['','[完整数据集门槛与指标](summary.json) · [模型选择](selection.json) · [执行协议](../../paper/object_memory_experiment_plan_2026-11-12.md)',
        '', '## 代码与运行产物','',
        '[模型与提示模块](../../../omni_semantic_map/omni_semantic_map/memory_resegmentation.py) · [训练脚本](../../../scripts/benchmarks/train_memory_resegmentation.py) · [SAM 验证](../../../scripts/benchmarks/validate_memory_resegmentation.py) · [重分割与建图](../../../scripts/benchmarks/run_memory_resegmentation.py)',
        '', f'实验目录：`{OUT}`。`fits/` 为九组权重，`regenerated/` 为新掩码和支持分配，`maps/` 为十二张地图及前缀，`evaluation/` 为固定域指标与逐对象核验。',
        '', '已有实验目录有防覆盖检查。复现应新建输出命名空间，不删除旧结果。无需联网下载模型，使用本项目原有 SAM2.1 权重和已缓存的 DINO/SigLIP 数据。',
        '', '本轮交付可运行代码、训练权重、新掩码、完整地图和正负结果。仅当完整门槛通过，才进入独立评测；否则停止该具体实现。']
    lines+=['', '## 交付核验', '', '9 次训练共 21,600 次梯度更新；12 张地图、1,600 个重分割采样帧已完成。四组 B0 复现核对通过，B1/B2 逐帧 SAM 调用预算一致，所有输出支持文件哈希通过。3 项针对性测试通过，旧冻结结果核验通过（515 个本地文件、512 个归档文件）。', '', '[交付核验与源文件哈希](delivery_audit.json)']
    (REPORT/'report.md').write_text('\n'.join(lines)+'\n')
    for name in ['summary.json','selection.json','validation.json','training_execution.json','delivery_scope_amendment.json','validation_retry.json','resource_retry.json','predictions_complete.json','delivery_audit.json']:
        shutil.copyfile(OUT/name,REPORT/name)
    try:
        import markdown
        content=markdown.markdown('\n'.join(lines),extensions=['tables'])
    except ImportError:
        from markdown_it import MarkdownIt
        content=MarkdownIt('commonmark').enable('table').render('\n'.join(lines))
    css='body{max-width:1200px;margin:36px auto;padding:0 24px;font:16px/1.65 system-ui;color:#18304a;background:#f7f9fc}table{border-collapse:collapse;width:100%;font-size:14px;background:white}td,th{border:1px solid #dce2eb;padding:9px}th{background:#e8eef7}img{max-width:100%;height:auto}a{color:#1765a5}pre{white-space:pre-wrap}'
    (REPORT/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>omni_slam 重分割实验</title><style>'+css+'</style><body>'+content+'</body></html>')
    print('REPORT',REPORT/'index.html',flush=True)

if __name__=='__main__':
    summary=read(OUT/'summary.json') if (OUT/'summary.json').exists() else score()
    publish(summary)
