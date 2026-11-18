"""Read-only posthoc audit and report; never feeds model fitting or inference."""
from pathlib import Path
from html import escape
import shutil
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from run_object_operation_gate import OUT,OLD,METHODS,CONTROLS,verify,arrays
from omni_semantic_map.learned_observation_revision import rollback
from omni_semantic_map.mesh_instance_evaluation import transfer_labels
from omni_semantic_map.operation_teacher import OperationTeacher

REPORT=ROOT/'output/object_operation_gate_v1'
NAMES=dict(native_replay='原归属回放',merge_only='原学习边：只合并',reassign_history='原历史重分配',
    gated_merge_only='对象收益筛选：只合并',utility_only='只预测对象收益',gated_revision='Ours 候选：对象收益 + 丢失风险')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=verify(True);summary=read(OUT/'summary.json');seal=read(OUT/'predictions_complete.json');trained=read(OUT/'trained.json')
    assert sha(OUT/'predictions_complete.json')==summary['prediction_seal_sha256']
    assert sha(OUT/'operation_models.joblib')==trained['model_sha256']
    for f,h in seal['files'].items():assert sha(f)==h,f
    REPORT.mkdir(parents=True,exist_ok=True);checks={};posthoc={};table='';details=''
    for s in p['scenes']:
        if s['role']=='train':continue
        key=s['key'];a=arrays(key);native=a['native'];scenecheck={};alllabels={};runs={}
        for method in CONTROLS+METHODS:
            folder=(OLD if method in CONTROLS else OUT)/key/method;labels=np.load(folder/'observation_identity.npy');events=read(folder/'events.json')
            with np.load(folder/'geometry.npz') as g:
                np.testing.assert_array_equal(g['xyz'],a['source_xyz']);np.testing.assert_array_equal(g['source_indices'],a['active_source_indices'])
                np.testing.assert_array_equal(g['instance'],labels[a['source_owner']])
            np.testing.assert_array_equal(rollback(labels,events),native)
            alllabels[method]=labels;runs[method]=read(folder/'run.json')
            scenecheck[method]=dict(source_verified=True,rollback_verified=True,changed_source_point_ids=int(np.count_nonzero(labels[a['source_owner']]!=native[a['source_owner']])),
                converged=runs[method]['converged'],events=len(events),objects=runs[method]['objects'])
            m=summary['results'][key]['methods'][method]
            if s['kind']=='indoor':
                metric=f"{100*m['observed_voxel_instance_pq']:.3f}";extra=f"TP {m['tp']} / FP {m['fp']} / FN {m['fn']}"
                change=summary['results'][key]['changes_vs_native'][method];extra+=f"；新增 {len(change['gained_matched_gt'])} / 丢失 {len(change['lost_matched_gt'])}"
            elif s['kind']=='outdoor':metric=f"{100*m['allAp50%']:.3f}";extra='AP50；本地适配开发窗口'
            else:metric='未标注';extra=f"{runs[method]['objects']} 个对象；{len(events)} 次操作"
            table+=f'<tr><td>{key}</td><td>{NAMES[method]}</td><td>{metric}</td><td>{extra}</td></tr>'
        same=np.array_equal(alllabels['utility_only'],alllabels['gated_revision'])
        assert read(OUT/key/'utility_only/events.json')==read(OUT/key/'gated_revision/events.json') if same else True
        scenecheck['risk_changes_final_partition']=not same;checks[key]=scenecheck
        if s['kind']=='indoor':
            ref=np.load(s['reference']);valid=ref['valid'] if 'valid' in ref else np.ones(len(ref['instance']),bool)
            truth=ref['instance'][valid];owner,_=transfer_labels(a['source_xyz'],a['source_owner']+1,ref['xyz'][valid],mode='nearest_5cm')
            gids=np.unique(np.r_[0,truth]);gt=np.searchsorted(gids,truth);ng=len(gids);known=owner>0
            counts=np.bincount((owner[known]-1)*ng+gt[known],minlength=len(native)*ng).reshape(len(native),ng)
            teacher=OperationTeacher(counts,np.bincount(gt,minlength=ng));labels=native.copy();records=[]
            for d in read(OUT/key/'gated_revision/decisions.json'):
                ids=np.asarray(d['indices']);assert np.all(labels[ids]==d['source'])
                reward=teacher.reward(labels,ids,d['target']);records.append(dict(**d,**reward))
                if d['accept']:labels[ids]=d['target']
            np.testing.assert_array_equal(labels,alllabels['gated_revision'])
            initial=teacher.metrics(native);final=teacher.metrics(labels)
            np.testing.assert_allclose(final['pq'],summary['results'][key]['methods']['gated_revision']['observed_voxel_instance_pq'],atol=1e-12)
            np.testing.assert_allclose(sum(r['delta_pq'] for r in records if r['accept']),final['pq']-initial['pq'],atol=1e-12)
            stats=dict(accepted_beneficial=sum(r['accept'] and r['delta_pq']>1e-12 for r in records),
                accepted_harmful=sum(r['accept'] and r['delta_pq']< -1e-12 for r in records),
                accepted_tp_loss=sum(r['accept'] and r['delta_tp']<0 for r in records),
                rejected_beneficial=sum(not r['accept'] and r['delta_pq']>1e-12 for r in records),
                rejected_harmful=sum(not r['accept'] and r['delta_pq']< -1e-12 for r in records))
            posthoc[key]=dict(stats=stats,records=records,scope='Posthoc GT counterfactuals on actual encountered states; overlapping operations, counts are NOT independent recoverable objects. Never model inputs.')
            aud={m:{g['gt_id']:g for g in read((OLD if m in CONTROLS else OUT)/key/m/'object_audit.json')['ground_truth']} for m in ['native_replay','merge_only','gated_merge_only','gated_revision']}
            gtrows=''
            for gid in sorted(aud['native_replay']):
                values=[aud[m][gid]['best_iou'] for m in aud];delta=values[-1]-values[0]
                gtrows+=f'<tr><td>{gid}</td>'+''.join(f'<td>{v:.4f}</td>' for v in values)+f'<td class="{"up" if delta>0 else "down" if delta<0 else ""}">{delta:+.4f}</td></tr>'
            details+=f'<h2>{key}：完整地图逐对象审计</h2><p>沿实际操作轨迹，接受了 {stats["accepted_beneficial"]} 次正收益、{stats["accepted_harmful"]} 次负收益操作，其中 {stats["accepted_tp_loss"]} 次降低正确匹配数；拒绝了 {stats["rejected_beneficial"]} 次正收益操作。这是评分后的机制诊断，操作之间有依赖，不能把次数当作可新增对象数。</p><details><summary>全部 GT 对象的最佳 IoU</summary><table><tr><th>GT ID</th><th>原归属</th><th>原只合并</th><th>筛选只合并</th><th>Ours 候选</th><th>相对原归属</th></tr>{gtrows}</table></details>'
        draw=['native_replay','merge_only','gated_merge_only','gated_revision']
        fig,axes=plt.subplots(1,4,figsize=(16,4),sharex=True,sharey=True);src=a['source_xyz']
        sample=np.linspace(0,len(src)-1,min(60000,len(src)),dtype=int)
        for ax,method in zip(axes,draw):
            labels=alllabels[method][a['source_owner']]
            ax.scatter(src[sample,0],src[sample,1],c=labels[sample]%20,cmap='tab20',vmin=0,vmax=19,s=.5,rasterized=True)
            ax.set_aspect('equal');ax.set_title(('Ours (candidate)' if method=='gated_revision' else method)+f"\n{runs[method]['objects']} objects",fontsize=10);ax.set_xlabel('x / m')
        axes[0].set_ylabel('y / m');fig.suptitle(key+' | identical measured source support');fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(REPORT/(key+'_maps.png'),dpi=150);plt.close(fig)
        details+=f'<h3>{key}：真实源点地图</h3><img src="{key}_maps.png"><p>所有方法的原始点坐标、支持集完全相同。颜色仅编码预测 ID，不能代替准确率；二维俯视抽样最多 60,000 点，指标使用完整参考域。Ours 候选改变 ID 的源点数：{scenecheck["gated_revision"]["changed_source_point_ids"]}。</p>'
    # Real robot frames, unchanged and linked to capture provenance.
    job=read(ROOT/'build/benchmarks/evidence_retention_20261111/jobs/livo_full_v2.json');frames=job['frames'];images=[]
    for idx in [0,len(frames)//2,len(frames)-1]:
        f=frames[idx];source=Path(f['rgb']);assert sha(source)==f['sha256'][str(source)]
        dest=REPORT/f'robot_frame_{f["frame_id"]:06d}.png';shutil.copyfile(source,dest)
        images.append(dict(frame_id=f['frame_id'],source=str(source),sha256=sha(source),export=dest.name))
    dump(REPORT/'posthoc_operation_audit.json',dict(analysis_only=True,not_training_input=True,scenes=posthoc,prediction_seal_sha256=sha(OUT/'predictions_complete.json')))
    dump(REPORT/'verification.json',dict(complete=True,prediction_files_verified=len(seal['files']),checks=checks,robot_images=images,
        source_sha256=sha(__file__),summary_sha256=sha(OUT/'summary.json'),model_sha256=trained['model_sha256']))
    for name in ['protocol.json','summary.json','trained.json','training_complete.json']:shutil.copyfile(OUT/name,REPORT/name)
    verdict='通过开发验收，尚需独立验证' if summary['advance_gate'] else '未通过冻结开发验收；保留强基线，不提升为主线 Ours'
    intro=f'''<h1>omni_slam：对象级修订收益学习</h1><p class="lead">{verdict}。</p>
<p>主线是持续对象级语义地图，LIVO 提供机器人位姿与几何。候选机制学习在提交历史观测归属变化前，预测完整对象地图的收益及正确对象丢失风险。不是再次调整边阈值；也不是新增类别读出。</p>
<h2>一次冻结实验</h2><p>四个训练场景交叉拟合产生 {trained['operations']} 个操作：{trained['positive_reward']} 个正 PQ 收益、{trained['negative_reward']} 个负收益、{trained['risk_labels']} 个降低 TP。仅训练一次收益回归器和风险分类器；风险阈值固定为 0.1。训练收益是整个参考域上的精确 PQ 变化，模块输入不含 GT。12 份预测全部封存后评分，旧三组控制重新评分复现。</p>
<p>所有目标场景此前已暴露，属于开发评测。室外为 KITTI-360 的一个本地适配窗口，不是整个大场景 benchmark 或官方榜单；车库为 rosbag2245 的 330 帧缓存。训练覆盖只有室内四场景，不能将室外无操作解释为有效泛化。</p>
<h2>完整结果</h2><table><tr><th>场景</th><th>方法</th><th>PQ/AP50 × 100</th><th>对象得失</th></tr>{table}</table>
<h2>如何解读</h2><p>风险分支与只预测收益在所有目标场景得到相同的操作和最终分区，没有独立贡献。KITTI-360 的 9 个候选及车库的 23 个候选全部未提交，输出与原归属一致。室内候选应同时比较原归属、原只合并和筛选只合并，不能只比较退步最严重的旧历史重分配。</p>
<p>冻结门槛：候选在每个室内场景分别超过上述三项控制至少 2 个 PQ 点，且不降低 TP；室外 AP50 不低于三项控制。本轮不按结果更换门槛、不挑选单场景替换主线、不继续阈值搜索。机器人缺独立标注，未声称准确率改善。</p>'''
    robot='<h2>车库原始图像与修订状态</h2><div class="frames">'+''.join(f'<figure><img src="{im["export"]}"><figcaption>真实采集帧 {im["frame_id"]}</figcaption></figure>' for im in images)+'</div><p>输入为实测 LIVO 图像与点云缓存。候选没有提交任何修订，因此地图与原归属回放完全相同；这组对照证明执行结果，不能证明机器人性能提升。</p>'
    footer='<p><a href="protocol.json">冻结协议</a> · <a href="summary.json">完整评分</a> · <a href="verification.json">源点与回滚核验</a> · <a href="posthoc_operation_audit.json">操作得失诊断</a></p>'
    (REPORT/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>omni_slam 对象级操作学习</title><style>body{max-width:1360px;margin:40px auto;padding:0 24px;font:16px/1.7 system-ui;color:#223547}h1,h2,h3{line-height:1.3}.lead{padding:18px;background:#eef5f4;border-left:4px solid #008d80}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:9px;border-bottom:1px solid #dbe2e8;text-align:left}img{max-width:100%}.frames{display:flex;gap:14px}.frames figure{flex:1;margin:0}.up{color:#008274}.down{color:#b32949}a{color:#007c70}details{padding:14px;background:#f7f9fb}</style>'+intro+robot+details+footer+'</html>')
    print(REPORT/'index.html',flush=True)
    print('POSTHOC', {k:v['stats'] for k,v in posthoc.items()},flush=True)


if __name__=='__main__':main()
