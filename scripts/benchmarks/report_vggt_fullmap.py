"""Read-only review of sealed six-map experiment; no new inference or fitting."""
from pathlib import Path
import html
import json
import sys
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from run_vggt_fullmap import OUT, ARMS, verify

REPORT=ROOT/'output/vggt_fullmap_v1'
NAMES=['SigLIP + history2','VGGT single','VGGT joint (4 views)']
COLORS=['#8b97a9','#167cb8','#00a58b']


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    from omni_semantic_map.mesh_instance_evaluation import transfer_labels
    from omni_semantic_map.full_ovo_evaluation import voxel_metrics
    from omni_semantic_map.shared_observations import observation_table
    p=verify(True);sealed=read(OUT/'predictions_complete.json');result=read(OUT/'evaluation.json')
    assert sealed['complete'] and result['complete']
    assert result['prediction_seal_sha256']==sha(OUT/'predictions_complete.json')
    for file,h in sealed['files'].items():assert sha(file)==h,file
    feature_manifest=read(OUT/'features/manifest.json')
    for file,h in feature_manifest['files'].items():assert sha(file)==h,file
    REPORT.mkdir(parents=True,exist_ok=True)
    dump(REPORT/'metrics.json',result)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,4))
    x=np.arange(3)
    for i,arm in enumerate(ARMS):
        vals=[s['methods'][arm]['metrics']['observed_voxel_instance_pq']*100 for s in result['results']]+[result['mean_pq100'][arm]]
        bars=ax.bar(x+(i-1)*.24,vals,.23,color=COLORS[i],label=NAMES[i]);ax.bar_label(bars,fmt='%.2f',padding=3)
    ax.set_xticks(x,['Replica office_0','SceneNN 014','Scene mean']);ax.set_ylabel('Instance PQ x 100');ax.set_ylim(0,42)
    ax.legend(ncol=3,loc='upper right',frameon=False);fig.tight_layout();fig.savefig(REPORT/'pq.png',dpi=170);plt.close(fig)
    all_objects=[];mapped={};audits={};classification={}
    for s in p['scenes']:
        key=s['key'];assert sha(s['reference'])==s['reference_sha256']
        with np.load(s['reference']) as z:ref={k:z[k] for k in z.files}
        valid=ref.get('valid',np.ones(len(ref['instance']),bool));truth=ref['instance'][valid]
        xyz=ref['xyz'][valid];center=xyz.mean(0);_,_,axes=np.linalg.svd(xyz-center,full_matrices=False)
        xy=(xyz-center)@axes[:2].T
        sample=np.linspace(0,len(xyz)-1,min(25000,len(xyz)),dtype=int)
        fig,axs=plt.subplots(1,4,figsize=(16,4),sharex=True,sharey=True)
        fields=[truth];scene_audits={}
        for arm in ARMS:
            folder=OUT/'maps'/key/arm/'history2'
            z=np.load(folder/'geometry.npz');rows=read(folder/'classification.json')['objects'];classification[key,arm]=rows
            kept=[r['object_id'] for r in rows if r['predictions']['mean']['retained']]
            pred,_=transfer_labels(z['xyz'],np.where(np.isin(z['instance'],kept),z['instance'],0),ref['xyz'],mode='nearest_5cm')
            metrics=voxel_metrics(truth,pred[valid]);expected=next(r for r in result['results'] if r['scene']==key)['methods'][arm]['metrics']
            assert abs(metrics['observed_voxel_instance_pq']-expected['observed_voxel_instance_pq'])<1e-12
            assert (metrics['tp'],metrics['fp'],metrics['fn'])==(expected['tp'],expected['fp'],expected['fn'])
            mapped[key,arm]=pred[valid];fields.append(pred[valid])
            scene_audits[arm]=read(OUT/'evaluation'/key/f'{arm}_objects.json');audits[key,arm]=scene_audits[arm]
        for ax,labels,title in zip(axs,fields,['GT instances']+NAMES):
            ids=labels[sample];color=plt.get_cmap('tab20')((ids*7%20)/19);color[ids==0]=[.88,.89,.90,.55]
            ax.scatter(xy[sample,0],xy[sample,1],s=1,c=color,rasterized=True);ax.set_aspect('equal');ax.set_title(title);ax.axis('off')
        fig.suptitle(f'{key}: fixed reference surface, identical projection / sample; colors = instance IDs within each panel')
        fig.tight_layout();fig.savefig(REPORT/f'{key}_map.png',dpi=180);plt.close(fig)
        byarm={a:{g['gt_id']:g for g in scene_audits[a]['ground_truth']} for a in ARMS}
        for gid in sorted(byarm[ARMS[0]]):
            row=dict(scene=key,gt_id=gid)
            for a in ARMS:row[a]=byarm[a][gid]
            row['joint_minus_single_iou']=row[ARMS[2]]['best_iou']-row[ARMS[1]]['best_iou']
            all_objects.append(row)
    dump(REPORT/'all_objects.json',all_objects)

    # Post-hoc selected illustrative case: joint retains a split that single resolves.
    key='scenenn_014';gid=745473;s=next(s for s in p['scenes'] if s['key']==key)
    ref=np.load(s['reference']);valid=ref['valid'] if 'valid' in ref else np.ones(len(ref['instance']),bool)
    xyz=ref['xyz'][valid];truth=ref['instance'][valid];object_xyz=xyz[truth==gid];center=object_xyz.mean(0)
    _,_,axes=np.linalg.svd(object_xyz-center,full_matrices=False);xy=(xyz-center)@axes[:2].T
    lo,hi=np.min(object_xyz,axis=0)-.1,np.max(object_xyz,axis=0)+.1;crop=np.all((xyz>=lo)&(xyz<=hi),axis=1)
    fig,axs=plt.subplots(1,4,figsize=(13,3.3),sharex=True,sharey=True)
    case=next(r for r in all_objects if r['scene']==key and r['gt_id']==gid)
    for ax,arm,title in zip(axs,[None]+ARMS,['GT object 745473']+NAMES):
        labels=np.where(truth==gid,1,0) if arm is None else mapped[key,arm]
        relevant=[1] if arm is None else [o['predicted_id'] for o in audits[key,arm]['predicted'] if o['dominant_gt']==gid]
        colors=np.tile(np.array([.85,.86,.87,1.]),(len(xyz),1))
        for j,oid in enumerate(relevant):colors[labels==oid]=plt.get_cmap('tab10')(j%10)
        ax.scatter(xy[crop,0],xy[crop,1],s=6,c=colors[crop]);ax.set_aspect('equal');ax.axis('off')
        ax.set_title(title+(f"\nbest IoU {case[arm]['best_iou']:.3f}" if arm else ''))
    fig.suptitle('Observed reference points: single joins two fragments; joint preserves the baseline split')
    fig.tight_layout();fig.savefig(REPORT/'object_745473.png',dpi=180);plt.close(fig)
    rows=classification[key,'vggt_joint'];parts=[next(r for r in rows if r['object_id']==i) for i in [1,17]]
    cache=Path(s['cache']);_,frames,areas=observation_table(cache);records=s['records']
    fig,axs=plt.subplots(1,2,figsize=(11,4));images=[]
    for ax,part in zip(axs,parts):
        oid=max(part['observation_ids'],key=lambda i:areas[i]);rec=next(r for r in records if r['frame_id']==int(frames[oid]))
        with np.load(cache/'frames'/rec['file']) as z:
            mask=np.unpackbits(z['masks_packed'][oid-rec['observation_offset']],axis=1)[:,:z['segmentation'].shape[1]].astype(bool)
        rgb=np.asarray(Image.open(rec['rgb']).convert('RGB'));assert mask.shape==rgb.shape[:2]
        ax.imshow(rgb);overlay=np.zeros((*mask.shape,4));overlay[mask]=[0,.8,.7,.4];ax.imshow(overlay)
        ax.set_title(f"Observed SAM mask / frame {rec['frame_id']} / observation {oid}\nJoint object {part['object_id']}; predicted class: {part['predictions']['mean']['name']}");ax.axis('off')
        images.append(dict(observation=oid,frame=rec['frame_id'],image=rec['rgb'],sha256=sha(rec['rgb']),joint_object=part['object_id']))
    fig.tight_layout();fig.savefig(REPORT/'real_observations.png',dpi=160);plt.close(fig)
    lineage={}
    for arm in ARMS:
        members=set(sum([r['observation_ids'] for r in parts],[]));folder=OUT/'maps'/key/arm
        nodes=[n for n in read(folder/'nodes.json') if set(n['members'])&members];aliases={n['alias'] for n in nodes}
        decisions=[d for d in read(folder/'history2/graph.json')['decisions'] if d['a'] in aliases and d['b'] in aliases]
        merges=[m for m in read(folder/'history2/final_native_merges.json') if set(m['members'])&members]
        lineage[arm]=dict(nodes=nodes,internal_graph_decisions=decisions,final_merges=merges)
    assert len(lineage['vggt_single']['nodes'])==1 and len(lineage['vggt_joint']['nodes'])==2
    assert all(not v['internal_graph_decisions'] and not v['final_merges'] for v in lineage.values())
    dump(REPORT/'case_evidence.json',dict(gt_id=gid,lineage=lineage,images=images,selection='Post-hoc largest single-versus-joint IoU loss among matched SceneNN objects; each fragment illustrated with its largest-area observation.',
        finding='Difference already exists before history graph and final merging. No internal graph proposal or final merge acts on these fragments. Does not establish why joint descriptors differ.'))
    # Exact arithmetic decomposition of the SceneNN PQ gap; not a causal intervention.
    m=next(r for r in result['results'] if r['scene']==key)['methods']
    a=m['vggt_single']['metrics'];b=m['vggt_joint']['metrics'];sa=sum(x['iou'] for x in a['matches']);sb=sum(x['iou'] for x in b['matches'])
    da=a['tp']+.5*(a['fp']+a['fn']);db=b['tp']+.5*(b['fp']+b['fn'])
    decomposition=dict(single_pq=sa/da,joint_pq=sb/db,fp_denominator_delta_pp=100*(sa/db-sa/da),
        matched_iou_delta_pp=100*(sb-sa)/db,object745473_delta_pp=100*case['joint_minus_single_iou']/db)
    dump(REPORT/'pq_decomposition.json',decomposition)
    table=''.join('<tr><td>'+html.escape(r['scene'])+'</td><td>'+str(r['gt_id'])+'</td>'+''.join(f"<td>{r[a]['best_iou']:.4f}<br><small>{r[a]['category']}</small></td>" for a in ARMS)+f"<td>{r['joint_minus_single_iou']:+.4f}</td></tr>" for r in all_objects)
    summary=''.join('<tr><td>'+r['scene']+'</td>'+''.join(f"<td>{r['methods'][a]['metrics']['observed_voxel_instance_pq']*100:.2f}</td>" for a in ARMS)+'</tr>' for r in result['results'])
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>VGGT 完整对象地图评测</title><style>
    body{max-width:1160px;margin:40px auto;padding:0 24px;font:16px/1.7 system-ui,sans-serif;color:#1e3042;background:#fafcfd}h1{font-size:34px}h2{margin-top:36px}img{width:100%;background:white;border-radius:12px}table{border-collapse:collapse;width:100%;background:white}th,td{text-align:left;padding:10px;border-bottom:1px solid #dde5ea}small{color:#667}a{color:#167cb8}.lead{padding:22px;border-left:5px solid #00a58b;background:#edf8f6}.note{color:#566579}details{margin:24px 0}</style>
    <h1>VGGT：从小窗口检索到完整对象地图</h1><p>6 张地图已完成 · 两条完整采样轨迹 · 固定配置 · 已有产物机制复核</p>
    <div class="lead"><b>VGGT 特征带来了整图收益；联合特征尚未超过单帧对照。</b><br>平均 PQ：原基线 23.40，VGGT 单帧 25.69（+2.29），联合 25.17（+1.77）。三组正确匹配总数均为 24。预设的联合组晋级门槛未通过，暂不扩大这条联合特征配置的测试，也不自动调参。</div>
    <p class="note">Replica office_0 和 SceneNN 014 均是已参与开发的场景，各 200 个采样帧；并非原始图像逐帧全量，也不是独立验证或官方榜单。本轮仅干预关联特征，未叠加旧可见性清理模块。VGGT 联合特征组是方法候选；未训练的对象记忆模块未参与，不标成已验证的 Ours。</p>
    <h2>整图分数</h2><table><tr><th>场景 / PQ × 100</th><th>SigLIP + history2</th><th>VGGT 单帧</th><th>VGGT 联合</th></tr>'''+summary+'''</table><img src="pq.png">
    <p>相较基线，单帧组 FP 总数从 72 降至 62，联合组降至 66；TP 均为 24。SceneNN 中两种 VGGT 都新增匹配 GT 311327，同时丢失 GT 189623，因此不能描述为“没有对象退步”。误合并仍存在。</p>
    <h2>真实数据与对象地图</h2><img src="replica_office_0_map.png"><img src="scenenn_014_map.png"><p class="note">显示固定参考表面上的实例预测，使用相同点采样和投影；每个面板颜色表示该面板内部的实例 ID，颜色不表示跨方法身份对应。灰色为背景或未赋予前景实例的点；图像不等同于原始完整稠密重建。</p>
    <h2>定位到的机制差异</h2><p>SceneNN 的 GT 745473：基线和联合组保留两个主要碎片，最佳 IoU 为 0.538；单帧组把对应观测合入一个对象，IoU 达到 0.835。差异在 history2 之前已存在，两个分支之间没有 history2 合并提案，最终合并也没有处理它们。因此，这个案例的单帧收益不是先产生、再被后续融合抵消。</p>
    <img src="object_745473.png"><img src="real_observations.png"><p class="note">这是评分后选出的诊断案例；RGB 来自数据集原始图像，青色为既有 SAM 观测掩码，图中类别为模型预测。仅据这个案例，不能断言失败由跨窗口特征漂移造成，也不能据此断言所有联合注意力无效。</p>
    <p>SceneNN 联合组比单帧低 1.21 个 PQ 点。其中，按固定单帧匹配 IoU、先改变分母再改变分子的算术分解，4 个额外 FP 对应 −0.67 点，匹配对象 IoU 总和下降对应 −0.54 点。这是指标分解，不是额外实验或因果证明。</p>
    <h2>结论与收敛</h2><p>保留 VGGT 单帧作为有实测收益的候选强基线；不宣称它已独立泛化，也不立即替换论文主线。当前四帧联合特征的直接替换不满足晋级条件，本轮封存。已有证据支持下一项研究聚焦“让跨帧信息形成可复用的对象表示，并参与历史归属修订”，而非继续扩大原样的联合特征替换；具体新模块须另行预设训练划分、消融及有限预算。</p>
    <details><summary>全部 68 个参考对象的 IoU 与得失（含无收益和退步）</summary><table><tr><th>场景</th><th>GT ID</th><th>基线最佳 IoU</th><th>单帧最佳 IoU</th><th>联合最佳 IoU</th><th>联合 − 单帧</th></tr>'''+table+'''</table></details>
    <p><a href="metrics.json">完整指标</a> · <a href="all_objects.json">所有对象</a> · <a href="case_evidence.json">节点及原图证据</a> · <a href="pq_decomposition.json">PQ 分解</a> · <a href="verification.json">完整性核验</a></p></html>'''
    (REPORT/'index.html').write_text(page)
    dump(REPORT/'verification.json',dict(complete=True,protocol_sha256=sha(OUT/'protocol.json'),
        prediction_seal_sha256=sha(OUT/'predictions_complete.json'),evaluation_sha256=sha(OUT/'evaluation.json'),
        checked_prediction_files=len(sealed['files']),all_input_and_source_hashes_verified=True,
        all_feature_hashes_verified=True,six_pq_tp_fp_fn_recomputed=True,new_mapping_runs=0,
        report_source_sha256=sha(__file__),report_sha256=sha(REPORT/'index.html')))
    print(json.dumps(dict(report=str(REPORT/'index.html'),decomposition=decomposition),indent=2))


if __name__=='__main__':main()
