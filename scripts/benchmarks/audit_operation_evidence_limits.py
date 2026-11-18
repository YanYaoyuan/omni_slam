"""Posthoc evidence/coverage audit only; no predictor, training, or new maps.

Counts all cached masks, rather than inventing co-visibility from final memberships.
Reference coverage is a necessary-condition oracle diagnostic, never an AP score.
"""
from pathlib import Path
from html import escape
import json
import numpy as np
from scipy.spatial import cKDTree
from paper_final_common import ROOT,read,dump,sha
from run_object_operation_gate import OUT as RUN,OLD,verify,arrays
from omni_semantic_map.learned_observation_revision import FEATURE_NAMES
from omni_semantic_map.kitti360_input import load_poses
from evaluate_kitti360_development import transfer

DEST=ROOT/'build/benchmarks/operation_evidence_audit_v1'
REPORT=ROOT/'output/operation_evidence_audit_v1'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    assert not (DEST/'summary.json').exists()
    p=verify(True)
    for f,h in read(RUN/'predictions_complete.json')['files'].items():assert sha(f)==h,f
    REPORT.mkdir(parents=True,exist_ok=True)
    scene=next(s for s in p['scenes'] if s['key']=='scenenn_014')
    post=read(ROOT/'output/object_operation_gate_v1/posthoc_operation_audit.json')['scenes'][scene['key']]
    harmful=[r for r in post['records'] if r['accept'] and r['delta_tp']<0]
    assert len(harmful)==1 and harmful[0]['kind']=='merge'
    event=harmful[0];group_ids=[event['target'],event['source']];a=arrays(scene['key']);cache=Path(scene['cache']);manifest=read(cache/'manifest.json')
    fullxyz=np.load(cache/'geometry.npz')['xyz'];old=np.load(OLD/scene['key']/'observations.npz');flat=old['source_point_ids'];offsets=old['offsets']
    group={};footprints={};members={}
    for oid in group_ids:
        ids=np.flatnonzero(a['native']==oid);members[oid]=ids
        support=np.unique(np.concatenate([flat[offsets[i]:offsets[i+1]] for i in ids]));membership=np.zeros(len(fullxyz),bool);membership[support]=True
        footprints[oid]=(support,membership)
        group[str(oid)]=dict(observations=a['observation_ids'][ids].tolist(),frame_ranks=a['frames'][ids].tolist(),
            frame_ids=[manifest['records'][int(i)]['frame_id'] for i in a['frames'][ids]],points=len(support),
            aabb=[fullxyz[support].min(0).tolist(),fullxyz[support].max(0).tolist()])
    observed=[]
    for rank,r in enumerate(manifest['records']):
        with np.load(cache/'frames'/r['file']) as f:
            points=f['point_ids'];off=f['offsets'];row=dict(frame_id=r['frame_id'],frame_rank=rank,groups={})
            for oid,(support,mask) in footprints.items():
                count=np.array([mask[points[off[j]:off[j+1]]].sum() for j in range(r['masks'])]);j=int(count.argmax()) if len(count) else -1
                row['groups'][str(oid)]=dict(mask_index=j,observation_id=r['observation_offset']+j,
                    matched_points=int(count.max(initial=0)),full_object_support_fraction=float(count.max(initial=0)/len(support)))
            if all(v['matched_points']>0 for v in row['groups'].values()):observed.append(row)
    pairs=np.load(OLD/scene['key']/'pairs.npz');prob=np.load(OLD/scene['key']/'pair_predictions.npz')['probability'];ij=pairs['pairs'];native=a['native']
    select=((native[ij[:,0]]==group_ids[0])&(native[ij[:,1]]==group_ids[1]))|((native[ij[:,0]]==group_ids[1])&(native[ij[:,1]]==group_ids[0]))
    edges=[dict(observation_ids=a['observation_ids'][pair].tolist(),probability=float(pr),features=dict(zip(FEATURE_NAMES,map(float,x))))
        for pair,pr,x in zip(ij[select],prob[select],pairs['features'][select])]
    # All retained observations of the two harmed objects, not cherry-picked crops.
    jobpath=ROOT/'build/benchmarks/evidence_retention_20261111/jobs/v1/014.json';job=read(jobpath);jobframes={r['frame_id']:r for r in job['frames']};image_provenance=[]
    fig,axes=plt.subplots(2,max(map(len,members.values())),figsize=(16,7))
    for row,(oid,indices) in enumerate(members.items()):
        for col,idx in enumerate(indices):
            rec=manifest['records'][int(a['frames'][idx])];fpath=cache/'frames'/rec['file'];assert sha(fpath)==rec['sha256']
            with np.load(fpath) as f:
                local=int(a['observation_ids'][idx]-rec['observation_offset']);mask=np.unpackbits(f['masks_packed'][local],axis=1)[:,:manifest['config']['cam']['W']].astype(bool)
            raw=Path(jobframes[rec['frame_id']]['rgb']);rgb=np.asarray(Image.open(raw).convert('RGB'));assert rgb.shape[:2]==mask.shape
            # Scientific overlay of the actual recorded segmentation; no generated imagery.
            ax=axes[row,col];ax.imshow(rgb);layer=np.zeros((*mask.shape,4));layer[mask]=[.1,.85,.75,.45] if row==0 else [1.,.2,.45,.45];ax.imshow(layer)
            ax.set_title(f"Object {oid} | frame {rec['frame_id']}\nobservation {int(a['observation_ids'][idx])}",fontsize=10);ax.axis('off')
            image_provenance.append(dict(frame_id=rec['frame_id'],observation_id=int(a['observation_ids'][idx]),rgb=str(raw),rgb_sha256=sha(raw),mask_file=str(fpath),mask_sha256=sha(fpath)))
    fig.suptitle('All retained masks of the two incorrectly merged objects');fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(REPORT/'harmful_merge_all_masks.png',dpi=130);plt.close(fig)
    case=dict(scene=scene['key'],event=event,groups=group,candidate_edges=edges,
        possible_cross_pairs=len(members[group_ids[0]])*len(members[group_ids[1]]),
        retained_shared_frames=sorted(set(group[str(group_ids[0])]['frame_ids'])&set(group[str(group_ids[1])]['frame_ids'])),
        intersecting_source_points=len(np.intersect1d(footprints[group_ids[0]][0],footprints[group_ids[1]][0])),
        full_cache_covisibility=observed,image_provenance=image_provenance,
        interpretation='Distinct selected frames do not prove objects were never co-visible. Full-cache support overlaps are weak and may be on masks of other objects; this is not a validated cannot-link constraint.')
    outdoor={}
    for key in ['2013_05_28_drive_0000_sync','rosbag2245']:
        decisions=read(RUN/key/'gated_revision/decisions.json');outdoor[key]=dict(candidates=len(decisions),accepted=sum(bool(d['accept']) for d in decisions),
            positive_utility=sum(d['predicted_utility']>0 for d in decisions),utility_range=[min(d['predicted_utility'] for d in decisions),max(d['predicted_utility'] for d in decisions)],
            risk_range=[min(d['predicted_tp_loss_probability'] for d in decisions),max(d['predicted_tp_loss_probability'] for d in decisions)],
            new_identity_candidates=sum(d['target']>int(arrays(key)['native'].max()) for d in decisions),decisions=decisions)
    # Fixed support ceiling for the outdoor developer window, including original blockers.
    s=next(s for s in p['scenes'] if s['kind']=='outdoor');key=s['key'];a=arrays(key);c=Path(s['cache']);man=read(c/'manifest.json');xyz=np.load(c/'geometry.npz')['xyz']
    ref=np.load(s['reference']);assert sha(s['reference'])==p['reference_hashes'][s['reference']]
    original=np.load(Path(s['fixed_map'])/'geometry.npz');rows=read(Path(s['fixed_map'])/'classification.json')['objects'];fg=[r['object_id'] for r in rows if r['predictions']['mean']['retained']]
    blockers=original['xyz'][~np.isin(original['instance'],fg)]
    select=np.isin(ref['train_instance']//1000,[2,13]);gt=ref['train_instance'][select];confidence=ref['confidence'][select];query=ref['xyz'][select]
    current=transfer(np.r_[a['source_xyz'],blockers],np.r_[np.ones(len(a['source_xyz']),np.int64),np.zeros(len(blockers),np.int64)],query,.2)>0
    any_measurement=np.isfinite(cKDTree(xyz).query(query,distance_upper_bound=.2,workers=1)[0])
    objects=[]
    for gid in np.unique(gt):
        hit=gt==gid;mass=float(confidence[hit].sum())
        if mass<100:continue
        objects.append(dict(gt_id=int(gid),category='building' if gid//1000==2 else 'car',confidence_mass=mass,
            all_measurement_coverage=float(confidence[hit&any_measurement].sum()/mass),frozen_foreground_coverage=float(confidence[hit&current].sum()/mass)))
    totals={category:dict(eligible=sum(o['category']==category for o in objects),
        all_measurements_over_half=sum(o['category']==category and o['all_measurement_coverage']>.5 for o in objects),
        fixed_support_over_half=sum(o['category']==category and o['frozen_foreground_coverage']>.5 for o in objects)) for category in ['car','building']}
    ceiling=dict(scene=key,totals=totals,objects=objects,all_observations=man['observations'],retained_observations=len(a['native']),
        all_source_points=len(xyz),retained_source_points=len(a['source_xyz']),
        meaning='Necessary coverage condition for perfect pointwise identity/background oracle on fixed projected foreground. NOT attainable AP, NOT a new method, and does not account for false positives or indivisible observations. GT used only after frozen predictions.')
    base=ROOT/'build/datasets/kitti360_prepare';selection=read(base/'window_selection.json');inventory={}
    sources={}
    for part in ['train','val']:
        path=base/'splits'/f'2013_05_28_drive_{part}.txt';lines=path.read_text().splitlines();sources[str(path)]=sha(path)
        inventory[part]=dict(windows=len(lines),sequences=sorted({Path(r).parts[2] for r in lines}))
    inventory['downloaded_windows']=[dict(scene=r['scene'],start=r['start'],end=r['end'],sampled_frames=len(r['input_frames']),role=r['role']) for r in selection['windows']]
    inventory['validation_windows_downloaded']=sum(r['member'] in (base/'splits/2013_05_28_drive_val.txt').read_text().splitlines() for r in selection['windows'])
    inventory['no_fresh_independent_predictions']=True
    inventory['requirements_for_expansion']=['Select unexposed windows from metadata before label decoding; downloaded 0002/0003 already exposed.',
        'Download RGB, measured LiDAR, calibration, poses and opaque references; verify archive CRC/hash.',
        'Audit spatial overlap and temporal pairing; sequence name alone is not independence.',
        'Freeze a successful development candidate and matched controls before inference; seal all predictions before reference scoring.',
        'For official leaderboard comparison, process the full required split and exact official representation/protocol; selected-window scores are insufficient.']
    payload=dict(complete=True,diagnostic_only=True,no_new_maps=True,no_training=True,source_sha256=sha(__file__),
        parent_prediction_seal_sha256=sha(RUN/'predictions_complete.json'),case=case,zero_edit_audit=outdoor,
        outdoor_support_ceiling=ceiling,dataset_inventory=inventory,split_sources=sources)
    dump(DEST/'summary.json',payload);dump(REPORT/'summary.json',payload)
    rows=''.join(f'<tr><td>{c}</td><td>{v["eligible"]}</td><td>{v["all_measurements_over_half"]}</td><td>{v["fixed_support_over_half"]}</td></tr>' for c,v in totals.items())
    noedit=''.join(f'<tr><td>{k}</td><td>{v["candidates"]}</td><td>{v["positive_utility"]}</td><td>{v["utility_range"][0]:.4f} ～ {v["utility_range"][1]:.4f}</td><td>{v["risk_range"][0]:.4f} ～ {v["risk_range"][1]:.4f}</td></tr>' for k,v in outdoor.items())
    body=f'''<h1>omni_slam：修订机制的输入证据与能力边界</h1>
<p>本轮只复核已封存产物和真实输入，没有新训练、调参或地图预测。主线仍是持续对象级语义建图；目的在于确定应修改的信息入口，避免在同一压缩表示上反复拟合。</p>
<h2>一个错误合并实际看到了什么</h2><p>SceneNN 014 中对象 {group_ids} 被并为一体，损失两个原有正确匹配。两个对象各有四个保留观测，观测帧没有交集；16 个可能跨对象对中只有两个候选边。两条边的测量支持交集均为零，但语义外观余弦为 0.874、0.864，模型同一对象概率为 0.863、0.785。这里的外观输入是原 SigLIP 描述子；不是新增密集 DINO/VGGT 对应点证据。</p>
<img src="harmful_merge_all_masks.png"><p>图中展示这两个对象全部八个保留掩码及真实 RGB。颜色是预测掩码，不是人工实例标注。完整缓存中有 {len(observed)} 帧同时触及两组源点，但部分匹配来自同一个大掩码或其他对象掩码，且覆盖偏低。因此不能把“曾同帧出现”直接当作经过验证的禁止合并约束。</p>
<h2>室外和车库为何零操作</h2><table><tr><th>场景</th><th>候选</th><th>预测正收益</th><th>收益范围</th><th>风险范围</th></tr>{noedit}</table><p>所有操作先被收益预测拒绝；放宽风险阈值不会改变结果。更重要的是，候选空间只包含已保留观测的重新归属，没有创建遗漏掩码、恢复被丢弃对象或更换语义类别的能力。</p>
<h2>KITTI-360 当前窗口的固定支持限制</h2><table><tr><th>类别</th><th>可评估 GT</th><th>全部测量覆盖大于一半</th><th>当前固定前景覆盖大于一半</th></tr>{rows}</table><p>从 {ceiling['all_observations']} 个缓存观测、{ceiling['all_source_points']:,} 个测量点，到当前修订输入仅 {ceiling['retained_observations']} 个保留观测、{ceiling['retained_source_points']:,} 个支持点。上表是允许完美点级归属和背景剔除后的覆盖必要条件，<strong>不是 AP 或可达到的真实召回</strong>；固定支持列还保留了原评分中的背景阻挡点。</p>
<h2>扩大室外评测还缺什么</h2><p>本地官方划分文件包含 {inventory['train']['windows']} 个训练窗口和 {inventory['val']['windows']} 个验证窗口。当前只下载了 3 个选定窗口、161 个采样帧，其中验证窗口 {inventory['validation_windows_downloaded']} 个，均已暴露。当前成绩不能代表完整室外 benchmark 或官方榜单。</p><ol>'''+''.join(f'<li>{escape(x)}</li>' for x in inventory['requirements_for_expansion'])+'''</ol>
<h2>明确的下一步约束</h2><p>停止继续改这版收益/风险阈值。若继续研究历史身份修订，应先证实新的观测级或点级身份信息确实能区别同类别相邻对象，再训练提交策略。针对室外的新增机制必须能访问当前前景筛选之前的对象证据，并与 masked-only/bbox-only 强读出及原 CG/history2 对照；仅重新排列现有保留对象无法修复支持范围以外的遗漏。</p><p><a href="summary.json">完整可复核记录</a> · <a href="../object_operation_gate_v1/index.html">上一轮完整指标与逐对象得失</a></p>'''
    (REPORT/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>对象修订证据审计</title><style>body{max-width:1250px;margin:40px auto;padding:0 24px;font:16px/1.7 system-ui;color:#243749}img{width:100%}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:10px;border-bottom:1px solid #dce4e8}a{color:#008577}</style>'+body+'</html>')
    print('CASE',group_ids,'candidate edges',len(edges),'shared selected frames',case['retained_shared_frames'],flush=True)
    print('OUTDOOR SUPPORT',totals,flush=True);print(REPORT/'index.html',flush=True)


if __name__=='__main__':main()
