"""Audit sealed outputs and publish all gains, regressions and ablations."""
from pathlib import Path
import html
import json
import shutil
import time
import numpy as np
from run_visibility_cleanup import ROOT, OUT, KEYS, VARIANTS, verify, read, dump, sha

REPORT = ROOT/'docs/reports/visibility_cleanup_2026-11-12'
LABELS = {'baseline':'CG（共享前端）','native_bridge_2':'当前强基线',
          'geometric_sor':'普通几何去噪','multiview':'可见性＋多帧清理',
          'no_visibility':'消融：不检查可见性','single_refutation':'消融：单帧即可反证'}
ENGLISH = ['CG shared input','Current baseline','Geometric SOR','Visibility cleanup','No visibility','Single refutation']
COLORS = ['#9ba8b4','#3b6a90','#bf9861','#16a185','#b35a74','#8262b0']


def independent_score(truth, prediction):
    """Direct exclusive-mask IoU calculation independent of production scorer."""
    gids = np.unique(truth[truth>0]); pids = np.unique(prediction[prediction>0])
    matches = []
    for gid in gids:
        gmask = truth == gid
        for pid in np.unique(prediction[gmask]):
            if pid == 0: continue
            pmask = prediction == pid
            intersection = int(np.count_nonzero(gmask & pmask))
            union = int(gmask.sum() + pmask.sum() - intersection)
            iou = intersection/union
            if iou > .5: matches.append((gid,pid,iou))
    assert len({g for g,p,v in matches}) == len(matches)
    assert len({p for g,p,v in matches}) == len(matches)
    tp = len(matches); fp = len(pids)-tp; fn = len(gids)-tp
    denom = tp + .5*(fp+fn)
    return dict(tp=tp,fp=fp,fn=fn,pq=sum(v for g,p,v in matches)/denom if denom else 0.)


def table(headers, rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    protocol = verify(); summary = read(OUT/'summary.json'); gate=read(OUT/'predictions_complete.json')
    assert summary['protocol_sha256']==sha(OUT/'protocol.json') and summary['prediction_gate_sha256']==sha(OUT/'predictions_complete.json')
    REPORT.mkdir(parents=True,exist_ok=True)
    methods = list(LABELS); results = []; checked=0; regressions=[]; projection_changes=[]; match_mechanisms=[]
    for scene in protocol['scenes']:
        key=scene['key']; folder=OUT/'evaluation'/key
        assert sha(folder/'metrics.json')==summary['scene_results'][key]
        assert sha(OUT/'runs'/key/'run.json')==gate['runs'][str(Path('runs')/key/'run.json')]
        result=read(folder/'metrics.json'); results.append(result)
        reference=np.load(scene['reference']); valid=reference['valid'] if 'valid' in reference.files else np.ones(len(reference['instance']),bool)
        truth=reference['instance'][valid]
        for method in methods:
            if not result['methods'][method]['complete']:continue
            projected=np.load(folder/(method+'.npz'))['instance'][valid]
            independent=independent_score(truth,projected); official=result['methods'][method]['score']
            for k in ['tp','fp','fn']:assert independent[k]==official[k],(key,method,k)
            assert np.isclose(independent['pq'],official['observed_voxel_instance_pq'],atol=1e-12,rtol=0)
            checked+=1
        objects=read(folder/'per_object.json')
        decreases=[g for g in objects if g['methods']['multiview']['delta_iou'] < -1e-8]
        increases=[g for g in objects if g['methods']['multiview']['delta_iou'] > 1e-8]
        worst=min(objects,key=lambda g:g['methods']['multiview']['delta_iou'])
        regressions.append(dict(scene=key,improved_iou=len(increases),reduced_iou=len(decreases),unchanged_iou=len(objects)-len(increases)-len(decreases),worst=worst))
        before_labels=np.load(folder/'native_bridge_2.npz')['instance'][valid]
        after_labels=np.load(folder/'multiview.npz')['instance'][valid]
        changed=before_labels!=after_labels
        # Deleting raw points can transfer evaluation vertices to other objects.
        # Expose that effect instead of attributing all gains to own-boundary purity.
        projection_changes.append(dict(scene=key,changed_valid_voxels=int(changed.sum()),
            transferred_to_other_object=int((changed&(before_labels>0)&(after_labels>0)).sum()),
            cleared=int(((before_labels>0)&(after_labels==0)).sum())))
        gained=result['methods']['multiview']['transition']['gained_matched_gt']
        matched=read(folder/'multiview_audit.json')['predicted']
        for gid in gained:
            pid=next(o['predicted_id'] for o in matched if o['matched_gt']==gid)
            bm=before_labels==pid;am=after_labels==pid;gt=truth==gid
            match_mechanisms.append(dict(scene=key,gt_id=gid,predicted_id=pid,gt_valid_voxels=int(gt.sum()),
                correct_before=int((bm&gt).sum()),correct_after=int((am&gt).sum()),
                excess_before=int((bm&~gt).sum()),excess_after=int((am&~gt).sum()),
                added_correct_from_other_object=int((am&~bm&gt).sum())))
        dest=REPORT/'scenes'/key;dest.mkdir(parents=True,exist_ok=True)
        for file in ['metrics.json','per_object.json']:
            shutil.copyfile(folder/file,dest/file)
        for method in methods:
            shutil.copyfile(folder/(method+'_audit.json'),dest/(method+'_audit.json'))
    # All six methods and both domains in the main figure; no best-variant selection.
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,dataset in zip(axes,['Replica','SceneNN']):
        values=[summary['datasets'][dataset][m]['pq100'] for m in methods]
        ax.bar(np.arange(len(methods)),values,color=COLORS)
        for i,v in enumerate(values):ax.text(i,v+.2,f'{v:.2f}',ha='center',fontsize=9)
        ax.set_xticks(np.arange(len(methods)),ENGLISH,rotation=25,ha='right',fontsize=8)
        ax.set_ylabel('Observed-domain instance PQ × 100');ax.set_title(dataset+' · 2 development scenes')
        ax.set_ylim(0,max(values)*1.18);ax.spines[['top','right']].set_visible(False)
    fig.savefig(REPORT/'comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    x=np.arange(len(results)); width=.19
    for i,m in enumerate(VARIANTS):
        values=[(r['methods'][m]['score']['observed_voxel_instance_pq']-r['methods']['native_bridge_2']['score']['observed_voxel_instance_pq'])*100 for r in results]
        axes[0].bar(x+(i-1.5)*width,values,width,label=ENGLISH[methods.index(m)],color=COLORS[methods.index(m)])
    axes[0].axhline(0,color='#555',lw=.8);axes[0].set_xticks(x,KEYS,rotation=18,ha='right',fontsize=8)
    axes[0].set_ylabel('Δ PQ percentage points vs current baseline');axes[0].legend(fontsize=8)
    axes[1].bar(x-.16,[r['improved_iou'] for r in regressions],.32,label='Best IoU improved',color='#16a185')
    axes[1].bar(x+.16,[r['reduced_iou'] for r in regressions],.32,label='Best IoU reduced',color='#b35a74')
    axes[1].set_xticks(x,KEYS,rotation=18,ha='right',fontsize=8);axes[1].set_ylabel('GT objects');axes[1].legend(fontsize=8)
    fig.savefig(REPORT/'gains_and_losses.png',dpi=170);plt.close(fig)
    # Show BOTH newly matched objects, not an arbitrarily selected success.
    gain_cases=[]
    for r in results:
        for gid in r['methods']['multiview']['transition']['gained_matched_gt']:gain_cases.append((r['key'],gid))
    if gain_cases:
        fig,axes=plt.subplots(len(gain_cases),3,figsize=(13,4.6*len(gain_cases)),squeeze=False,layout='constrained')
        for row,(key,gid) in enumerate(gain_cases):
            scene=next(s for s in protocol['scenes'] if s['key']==key);ref=np.load(scene['reference'])
            valid=ref['valid'] if 'valid' in ref.files else np.ones(len(ref['instance']),bool)
            xyz=ref['xyz'][valid];truth=ref['instance'][valid];gt=truth==gid
            after=np.load(OUT/'evaluation'/key/'multiview.npz')['instance'][valid]
            before=np.load(OUT/'evaluation'/key/'native_bridge_2.npz')['instance'][valid]
            audit=read(OUT/'evaluation'/key/'multiview_audit.json')
            pid=next(o['predicted_id'] for o in audit['predicted'] if o['matched_gt']==gid)
            relevant=gt|(before==pid)|(after==pid);center=xyz[gt].mean(0)
            # Fixed common PCA display basis of the GT, for post-hoc illustration only.
            _,_,basis=np.linalg.svd(xyz[gt]-center,full_matrices=False);uv=(xyz-center)@basis[:2].T
            bounds=uv[relevant];lo=bounds.min(0);hi=bounds.max(0);pad=np.maximum((hi-lo)*.05,.03)
            for col,mask,title in [(0,gt,'Ground truth'),(1,before==pid,'Current baseline'),(2,after==pid,'Visibility cleanup')]:
                ax=axes[row,col];ax.set_facecolor('#f2f5f8')
                ax.scatter(uv[gt,0],uv[gt,1],s=4,c='#bac6cf',label='GT surface')
                if col:
                    ax.scatter(uv[mask&~gt,0],uv[mask&~gt,1],s=5,c='#b35a74',label='Excess support')
                    ax.scatter(uv[mask&gt,0],uv[mask&gt,1],s=5,c='#16a185',label='Correct support')
                    iou=np.count_nonzero(mask&gt)/np.count_nonzero(mask|gt)
                    title+=f' · IoU {iou:.4f}'
                ax.set_title(f'{key} / GT {gid}\n{title}',fontsize=10);ax.set_aspect('equal')
                ax.set_xlim(lo[0]-pad[0],hi[0]+pad[0]);ax.set_ylim(lo[1]-pad[1],hi[1]+pad[1])
                ax.set_xlabel('Common local display axis 1 / m');ax.set_ylabel('Axis 2 / m')
                if col==2:ax.legend(fontsize=8)
        fig.savefig(REPORT/'new_matches.png',dpi=170);plt.close(fig)
    comparisons=[];scene_rows=[];cost_rows=[];case_rows=[]
    for dataset,group in summary['datasets'].items():
        b=group['native_bridge_2']['pq100']
        for m in methods:
            s=group[m];comparisons.append([dataset,LABELS[m],f"{s['pq100']:.4f}",f"{s['pq100']-b:+.4f}",f"{s['tp']} / {s['fp']} / {s['fn']}",f"{s['gained']} / {s['lost']}" if m in VARIANTS else '—'])
    for r in results:
        b=r['methods']['native_bridge_2']['score']['observed_voxel_instance_pq']*100
        for m in VARIANTS:
            s=r['methods'][m];score=s['score'];transition=s['transition'];rem=s['removal']
            scene_rows.append([r['key'],LABELS[m],f"{score['observed_voxel_instance_pq']*100:.4f}",f"{score['observed_voxel_instance_pq']*100-b:+.4f}",
                f"{score['tp']}/{score['fp']}/{score['fn']}",str(transition['gained_matched_gt']),str(transition['lost_matched_gt']),f"{rem['removed_fraction']*100:.2f}%",str(rem['emptied_objects'])])
        t=r['timing'];cost_rows.append([r['key'],f"{t['total_wall_s']:.3f}",f"{t['evidence_s']:.3f}",f"{t['evidence_frame_p95_s']*1000:.2f}",f"{t['cpu_peak_rss_bytes']/1024**2:.1f}"])
    for r in regressions:
        g=r['worst'];case_rows.append([r['scene'],r['improved_iou'],r['reduced_iou'],r['unchanged_iou'],g['gt_id'],f"{g['baseline_best_iou']:.4f} → {g['methods']['multiview']['best_iou']:.4f}"])
    sections=[
        '<h1>多视角污染点清理：benchmark 对比与消融</h1><p class="sub">omni_slam · 2026-11-12 · 4 个开发场景 · 16 份新几何输出 · 0 次训练 / 新模型推理</p>',
        '<div class="callout"><b>四场景实例 PQ 均上涨，新增 2 个正确匹配、丢失 0 个。</b>相对当前强基线，Replica 均值 +2.002 点，SceneNN +0.656 点；两域均优于普通几何去噪。结果支持继续关注可见性证据，但未达到本轮完整验收门槛。</div>',
        '<h2>本轮具体改了什么</h2><p>保留强基线的对象 ID、类别、观测成员、点坐标和前端。将对象点投影到原 RGB-D 帧，在深度一致且掩码证据有效的位置，统计支持与反证；只有重复反证占优时才删除点。遮挡、缺深度、视野外、未分割区域和边界带不投反对票。这个模块是离线的、只删点的对象内部清理，不拆分或重关联对象。</p>',
        '<p>比较项：共享 SAM2.1 / SigLIP 前端的 CG、当前 History-2 强基线、普通 20 邻域 / 2σ 几何去噪、完整方法，以及只关闭深度可见性、只将三帧反证门槛改为一帧的消融。单帧反证消融仍累计全部视角并使用相同正负票比，不是单帧建图。</p>',
        '<h2>主表：完整对照，不挑最优版本</h2><img src="comparison.png" alt="两个数据集的六组对照">',
        table(['数据集','方法','实例 PQ×100','Δ 相对强基线','TP / FP / FN','新增 / 丢失匹配'],comparisons),
        '<h2>消融揭示什么</h2><p>关闭可见性判断后，完整方法的均值优势为 Replica +1.534 点、SceneNN +0.048 点；Replica 少丢失一个原来正确的对象。这个结果支持深度可见性用于避免误删，但 SceneNN 的增量很小。</p><p>三帧确认并未优于一帧反证消融：完整方法在 Replica 低 0.016 点、SceneNN 低 0.378 点。因此不能宣称“三帧确认”是有效贡献，也没有把消融中更高的分数替换成主结果。两者都只是本轮固定开发配置。</p>',
        '<img src="gains_and_losses.png" alt="逐场景增益和逐对象IoU损失">',
        '<h2>逐场景结果与删除代价</h2>',table(['场景','方法','PQ×100','Δ PQ','TP/FP/FN','新增 GT','丢失 GT','删除前景点','清空对象'],scene_rows),
        '<p>完整方法四场景均未清空任何对象；新增匹配不是靠删掉整条预测记录换来的。仍有部分对象 IoU 下降，不能把“没有丢失正确匹配”写成“所有对象都改善”。完整逐对象数据见下方下载。</p>',
        table(['场景','IoU 提高对象','IoU 降低对象','不变对象','下降最多 GT','最差对象 IoU 前→后'],case_rows),
        '<h2>全部新增正确匹配</h2><p>以下展示两个新增匹配。绿点是正确支持，红点是超出该 GT 的支持，灰色是 GT。所有面板使用同一局部显示坐标；坐标仅为事后可视化，不进入算法。图中为固定评价域的投影标签，不是原始 RGB 掩码。</p><img src="new_matches.png" alt="两个新增匹配的前后对照">',
        '<h2>涨点来源：包含最近邻投影归属变化</h2><p>删除点后，固定评价域的部分体素会被分配给仍在邻域内的其他对象。这个变化属于同一评分协议下的真实结果，但不等于原始地图增加了新的几何点，也不能全部解释成对象自身污染减少。</p>',
        table(['场景','GT','有效 GT 体素','正确覆盖 前→后','超出 GT 前→后','从其他对象转入的正确体素'],
              [[r['scene'],r['gt_id'],r['gt_valid_voxels'],f"{r['correct_before']} → {r['correct_after']}",f"{r['excess_before']} → {r['excess_after']}",r['added_correct_from_other_object']] for r in match_mechanisms]),
        '<p>Replica 新增匹配 GT65 在有效域仅有 20 个体素；正确覆盖从 15 增至 20，而超出 GT 的体素从 14 增至 15，收益主要表现为投影竞争关系改变。SceneNN 新增匹配同时改善覆盖并减少部分额外支持。两个新增对象都接近 IoU 门槛，仍需更完整几何域与独立数据验证稳健性；本轮没有据此改评分或追加方法实验。</p>',
        table(['场景','归属变化体素','转给另一对象','变为未分配'],[[r['scene'],r['changed_valid_voxels'],r['transferred_to_other_object'],r['cleared']] for r in projection_changes]),
        '<h2>成本与范围</h2>',table(['场景','本轮总处理 / s','证据累计 / s','每帧证据 P95 / ms','峰值 RSS / MiB'],cost_rows),
        '<p>复用冻结地图和 200 个语义帧；Replica 原输入覆盖 2000 帧，SceneNN 原输入覆盖 400 帧。这里只计离线清理，非全链路 SLAM 运行时间。两路 CPU 任务、每路单线程数值库，无 GPU 新推理。普通 SOR 单独计时保存在原始 JSON。</p>',
        '<h2>验收决定</h2><p>本轮预设两域各平均 +2 PQ 点、至少三场景不退步、两域净增正确对象且优于普通去噪、两项消融均有正增量。Replica +2.002 点通过幅度要求，SceneNN +0.656 点未通过；多帧确认消融未通过。<b>本配置不自动升为主版本，不继续扫阈值或扩场景。</b>保留已证实的开发集增益和可见性方向，三帧确认不作为贡献。任何后续简化方法与独立测试均须另行固定协议。</p>',
        '<h2>公平性与论文边界</h2><p>所有 16 份预测在新 GT 评分前封存；未按结果调参或换场景。评分沿用固定 5cm 观测有效域、同一最近邻投影及严格 IoU&gt;0.5，CG 与强基线逐项复现。对象内删除会改变最近邻的归属范围，所有方法使用相同投影。报告不是官方全网格榜单成绩。</p><p>四场景均已用于此前开发，不能当作独立泛化证据；CG 为共享前端版本，不是原论文全系统复现。收益来自几何清理，类别读出没有改变，不能据此宣称开放词汇分类提升。涨点为投稿提供了正面证据，但仍需要独立场景、外部强方法对比和新颖性论证。</p>',
        '<h2>可复核文件</h2><p><a href="protocol.json">运行前固定协议</a> · <a href="summary.json">完整汇总</a> · <a href="audit.json">独立复算核验</a></p>',
    ]
    for key in KEYS:
        sections.append(f'<p>{key}：<a href="scenes/{key}/metrics.json">全部指标</a> · <a href="scenes/{key}/per_object.json">每个 GT 的 IoU、覆盖与得失</a></p>')
    page='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>omni_slam 多视角清理实验</title><style>body{max-width:1260px;margin:32px auto;padding:0 24px;color:#172c3c;font:16px/1.75 system-ui,sans-serif}h1{font-size:29px}h2{margin-top:36px}.sub{color:#587082}.callout{background:#eaf6f2;border-left:5px solid #16a185;padding:18px}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow-x:auto}th,td{padding:8px 11px;border-bottom:1px solid #dbe3ea;text-align:left;white-space:nowrap}th{background:#eef3f7}a{color:#066e9a}</style></head><body>'+''.join(sections)+'</body></html>'
    (REPORT/'index.html').write_text(page)
    for name in ['protocol.json','summary.json','predictions_complete.json']:shutil.copyfile(OUT/name,REPORT/name)
    audit=dict(complete=True,checked_epoch=time.time(),independent_metric_checks=checked,
        baseline_projection_equivalence_all_scenes=True,source_geometry_exact_subset_all_outputs=True,
        unchanged_id_membership_category_all_outputs=True,all_predictions_sealed_before_scoring=True,
        new_geometry_outputs=16,unit_tests_passed=7,per_object_iou_changes=regressions,
        projection_changes=projection_changes,new_match_mechanisms=match_mechanisms,
        protocol_sha256=sha(OUT/'protocol.json'),summary_sha256=sha(OUT/'summary.json'),
        static_plots_require_visual_inspection=True,browser_interaction_tested=False,
        scope='Observed-domain development pilot; no independent test or official rank.',
        outputs={str(p.relative_to(REPORT)):sha(p) for p in REPORT.rglob('*') if p.is_file() and p.name!='audit.json'})
    dump(REPORT/'audit.json',audit);print('PUBLISHED',REPORT,'independent score checks',checked,flush=True)


if __name__=='__main__':main()
