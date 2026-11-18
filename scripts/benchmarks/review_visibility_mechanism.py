"""Read-only post-hoc analysis of the sealed visibility experiment.

Never builds maps, changes predictions, tunes thresholds or invokes a model.
Writes only a separate report and its reproducible numerical evidence.
"""
from pathlib import Path
import hashlib
import html
import json
import time

import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'build/benchmarks/visibility_independent_20261112'
OUT = ROOT / 'docs/reports/visibility_mechanism_review_2026-11-12'
SCENES = ['052', '054', '057', '061']
METHODS = ['cg', 'history2', 'sor', 'visibility1', 'visibility3',
           'visibility1_no_depth', 'maskclustering']
NAMES = dict(cg='CG · shared input', history2='omni_slam strong baseline',
             sor='Baseline + SOR', visibility1='omni_slam + Visibility Cleanup (Ours)',
             visibility3='Ours · 3-frame control',
             visibility1_no_depth='Ours · without depth visibility',
             maskclustering='MaskClustering · shared input')
INPUT_HASHES = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def remember(p):
    p = Path(p)
    INPUT_HASHES[str(p.relative_to(ROOT))] = sha(p)
    return p


def read(p):
    return json.loads(remember(p).read_text())


def npz(p):
    with np.load(remember(p)) as z:
        return {k: z[k] for k in z.files}


def dump(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def pair_stats(truth, pred, gid, pid):
    gt, pm = truth == gid, pred == pid
    correct = int((gt & pm).sum())
    excess = int((~gt & pm).sum())
    missing = int((gt & ~pm).sum())
    return dict(correct=correct, excess=excess, missing=missing,
                iou=correct / (correct + excess + missing))


def table(head, rows):
    def cells(tag, row):
        return ''.join(f'<{tag}>{html.escape(str(x))}</{tag}>' for x in row)
    return '<table><thead><tr>' + cells('th', head) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + cells('td', row) + '</tr>' for row in rows) + '</tbody></table>'


def collect():
    protocol = read(BASE / 'protocol.json')
    for path, digest in protocol['source_sha256'].items():
        assert sha(remember(ROOT / path)) == digest, path
    summary = read(BASE / 'summary.json')
    status = read(BASE / 'queue_status.json')
    assert status['complete'] and sha(BASE / 'summary.json') == status['summary_sha256']
    barrier = read(BASE / 'predictions_complete.json')
    assert barrier['complete']
    scenes, rows = {}, []
    for scene in SCENES:
        folder = BASE / 'evaluation' / scene
        seal = read(BASE / 'predictions' / (scene + '.json'))
        assert sha(BASE / 'predictions' / (scene + '.json')) == barrier['files'][scene]
        for method, record in seal['methods'].items():
            for field in ['geometry', 'classification']:
                assert sha(remember(record[field + '_path'])) == record[field + '_sha256']
        metrics = read(folder / 'metrics.json')['methods']
        ref = npz(folder / 'reference.npz')
        valid = ref['valid']; truth = ref['instance'][valid]; xyz = ref['xyz'][valid]
        preds = {n: npz(folder / (n + '.npz'))['instance'][valid] for n in METHODS}
        for n in ['sor', 'visibility1', 'visibility3', 'visibility1_no_depth']:
            preds[n + '_locked'] = npz(folder / (n + '_locked.npz'))['instance'][valid]
        denoms = {n: .5 * (len(np.unique(a[a > 0])) + len(np.unique(truth[truth > 0])))
                  for n, a in preds.items()}
        objects = []
        for gid in np.unique(truth[truth > 0]):
            obj = dict(scene=scene, gt_id=int(gid), gt_voxels=int((truth == gid).sum()), methods={})
            for n, a in preds.items():
                candidates = [int(pid) for pid in np.unique(a[truth == gid]) if pid > 0]
                pairs = [(pair_stats(truth, a, gid, pid), pid) for pid in candidates]
                best, pid = max(pairs, key=lambda x: (x[0]['iou'], -x[1])) if pairs else (
                    dict(correct=0, excess=0, missing=obj['gt_voxels'], iou=0.), 0)
                obj['methods'][n] = dict(best, predicted_id=pid, matched=best['iou'] > .5,
                    pq_contribution100=100 * best['iou'] / denoms[n] if best['iou'] > .5 else 0.)
            obj['delta_ours_baseline'] = obj['methods']['visibility1']['iou'] - obj['methods']['history2']['iou']
            obj['delta_ours_sor'] = obj['methods']['visibility1']['iou'] - obj['methods']['sor']['iou']
            obj['delta_pq_ours_sor'] = obj['methods']['visibility1']['pq_contribution100'] - obj['methods']['sor']['pq_contribution100']
            objects.append(obj)
        # Every per-object PQ contribution must reconstruct the frozen metric.
        for n in preds:
            record = metrics[n[:-7]]['locked_assignment_score'] if n.endswith('_locked') else metrics[n]['score']
            reconstructed = sum(o['methods'][n]['pq_contribution100'] for o in objects)
            assert np.isclose(reconstructed, record['observed_voxel_instance_pq'] * 100, atol=1e-10, rtol=0), (scene, n)
            assert sum(o['methods'][n]['matched'] for o in objects) == record['tp']
        scenes[scene] = dict(truth=truth, xyz=xyz, preds=preds, metrics=metrics, objects=objects,
                             denoms=denoms)
        rows.extend(objects)
    dump(OUT / 'per_object.json', rows)
    return protocol, summary, scenes, rows


def diagnosis(scenes):
    scene = scenes['057']; gt, preds = scene['truth'], scene['preds']; gid, pid = 136313, 7
    pair = {n: pair_stats(gt, preds[n], gid, pid)
            for n in ['history2', 'sor', 'visibility1', 'sor_locked', 'visibility1_locked']}
    main = preds['visibility1'] == pid; locked = preds['visibility1_locked'] == pid
    added = main & ~locked
    g = npz(BASE / 'maps/057/cg/history2/geometry.npz')
    evidence = npz(BASE / 'maps/057/cleanup/evidence.npz')
    indices = evidence['source_indices']; owners = g['instance'][indices]
    positive, negative = evidence['positive'], evidence['negative']
    rejected = (negative >= 1) & (negative >= 2 * positive)
    retained = np.load(remember(BASE / 'maps/057/cleanup/visibility1/retained_source_indices.npy'))
    assert np.array_equal(rejected, ~np.isin(indices, retained))
    # Assign each valid reference voxel to its original foreground source point.
    distances, nearest = cKDTree(g['xyz'][indices]).query(scene['xyz'], workers=1)
    predicted_baseline = np.where(distances <= .05, owners[nearest], 0)
    assert np.array_equal(predicted_baseline, preds['history2'])
    reason = np.full(len(indices), 'below_2x_support', dtype='<U30')
    reason[negative == 0] = 'no_negative_with_positive'
    reason[(negative == 0) & (positive == 0)] = 'no_votes'
    reason[rejected] = 'rejected'
    reason_hist = lambda ids: {str(k): int(v) for k, v in zip(*np.unique(reason[ids], return_counts=True))}
    excess = (preds['history2'] == pid) & (gt != gid)
    owner = owners == pid
    target = next(o for o in scene['objects'] if o['gt_id'] == gid)
    gap = (scene['metrics']['visibility1']['score']['observed_voxel_instance_pq'] -
           scene['metrics']['sor']['score']['observed_voxel_instance_pq']) * 100
    critical = target['delta_pq_ours_sor']
    result = dict(scene='057', gt_id=gid, predicted_id=pid, pair_stats=pair,
        strict_iou_threshold=.5, pq_denominator=scene['denoms']['visibility1'],
        ours_minus_sor_pq100=gap, critical_object_contribution_pq100=critical,
        all_other_objects_contribution_pq100=gap-critical,
        reassigned_into_object=dict(total=int(added.sum()), correct=int((added & (gt == gid)).sum()),
                                    excess=int((added & (gt != gid)).sum())),
        reassigned_out=int((locked & ~main).sum()),
        source_point_evidence=reason_hist(np.flatnonzero(owner)),
        baseline_excess_reference_voxel_evidence=reason_hist(nearest[excess]),
        raw_points=int(owner.sum()),
        raw_removed_sor=int((owner & ~np.isin(indices, np.load(remember(BASE/'maps/057/cleanup/sor/retained_source_indices.npy')))).sum()),
        raw_removed_ours=int((owner & rejected).sum()),
        interpretation='Post-hoc mechanism diagnosis only. No threshold or evaluator change. '
                       'Vote totals cannot distinguish persistent mask contamination from insufficient viewpoint evidence.')
    dump(OUT / 'scene057_diagnosis.json', result)
    return result


def savefig(fig, name):
    fig.savefig(OUT / (name + '.png'), dpi=170)
    fig.savefig(OUT / (name + '.svg'))
    plt.close(fig)


def point_panels(scenes, cases, methods, name):
    fig, axes = plt.subplots(len(cases), len(methods), figsize=(4.1*len(methods), 3.9*len(cases)),
                             squeeze=False, layout='constrained')
    titles = dict(history2='Strong baseline', visibility1='Ours', sor='SOR', visibility1_locked='Ours · locked assignment')
    for row, (scene, gid, pid, purpose) in enumerate(cases):
        s = scenes[scene]; xyz, truth = s['xyz'], s['truth']; gt = truth == gid
        _, _, basis = np.linalg.svd(xyz[gt] - xyz[gt].mean(0), full_matrices=False)
        uv = (xyz - xyz[gt].mean(0)) @ basis[:2].T
        relevant = gt.copy()
        for m in methods: relevant |= s['preds'][m] == pid
        lo, hi = uv[relevant].min(0), uv[relevant].max(0); pad = np.maximum((hi-lo)*.07, .03)
        for col, m in enumerate(methods):
            ax = axes[row, col]; pm = s['preds'][m] == pid
            stats = pair_stats(truth, s['preds'][m], gid, pid)
            ax.set_facecolor('#f4f6f8')
            ax.scatter(uv[gt & ~pm, 0], uv[gt & ~pm, 1], s=9, color='#a9b5c4', label='Missed GT')
            ax.scatter(uv[pm & ~gt, 0], uv[pm & ~gt, 1], s=9, color='#c45771', label='Excess')
            ax.scatter(uv[pm & gt, 0], uv[pm & gt, 1], s=9, color='#159b84', label='Correct')
            ax.set_title(f'{titles[m]} | IoU {stats["iou"]:.4f}\n{scene} / GT {gid} / pred {pid}', fontsize=10)
            ax.set_xlim(lo[0]-pad[0], hi[0]+pad[0]); ax.set_ylim(lo[1]-pad[1], hi[1]+pad[1])
            ax.set_aspect('equal'); ax.set_xlabel('Common PCA display axis / m', fontsize=8)
            if col == 0: ax.set_ylabel(purpose + '\nAxis 2 / m', fontsize=9)
            ax.tick_params(labelsize=8)
            if row == 0 and col == len(methods)-1: ax.legend(fontsize=8, loc='best')
    savefig(fig, name)


def charts(scenes, diagnostic):
    # All GT objects, including regressions and unmatched ones.
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
    for ax, (key, s) in zip(axes.flat, scenes.items()):
        objects = sorted(s['objects'], key=lambda o: o['delta_ours_baseline'])
        values = [100*o['delta_ours_baseline'] for o in objects]
        ax.bar(range(len(values)), values, color=['#159b84' if v >= 0 else '#c45771' for v in values])
        ax.axhline(0, color='#46586a', lw=.7)
        ax.set_title(f'{key} | all {len(values)} GT instances')
        ax.set_ylabel('Best-IoU change / percentage points')
        ax.set_xlabel('GT instances sorted by change (IDs in per_object.json)')
    savefig(fig, 'all_object_changes')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout='constrained')
    d = diagnostic
    vals = [d['critical_object_contribution_pq100'], d['all_other_objects_contribution_pq100'], d['ours_minus_sor_pq100']]
    axes[0].bar(['GT 136313\nthreshold crossing', 'All other\nGT objects', 'Total gap'], vals,
                color=['#c45771', '#159b84', '#647b91'])
    axes[0].axhline(0, color='#46586a', lw=.7)
    for i, v in enumerate(vals): axes[0].text(i, v, f'{v:+.3f}', ha='center', va='bottom' if v>0 else 'top')
    axes[0].set_ylim(-2.35, 1.1); axes[0].set_ylabel('Ours − SOR / PQ percentage points')
    axes[0].set_title('057: exact additive PQ decomposition')
    for m, color in [('sor', '#bf9861'), ('visibility1', '#159b84')]:
        values = [s['metrics'][m]['score']['observed_voxel_instance_pq']*100 for s in scenes.values()]
        locked = [s['metrics'][m]['locked_assignment_score']['observed_voxel_instance_pq']*100 for s in scenes.values()]
        axes[1].plot([0,1], [np.mean(values), np.mean(locked)], marker='o', color=color, label='Ours' if m=='visibility1' else 'SOR')
        for i,v in enumerate([np.mean(values),np.mean(locked)]): axes[1].annotate(f'{v:.3f}', (i,v), xytext=(8,5), textcoords='offset points')
    axes[1].set_xticks([0,1], ['Primary: nearest 5 cm', 'Diagnostic: locked ownership'])
    axes[1].set_xlim(-.2,1.5); axes[1].set_ylabel('Mean instance PQ × 100'); axes[1].legend()
    axes[1].set_title('Scoring sensitivity: report both; do not replace primary')
    savefig(fig, 'mechanism_decomposition')
    point_panels(scenes, [('057',136313,7,'Threshold case')],
                 ['history2','visibility1','sor','visibility1_locked'], '057_threshold_case')
    point_panels(scenes, [('057',59005,29,'Improvement'),('061',64749,10,'Coverage retained'),
                         ('057',208002,43,'Regression retained')],
                 ['history2','visibility1','sor'], 'representative_objects')


def architecture():
    fig, ax = plt.subplots(figsize=(15, 7.3)); ax.set_xlim(0,15); ax.set_ylim(0,7.3); ax.axis('off')
    def box(x,y,w,h,title,body,color='#eaf0f6',edge='#647b91'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.12,rounding_size=0.10',
                                   facecolor=color,edgecolor=edge,lw=1.3))
        ax.text(x+.16,y+h-.25,title,fontsize=12,fontweight='bold',va='top',color='#183247')
        ax.text(x+.16,y+h-.65,body,fontsize=10,va='top',linespacing=1.5,color='#31485b')
    def arrow(a,b):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,color='#647b91',lw=1.5))
    ax.text(.2,6.95,'omni_slam | Visibility-guided object geometry revision',fontsize=19,fontweight='bold',color='#183247')
    ax.text(.2,6.5,'Evaluated path: fixed RGB-D observations and given poses · no training · no identity reassignment',fontsize=11,color='#647b91')
    box(.2,4.1,3,1.85,'01  Shared observations','RGB + measured depth + poses\nSAM2.1 instance proposals\nSigLIP-384 semantic descriptors')
    box(3.8,4.1,4.6,1.85,'02  Frozen strong baseline','CG association + history2 graph\nTwo-frame bridge evidence; coframe veto\nFinal object IDs, masks and class readout')
    box(9,4.1,5.6,1.85,'03  Evidence ledger (Ours)','Project each source point into its object views\nDepth agreement: |z − D| ≤ 0.05 + 0.01 D\nOwn interior: support; other known mask: refute', '#e4f4ef','#159b84')
    box(9,.95,5.6,2.15,'04  Geometry revision (Ours)','Accumulate one vote per point per frame\nReject only if N ≥ 1 and N ≥ 2P\nKeep object identities, memberships and classes\nOutput: cleaned object point sets', '#e4f4ef','#159b84')
    box(3.8,.95,4.6,2.15,'05  Evaluation and diagnostics','Primary: shared observed domain, 5 cm\nInstance PQ; strict IoU > 0.5\nSecondary: lock original nearest ownership\nAll objects, gains and regressions reported')
    box(.2,.95,3,2.15,'Abstain when uncertain','No own mask in this frame\nMissing / disagreeing depth\nOutside view or 2 px boundary\nUnsegmented pixel')
    arrow((3.32,5.05),(3.68,5.05));arrow((8.52,5.05),(8.88,5.05));arrow((11.8,3.98),(11.8,3.22));arrow((8.88,2),(8.52,2))
    ax.text(.2,.35,'System context: LIVO supplies poses in the robot pipeline. This benchmark uses given poses; DINO is not enabled in this frozen branch.',fontsize=10,color='#647b91')
    savefig(fig, 'method_architecture')


def publish(protocol, summary, scenes, objects, d):
    method_rows=[]
    for n in METHODS:
        vals=[s['metrics'][n]['score']['observed_voxel_instance_pq']*100 for s in scenes.values()]
        counts=[sum(s['metrics'][n]['score'][k] for s in scenes.values()) for k in ['tp','fp','fn']]
        locked=np.mean([s['metrics'][n]['locked_assignment_score']['observed_voxel_instance_pq']*100 for s in scenes.values()]) if n in ['sor','visibility1','visibility3','visibility1_no_depth'] else None
        method_rows.append([NAMES[n],*[f'{v:.3f}' for v in vals],f'{np.mean(vals):.3f}',
                            '/'.join(map(str,counts)),f'{locked:.3f}' if locked is not None else '—'])
    per_scene=[]
    for key,s in scenes.items():
        changes=np.array([o['delta_ours_baseline'] for o in s['objects']]);worst=min(s['objects'],key=lambda o:o['delta_ours_baseline'])
        per_scene.append([key,len(changes),int((changes>1e-10).sum()),int((changes < -1e-10).sum()),int((abs(changes)<=1e-10).sum()),worst['gt_id'],f'{worst["delta_ours_baseline"]*100:+.3f}'])
    pair_rows=[[n, v['correct'],v['excess'],v['missing'],f'{v["iou"]:.6f}', '通过' if v['iou']>.5 else '未通过'] for n,v in d['pair_stats'].items()]
    object_rows = [[o['scene'], o['gt_id'], o['gt_voxels'],
                    *[f'{o["methods"][n]["iou"]:.6f}' for n in ['history2','visibility1','sor']],
                    f'{o["delta_ours_baseline"]*100:+.3f}',
                    '匹配' if o['methods']['visibility1']['matched'] else '未匹配'] for o in objects]
    reason_names = dict(no_negative_with_positive='有支持、无反证', no_votes='无正负票',
                        below_2x_support='有反证，但不足支持票的两倍', rejected='满足拒绝条件')
    lead='冻结 Ours；保留已验证增益。057 的主要差距来自一个临界对象跨越匹配阈值，不能概括为整张地图越改越差。'
    body=[f'<h1>可见性证据驱动的对象几何修订</h1><p class="sub">机制复核与论文材料 · 2026-11-12 · 仅使用已有产物，0 次新建图 / 推理 / 训练 / 调参</p><div class="callout">{lead}</div>',
      '<h2>1. 本轮结论与去留决定</h2><p>四个新 SceneNN 场景相对 history2 全涨，均值 +0.841 PQ 百分点；66 个正确匹配均保留。Ours 相对 SOR 为 −0.050，原预设验收仍未全部通过。继续以对象几何修订为论文候选贡献，当前算法与评分协议冻结，不因本次事后分析改变阈值或切换主指标。</p>',
      '<h2>2. 方法与真实执行路径</h2><img src="method_architecture.png" alt="实际评测方法架构"><p>当前强基线为 CG 共享前端关联 + history2 历史图合并：两帧桥接证据、同帧冲突否决，再执行既有最终处理。新模块只清理已归属点，不拆分/合并对象，不改变类别或观测归属。共享前端是 SAM2.1 与 SigLIP-384；这轮未启用 DINO，也未训练注意力。机器人系统可由 LIVO 提供位姿，但本轮 RGB-D benchmark 使用给定位姿，不能据此宣称 LIVO 精度或端到端实时优势。</p>',
      '<p>对对象 o 的源点 p，将其投影到存在该对象观测的帧 t。深度一致条件为 |z−D|≤0.05+0.01D（米），且 z&gt;0.2 m、像素在图内。点落入该对象掩码并集腐蚀后的内部时支持；落在其膨胀区域以外、且属于任一已分割掩码时反证。边界采用 2 像素方形核。无对象观测、无有效深度、深度不一致、视野外或未知区域均不投票。每点每帧最多一票。</p>',
      '<div class="formula">P(p)=Σ supportₜ(p), N(p)=Σ refuteₜ(p)<br>reject(p) ⇔ N(p) ≥ 1 且 N(p) ≥ 2P(p)</div><p>“1”是累计反证帧数的最低门槛，并非只看单帧；所有 200 个语义帧参与证据累计。三帧版只把 N≥1 改为 N≥3。无深度消融移除深度门控，仍保留视野、掩码、边界及同一拒绝规则。SOR 对每对象使用 20 近邻均距、均值+2σ 阈值。</p>',
      '<h2>3. 主表与必要消融</h2>'+table(['方法','052','054','057','061','均值 PQ×100','TP/FP/FN 合计','固定归属均值'],method_rows),
      '<p>主表为固定观测有效域的实例 PQ，严格 IoU&gt;0.5；不是完整网格官方排行榜，也不是带类别正确性约束的开放词汇语义 PQ。所有场景保留，无测试后挑选。Ours 比无深度消融均值高 0.454（两胜两负），比三帧控制高 0.136；三帧确认不作为贡献。MaskClustering 为官方核心共享输入适配版，非原论文完整管线，不能把其差距写成原版 SOTA 排名。</p>',
      '<h2>4. 057：差距可以精确分解</h2>'+table(['配置 / 指标','正确支持体素','多余支持体素','漏失 GT 体素','该对象 IoU','严格 0.5 门槛'],pair_rows),
      f'<p>同一 GT 136313 / 预测对象 7：基线 0.492718，Ours 0.496947，SOR 0.535340。该场景所有三种方法的 PQ 分母均为 28。因此这个对象给 Ours−SOR 带来 {d["critical_object_contribution_pq100"]:+.6f} 点，其余对象合计 {d["all_other_objects_contribution_pq100"]:+.6f} 点，总差距 {d["ours_minus_sor_pq100"]:+.6f} 点。Ours 在其他对象上总体更好，但一个匹配状态改变就超过了这部分收益。这是指标的真实阈值效应，不应为了消除差距而调整评分。</p>',
      '<img src="mechanism_decomposition.png" alt="逐对象 PQ 差距分解与评分敏感性"><img src="057_threshold_case.png" alt="057 临界对象统一视角对照">',
      '<h3>最近邻重分配的影响</h3><p>固定原始点归属时，Ours 对该对象 IoU 为 0.501247；主评分允许删除后重新找最近点，有 22 个体素重新进入该对象，其中 5 个正确、17 个多余，IoU 降回 0.496947。固定归属下 057 的 Ours PQ=33.436，SOR=33.106；排名与主评分相反。两套分数必须并列报告，固定归属只是诊断，不能事后替换主评分。</p>',
      '<h3>为什么没有把多余点全部清掉</h3><p>对象 7 原始有 2669 个前景源点；Ours 删除 92 个，SOR 删除 97 个，总数量接近但位置不同。Ours 的 2454 个源点没有反证，另有 123 个虽有反证但不足支持票的两倍；115 个源点完全没有正负票。这些是整个对象的源点统计，不等同于错误点比例。</p>'+
      table(['基线 410 个多余参考体素的原始最近点证据','数量'], [(reason_names[k],v) for k,v in d['baseline_excess_reference_voxel_evidence'].items()])+
      '<p>410 个多余参考体素中，304 个的源点有支持却无反证，73 个无正负票，11 个反证不足支持票的两倍，22 个满足拒绝条件。这里定位到了具体的证据缺口：当前规则仍会保留被掩码支持或没有充分反证的污染。累计票数不能单独区分“视角不足”与“掩码持续污染”，本轮不把推测写成已证实根因，也不追加逐帧模型推理。</p>',
      '<h2>5. 全部对象得失与典型图</h2>'+table(['场景','GT 数','IoU 改善','IoU 退步','不变','最差 GT','最差 ΔIoU 百分点'],per_scene)+
      '<p>统计每个 GT 的最佳对象 IoU，包含未匹配对象；合计 100 个改善、27 个退步、94 个不变。可视化案例在评分后选择用于解释：057/59005 展示改善，061/64749 展示相对 SOR 的覆盖保留，057/208002 保留退步案例。不得把这些插图当作额外独立测试。</p><img src="all_object_changes.png" alt="四场景全部对象改善和退步"><img src="representative_objects.png" alt="改善、覆盖保留与退步案例"><p>点图展示固定观测有效域，绿色为正确支持，红色为多余支持，灰色为漏失 GT。每行统一 GT-PCA 视角与坐标范围，供事后展示；不参与建图或坐标对齐，二维投影可能遮叠。</p><details><summary>展开全部 221 个对象的得失</summary>'+
      table(['场景','GT ID','GT 体素','基线最佳 IoU','Ours 最佳 IoU','SOR 最佳 IoU','Ours ΔIoU 百分点','Ours 状态'],object_rows)+'</details>',
      '<h2>6. 论文可写的论点与剩余边界</h2><ul><li><b>已支持：</b>固定配置在四个新序列改善对象几何，维持匹配对象集合；不需要额外训练。用全对象表、深度消融和固定归属诊断支撑。</li><li><b>候选方法贡献：</b>在固定对象身份上，利用可见性门控的支持/反证累计实现保守几何修订。现阶段称为候选贡献，既有方法已有多视角点过滤，新颖性仍需机制级区分。</li><li><b>尚未支持：</b>普遍优于 SOR、跨数据集独立泛化、原版外部 SOTA 排名、开放词汇类别准确率提升、LIVO 位姿提升或端到端实时性能。</li><li><b>下一步限定：</b>先完成这套固定算法的论文方法与实验草稿。若进入投稿补测，优先补原版外部方法及第二数据集未参与开发的场景；另立协议与数据划分，不在已看过的 057 上继续调规则。</li></ul>',
      '<h2>7. 冻结与复现</h2><p><a href="freeze_manifest.json">冻结输入与审计清单</a> · <a href="per_object.json">全对象数据</a> · <a href="scene057_diagnosis.json">057 精确机制数据</a> · <a href="paper_method.md">论文方法与实验草稿</a> · <a href="method_architecture.svg">矢量架构图</a> · <a href="../visibility_independent_2026-11-12/index.html">原始独立验证报告</a></p>']
    page='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>omni_slam 可见性修订机制复核</title><style>body{max-width:1240px;margin:40px auto;padding:0 24px;font:16px/1.8 system-ui;color:#203449;background:#fafbfd}h1,h2,h3{line-height:1.4}h2{margin-top:44px;border-top:1px solid #dce4ec;padding-top:24px}img{max-width:100%;height:auto;background:white;margin:16px 0}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px 8px;border-bottom:1px solid #dce4ec;text-align:left}th{background:#eaf0f6}a{color:#087f70}.sub{color:#647b91}.callout,.formula{padding:20px;background:#e4f4ef;border-left:4px solid #159b84}.formula{font:19px/1.8 monospace}</style></head><body>'+''.join(body)+'</body></html>'
    (OUT/'index.html').write_text(page)
    md='''# Visibility-guided Object Geometry Revision — frozen method draft

## Scope and implementation

The evaluated method is **omni_slam + Visibility Cleanup (Ours)** (`visibility1`).
It refines the geometry of existing objects without changing object identities,
observation memberships, semantic labels, or poses. This is an offline revision
module in the current implementation, not a demonstrated online real-time system.

The frozen benchmark frontend uses SAM2.1 proposals and SigLIP-384 descriptors.
The strong baseline comprises shared-input CG association, a history graph with
two-frame bridge evidence and coframe conflict rejection, and inherited final
processing. DINO is disabled in this branch. RGB-D benchmark poses are given;
the robot pipeline's LIVO pose estimator is outside this experiment's claims.

## Visibility-conditioned support and refutation

For each source point p assigned to object o, project p into a frame t only when
that frame contains an observation assigned to o. Form U(o,t), the union of all
of o's masks, and K(t), the union of all segmented masks. Let V(p,t) require
in-image projection, z > 0.2 m, valid measured depth D, and
|z-D| <= 0.05 m + 0.01 D. These depth conditions test surface consistency; they
do not establish perfect visibility under inaccurate depth or camera poses.

Support is V AND membership in the 2-pixel eroded U. Refutation is V AND being
outside the 2-pixel dilated U AND inside K. Invalid depth, depth disagreement,
out-of-view points, uncertain boundaries, unsegmented pixels, and absent object
observations abstain. Each point receives at most one vote per frame, regardless
of overlapping masks. Sum support P and refutation N over all sampled frames.

    reject(p) = [N(p) >= 1] AND [N(p) >= 2 P(p)]

The one-frame minimum does not discard other views: it changes only the minimum
refutation count. The three-frame control uses N >= 3. The no-depth ablation
removes V's depth conditions while retaining the same projection, masks, border
handling and rejection rule. The geometric SOR control uses per-object mean
20-neighbor distance and a mean + 2 standard deviation rejection threshold.

## Experimental design and results

Four new SceneNN sequences (052, 054, 057, 061) were fixed before processing.
Each uses 400 sampled geometry frames and 200 semantic frames. All predictions
were sealed before scoring. All 28 scored outputs are reported, with no new
parameter search. This tests unseen sequences within one dataset, not independent
cross-dataset generalization. Earlier Replica/SceneNN pilot scenes are development
data and must be kept separate.

Primary scoring uses the frozen observed-valid domain, exclusive nearest-point
assignment within 5 cm, and strict instance IoU > 0.5. This is instance geometry
PQ, not class-aware semantic PQ or the official full-mesh leaderboard protocol.
The secondary diagnostic freezes each reference voxel's original nearest source
point; removing that point makes the voxel unassigned rather than reassigning it.
The secondary metric must not replace the primary after observing results.

| Method | 052 | 054 | 057 | 061 | Mean PQ x 100 |
|---|---:|---:|---:|---:|---:|
'''
    for row in method_rows: md+='| '+' | '.join(row[:6])+' |\n'
    md+='''
Ours improves the strong baseline in all four sequences, by 0.841 PQ percentage
points on average, with the same 66 matched GT instances and no gained or lost
matches. Mean PQ exceeds the no-depth ablation by 0.454 points (two wins and two
losses). Three-frame confirmation has no independent advantage in this test.
Ours trails SOR by 0.050 points on the primary mean, so the prespecified full
acceptance gate remains failed. Under locked ownership, Ours gains 1.095 points
over the strong baseline; this supports a geometry effect beyond reassignment,
but does not establish superiority under the primary metric.

## Mechanism case: SceneNN 057

For GT 136313 / predicted object 7, IoU is 0.492718 (baseline), 0.496947 (Ours)
and 0.535340 (SOR). With a common PQ denominator of 28, the SOR-only match gives
Ours minus SOR a -1.911930 point contribution; all other objects contribute
+0.648102 points, yielding the -1.263827 scene gap.

Under locked ownership, the same Ours object reaches 0.501247 IoU. Nearest-point
reassignment adds 22 voxels to it (5 correct, 17 excess), bringing it back below
the matching threshold. This is a documented metric sensitivity, not a reason
to change the evaluation protocol. Ours and SOR delete 92 and 97 raw source
points respectively from this object; deletion locations, not only counts,
matter. Accumulated evidence alone cannot separate inadequate views from
persistent mask contamination. Case illustrations are post-hoc explanations,
not additional independent validation.

## Claims and limitations

The supported claim is that frozen visibility-conditioned revision improves
object geometry across four unseen sequences while retaining existing matches.
The candidate contribution is the specified support/refutation mechanism under
fixed object identities, supported by depth ablation and attribution diagnostics.
Point filtering itself is not new: external methods also employ multiview filters.
Novelty and a general advantage over geometric denoising remain unproven.

Shared-input CG and MaskClustering scores must be labeled as adapted comparisons.
The latter replaces CropFormer/OpenCLIP and native backprojection supports with
shared inputs and exports exclusive labels; it is not an original full-pipeline
reproduction. Do not claim original-method SOTA, localization improvement,
semantic category accuracy improvement, real-time performance or publication
readiness from these results alone.

Freeze the present algorithm and primary metric. Prepare the paper around this
evidence before further changes. If submission validation proceeds, prioritize
an original external pipeline and a second dataset with a genuinely unexposed
test split under a separate preregistered protocol. The already examined 057
sequence must not be reused as an untouched test for a modified algorithm.

## Artifacts

- [Method architecture (SVG)](method_architecture.svg)
- [Exact mechanism data](scene057_diagnosis.json)
- [All objects, including regressions](per_object.json)
- [Input hashes and verification](freeze_manifest.json)
- Reproduction script: `scripts/benchmarks/review_visibility_mechanism.py`
'''
    (OUT/'paper_method.md').write_text(md)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol, summary, scenes, objects = collect()
    d = diagnosis(scenes)
    charts(scenes, d); architecture(); publish(protocol, summary, scenes, objects, d)
    # Analysis may only add derived artifacts. Verify every source remained intact.
    for path, digest in INPUT_HASHES.items(): assert sha(ROOT / path) == digest, path
    freeze=dict(complete=True, created_epoch=time.time(), primary_method='visibility1',
                display_name=NAMES['visibility1'], scope='Post-hoc diagnosis only; no new map, inference, training or tuning.',
                protocol_sha256=sha(BASE/'protocol.json'), summary_sha256=sha(BASE/'summary.json'),
                frozen_source_count=len(protocol['source_sha256']), source_and_evaluation_inputs=INPUT_HASHES,
                inputs_unchanged_after_analysis=True, per_object_rows=len(objects),
                scored_arrays_reconstructed=44, primary_validation_passed=summary['primary_validation_passed'],
                script_sha256=sha(__file__), artifacts={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='freeze_manifest.json'})
    dump(OUT/'freeze_manifest.json',freeze)
    print(json.dumps(dict(report=str(OUT/'index.html'),diagnosis=d,objects=len(objects),inputs_checked=len(INPUT_HASHES)),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
