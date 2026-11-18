"""Consolidate the bounded read-only audit and render actual GT-query changes."""
from pathlib import Path
import html
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from audit_anchor_retraction_failures import OUT, verify, reference

PARENT = ROOT / 'build/benchmarks/anchor_counterevidence_v1'
DEST = ROOT / 'output/anchor_retraction_failure_audit_v1'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = verify(); s = read(OUT / 'summary.json')
    compression = read(OUT / 'compression/summary.json')
    semantic = read(OUT / 'semantic_independence.json')
    ordinal = read(OUT / 'object_score_offsets.json')
    subtype = read(OUT / 'vehicle_subtype_diagnostic.json')
    for seal in [read(OUT / 'compression/scores_complete.json'), ordinal]:
        for path, expected in seal['files'].items(): assert sha(path) == expected
    assert not read(PARENT / 'summary.json')['development_gate']
    assert all(s['complete'] for s in [s, compression, semantic, ordinal])
    DEST.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, key in zip(axes, ['kitti360_0000', 'kitti360_0002']):
        original = s['results'][key]['anchor_diagnostic']['groups']['car']['auroc']['mean_margin']
        local = compression['results'][key]['car']['auroc']['local_support_margin']
        rank = ordinal['results'][key]['car']['within_object_rank_auc']
        vision = semantic['results'][key]['common_source_controls']['car']['identity_free_semantic_auc_same_subset']
        assert semantic['results'][key]['common_source_controls']['car']['known_anchors'] == semantic['results'][key]['common_source_controls']['car']['semantic_available']
        values = [original, local, rank, vision]
        ax.bar(np.arange(4), values, color=['#8b9bab', '#bf7d85', '#548bb4', '#168f88'])
        ax.axhline(.5, color='#777', linestyle='--', linewidth=1)
        ax.set_ylim(0, 1.06); ax.set_xticks(np.arange(4), ['Original\nmean margin', 'Local-unit\nsupport', 'Within-object\nrank', 'Existing visual\nsemantic evidence'])
        ax.set_title(key + ' / car anchors'); ax.set_ylabel('Wrong-class detection AUROC')
        for i, value in enumerate(values): ax.text(i, value + .02, f'{value:.3f}', ha='center')
    fig.suptitle('Same actual anchor queries: no alternative decisions or map AP scores')
    fig.savefig(DEST / 'car_evidence.png', dpi=180); plt.close(fig)
    all_objects = read(PARENT / 'per_object.json')['objects']
    cases = []
    for key in ['kitti360_0000', 'kitti360_0002']:
        cars = [r for r in all_objects[key] if r['gt_id'] // 1000 == 13 and r['methods']['baseline']['predicted_id'] is not None]
        ordered = sorted(cars, key=lambda r: r['methods']['ours_retract_propagation']['iou'] - r['methods']['baseline']['iou'])
        cases.extend([(key, ordered[-1])] if key.endswith('0000') else [(key, r) for r in ordered[:2]])
    fig, axes = plt.subplots(len(cases), 2, figsize=(11, 10), constrained_layout=True)
    case_meta = []
    for row, (key, obj) in enumerate(cases):
        c = next(c for c in p['contexts'] if c['key'] == key)
        ref, train = reference(c, p); gid = obj['gt_id']; oid = obj['methods']['baseline']['predicted_id']
        old = np.load(Path(c['ours']) / 'reference_prediction.npz')['instance']
        new = np.load(PARENT / 'maps' / key / 'ours_retract_propagation/reference_prediction.npz')['instance']
        gt = ref['train_instance'] == gid
        # Same fixed baseline object in both panels, no post-change ID switching.
        region = gt | (old == oid)
        center = ref['xyz'][gt].mean(0)
        xy = ref['xyz'][gt, :2] - center[:2]
        _, vectors = np.linalg.eigh(xy.T @ xy)
        horizontal = (ref['xyz'][region, :2] - center[:2]) @ vectors[:, -1]
        vertical = ref['xyz'][region, 2] - center[2]
        ids = np.flatnonzero(region)
        if len(ids) > 40000:
            sample = np.linspace(0, len(ids) - 1, 40000, dtype=int)
        else: sample = np.arange(len(ids))
        for col, (name, pred) in enumerate([('Frozen baseline', old), ('Ours (closed candidate)', new)]):
            good = (pred[ids] == oid) & gt[ids]
            bad = (pred[ids] == oid) & ~gt[ids] & (train[ids] != 255)
            unknown = (pred[ids] == oid) & (train[ids] == 255)
            colors = np.full(len(ids), '#c3cad0', dtype='<U7')
            colors[good] = '#119b86'; colors[bad] = '#e36b4f'; colors[unknown] = '#9467bd'
            axes[row, col].scatter(horizontal[sample], vertical[sample], c=colors[sample], s=2, rasterized=True)
            axes[row, col].set_aspect('equal', adjustable='datalim')
            axes[row, col].set_xlabel('Horizontal GT-axis projection (m)'); axes[row, col].set_ylabel('Relative height (m)')
            method = 'baseline' if col == 0 else 'ours_retract_propagation'
            axes[row, col].set_title(f'{key} / GT {gid} / fixed predicted ID {oid}\n{name}; best same-class IoU {obj["methods"][method]["iou"]:.3f}', fontsize=10)
        case_meta.append(dict(key=key, gt_id=gid, fixed_predicted_id=oid, reference_queries=len(ids),
                              display_queries=len(sample), view='GT principal XY axis vs height; visualization only'))
    fig.suptitle('Assigned: green target GT / orange other known GT / purple ignored; gray unassigned\nLargest car gain in0000 and two largest car regressions in0002; fixed old object IDs', fontsize=11)
    fig.savefig(DEST / 'actual_vehicle_changes.png', dpi=170); plt.close(fig)
    rows = []
    for key in ['kitti360_0000', 'kitti360_0002']:
        for name in ['building', 'car']:
            a = s['results'][key]['anchor_diagnostic']['groups'][name]
            sem = semantic['results'][key]['common_source_controls'][name]
            local = compression['results'][key][name]['auroc']['local_support_margin']
            rank = ordinal['results'][key][name]['within_object_rank_auc']
            rows.append(f'<tr><td>{key}</td><td>{name}</td><td>{a["auroc"]["mean_margin"]:.3f}</td><td>{local:.3f}</td><td>{rank:.3f}</td><td>{sem["identity_free_semantic_auc_same_subset"]:.3f}</td><td>{100*a["correct_class_rejection_rate"]:.2f}%</td></tr>')
    page = f'''<!doctype html><meta charset="utf-8"><title>omni_slam 撤回机制复核</title>
<style>body{{font:16px system-ui;max-width:1250px;margin:36px auto;padding:0 24px;color:#193246;line-height:1.6}}img{{width:100%}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:8px;border-bottom:1px solid #dce3e8;text-align:left}}</style>
<h1>omni_slam：锚点撤回为什么在0002退步</h1>
<p><b>主要损失已出现在锚点直接撤回阶段。对象内一致性不能直接作为跨对象的语义正确率。</b>本轮复用缓存，0新模型前向、0训练、0新地图、0新AP。失败门槛及旧地图未改。</p>
<p>0002中，对基线具有可靠GT对应的对象，直接撤回损失7458.38置信度质量，额外传播损失562.06；两者分别是已确认正确实例归属的参考查询损失，不能把它们当作传感器点数。</p>
<img src="car_evidence.png"><table><tr><th>窗口</th><th>类别</th><th>原型margin</th><th>局部支持诊断</th><th>对象内排序诊断</th><th>已有视觉语义</th><th>正确类别锚点误撤回</th></tr>{''.join(rows)}</table>
<p>表中前四种分数为错误类别检测AUROC，不是地图AP。局部支持只替换前景压缩，背景原型固定，仅定位信息损失，不是公平的新完整分类器。视觉语义在相同可用来源上比较：车辆全部可用；建筑分别缺8/3个来源，精确匹配对照见JSON。任何分数都没有被用于产生新撤回决定。</p>
<h2>实际车辆几何变化</h2><p>采用此前全部GT表的固定最大增益/退步案例，不筛选新的模型结果。绿色为分配给该对象且属于选定GT的点，橙色为分配给该对象但属于其他已知GT的点，紫色为分配到忽略标注的点；灰色为未分配，既可能是漏掉的目标表面，也可能是成功撤回的其他表面。橙色的语义归属差异不等同于官方AP中的全部FP，后者还受忽略类别规则影响。GT仅用于颜色与显示视角，不进入预测。所有94个GT对象仍保留在父报告。</p><img src="actual_vehicle_changes.png">
<h2>类别错不等于对象几何坏</h2><p>0002预测对象1被标为car，但5976个有参考标签的锚点中5965个属于truck；预测对象34的279个有参考标签锚点中277个属于motorcycle。忽略类别后的最佳GT对应IoU分别为0.818和0.607。这是两个有效车辆对象的细分类别问题，不能把这些点统称为几何噪声。这些诊断IoU不是官方新增跑分；改正类别能否提高AP仍未验证。</p>
<h2>研究决定</h2><p>不继续全局余弦投票、原型变体或对象内归一化单独替换。下一机制需要区分对象类别修订与局部几何修订，并将仅改类别、仅改几何、相同证据的简单融合列为对照。类别适配本身不是创新。尚未训练、生成新地图或证明该假设；不能将诊断AUROC写成论文涨点。</p>
<p><a href="../../build/benchmarks/anchor_retraction_failure_audit_v1/summary.json">全部来源/对象归因</a> · <a href="../../build/benchmarks/anchor_retraction_failure_audit_v1/semantic_independence.json">同来源语义与逐对象诊断</a> · <a href="../anchor_counterevidence_v1/index.html">完整地图评测</a></p>'''
    (DEST / 'index.html').write_text(page)
    paths = [OUT / 'summary.json', OUT / 'compression/summary.json', OUT / 'semantic_independence.json', OUT / 'object_score_offsets.json', OUT / 'earlier_probe_by_class.json', OUT / 'vehicle_subtype_diagnostic.json']
    dump(OUT / 'delivery_audit.json', dict(complete=True, parent_inputs_and_failed_gate_preserved=True,
        parent_development_gate=False, source_sha256=sha(__file__), cases=case_meta,
        diagnostic_hashes={str(f): sha(f) for f in paths}, report_hashes={str(f): sha(f) for f in DEST.iterdir()},
        note='Legacy summary field parent_gate_unchanged stores the retained gate value False, not a mutation indicator; this audit explicitly verifies unchanged inputs and failed gate.',
        new_encoder_calls=0, new_training=0, new_maps=0, new_ap_scores=0))
    print('READ-ONLY FAILURE AUDIT REPORT COMPLETE', flush=True)


if __name__ == '__main__': main()
