"""Report and export audit for fixed equal-count controls, all cases retained."""
import html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper_final_common import ROOT, read, dump, sha
from audit_matched_transport import OUT, METHODS, verify, checked_seal

DEST = ROOT / 'output/matched_transport_audit_v1'
NAMES = dict(ours_frame='Ours: frame certificate', nearest_distance='Matched nearest distance', normalized_margin='Matched normalized margin')
COLORS = ['#008f7a', '#4178c0', '#bd8040']


def check_exports(p, sealed, stability):
    results = {}
    for c in p['contexts']:
        key = c['key']; state = np.load(OUT / key / 'readout.npz')
        f = np.load(c['field']); cells = f['source_to_cell']; owner = state['nearest_owner']
        full = np.load(Path(c['full']) / 'geometry.npz'); base = np.load(Path(c['graph']) / 'geometry.npz')
        sem = np.load(c['sem']); mapping = np.load(Path(c['full']) / 'old_to_full.npy')
        original_owner = np.zeros(len(cells), np.int64); original_owner[mapping] = sem['owners']
        nbase = len(base['xyz']); target = state['ours_frame_sources']; expected = np.bincount(owner[cells[target]])
        failure = np.load(OUT / key / 'deletion_failures.npz')['failure_count']
        universe = np.flatnonzero((original_owner == 0) & (owner[cells] > 0)); distances = state['label_distance']
        selected_stats = {}
        for method in METHODS:
            row = sealed['results'][key][method]; g = np.load(row['geometry']); selected = g['added_full_source_ids']
            np.testing.assert_array_equal(selected, state[method + '_sources'])
            np.testing.assert_array_equal(g['xyz'][:nbase], base['xyz']); np.testing.assert_array_equal(g['instance'][:nbase], base['instance'])
            np.testing.assert_array_equal(g['xyz'][nbase:], full['xyz'][selected])
            np.testing.assert_array_equal(g['instance'][nbase:], owner[cells[selected]])
            np.testing.assert_array_equal(np.bincount(g['instance'][nbase:], minlength=len(expected)), expected)
            np.testing.assert_array_equal(selected, np.unique(selected)); assert not original_owner[selected].any()
            assert sha(row['classification']) == sha(Path(c['graph']) / 'classification.json')
            if method != 'ours_frame':
                # Check optimal ranking through a boundary property, rather than
                # calling the selection implementation again.
                retained = np.zeros(len(cells), bool); retained[selected] = True
                for oid in np.flatnonzero(expected):
                    ids = universe[owner[cells[universe]] == oid]
                    d = distances[cells[ids]]
                    if method == 'nearest_distance': quality = -d[:, 0]
                    else:
                        quality = np.ones(len(ids)); finite = np.isfinite(d[:, 1])
                        quality[finite] = (d[finite, 1] - d[finite, 0]) / (d[finite, 1] + d[finite, 0])
                    a = retained[ids]
                    if a.any() and (~a).any():
                        low = quality[a].min(); high = quality[~a].max()
                        assert low >= high
                        if low == high: assert ids[a & (quality == low)].max() < ids[~a & (quality == high)].min()
            sums = sum(r['methods'][method]['changed'] for r in stability['results'][key]['per_frame'])
            assert sums == int(failure[cells[selected]].sum())
            actual = int((failure[cells[selected]] > 0).sum())
            assert actual == stability['results'][key]['methods'][method]['vulnerable_sources']
            selected_stats[method] = dict(points=len(g['xyz']), added_sources=len(selected),
                old_prefix_ids_scores_exact=True, original_unowned_sources_exact=True,
                per_object_count_exact=True, ranking_boundary_valid=True, failure_counts_reconciled=True)
        results[key] = selected_stats
    return results


def main():
    p = verify(); predictions = checked_seal('predictions_complete')
    stability = checked_seal('stability_complete'); scores = checked_seal('scores_complete')
    audit = check_exports(p, predictions, stability); DEST.mkdir(parents=True, exist_ok=True)
    rows = []; plots = []; envelope = {}; envelope_rows = []; difference_examples = []
    diff_fig, diff_axes = plt.subplots(4, 2, figsize=(13, 15), layout='constrained')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    for context_index, c in enumerate(p['contexts']):
        key = c['key']; score = scores['results'][key]; st = stability['results'][key]
        # Posthoc interpretation of the already completed counterfactuals.
        # This is not an additional predictor, map, score, or selection control.
        state = np.load(OUT / key / 'readout.npz'); f = np.load(c['field'])
        failures = np.load(OUT / key / 'deletion_failures.npz')['failure_count']
        cells = f['source_to_cell']; sem = np.load(c['sem'])
        original_owner = np.zeros(len(cells), np.int64)
        original_owner[np.load(Path(c['full']) / 'old_to_full.npy')] = sem['owners']
        missed = (state['nearest_owner'] > 0) & (state['certified_owner'] < 0) & (failures == 0)
        ids = np.flatnonzero((original_owner == 0) & missed[cells])
        oids, counts = np.unique(state['nearest_owner'][cells[ids]], return_counts=True)
        envelope[key] = dict(stable_but_uncertified_cells=int(missed.sum()),
            stable_but_unselected_unowned_sources=len(ids),
            by_object={str(int(o)): int(n) for o, n in zip(oids, counts)},
            no_ground_truth_in_this_derivation=True, map_quality_unmeasured=True)
        envelope_rows.append(f'<tr><td>{key}</td><td>{int(missed.sum()):,}</td><td>{len(ids):,}</td><td>{len(oids)}</td></tr>')
        for m in METHODS:
            s = st['methods'][m]; ap = score.get('methods', {}).get(m, {}).get('metrics', {}).get('allAp')
            ap_text = 'N/A' if ap is None else f'{100 * ap:.3f}'
            rows.append(f'<tr><td>{key}</td><td>{NAMES[m]}</td><td>{s["added_sources"]:,}</td>'
                        f'<td>{ap_text}</td><td>{s["vulnerable_sources"]:,} ({100*s["vulnerable_fraction"]:.3f}%)</td>'
                        f'<td>{100*s["mean_failure_fraction"]:.4f}%</td><td>{100*s["macro_object_vulnerable_fraction"]:.3f}%</td></tr>')
        # Fixed whole-scene views, same axis limits and deterministic subsampling.
        full = np.load(Path(c['full']) / 'geometry.npz')['xyz']
        # Explain why whole-scene images look similar. Select the object with
        # most changed source memberships, without GT or quality selection.
        source_sets = {m: state[m + '_sources'] for m in METHODS}
        winning = state['nearest_owner'][cells]
        candidate_objects = np.unique(winning[source_sets['ours_frame']])
        differences = []
        for oid in candidate_objects:
            original = source_sets['ours_frame'][winning[source_sets['ours_frame']] == oid]
            total = sum(len(np.setxor1d(original, source_sets[m][winning[source_sets[m]] == oid], assume_unique=True)) for m in METHODS[1:])
            differences.append((total, int(oid)))
        _, example_id = max(differences, key=lambda x: (x[0], -x[1]))
        parts = {m: s[winning[s] == example_id] for m, s in source_sets.items()}
        union = np.unique(np.concatenate(list(parts.values())))
        points = full[union]; origin = points.mean(axis=0)
        _, _, vt = np.linalg.svd((points - origin)[::max(1, len(points) // 20000)], full_matrices=False)
        basis = vt[:2].T; coordinates = (points - origin) @ basis
        lower, upper = coordinates.min(axis=0), coordinates.max(axis=0)
        padding = np.maximum((upper - lower) * .06, .1)
        example = dict(scene=key, object_id=example_id, selection='Max summed source symmetric difference, tie smallest object ID; no GT.', comparisons={})
        for j, m in enumerate(METHODS[1:]):
            a, b = parts['ours_frame'], parts[m]
            common = np.intersect1d(a, b, assume_unique=True)
            only_a = np.setdiff1d(a, b, assume_unique=True); only_b = np.setdiff1d(b, a, assume_unique=True)
            ax = diff_axes[context_index, j]
            for ids, color, label, size in [(common, '#c7d1da', 'Shared', 1), (only_a, COLORS[0], 'Ours only', 3), (only_b, '#d87936', 'Control only', 3)]:
                step = max(1, (len(ids) + 39999) // 40000)
                q = (full[ids[::step]] - origin) @ basis
                ax.scatter(q[:, 0], q[:, 1], s=size, color=color, label=label, rasterized=True)
            ax.set(xlim=(lower[0] - padding[0], upper[0] + padding[0]), ylim=(lower[1] - padding[1], upper[1] + padding[1]),
                   xlabel='Object principal axis 1 / m', ylabel='Object principal axis 2 / m')
            ax.set_aspect('equal'); ax.set_title(f'{key} | predicted ID {example_id} | {NAMES[m]}\n{len(a):,} sources each; {len(only_a):,} Ours-only / {len(only_b):,} control-only')
            ax.legend(loc='best', markerscale=3, fontsize=8)
            example['comparisons'][m] = dict(sources_each=len(a), shared=len(common), ours_only=len(only_a), control_only=len(only_b))
        difference_examples.append(example)
        lo, hi = np.quantile(full[:, :2], [.005, .995], axis=0)
        background = full[::max(1, (len(full) + 99999) // 100000)]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
        for ax, m in zip(axes, METHODS):
            g = np.load(predictions['results'][key][m]['geometry']); xyz = g['xyz']; ids = g['instance']
            stride = max(1, (len(xyz) + 99999) // 100000)
            ax.scatter(background[:, 0], background[:, 1], s=.15, color='#d9dee3', alpha=.4, rasterized=True)
            ax.scatter(xyz[::stride, 0], xyz[::stride, 1], s=.7, c=ids[::stride] % 20, cmap='tab20', vmin=0, vmax=19, rasterized=True)
            ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), xlabel='x / m', ylabel='y / m')
            ax.set_aspect('equal'); ax.set_title(f'{NAMES[m]}\n{len(xyz):,} total points')
        fig.suptitle(key + ' | actual measured geometry; identical point counts per object')
        path = DEST / (key + '.png'); fig.savefig(path, dpi=130); plt.close(fig); plots.append(path.name)
    diff_fig.savefig(DEST / 'actual_source_differences.png', dpi=140); plt.close(diff_fig)
    dump(DEST / 'difference_examples.json', difference_examples)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), layout='constrained')
    for k, c in enumerate(p['contexts']):
        key = c['key']; st = stability['results'][key]
        risks = [100 * st['methods'][m]['vulnerable_fraction'] for m in METHODS]
        axes[0, k].bar(np.arange(3), risks, color=COLORS)
        axes[0, k].set_ylim(0, max(risks) * 1.2 if max(risks) else 1)
        for i, v in enumerate(risks): axes[0, k].text(i, v, f'{v:.2f}%', ha='center', va='bottom', fontsize=9)
        axes[0, k].set_title(key); axes[0, k].set_ylabel('Sources vulnerable to >=1 deletion / %')
        if 'methods' in scores['results'][key]:
            vals = [100 * scores['results'][key]['methods'][m]['metrics']['allAp'] for m in METHODS]
            axes[1, k].bar(np.arange(3), vals, color=COLORS)
            axes[1, k].set_ylim(0, max(vals) * 1.2)
            for i, v in enumerate(vals): axes[1, k].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        else:
            axes[1, k].text(.5, .5, 'No independent robot GT\nAP: N/A', ha='center', va='center', transform=axes[1, k].transAxes)
        axes[1, k].set_ylabel('Clean full-map AP50:90 x100')
        for ax in axes[:, k]: ax.set_xticks(np.arange(3), ['Ours', 'Distance', 'Margin']); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    file = DEST / 'quality_and_stability.png'; fig.savefig(file, dpi=145); plt.close(fig); plots.insert(0, file.name)
    objects = read(OUT / 'objects.json'); detail_rows = []; comparisons = []
    for key, items in objects.items():
        for row in items:
            detail_rows.append(f'<tr><td>{key}</td><td>{row["gt_id"]}</td>' + ''.join(f'<td>{row["methods"][m]["iou"]:.5f}</td>' for m in METHODS) + '</tr>')
        for m, d in scores['results'][key]['per_object_comparisons'].items():
            comparisons.append(f'<li>{key}，Ours 相对 {NAMES[m]}：IoU 改善 {d["ours_better"]}、下降 {d["ours_worse"]}、不变 {d["equal"]}；'
                               f'仅 Ours 有 AP50 匹配 {d["ours_only_ap50"]}，仅对照有匹配 {d["control_only_ap50"]}。</li>')
    frame_count = sum(r['frames'] for r in stability['results'].values())
    body = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>omni_slam · 等覆盖量机制核查</title>
<style>body{{max-width:1250px;margin:40px auto;padding:0 20px;color:#20364a;font:16px/1.6 system-ui}}h1,h2{{line-height:1.3}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #dbe3e9;text-align:right}}td:first-child,th:first-child,td:nth-child(2){{text-align:left}}img{{width:100%;margin:18px 0}}.note{{padding:18px;background:#edf7f4;border-left:4px solid #008f7a}}summary{{cursor:pointer;font-weight:600}}a{{color:#087f80}}</style>
<h1>omni_slam：等点数条件下，来源帧约束是否有独立价值？</h1>
<p>主线：对象级语义建图。这里固定原对象身份、语义前端、实测几何和分类分数，比较几何补全的选择规则。</p>
<div class="note">四个已暴露场景，两个固定控制，共八份新对照地图。每个对象的新增点数与冻结 Ours 完全相同。完成 {frame_count} 次真实锚点批次撤除、三段公开数据的九次地图评分（含三次 Ours 精确重放）。没有新训练或模型推理。</div>
<h2>质量与归属稳定性一起看</h2>
<p>来源帧证书按设计保护固定图上的归属。稳定性收益不能替代语义质量收益。AP 为 KITTI-360 建筑／车辆本地适配评分；原开发与公开质量门槛仍未通过，不以本次事后诊断改写。</p>
<img src="quality_and_stability.png" alt="完整地图质量与来源帧撤除后的归属变化">
<table><tr><th>场景</th><th>方法</th><th>新增点数</th><th>AP50:90 ×100</th><th>至少一次改变的来源点</th><th>每次撤除平均变化率</th><th>逐对象宏平均脆弱率</th></tr>{''.join(rows)}</table>
<h2>定位到的保守性缺口</h2><p>对已完成的全部撤除结果作事后读取，发现有些点未获原证书，但在每一次真实来源帧锚点撤除后归属都不变。原充分条件将竞争对象证据视为始终保留，因此会拒绝一部分实际稳定的点。下表不使用 GT，也没有据此生成或评分新地图；稳定不代表语义正确。</p>
<table><tr><th>场景</th><th>稳定但未获证书的单元</th><th>其中未归属的真实来源点</th><th>涉及预测对象</th></tr>{''.join(envelope_rows)}</table>
<p>可检验的下一机制是显式处理同一批次内本对象与竞争对象证据的共同撤除，收紧当前保守判据；不是降低稳健性门槛。需用本次穷举结果校验高效实现，并另外冻结一次完整地图对照，检验恢复覆盖是否真的提高精度。该机制尚未实现或通过验收。</p>
<h2>实际地图</h2><p>颜色表示固定对象 ID；灰色为共享实测背景。三组使用相同视野与点数，按源点顺序等间隔显示。点数相同不代表空间覆盖相同，也不代表所有对象都正确。</p>
{''.join(f'<img src="{name}" alt="{name}">' for name in plots[1:])}
<h2>为什么整图看着接近：放大真实差异</h2><p>每个场景选取与两种对照合计来源点差异最多的预测对象，平局取较小 ID，不按 GT 或涨点选例。相同主轴投影下，灰色为共有点，绿色为仅 Ours 保留，橙色为仅对照保留。配对点数完全相同；这些颜色表示选择差异，不表示语义正确或错误。</p>
<img src="actual_source_differences.png" alt="四个场景的实际来源点选择差异">
<h2>全部对象的得失</h2><ul>{''.join(comparisons)}</ul>
<details><summary>展开所有合格 GT 对象的最佳同类 IoU</summary><table><tr><th>场景</th><th>GT ID</th><th>Ours</th><th>Distance</th><th>Margin</th></tr>{''.join(detail_rows)}</table></details>
<h2>对照与边界</h2><p>控制一按最近归属锚点的距离从小到大选点。控制二按 (竞争距离 − 归属距离) / (竞争距离 + 归属距离) 从大到小选点；无可达竞争者记 1。逐对象截取冻结 Ours 的点数，分数相同按来源 ID 选取，不读取 GT 选择阈值。</p>
<p>撤除的是单个采集帧提供的全部前景／背景锚点，几何仍固定。观察已选点的最近归属是否变成其他对象、背景或未知；不代表重新认证仍接纳，不是原始传感器丢帧或定位鲁棒性测试。所有场景已经用于分析，不是独立测试；车库没有独立 GT，AP 为 N/A。0007 的原始 N/A 及验收失败保持不变。</p>
<p><a href="../../build/benchmarks/matched_transport_audit_v1/protocol.json">冻结协议</a> · <a href="../../build/benchmarks/matched_transport_audit_v1/stability_complete.json">完整逐帧撤除记录</a> · <a href="../../build/benchmarks/matched_transport_audit_v1/scores_complete.json">完整评分</a> · <a href="../../build/benchmarks/matched_transport_audit_v1/objects.json">逐对象数据</a></p></html>'''
    (DEST / 'index.html').write_text(body)
    dump(OUT / 'posthoc_mechanism.json', dict(complete=True, results=envelope,
        timing='Derived after clean scores and exhaustive deletion outcomes; no added prediction or score. Hypothesis generation only.',
        source_sha256=sha(__file__), stability_sha256=sha(OUT / 'stability_complete.json'), original_gates_unchanged=True))
    dump(OUT / 'delivery_audit.json', dict(complete=True, exports=audit, actual_frame_deletions=frame_count,
        protocol_sha256=sha(OUT / 'protocol.json'), prediction_sha256=sha(OUT / 'predictions_complete.json'),
        source_sha256=sha(__file__), files={str(path): sha(path) for path in sorted(DEST.iterdir())},
        old_inputs_and_results_unchanged=True, original_gates_unchanged=True, goal_complete=False))
    print('MATCHED AUDIT REPORT', DEST / 'index.html', 'deletions', frame_count, flush=True)


if __name__ == '__main__': main()
