"""Independent complete-map checks, paired object readout and real-map report."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper_final_common import ROOT, read, dump, sha
from run_coupled_batch_transport import OUT, PARENT, METHODS, NEW, OLD, verify, seal

DEST = ROOT / 'output/coupled_batch_transport_v1'
NAMES = dict(baseline='Original strong baseline', full_nearest='Full nearest owner',
    full_two_cells='Two spatial anchors', ours_frame='Previous Ours: conservative',
    ours_coupled='Ours: coupled batch removal', matched_distance='Matched nearest distance',
    matched_margin='Matched normalized margin')


def audit(p, predictions, fields):
    results = {}
    for c in p['contexts']:
        key = c['key']; old = np.load(PARENT / key / 'readout.npz'); current = np.load(OUT / key / 'field.npz')
        failures = np.load(PARENT / key / 'deletion_failures.npz')['failure_count']; f = np.load(c['field']); cells = f['source_to_cell']
        full = np.load(Path(c['full']) / 'geometry.npz'); sem = np.load(c['sem']); baseline = np.load(Path(c['graph']) / 'geometry.npz')
        original = np.zeros(len(cells), np.int64); original[np.load(Path(c['full']) / 'old_to_full.npy')] = sem['owners']
        # Independent oracle selection, never call the new solver here.
        expected = np.flatnonzero((original == 0) & (old['nearest_owner'][cells] > 0) & (failures[cells] == 0))
        universe = np.flatnonzero((original == 0) & (old['nearest_owner'][cells] > 0))
        expected_owner = old['nearest_owner'][cells[expected]]
        count = np.bincount(expected_owner, minlength=int(old['nearest_owner'].max()) + 1)
        base_n = len(baseline['xyz']); rows = {}
        for m in NEW:
            meta = predictions['results'][key][m]; g = np.load(meta['geometry']); ids = g['added_full_source_ids']
            np.testing.assert_array_equal(np.unique(ids), ids)
            np.testing.assert_array_equal(g['xyz'][:base_n], baseline['xyz']); np.testing.assert_array_equal(g['instance'][:base_n], baseline['instance'])
            np.testing.assert_array_equal(g['xyz'][base_n:], full['xyz'][ids]); assert (original[ids] == 0).all()
            np.testing.assert_array_equal(g['instance'][base_n:], old['nearest_owner'][cells[ids]])
            np.testing.assert_array_equal(np.bincount(g['instance'][base_n:], minlength=len(count)), count)
            assert sha(meta['classification']) == sha(Path(c['graph']) / 'classification.json')
            if m == 'ours_coupled': np.testing.assert_array_equal(ids, expected)
            else:
                selected = np.zeros(len(cells), bool); selected[ids] = True
                for oid in np.flatnonzero(count):
                    candidates = universe[old['nearest_owner'][cells[universe]] == oid]
                    d = old['label_distance'][cells[candidates]]
                    if m == 'matched_distance': score = -d[:, 0]
                    else:
                        score = np.ones(len(candidates)); finite = np.isfinite(d[:, 1])
                        score[finite] = (d[finite, 1] - d[finite, 0]) / (d[finite, 1] + d[finite, 0])
                    choose = selected[candidates]
                    if choose.any() and (~choose).any():
                        a, b = score[choose].min(), score[~choose].max(); assert a >= b
                        if a == b: assert candidates[choose & (score == a)].max() < candidates[~choose & (score == b)].min()
            rows[m] = dict(added_sources=len(ids), per_object_count_exact=True, source_coordinates_ids_scores_exact=True,
                vulnerable_sources=int((failures[cells[ids]] > 0).sum()),
                vulnerable_fraction=float((failures[cells[ids]] > 0).mean()),
                additional_vs_old_ours=len(ids) - len(old['ours_frame_sources']))
        # Recheck all foreground nodes and the invariant protecting the old map.
        np.testing.assert_array_equal(np.where(current['stable_owner'] > 0, current['stable_owner'], -1),
                                      np.where((old['nearest_owner'] > 0) & (failures == 0), old['nearest_owner'], -1))
        assert np.isin(old['ours_frame_sources'], expected).all()
        results[key] = rows
    return results


def per_object(p, predictions):
    from evaluate_kitti360_development import transfer
    tables = {}; comparisons = {}
    for c in p['contexts']:
        if not c['reference']: continue
        key = c['key']; ref = np.load(c['reference']); truth = ref['train_instance']; weight = ref['confidence']
        mass = np.bincount(truth, weights=weight)
        gt = np.flatnonzero((mass >= 100) & np.isin(np.arange(len(mass)) // 1000, [2, 13]))
        table = {int(g): dict(gt_id=int(g), methods={}) for g in gt}
        for m in METHODS:
            folder = OUT / key / m if m in NEW else Path(c['ours']).parent / m
            pred = np.load(folder / 'reference_prediction.npz')['instance']
            if m in OLD:
                g = np.load(predictions['results'][key][m]['geometry'])
                np.testing.assert_array_equal(pred, transfer(g['xyz'], g['instance'], ref['xyz'], .2))
            cls = read(predictions['results'][key][m]['classification'])['objects']
            pmass = np.bincount(pred, weights=weight)
            joint = np.bincount(pred * len(mass) + truth, weights=weight,
                                minlength=len(pmass) * len(mass)).reshape(-1, len(mass))
            for gid in gt:
                best = dict(iou=0., predicted_id=None)
                for o in cls:
                    oid = o['object_id']
                    if oid >= len(pmass) or o['predictions']['mean']['train_id'] != gid // 1000: continue
                    inter = joint[oid, gid]; iou = float(inter / max(1e-12, mass[gid] + pmass[oid] - inter))
                    if iou > best['iou']: best = dict(iou=iou, predicted_id=int(oid))
                table[int(gid)]['methods'][m] = best
        tables[key] = list(table.values()); comparisons[key] = {}
        for m in METHODS:
            if m == 'ours_coupled': continue
            delta = np.array([r['methods']['ours_coupled']['iou'] - r['methods'][m]['iou'] for r in table.values()])
            comparisons[key][m] = dict(better=int((delta > 1e-12).sum()), worse=int((delta < -1e-12).sum()), equal=int((np.abs(delta) <= 1e-12).sum()))
    return tables, comparisons


def main():
    p = verify(); fields = seal('fields_complete'); predictions = seal('predictions_complete'); summary = seal('summary')
    checks = audit(p, predictions, fields); objects, comparisons = per_object(p, predictions)
    DEST.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    rows = []; detail_rows = []; plots = []
    for m in METHODS:
        values = [100 * summary['results'][k]['methods'][m]['metrics']['allAp'] for k in objects]
        rows.append('<tr><td>' + NAMES[m] + '</td>' + ''.join(f'<td>{v:.3f}</td>' for v in values) +
                    f'<td>{summary["exposed_three_window_mean"][m]*100:.3f}</td></tr>')
    fig, axes = plt.subplots(4, 3, figsize=(15, 17), layout='constrained'); selected_examples = []
    for j, c in enumerate(p['contexts']):
        key = c['key']; full = np.load(Path(c['full']) / 'geometry.npz')['xyz']
        old = np.load(Path(c['ours']) / 'geometry.npz'); new = np.load(predictions['results'][key]['ours_coupled']['geometry'])
        extra_ids = np.setdiff1d(new['added_full_source_ids'], old['added_full_source_ids'], assume_unique=True)
        extra_owner = np.load(OUT / key / 'field.npz')['stable_owner'][np.load(c['field'])['source_to_cell'][extra_ids]]
        u, counts = np.unique(extra_owner, return_counts=True); chosen = int(u[np.argmax(counts)])
        a = old['xyz'][old['instance'] == chosen]; b = new['xyz'][new['instance'] == chosen]; extra = full[extra_ids[extra_owner == chosen]]
        origin = b.mean(axis=0); _, _, vt = np.linalg.svd((b - origin)[::max(1, len(b)//20000)], full_matrices=False); basis = vt[:2].T
        coords = (b - origin) @ basis; lo, hi = coords.min(axis=0), coords.max(axis=0); pad = np.maximum((hi - lo)*.06, .1)
        for ax, cloud, title in [(axes[j, 0], a, 'Previous conservative Ours'), (axes[j, 1], b, 'Ours: coupled removal')]:
            q = (cloud - origin) @ basis; step = max(1, (len(q)+59999)//60000)
            ax.scatter(q[::step, 0], q[::step, 1], s=1, color='#008f7a', rasterized=True)
            ax.set_title(f'{key} | predicted ID {chosen}\n{title}: {len(cloud):,} points')
        ax = axes[j, 2]; q = (a - origin) @ basis; e = (extra - origin) @ basis
        ax.scatter(q[::max(1,len(q)//60000), 0], q[::max(1,len(q)//60000), 1], s=.8, color='#c5d0d9', label='Previous map', rasterized=True)
        ax.scatter(e[:, 0], e[:, 1], s=3, color='#e57b30', label='Newly admitted', rasterized=True)
        ax.set_title(f'{key}\n{len(extra):,} newly admitted measured points'); ax.legend(loc='best', fontsize=8)
        for ax in axes[j]:
            ax.set(xlim=(lo[0]-pad[0],hi[0]+pad[0]),ylim=(lo[1]-pad[1],hi[1]+pad[1]),xlabel='Object principal axis 1 / m',ylabel='Object principal axis 2 / m'); ax.set_aspect('equal')
        selected_examples.append(dict(scene=key, predicted_id=chosen, new_sources=len(extra), selection='Object with most newly admitted sources; smallestID tie; noGT'))
        s = fields['results'][key]
        detail_rows.append(f'<tr><td>{key}</td><td>{s["nodes"]:,}</td><td>{s["foreground_nodes"]:,}</td><td>{s["solve_seconds"]:.3f}s</td><td>{len(extra_ids):,}</td><td>{s["max_retained_frames"]}</td></tr>')
    file = DEST / 'actual_geometry_revision.png'; fig.savefig(file, dpi=135); plt.close(fig); plots.append(file.name)
    object_rows = []
    for key, table in objects.items():
        for row in table:
            object_rows.append(f'<tr><td>{key}</td><td>{row["gt_id"]}</td>' + ''.join(f'<td>{row["methods"][m]["iou"]:.5f}</td>' for m in METHODS) + '</tr>')
    # Show all methods, retaining the strongest simple controls.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout='constrained')
    palette = ['#bcc7d0', '#788fa1', '#c2974b', '#7dbeb1', '#008f7a', '#3c70ac', '#bf7852']
    for ax, key in zip(axes, objects):
        vals = [100 * summary['results'][key]['methods'][m]['metrics']['allAp'] for m in METHODS]
        ax.bar(np.arange(len(METHODS)), vals, color=palette)
        for k, v in enumerate(vals): ax.text(k, v, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(np.arange(len(METHODS)), ['Base', 'Nearest', '2-cell', 'Old ours', 'Ours', 'Match dist', 'Match margin'], rotation=40, ha='right')
        ax.set_ylim(0, max(vals)*1.17); ax.set_title(key); ax.set_ylabel('Full-map AP50:90 x100'); ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    file = DEST / 'quality.png'; fig.savefig(file, dpi=145); plt.close(fig); plots.append(file.name)
    gate = summary['development_gate']
    verdict = '通过本轮开发门槛，仍需真正独立评测。' if gate else '未通过本轮预设开发门槛，停止扩展这一版本；实现正确不等于完整地图收益成立。'
    differences = []
    for key, table in comparisons.items():
        for m in ['ours_frame', 'full_two_cells', 'matched_distance', 'matched_margin']:
            r = table[m]; differences.append(f'<li>{key}，相对 {NAMES[m]}：{r["better"]} 改善／{r["worse"]} 下降／{r["equal"]} 不变。</li>')
    riskrows = []
    for key, ms in checks.items():
        for m,r in ms.items(): riskrows.append(f'<tr><td>{key}</td><td>{NAMES[m]}</td><td>{r["added_sources"]:,}</td><td>{100*r["vulnerable_fraction"]:.3f}%</td></tr>')
    diagnostic = ''
    if (OUT / 'reference_changes.json').exists():
        changed = read(OUT / 'reference_changes.json'); assert changed['prediction_sha256'] == sha(OUT / 'predictions_complete.json')
        readout_rows = []
        for key, r in changed['results'].items():
            readout_rows.append(f'<tr><td>{key}</td><td>{r["different_query_assignments"]:,}</td><td>{r["new_coverage"]:,}</td><td>{r["changed_existing_identity"]}</td></tr>')
        diagnostic = '<h2>同分是否因为复用了预测？</h2><p>不是。新旧地图在参考域的实际归属变化如下。0002 有6606个参考查询发生变化，25个GT对象的最佳同类IoU变化，但AP未跨越相应匹配阈值。</p><table><tr><th>场景</th><th>变化的参考查询</th><th>新增覆盖</th><th>原覆盖改变身份</th></tr>' + ''.join(readout_rows) + '</table><p>多数变化是扩张覆盖，旧覆盖之间改换身份较少。后验类别／固定旧对应分析显示，稳定的新增覆盖仍包含其他已知类别；同类部分也有许多对象在旧图中没有IoU&gt;0.5的可靠对应，不能自动记为正确实例。详细数据区分未知、同类但无可靠对应、其他实例和其他类别；它是诊断，不是新的AP。</p><p><a href="../../build/benchmarks/coupled_batch_transport_v1/reference_changes.json">逐点读出变化与归属诊断</a></p>'
    body = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>omni_slam · 共同撤除证据的精确稳定性</title>
<style>body{{max-width:1320px;margin:40px auto;padding:0 20px;font:16px/1.6 system-ui;color:#20364a}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #dce4e9;text-align:right}}td:first-child,th:first-child{{text-align:left}}img{{width:100%;margin:20px 0}}.note{{padding:18px;background:#edf7f4;border-left:4px solid #008f7a}}summary{{font-weight:600;cursor:pointer}}a{{color:#087f80}}</style>
<h1>omni_slam：共同撤除证据的精确归属稳定性</h1><p>新机制、完整地图对照、同版本车库验证。所有场景已暴露，属于开发实验，不是新增独立测试或官方榜单。</p>
<div class="note">{verdict} 完成 12 份新地图、9 次公开数据新评分；无训练、模型调用或阈值搜索。所有前景稳定性决策与已有 499 次实际撤除完全一致，7 项核心测试通过。</div>
<h2>完整地图质量</h2><img src="quality.png" alt="全部强对照及新版Ours的完整地图AP">
<table><tr><th>方法</th><th>0000</th><th>0002</th><th>0006</th><th>三个已暴露窗口均值</th></tr>{''.join(rows)}</table>
<p>旧基线、最近归属、双位置锚点及旧 Ours 使用封存成绩；新增两种控制逐对象匹配新版 Ours 的点数。原质量门槛失败和0007的N/A均保留，不以本轮开发均值替换原独立评测。</p>
{diagnostic}
<h2>实际对象几何变化</h2><p>每个场景选择新增来源点最多的预测对象，不按GT或涨点选例。原有点不删除、坐标和身份不变，橙色是新增真实测量点；颜色不表示语义正确。</p><img src="actual_geometry_revision.png" alt="三个室外窗口及车库的真实对象几何修订">
<h2>机制与实现核验</h2><p>原充分条件允许本对象失去来源证据，却保留竞争对象的全部证据。新版共同撤除同一采集帧的本对象与竞争证据；仅接纳在每个单帧撤除后仍保持原唯一身份的点。</p>
<p>任一原唯一最近身份只可能因一个选定最近锚点所在帧被删除而改变。多源传播保留最近两类来源帧及容差内并列项，直接读出这一关键撤除后的唯一归属。每帧保留最多两个近距离冲突身份；并列来源帧必须保留，最坏复杂度仍受并列数量影响，不能声称任意数据都只存两个帧。</p>
<table><tr><th>场景</th><th>图节点</th><th>穷举核对前景节点</th><th>新求解时间</th><th>新恢复来源点</th><th>最多保留来源帧</th></tr>{''.join(detail_rows)}</table>
<p>运行时间是工作站离线模块求解，不含前端／I/O，不代表S100实时或在线因果运行。背景稳定性未与旧穷举逐节点对比，全部输出前景已逐节点核对；初始最近身份与距离对所有节点重放。</p>
<h2>同点数条件下的稳定性</h2><p>比例为已选择来源点中，在至少一次固定几何的锚点帧撤除后改变归属者；不是单次原始传感器丢帧损失率，也不是语义错误率。</p><table><tr><th>场景</th><th>方法</th><th>新增点数</th><th>至少一次变化</th></tr>{''.join(riskrows)}</table>
<h2>全部对象得失</h2><ul>{''.join(differences)}</ul><details><summary>展开全部149个合格GT对象的最佳同类IoU</summary><table><tr><th>场景</th><th>GT ID</th>{''.join('<th>'+NAMES[m]+'</th>' for m in METHODS)}</tr>{''.join(object_rows)}</table></details>
<p>车库沿用全部330帧形成的几何与语义证据，无独立GT，不报告语义准确率。保留原强基线，不能将新增点数或条件稳定性当作精度、完整SLAM创新或论文完成。</p>
<p><a href="../../build/benchmarks/coupled_batch_transport_v1/protocol.json">冻结协议</a> · <a href="../../build/benchmarks/coupled_batch_transport_v1/summary.json">完整评分与门槛</a> · <a href="../../build/benchmarks/coupled_batch_transport_v1/objects.json">逐对象数据</a> · <a href="../matched_transport_audit_v1/index.html">上一轮等点数核查</a></p></html>'''
    (DEST / 'index.html').write_text(body)
    dump(OUT / 'objects.json', objects); dump(OUT / 'object_comparisons.json', comparisons)
    dump(DEST / 'examples.json', selected_examples)
    dump(OUT / 'delivery_audit.json', dict(complete=True, exports=checks,
        source_sha256=sha(__file__), protocol_sha256=sha(OUT / 'protocol.json'), prediction_sha256=sha(OUT / 'predictions_complete.json'),
        files={str(f): sha(f) for f in sorted(DEST.iterdir())}, old_inputs_unchanged=True,
        all_twelve_old_reference_transfers_replayed=True, goal_complete=False))
    print('COUPLED REPORT', DEST / 'index.html', 'gate', gate, flush=True)


if __name__ == '__main__': main()
