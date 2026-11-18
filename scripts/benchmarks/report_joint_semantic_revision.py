"""Verify map/type differences, real observations and every target GT outcome."""
from itertools import combinations
from pathlib import Path
import html
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from run_joint_semantic_revision import OUT, PARENT, METHODS, verify, seal

DEST = ROOT / 'output/joint_semantic_revision_v1'
NAMES = dict(baseline='Frozen baseline', geometry_only='Geometry only', class_only='Class only',
             simple_fusion='Simple fusion', ours_joint='Ours (candidate): joint revision')


def audit(p, predictions, summary):
    from evaluate_kitti360_development import transfer
    stats, all_objects, comparisons = {}, {}, {}
    for c in p['contexts']:
        key = c['key']; maps = predictions['results'][key]['methods']
        old = np.load(Path(c['ours']) / 'geometry.npz'); xyz, labels = old['xyz'], old['instance']
        old_cls = {o['object_id']: o['predictions']['mean'] for o in read(Path(c['ours']) / 'classification.json')['objects']}
        nids = int(labels.max()) + 1
        geometries, classes, semantic = {}, {}, {}
        for m in METHODS:
            g = np.load(maps[m]['geometry']); cls = read(maps[m]['classification'])['objects']
            np.testing.assert_array_equal(g['xyz'], xyz)
            current = g['instance']; assert np.all((current == labels) | (current == 0))
            lookup = {o['object_id']: o['predictions']['mean'] for o in cls}
            assert set(lookup) == set(old_cls)
            for oid, row in lookup.items():
                assert row['score'] == old_cls[oid]['score']
                if row['train_id'] in [2, 13]:
                    assert row['semantic_id'] == {2: 11, 13: 26}[row['train_id']]
                if row['train_id'] == old_cls[oid]['train_id']:
                    assert row == old_cls[oid], 'Unchanged categories must preserve evaluator metadata'
            geometries[m], classes[m] = current, lookup
            lut = np.full(nids, -1, int)
            for oid, row in lookup.items(): lut[oid] = row['train_id']
            semantic[m] = lut[current]
        np.testing.assert_array_equal(geometries['baseline'], geometries['class_only'])
        np.testing.assert_array_equal(geometries['geometry_only'], geometries['simple_fusion'])
        assert classes['class_only'] == classes['simple_fusion']
        obs = np.load(OUT / 'evidence' / key / 'observations.npz')
        before = np.load(OUT / 'evidence' / key / 'before.npz')
        check = np.unique(np.linspace(0, len(before['cell_owner']) - 1, min(24, len(before['cell_owner'])), dtype=int))
        for i in check:
            cell, owner = before['cell_owner'][i]
            take = (obs['cell_owner'][:, 1] == owner) & (obs['cell_owner'][:, 0] != cell)
            q = obs['probabilities']; q = q / q.sum(1)[:, None]
            parts = [q[take & (obs['observation_frame'] == f)].mean(0) for f in np.unique(obs['observation_frame'][take])]
            expected = np.mean(parts, axis=0) if parts else np.zeros(19)
            np.testing.assert_allclose(before['leave_cell_out_probability'][i], expected, atol=1e-11)
        decisions = np.load(OUT / 'evidence' / key / 'decision.npz')
        rejected = {tuple(x) for x in decisions['rejected_units']}
        parent = np.load(PARENT / 'decisions' / key / 'anchors.npz')
        proposals = {tuple(x) for x in parent['units'][parent['retract']]}
        conflict = {tuple(x) for x in before['cell_owner'][before['semantic_conflict']]}
        assert rejected == {x for x in proposals if x[:2] in conflict}
        remaining = {tuple(x[:2]) for x in decisions['retained_seeds']}
        assert {tuple(x) for x in decisions['revoked_cell_owner']} == {x[:2] for x in rejected if x[:2] not in remaining}
        pairwise = {}
        for a, b in combinations(METHODS, 2):
            pairwise[a + '__' + b] = dict(changed_geometry_rows=int((geometries[a] != geometries[b]).sum()),
                changed_semantic_rows=int((semantic[a] != semantic[b]).sum()),
                changed_object_classes=sum(classes[a][o]['train_id'] != classes[b][o]['train_id'] for o in old_cls))
        stats[key] = dict(coordinates_identity_scores_verified=True, real_leave_cell_out_checks=len(check),
                          exact_joint_rejection_and_full_cell_revocation_verified=True, pairwise=pairwise,
                          per_predicted_object=[dict(object_id=int(o), classes={m: classes[m][o]['train_id'] for m in METHODS},
                              assigned_points={m: int((geometries[m] == o).sum()) for m in METHODS}) for o in sorted(old_cls)])
        if not c['reference']: continue
        ref = np.load(c['reference']); truth, weight = ref['train_instance'], ref['confidence']
        nearest = transfer(xyz, np.arange(1, len(xyz) + 1), ref['xyz'], .2)
        mass = np.bincount(truth, weights=weight)
        gids = np.flatnonzero((mass >= 100) & np.isin(np.arange(len(mass)) // 1000, [2, 13]))
        table = {int(g): dict(gt_id=int(g), methods={}) for g in gids}
        reference_predictions = {}
        for m in METHODS:
            expected = np.r_[0, geometries[m]][nearest]
            path = Path(c['ours']) / 'reference_prediction.npz' if m == 'baseline' else Path(maps[m]['geometry']).parent / 'reference_prediction.npz'
            pred = np.load(path)['instance']; np.testing.assert_array_equal(pred, expected)
            reference_predictions[m] = pred
            pmass = np.bincount(pred, weights=weight, minlength=nids)
            joint = np.bincount(pred * len(mass) + truth, weights=weight, minlength=nids * len(mass)).reshape(nids, len(mass))
            for gid in gids:
                best = dict(iou=0., predicted_id=None)
                for oid, cls in classes[m].items():
                    if cls['train_id'] != gid // 1000: continue
                    inter = joint[oid, gid]; iou = float(inter / max(1e-12, pmass[oid] + mass[gid] - inter))
                    if iou > best['iou']: best = dict(iou=iou, predicted_id=int(oid))
                table[int(gid)]['methods'][m] = best
        for a, b in combinations(METHODS, 2):
            diff = np.array([r['methods'][a]['iou'] - r['methods'][b]['iou'] for r in table.values()])
            pairwise[a + '__' + b].update(changed_reference_ids=int((reference_predictions[a] != reference_predictions[b]).sum()),
                changed_best_gt_iou=int((np.abs(diff) > 1e-12).sum()),
                exactly_equal_ap=summary['results'][key]['methods'][a]['metrics']['allAp'] == summary['results'][key]['methods'][b]['metrics']['allAp'])
        all_objects[key] = list(table.values()); comparisons[key] = {}
        ours = summary['results'][key]['methods']['ours_joint']
        for m in METHODS[:-1]:
            diff = np.array([r['methods']['ours_joint']['iou'] - r['methods'][m]['iou'] for r in table.values()])
            baseline = summary['results'][key]['methods'][m]
            comparisons[key][m] = dict(better=int((diff > 1e-12).sum()), worse=int((diff < -1e-12).sum()), equal=int((np.abs(diff) <= 1e-12).sum()),
                new_matched_gt=sorted(set(ours['matched_gt_ids']) - set(baseline['matched_gt_ids'])),
                lost_matched_gt=sorted(set(baseline['matched_gt_ids']) - set(ours['matched_gt_ids'])))
        stats[key]['all_five_reference_transfers_replayed'] = True
    dump(OUT / 'per_object.json', dict(objects=all_objects, comparisons=comparisons,
        scope='Best same-class IoU for all eligible GT, distinct from one-to-one AP matching.'))
    return stats, all_objects, comparisons


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    p = verify(); seal('prepared'); predictions = seal('predictions_complete'); summary = seal('summary')
    stats, objects, comparisons = audit(p, predictions, summary)
    DEST.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 4, figsize=(17, 10), constrained_layout=True)
    for row, c in enumerate(p['contexts']):
        key = c['key']; frames = read(c['job'])['frames']; rgb = frames[len(frames) // 2]['rgb']
        with Image.open(rgb) as im: axes[row, 0].imshow(im)
        axes[row, 0].set_title(key + '\nActual midpoint camera image'); axes[row, 0].axis('off')
        base = np.load(Path(c['ours']) / 'geometry.npz'); xyz, old = base['xyz'], base['instance']
        sample = np.unique(np.linspace(0, len(xyz)-1, min(40000, len(xyz)), dtype=int))
        for col, m in enumerate(['baseline', 'simple_fusion', 'ours_joint'], start=1):
            r = predictions['results'][key]['methods'][m]; g = np.load(r['geometry'])
            cls_change = np.array([d['object_id'] for d in r['class_changes']], int)
            removed = g['instance'] == 0; changed = np.isin(old, cls_change)
            colors = np.where(removed, '#e65465', np.where(changed, '#8666d3', '#069a8d'))
            ax = axes[row, col]; ax.scatter(xyz[sample, 0], xyz[sample, 1], c=colors[sample], s=.5, rasterized=True)
            ax.set_aspect('equal', adjustable='box'); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
            ax.set_title(f'{NAMES[m]}\n{r["removed"]:,} unassigned; {len(cls_change)} type changes', fontsize=10)
    fig.suptitle('Actual map rows: teal unchanged; purple type changed; red ownership withdrawn. Changes are not correctness.', fontsize=12)
    fig.savefig(DEST / 'actual_maps.png', dpi=160); plt.close(fig)
    rows = []
    for c in p['contexts']:
        key = c['key']
        for m in METHODS:
            r = predictions['results'][key]['methods'][m]
            met = summary['results'][key].get('methods', {}).get(m, {}).get('metrics', {})
            rows.append(f'<tr><td>{key}</td><td>{NAMES[m]}</td><td>{100*met["allAp"]:.6f}</td><td>{100*met["allAp50%"]:.6f}</td><td>{r["removed"]}</td><td>{len(r["class_changes"])}</td></tr>' if met else
                        f'<tr><td>{key}</td><td>{NAMES[m]}</td><td>N/A</td><td>N/A</td><td>{r["removed"]}</td><td>{len(r["class_changes"])}</td></tr>')
    object_rows = []
    for key, table in objects.items():
        for r in table:
            object_rows.append('<tr><td>' + key + '</td><td>' + str(r['gt_id']) + '</td>' + ''.join(f'<td>{r["methods"][m]["iou"]:.6f}</td>' for m in METHODS) + '</tr>')
    body = f'''<!doctype html><meta charset="utf-8"><title>omni_slam — joint semantic revision</title>
<style>body{{font:16px/1.6 system-ui;color:#183045;max-width:1440px;margin:40px auto;padding:0 25px}}table{{border-collapse:collapse;font-size:14px;width:100%}}td,th{{padding:7px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#eef6f5}}img{{width:100%}}pre{{white-space:pre-wrap}}strong{{color:#08796d}}</style>
<h1>omni_slam：对象类别验证与局部几何修订</h1>
<p>主线是对象级语义建图；LIVO支撑定位与实测几何。本轮为固定身份的回顾性修订，不能代表对象发现或完整SLAM创新。两个公共窗口已经暴露；本地适配KITTI-360评分，不是官方榜单。</p>
<p><strong>预设开发门槛：{summary['development_gate']}</strong>。9份有效新地图、6次有效新公共评分，0训练/编码器前向。首次导出曾将train ID2的building映射成garage原始ID，导致建筑被评测器忽略；9份错误导出与6次错误评分已存档作废。修正后重导出相同方法，未改判据、窗口或排序分数。保留所有失败、退步及无变化结果；没有机器人独立GT。</p>
<p>每个对象先按实际观测帧均衡类别证据。查询局部单元时从对象证据中移除该单元；只有局部类别冲突同时支持旧几何反证时才撤回锚点，再更新传播与类别。兼容分数是模型假设，不是独立证据保证。仅改类别及简单组合是必要控制，不能写成创新。</p>
<h2>完整对照</h2><table><tr><th>场景</th><th>方法</th><th>AP50:90 %</th><th>AP50 %</th><th>撤回归属点</th><th>类别变化对象</th></tr>{''.join(rows)}</table>
<h2>真实图像与地图</h2><p>统一坐标、固定等间距采样。紫色为类别改变，红色为撤回归属；颜色表示修改，不表示正确。没有删除测量坐标。</p><img src="actual_maps.png">
<h2>Ours 相对各控制的逐对象得失</h2><pre>{html.escape(str(comparisons))}</pre>
<h2>为何同分：逐点与类别独立核查</h2><pre>{html.escape(str({k: v['pairwise'] for k, v in stats.items()}))}</pre>
<h2>全部目标GT的最佳同类IoU（不等于AP匹配）</h2><table><tr><th>场景</th><th>GT ID</th>{''.join('<th>'+NAMES[m]+'</th>' for m in METHODS)}</tr>{''.join(object_rows)}</table>
<p>输入及代码哈希冻结；全部坐标/身份/排序分数核验；72个真实单元留一重算；十份公共参考投影重放。分类只改19类标签，旧排序分数保持，未重新校准。</p>'''
    (DEST / 'index.html').write_text(body)
    dump(OUT / 'delivery_audit.json', dict(complete=True, stats=stats, development_gate=summary['development_gate'],
        parent_failed_gate_preserved=read(PARENT / 'summary.json')['development_gate'] is False,
        artifacts={str(f): sha(f) for f in [DEST / 'index.html', DEST / 'actual_maps.png', OUT / 'per_object.json']},
        protocol_sha256=sha(OUT / 'protocol.json'), predictions_sha256=sha(OUT / 'predictions_complete.json'),
        taxonomy_fix_sha256=sha(OUT / 'taxonomy_export_fix.json')))
    print('JOINT DELIVERY VERIFIED', summary['means'], 'gate', summary['development_gate'], flush=True)


if __name__ == '__main__': main()
