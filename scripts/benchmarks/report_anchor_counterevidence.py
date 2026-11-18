"""Audit full exports, sampled independent votes, all GT objects and real maps."""
from pathlib import Path
import html
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from run_anchor_counterevidence import OUT, METHODS, verify, seal

DEST = ROOT / 'output/anchor_counterevidence_v1'
NAMES = dict(baseline='Frozen baseline / no retraction', direct_only='Same evidence: direct only',
             ours_retract_propagation='Ours: retraction + propagation', matched_distance='Matched-count distance control')


def independent_votes(c):
    all_units, all_features, proto, labels, frames = [], [], [], [], []
    for frame in read(c['job'])['frames']:
        d = np.load(OUT / 'features' / c['key'] / f'{frame["frame_id"]:06d}.npz')
        all_units.append(d['units']); all_features.append(d['features'])
        proto.append(d['prototype_features']); labels.append(d['prototype_label'])
        frames.append(np.full(len(d['prototype_label']), frame['frame_id']))
    units = np.concatenate(all_units); q = np.concatenate(all_features)
    p, label, pf = np.concatenate(proto), np.concatenate(labels), np.concatenate(frames)
    q /= np.linalg.norm(q, axis=1)[:, None]; p /= np.linalg.norm(p, axis=1)[:, None]
    output = np.load(OUT / 'decisions' / c['key'] / 'anchors.npz')
    np.testing.assert_array_equal(output['units'], units)
    checked = np.unique(np.linspace(0, len(units) - 1, min(32, len(units)), dtype=int))
    for i in checked:
        n = neg = pos = 0
        for frame in np.unique(pf):
            if frame == units[i, 2]: continue
            own = np.flatnonzero((pf == frame) & (label == units[i, 1]))
            bg = np.flatnonzero((pf == frame) & (label < 0))
            if not len(own) or not len(bg): continue
            assert len(own) == 1
            delta = max(float(np.dot(q[i], p[j])) for j in bg) - float(np.dot(q[i], p[own[0]]))
            n += 1; neg += delta > 1e-6; pos += delta < -1e-6
        assert (n, neg, pos, n >= 2 and 2 * neg > n) == tuple(output[k][i] for k in ['eligible', 'refute', 'support', 'retract']), (c['key'], i)
    return dict(sampled_units=len(checked), checked_indices=checked.tolist(), all_other_frames_exhausted=True)


def audit(p, predictions, summary):
    from evaluate_kitti360_development import transfer
    from scipy.spatial import cKDTree
    result, per_object, comparisons = {}, {}, {}
    for c in p['contexts']:
        key = c['key']; base = np.load(Path(c['ours']) / 'geometry.npz'); old = base['instance']; xyz = base['xyz']
        field = np.load(c['field']); prop = np.load(OUT / 'decisions' / key / 'propagation.npz')
        np.testing.assert_array_equal(prop['original_winner'], field['nearest_owner'])
        r = predictions['results'][key]['methods']; masks = {}
        for m in METHODS:
            g = np.load(r[m]['geometry'])
            np.testing.assert_array_equal(g['xyz'], xyz)
            np.testing.assert_array_equal(g['original_row_id'], np.arange(len(old)))
            assert sha(r[m]['classification']) == sha(Path(c['ours']) / 'classification.json')
            mask = g['instance'] == 0
            np.testing.assert_array_equal(g['retracted_rows'], np.flatnonzero(mask))
            np.testing.assert_array_equal(g['instance'][~mask], old[~mask])
            masks[m] = mask
        assert not masks['baseline'].any()
        assert np.all(~masks['direct_only'] | masks['ours_retract_propagation'])
        assert np.all(~masks['direct_only'] | masks['matched_distance'])
        nids = int(old.max()) + 1
        np.testing.assert_array_equal(np.bincount(old[masks['ours_retract_propagation']], minlength=nids),
                                      np.bincount(old[masks['matched_distance']], minlength=nids))
        for oid in np.unique(old):
            candidates = np.flatnonzero((old == oid) & ~masks['direct_only'])
            selected = masks['matched_distance'][candidates]
            if not selected.any() or selected.all(): continue
            revoked = prop['revoked_cell_owner']; nodes = revoked[revoked[:, 1] == oid, 0]
            if not len(nodes): nodes = revoked[:, 0]
            distance = cKDTree(field['centers'][nodes]).query(xyz[candidates], workers=1)[0]
            a, b = distance[selected].max(), distance[~selected].min()
            assert a <= b
            if a == b: assert candidates[selected & (distance == a)].max() < candidates[~selected & (distance == b)].min()
        result[key] = dict(independent_votes=independent_votes(c), original_coordinates_ids_scores_preserved=True,
                           matched_counts_and_order_checked=True,
                           methods={m: {k: r[m][k] for k in ['removed', 'removed_original', 'removed_propagated']} for m in METHODS})
        if not c['reference']: continue
        ref = np.load(c['reference']); truth, weight = ref['train_instance'], ref['confidence']
        # All methods have identical coordinates, so one deterministic nearest
        # row query must reproduce every independently exported prediction.
        nearest = transfer(xyz, np.arange(1, len(xyz) + 1), ref['xyz'], .2)
        mass = np.bincount(truth, weights=weight)
        gids = np.flatnonzero((mass >= 100) & np.isin(np.arange(len(mass)) // 1000, [2, 13]))
        table = {int(g): dict(gt_id=int(g), methods={}) for g in gids}
        for m in METHODS:
            g = np.load(r[m]['geometry']); cls = read(r[m]['classification'])['objects']
            expected = np.r_[0, g['instance']][nearest]
            path = Path(c['ours']) / 'reference_prediction.npz' if m == 'baseline' else Path(r[m]['geometry']).parent / 'reference_prediction.npz'
            pred = np.load(path)['instance']; np.testing.assert_array_equal(expected, pred)
            pmass = np.bincount(pred, weights=weight, minlength=nids)
            joint = np.bincount(pred * len(mass) + truth, weights=weight, minlength=nids * len(mass)).reshape(nids, len(mass))
            for gid in gids:
                candidates = [o['object_id'] for o in cls if o['predictions']['mean']['train_id'] == gid // 1000]
                best = dict(iou=0., predicted_id=None)
                for oid in candidates:
                    inter = joint[oid, gid]; iou = float(inter / max(1e-12, mass[gid] + pmass[oid] - inter))
                    if iou > best['iou']: best = dict(iou=iou, predicted_id=int(oid))
                table[int(gid)]['methods'][m] = best
        per_object[key] = list(table.values()); comparisons[key] = {}
        for m in ['baseline', 'direct_only', 'matched_distance']:
            d = np.array([r['methods']['ours_retract_propagation']['iou'] - r['methods'][m]['iou'] for r in table.values()])
            comparisons[key][m] = dict(better=int((d > 1e-12).sum()), worse=int((d < -1e-12).sum()), equal=int((np.abs(d) <= 1e-12).sum()))
        result[key]['all_reference_transfers_replayed'] = True
    dump(OUT / 'per_object.json', dict(objects=per_object, comparisons=comparisons,
        scope='Best same-class IoU for every eligible GT, not one-to-one AP matching. All outcomes retained.'))
    return result, per_object, comparisons


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    p = verify(); seal('prepared'); seal('features_complete'); seal('decisions_complete')
    predictions = seal('predictions_complete'); summary = seal('summary')
    checked, objects, comparisons = audit(p, predictions, summary)
    DEST.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), constrained_layout=True)
    for row, c in enumerate(p['contexts']):
        key = c['key']; frames = read(c['job'])['frames']; rgb = frames[len(frames) // 2]['rgb']
        with Image.open(rgb) as image: axes[row, 0].imshow(image)
        axes[row, 0].set_title(key + '\nActual midpoint RGB'); axes[row, 0].axis('off')
        base = np.load(Path(c['ours']) / 'geometry.npz'); xyz = base['xyz']; old = base['instance']
        sample = np.unique(np.linspace(0, len(xyz) - 1, min(30000, len(xyz)), dtype=int))
        for col, m in enumerate(['baseline', 'ours_retract_propagation', 'matched_distance'], start=1):
            ax = axes[row, col]; g = np.load(predictions['results'][key]['methods'][m]['geometry']); removed = g['instance'] == 0
            ax.scatter(xyz[sample, 0], xyz[sample, 1], c=np.where(removed[sample], '#e14373', '#0a9687'), s=.5, rasterized=True)
            ax.set_aspect('equal', adjustable='box'); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
            title = 'Frozen baseline' if m == 'baseline' else 'Ours: retraction\n+ propagation' if m == 'ours_retract_propagation' else 'Same-count distance\ncontrol'
            ax.set_title(f'{title}\n{int(removed.sum()):,} rows unassigned', fontsize=10)
    fig.suptitle('Actual fixed map coordinates: teal retained; red withdrawn ownership (not removed measurements)', fontsize=13)
    fig.savefig(DEST / 'actual_maps.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, c in zip(axes, [c for c in p['contexts'] if c['reference']]):
        rows = summary['results'][c['key']]['methods']
        values = [100 * rows[m]['metrics']['allAp'] for m in METHODS]
        ax.bar(np.arange(4), values, color=['#99a5b4', '#5482aa', '#0a9687', '#d29b53'])
        ax.set_xticks(np.arange(4), ['Frozen\nbaseline', 'Direct\nonly', 'Ours', 'Matched\ndistance'])
        ax.set_title(c['key']); ax.set_ylabel('Local adapted AP50:90 (%)')
        ax.set_ylim(0, max(values) * 1.2)
        for i, value in enumerate(values): ax.text(i, value + .2, f'{value:.3f}', ha='center', fontsize=9)
    fig.suptitle('Fixed development comparison — not an official leaderboard result')
    fig.savefig(DEST / 'quality.png', dpi=180); plt.close(fig)
    rows = []
    for c in p['contexts']:
        key = c['key']
        for m in METHODS:
            r = predictions['results'][key]['methods'][m]
            metrics = summary['results'][key].get('methods', {}).get(m, {}).get('metrics')
            ap = f"{100 * metrics['allAp']:.6f}" if metrics else 'N/A'
            rows.append(f'<tr><td>{key}</td><td>{html.escape(NAMES[m])}</td><td>{ap}</td><td>{r["removed_original"]}</td><td>{r["removed_propagated"]}</td></tr>')
    object_tables = []
    for key, table in objects.items():
        trs = []
        for r in table:
            cells = ''.join(f'<td>{r["methods"][m]["iou"]:.6f}</td>' for m in METHODS)
            trs.append(f'<tr><td>{r["gt_id"]}</td>{cells}</tr>')
        object_tables.append(f'<details><summary>{key}: all {len(table)} eligible GT objects</summary><table><tr><th>GT ID</th>' + ''.join(f'<th>{html.escape(NAMES[m])}</th>' for m in METHODS) + '</tr>' + ''.join(trs) + '</table></details>')
    report = f'''<!doctype html><meta charset="utf-8"><title>omni_slam: anchor retraction</title>
<style>body{{font:16px system-ui;max-width:1400px;margin:35px auto;padding:0 24px;color:#163047}}img{{width:100%}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #dce3e8;text-align:left}}summary{{padding:16px;background:#eef6f5;cursor:pointer}}</style>
<h1>omni_slam — Ours: anchor retraction and propagation cleanup</h1>
<p>Fixed development gate: <b>{summary['development_gate']}</b>. Two exposed public windows and all330 archived robot frames. Six new public scores; nine new maps. Robot has no independent GT.</p>
<p>Same native Utonia features; acquisition-frame-excluded object/background votes. Rejecting a seed does not declare it background. Original coordinates, object IDs and class scores stay fixed; only ownership becomes unassigned. No new objects, blind-region expansion or online-SLAM claim.</p>
<p>Frame exclusion removes the query's own feature context, not all shared semantic-label provenance. Prototypes use completed-window predictions and future frames: this is retrospective, not independent-view statistical evidence.</p>
<img src="quality.png"><table><tr><th>Context</th><th>Method</th><th>AP50:90 (%)</th><th>Original rows withdrawn</th><th>Propagated rows withdrawn</th></tr>{''.join(rows)}</table>
<h2>Actual imagery and complete map views</h2><p>Fixed midpoint RGB for context; XY projections show all object regions with deterministic point subsampling. Red means withdrawn semantic ownership, not a removed sensor measurement. Every compared method has the same coordinate array.</p><img src="actual_maps.png">
<h2>All-object readout</h2><p>Best same-class IoU for each eligible GT; this is not one-to-one AP matching. All improvements, regressions and unchanged cases are retained.</p>{''.join(object_tables)}
<p><a href="../../build/benchmarks/anchor_counterevidence_v1/protocol.json">Frozen protocol</a> · <a href="../../build/benchmarks/anchor_counterevidence_v1/summary.json">Scores</a> · <a href="../../build/benchmarks/anchor_counterevidence_v1/delivery_audit.json">Audit</a></p>'''
    (DEST / 'index.html').write_text(report)
    dump(OUT / 'delivery_audit.json', dict(complete=True, results=checked, comparisons=comparisons,
        frozen_inputs_unchanged=True, source_sha256=sha(__file__), new_map_count=9, new_public_scores=6,
        report_hashes={str(f): sha(f) for f in DEST.iterdir() if f.is_file()},
        protocol_sha256=sha(OUT / 'protocol.json'), summary_sha256=sha(OUT / 'summary.json')))
    print('RETRACTION COMPLETE AUDIT', comparisons, flush=True)


if __name__ == '__main__': main()
