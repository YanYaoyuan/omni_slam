"""Independent numeric checks and complete-frame diagnostic figure."""
from pathlib import Path
import json
import numpy as np
from paper_final_common import ROOT, dump, sha
from probe_utonia_ownership_evidence import OUT, verify


def weighted_auc(y, scores, weight):
    # Weighted Mann-Whitney statistic, independent of sklearn ROC integration.
    if not y.any() or y.all(): return None
    order = np.argsort(scores, kind='stable')
    y, scores, weight = y[order], scores[order], weight[order]
    starts = np.r_[0, np.flatnonzero(np.diff(scores)) + 1]
    pos = np.add.reduceat(weight * y, starts)
    neg = np.add.reduceat(weight * ~y, starts)
    lower_neg = np.cumsum(neg) - neg
    return float(np.sum(pos * (lower_neg + .5 * neg)) / (pos.sum() * neg.sum()))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = verify()
    summary = json.loads((OUT / 'summary.json').read_text())
    for name in ['prepared', 'scores_complete']:
        seal = json.loads((OUT / f'{name}.json').read_text())
        for file, expected in seal['files'].items(): assert sha(file) == expected
    for file, expected in summary['readout_hashes'].items(): assert sha(file) == expected
    audit = {}
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, c in zip(axes, p['scenes']):
        key = c['key']; ref = np.load(c['reference'])
        mass = np.bincount(ref['train_instance'], weights=ref['confidence'])
        entries = []
        for frame, stats in zip(c['frames'], summary['results'][key]['frames']):
            fid = frame['frame_id']
            d = np.load(OUT / 'inputs' / key / f'{fid:06d}.npz')
            s = np.load(OUT / 'scores' / key / f'{fid:06d}.npz')
            r = np.load(OUT / 'readout' / key / f'{fid:06d}.npz')
            left, right = d['left'], d['right']
            actual_distance = np.linalg.norm(d['xyz'][left] - d['xyz'][right], axis=1)
            np.testing.assert_allclose(actual_distance, d['distance'], rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(s['negative_distance'], -d['distance'])
            assert (d['owner'][left] == 0).all() and (d['owner'][right] > 0).all()
            ri, rd = r['reference_index'], r['reference_distance']
            label, w = ref['train_instance'][ri], ref['confidence'][ri]
            np.testing.assert_allclose(np.linalg.norm(d['xyz'] - ref['xyz'][ri], axis=1), rd, atol=1e-12)
            # Small deterministic exhaustive nearest-reference check.
            for index in np.linspace(0, len(ri) - 1, 7, dtype=int):
                distance = np.linalg.norm(ref['xyz'] - d['xyz'][index], axis=1)
                assert ri[index] == np.argmin(distance)
            reasons = [
                ('outside_reference_20cm', (rd[left] > .2) | (rd[right] > .2)),
                ('unknown_label_or_confidence', (label[left] < 1000) | (label[left] >= 19000) |
                 (label[right] < 1000) | (label[right] >= 19000) | (w[left] <= 0) | (w[right] <= 0)),
                ('anchor_not_building_or_car', ~np.isin(label[right] // 1000, [2, 13])),
                ('anchor_gt_mass_below_100', mass[label[right]] < 100),
            ]
            remaining = np.ones(len(left), bool); excluded = {}
            for name, mask in reasons:
                excluded[name] = int((remaining & mask).sum()); remaining &= ~mask
            np.testing.assert_array_equal(remaining, r['eligible'])
            np.testing.assert_array_equal(label[left[remaining]] == label[right[remaining]], r['same_instance'])
            np.testing.assert_allclose(np.minimum(w[left[remaining]], w[right[remaining]]), r['weights'])
            for field, score in [('cosine_auc', s['cosine'][remaining]), ('distance_auc', -d['distance'][remaining])]:
                value = weighted_auc(r['same_instance'], score, r['weights'])
                assert abs(value - stats['all_pairs'][field]) < 1e-12
            entries.append(dict(frame_id=fid, exclusions=excluded, eligible_pairs=int(remaining.sum()),
                                candidate_pairs=len(left), exhaustive_reference_checks=7,
                                weighted_auc_independent_check=True))
        audit[key] = entries
        rows = summary['results'][key]['frames']
        x = np.arange(len(rows)); width = .34
        ax.bar(x - width / 2, [r['all_pairs']['distance_auc'] for r in rows], width, color='#a3afbd', label='Distance control')
        ax.bar(x + width / 2, [r['all_pairs']['cosine_auc'] for r in rows], width, color='#028b86', label='Utonia cosine (external feature)')
        ax.set_xticks(x, [str(r['frame_id']) for r in rows]); ax.set_xlabel('Fixed raw scan ID')
        ax.set_ylim(.5, 1.02); ax.set_ylabel('Confidence-weighted surface-pair AUROC')
        ax.set_title(key + ' | all eligible pairs')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=8, loc='lower right')
    figure.suptitle('Six-frame evidence probe: not map AP; predeclared information gate FAILED', fontsize=12)
    target = ROOT / 'output/utonia_ownership_probe_v1'
    target.mkdir(parents=True, exist_ok=True)
    figure.savefig(target / 'all_frames.png', dpi=180)
    plt.close(figure)
    dump(OUT / 'delivery_audit.json', dict(complete=True, results=audit,
        original_inputs_unchanged=True, independent_auc=True, source_sha256=sha(__file__),
        summary_sha256=sha(OUT / 'summary.json'), figure_sha256=sha(target / 'all_frames.png')))
    print('All six frame readouts independently checked; 42 exhaustive nearest-reference checks passed.', flush=True)


if __name__ == '__main__': main()
