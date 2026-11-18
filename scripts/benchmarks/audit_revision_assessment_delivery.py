"""Read-only replay of saved scores and output differences; no new experiments."""
import itertools
import json
from html.parser import HTMLParser
from pathlib import Path
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from omni_semantic_map.kitti360_evaluation import kernel

OUT = ROOT / 'build/benchmarks/revision_assessment_v1'
REPORT = ROOT / 'output/revision_assessment_v1'
METHODS = ['strong_baseline', 'unconditional', 'history_only', 'full_authorization']


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        self.paths.extend(v for k, v in attrs if k in ('src', 'href') and v)


def main():
    summary = read(OUT / 'summary.json')
    scoring = read(OUT / 'scoring_protocol.json')
    seal = read(OUT / 'comparison_maps_complete.json')
    robot = read(OUT / 'robot_correspondence_audit.json')
    report = read(OUT / 'report_audit.json')
    checks = {}
    for source in [scoring['sources'], scoring['inputs'], seal['files'],
                   robot['sources'], robot['shared_kernel_sha256'], report['files']]:
        for filename, expected in source.items():
            assert sha(filename) == expected, filename
            checks[filename] = expected
    for method, record in robot['methods'].items():
        for name in ['geometry', 'classification']:
            filename = Path(record['folder']) / (name + ('.npz' if name == 'geometry' else '.json'))
            assert sha(filename) == record[name + '_sha256']
            checks[str(filename)] = record[name + '_sha256']
    assert sha(OUT / 'protocol.json') == summary['protocol_sha256']
    assert sha(OUT / 'comparison_maps_complete.json') == summary['maps_seal_sha256']
    assert sha(OUT / 'scoring_protocol.json') == summary['scoring_protocol_sha256']
    assert sha(OUT / 'summary.json') == report['summary_sha256']
    assert sha(OUT / 'per_object.json') == report['per_object_sha256']
    assert seal['reference_labels_decoded'] is False
    assert scoring['reference_labels_decoded'] is False
    assert scoring['created_epoch'] < read(OUT / 'scoring_started.json')['started_epoch']
    args, funcs = kernel(ROOT / 'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py')
    objects = read(OUT / 'per_object.json')['results']
    difference, class_scores, changes = {}, {}, []
    replay_count, object_count = 0, 0
    scores = {m: [] for m in METHODS}
    for key, rows in summary['results'].items():
        predictions = {}
        difference[key], class_scores[key] = {}, {}
        for method in METHODS:
            folder = OUT / 'evaluation' / key / method
            predictions[method] = np.load(folder / 'reference_prediction.npz')['instance']
            for mode, recorded in rows[method].items():
                matches = read(folder / (mode + '_matches.json'))
                ap = funcs['evaluateMatches'](matches, args)
                result = funcs['computeAverages'](ap, args)
                for metric in ['allAp', 'allAp50%', 'allAp25%']:
                    np.testing.assert_allclose(result[metric], recorded['metrics'][metric], rtol=0, atol=1e-12)
                gained = set()
                for records in matches['window']['groundTruth'].values():
                    for gt in records:
                        if gt['vert_count'] < 100:
                            continue
                        for pred in gt['matched_pred']:
                            iou = pred['intersection'] / (gt['vert_count'] + pred['vert_count'] - pred['intersection'])
                            if pred['vert_count'] >= 100 and iou > .5:
                                gained.add(gt['instance_id'])
                assert gained == set(recorded['matched_gt_ids'])
                replay_count += 1
            class_scores[key][method] = rows[method]['primary']['metrics']['classes']
            scores[method].append(rows[method]['primary']['metrics']['allAp'])
        for a, b in itertools.combinations(METHODS, 2):
            different = int(np.count_nonzero(predictions[a] != predictions[b]))
            difference[key][a + '__vs__' + b] = different
        # Recompute every per-object entry from saved spatial labels and reference.
        reference = dict(np.load(OUT / 'evaluation' / key / 'reference.npz'))
        for mode, field in [('primary', 'train_instance'), ('raw_targets', 'raw_target_instance')]:
            gt, weights = reference[field], reference['confidence']
            for method in METHODS:
                pred = predictions[method]
                metadata = read(OUT / 'comparison_maps' / key / method / 'classification.json')['objects']
                info = {r['object_id']: r['predictions']['mean'] for r in metadata}
                mass = np.bincount(pred, weights=weights, minlength=max(info, default=0)+1)
                for obj in objects[key][mode]:
                    gid = obj['gt_id']
                    ids = np.array([i for i, v in info.items() if v['train_id'] == gid // 1000], dtype=int)
                    take = gt == gid
                    inter = np.bincount(pred[take], weights=weights[take], minlength=len(mass))
                    gt_mass = float(weights[take].sum())
                    values = inter[ids] / np.maximum(1e-12, gt_mass + mass[ids] - inter[ids])
                    best = int(np.argmax(values)) if len(values) else None
                    expected = obj['methods'][method]
                    assert expected['object_id'] == (int(ids[best]) if best is not None else None)
                    np.testing.assert_allclose(expected['best_iou'], values[best] if best is not None else 0., atol=1e-12, rtol=0)
                object_count += len(objects[key][mode])
        old = set(rows['strong_baseline']['raw_targets']['matched_gt_ids'])
        new = set(rows['full_authorization']['raw_targets']['matched_gt_ids'])
        changes.append(dict(key=key, gained=sorted(new-old), lost=sorted(old-new)))
    means = {m: float(np.mean(v)) for m, v in scores.items()}
    for m, value in means.items():
        np.testing.assert_allclose(value, summary['mean_ap']['primary'][m], rtol=0, atol=1e-12)
    noharm = all(rows['full_authorization']['primary']['metrics'][metric] >= rows['strong_baseline']['primary']['metrics'][metric]
                 for rows in summary['results'].values() for metric in ['allAp', 'allAp50%'])
    edits = sum(v['full_authorization']['old_point_id_changes'] for v in seal['results'].values())
    gate = bool(len(scores['full_authorization']) == 2 and noharm and
                means['full_authorization'] >= means['strong_baseline'] + .01 and
                all(means['full_authorization'] > means[m] for m in ['unconditional', 'history_only']) and
                sum(len(c['gained']) for c in changes) >= 1 and not any(c['lost'] for c in changes) and edits > 0)
    assert gate == summary['prospective_gate']
    assert read(ROOT / 'build/benchmarks/revision_authorization_v1/summary.json')['development_gate'] is False
    parser = Links()
    parser.feed((REPORT / 'index.html').read_text())
    for link in parser.paths:
        assert (REPORT / link).resolve().exists(), link
    assert len(list(REPORT.glob('*.png'))) == 9
    dump(OUT / 'delivery_audit.json', dict(complete=True, source_sha256=sha(__file__),
        verified_hashes=checks, saved_score_replays=replay_count,
        per_object_method_replays=object_count,
        pairwise_reference_label_changes=difference, per_class_primary=class_scores,
        primary_means=means, raw_match_changes=changes, prospective_gate_recomputed=gate,
        original_failed_development_gate_preserved=True, robot_maps_verified=4,
        robot_semantic_accuracy_available=False, linked_assets_exist=True,
        visual_review=dict(completed=True, images=9,
            findings='Readable labels and common map coordinates; six fixed first/median/last RGB views. '
                     '0010 first view has no colored predictions; retained as observed, not replaced. '
                     'Overlays are measured-source projections, not dense masks or GT.'),
        no_new_model_inference=True, no_new_maps=True,
        goal_scope='New mechanism implemented, same-version robot comparison and prospective assessment delivered; '
                   'does not assert universal novelty, official leaderboard score or publication readiness.'))
    print(json.dumps(dict(score_replays=replay_count, object_method_replays=object_count,
                          means=means, changes=difference, prospective_gate=gate), indent=2))


if __name__ == '__main__':
    main()
