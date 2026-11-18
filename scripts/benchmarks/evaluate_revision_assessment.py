"""Audit all frozen exports, then decode references and score all four arms.

The scoring seal precedes reference decoding. Missing-class/window AP stays NA;
it cannot satisfy the two-window acceptance criterion or trigger a replacement.
"""
import argparse
from pathlib import Path
import runpy
import time
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from run_revision_assessment import OUT, FULL, COVER, AUTH, verify, sealed, contexts
from evaluate_kitti360_development import read_static_ply, transfer, clean_json
from omni_semantic_map.kitti360_evaluation import evaluate, EVALUATOR_SHA256
from omni_semantic_map.kitti360_reference_identity import encode_reference
from run_discovery_query_benchmark import matched_ids

METHODS = ['strong_baseline', 'unconditional', 'history_only', 'full_authorization']
MODES = ['primary', 'raw_targets', 'uniform_scores']
META = ROOT / 'build/datasets/kitti360_prepare'


def freeze():
    p = verify(True)
    assert not (OUT / 'scoring_protocol.json').exists()
    seal = sealed(OUT / 'comparison_maps_complete.json')
    assert seal['final_maps'] == 8
    results = {}
    for c in contexts():
        key = c['key']
        base = dict(np.load(Path(c['ours']) / 'geometry.npz'))
        parent = dict(np.load(COVER / 'maps' / key / 'cover_history/geometry.npz'))
        full = dict(np.load(AUTH / 'maps' / key / 'ours/geometry.npz'))
        decision = dict(np.load(AUTH / 'prepared' / key / 'write_decisions.npz'))
        source = np.load(FULL / key / 'geometry.npz')['xyz']
        n = len(base['xyz'])
        history = dict(parent)
        history['instance'] = np.r_[decision['authorized_history'], parent['instance'][n:]]
        expected = dict(strong_baseline=base, unconditional=parent, history_only=history,
                        full_authorization=full)
        results[key] = {}
        for method in METHODS:
            folder = OUT / 'comparison_maps' / key / method
            got = dict(np.load(folder / 'geometry.npz'))
            assert set(got) == set(expected[method])
            for field in got:
                np.testing.assert_array_equal(got[field], expected[method][field])
            np.testing.assert_array_equal(got['xyz'][:n], base['xyz'])
            assert np.isfinite(got['xyz']).all() and np.all(got['instance'] >= 0)
            if method != 'strong_baseline':
                np.testing.assert_array_equal(got['xyz'][n:], source[got['added_full_source_ids']])
            elif len(got['added_full_source_ids']):
                count = len(got['added_full_source_ids'])
                np.testing.assert_array_equal(got['xyz'][-count:], source[got['added_full_source_ids']])
            metadata = read(folder / 'classification.json')
            ids = {r['object_id'] for r in metadata['objects']}
            assert set(got['instance']) - {0} <= ids
            expected_cls = Path(c['ours']) / 'classification.json' if method == 'strong_baseline' else COVER / 'maps' / key / 'cover_history/classification.json'
            assert metadata == read(expected_cls)
            results[key][method] = dict(all_export_arrays_exact=True,
                class_scores_exact=True, all_added_xyz_from_measured_sources=True,
                points=len(got['xyz']), nonempty_ids=len(set(got['instance']) - {0}))
        np.testing.assert_array_equal(full['instance'][:n], decision['authorized_history'])
    paths = [Path(__file__), META / 'evalInstanceLevelSemanticLabeling.py', META / 'labels.py',
             ROOT / 'scripts/benchmarks/evaluate_kitti360_development.py',
             ROOT / 'omni_semantic_map/omni_semantic_map/kitti360_evaluation.py',
             ROOT / 'omni_semantic_map/omni_semantic_map/kitti360_reference_identity.py',
             ROOT / 'scripts/benchmarks/run_discovery_query_benchmark.py']
    assert sha(paths[1]) == EVALUATOR_SHA256
    inputs = {str(OUT / 'comparison_maps_complete.json'): sha(OUT / 'comparison_maps_complete.json')}
    inputs.update(seal['files'])
    for s in p['scenes']:
        for f in [s['opaque_reference'], s['job']]:
            inputs[f] = sha(f)
    dump(OUT / 'scoring_protocol.json', dict(complete=True, created_epoch=time.time(),
        sources={str(f): sha(f) for f in paths}, inputs=inputs, export_audit=results,
        methods=METHODS, modes=MODES, reference_labels_decoded=False,
        rule='Exact registered local full-reference building/car AP50:90/AP50/AP25. '
            'All8maps exported and audited before either reference decode. '
            'Primary upstream_train, target_only and uniform-score diagnostics. '
            'NA retained; require2finite primary windows for acceptance. '
            'All per-object same-class best confidence-weighted reference IoU saved.',
        readout_check='16 fixed equally spaced reference indices per map, brute force '
            'distance with strict20cm and original source-index tie rule. This is a '
            'sample check, not exhaustive validation of all reference predictions.'))
    print('ALL EXPORTS AUDITED; SCORING FROZEN BEFORE REFERENCES', flush=True)


def mean_complete(values):
    return float(np.mean(values)) if values and all(v is not None and np.isfinite(v) for v in values) else None


def score():
    p = verify(True)
    s = read(OUT / 'scoring_protocol.json')
    assert s['complete'] and not (OUT / 'summary.json').exists()
    assert not (OUT / 'scoring_started.json').exists(), 'Inspect live/terminal attempt before recovery'
    for f, h in {**s['sources'], **s['inputs']}.items():
        assert sha(f) == h, f
    dump(OUT / 'scoring_started.json', dict(started_epoch=time.time(),
         scoring_protocol_sha256=sha(OUT / 'scoring_protocol.json')))
    taxonomy = runpy.run_path(str(META / 'labels.py'))['id2label']
    evaluator = META / 'evalInstanceLevelSemanticLabeling.py'
    results, per_object, reference_counts = {}, {}, {}
    count, brute = 0, 0
    for scene in p['scenes']:
        key = scene['key']
        job = read(scene['job'])
        raw = read_static_ply(scene['opaque_reference'])
        xyz = np.column_stack([raw[k] for k in ['x', 'y', 'z']]).astype(float) - np.asarray(job['origin_world'])
        truth, encoding = encode_reference(raw['semantic'], raw['instance'], taxonomy, 'upstream_train')
        physical, raw_encoding = encode_reference(raw['semantic'], raw['instance'], taxonomy, 'target_only')
        weight = raw['confidence'].astype(float)
        folder = OUT / 'evaluation' / key
        folder.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(folder / 'reference.npz', xyz=xyz, train_instance=truth,
                            raw_target_instance=physical, confidence=weight)
        dump(folder / 'reference_encodings.json', dict(primary=encoding, raw_targets=raw_encoding))
        targets = {'primary': truth, 'raw_targets': physical}
        tables = {}
        reference_counts[key] = {}
        for mode, gt in targets.items():
            mass = np.bincount(np.maximum(gt, 0), weights=weight)
            gids = np.flatnonzero(np.isin(np.arange(len(mass)) // 1000, [2, 13]) & (mass >= 100))
            tables[mode] = {int(g): dict(gt_id=int(g), confidence_mass=float(mass[g]), methods={}) for g in gids}
            reference_counts[key][mode] = {str(cls): int(np.count_nonzero(gids // 1000 == cls)) for cls in [2, 13]}
        rows = {}
        for method in METHODS:
            path = OUT / 'comparison_maps' / key / method
            geom = dict(np.load(path / 'geometry.npz'))
            records = read(path / 'classification.json')['objects']
            info = {r['object_id']: r['predictions']['mean'] for r in records}
            pred = transfer(geom['xyz'], geom['instance'], xyz, .2)
            dest = folder / method
            dest.mkdir()
            np.savez_compressed(dest / 'reference_prediction.npz', instance=pred)
            for i in np.unique(np.linspace(0, len(xyz)-1, min(16, len(xyz)), dtype=np.int64)):
                if len(geom['xyz']):
                    d2 = np.sum((geom['xyz']-xyz[i])**2, axis=1)
                    nearest = int(d2.argmin())
                    value = int(geom['instance'][nearest]) if d2[nearest] < .2**2 else 0
                else:
                    value = 0
                assert pred[i] == value, (key, method, int(i))
                brute += 1
            row = {}
            for mode, gt, uniform in [('primary', truth, False), ('raw_targets', physical, False),
                                      ('uniform_scores', truth, True)]:
                predictions = {o: dict(v, score=1.) for o, v in info.items()} if uniform else info
                result = evaluate(gt, weight, pred, predictions, evaluator)
                dump(dest / (mode + '_matches.json'), clean_json(result['matches']))
                row[mode] = dict(metrics=clean_json(result['averages']),
                                 matched_gt_ids=sorted(matched_ids(result['matches'])))
                count += 1
            pm = np.bincount(pred, weights=weight, minlength=max(info, default=0)+1)
            for mode, gt in targets.items():
                for gid, item in tables[mode].items():
                    ids = np.array([o for o, v in info.items() if v['train_id'] == gid // 1000], dtype=np.int64)
                    take = gt == gid
                    inter = np.bincount(pred[take], weights=weight[take], minlength=len(pm))
                    values = inter[ids] / np.maximum(1e-12, item['confidence_mass'] + pm[ids] - inter[ids])
                    best = int(np.argmax(values)) if len(values) else None
                    item['methods'][method] = dict(object_id=int(ids[best]) if best is not None else None,
                        best_iou=float(values[best]) if best is not None else 0.)
            rows[method] = row
            print('NEW WINDOW SCORE', key, method,
                  {m: row[m]['metrics']['allAp'] for m in MODES}, flush=True)
        for mode in MODES:
            old = set(rows['strong_baseline'][mode]['matched_gt_ids'])
            for row in rows.values():
                new = set(row[mode]['matched_gt_ids'])
                row[mode]['changes_vs_baseline'] = dict(gained=sorted(new-old), lost=sorted(old-new))
        results[key] = rows
        per_object[key] = {mode: list(rows.values()) for mode, rows in tables.items()}
    assert count == 24
    mean = {mode: {method: mean_complete([r[method][mode]['metrics']['allAp'] for r in results.values()])
                   for method in METHODS} for mode in MODES}
    finite_windows = sum(r['full_authorization']['primary']['metrics']['allAp'] is not None and
                         r['strong_baseline']['primary']['metrics']['allAp'] is not None for r in results.values())
    noharm = all(r['full_authorization']['primary']['metrics'][m] is not None and
                 r['strong_baseline']['primary']['metrics'][m] is not None and
                 r['full_authorization']['primary']['metrics'][m] >= r['strong_baseline']['primary']['metrics'][m]
                 for r in results.values() for m in ['allAp', 'allAp50%'])
    values = mean['primary']
    gain_ok = all(v is not None for v in values.values()) and values['full_authorization'] >= values['strong_baseline']+.01
    beats = all(v is not None for v in values.values()) and all(values['full_authorization'] > values[m] for m in ['unconditional', 'history_only'])
    changes = [r['full_authorization']['raw_targets']['changes_vs_baseline'] for r in results.values()]
    gained, lost = sum(len(c['gained']) for c in changes), sum(len(c['lost']) for c in changes)
    edits = sum(v['full_authorization']['old_point_id_changes'] for v in read(OUT / 'comparison_maps_complete.json')['results'].values())
    gate = bool(finite_windows == 2 and noharm and gain_ok and beats and gained >= 1 and lost == 0 and edits > 0)
    dump(OUT / 'per_object.json', dict(complete=True, results=per_object))
    dump(OUT / 'summary.json', dict(complete=True, results=results, mean_ap=mean,
        reference_target_counts=reference_counts, efficacy=dict(eligible_windows=finite_windows,
            per_window_nonregression=bool(noharm), mean_at_least_one_pp=bool(gain_ok),
            exceeds_controls=bool(beats), raw_gained=gained, raw_lost=lost, historical_edits=edits),
        prospective_gate=gate, original_development_gate=False, original_gate_preserved=True,
        candidate_promoted=False, new_scores=count, new_final_maps=8, new_public_windows=2,
        brute_force_reference_samples=brute, official_leaderboard=False,
        robot_semantic_accuracy_verified=False, goal_complete=False,
        protocol_sha256=sha(OUT / 'protocol.json'), maps_seal_sha256=sha(OUT / 'comparison_maps_complete.json'),
        scoring_protocol_sha256=sha(OUT / 'scoring_protocol.json')))
    print('PROSPECTIVE FOUR-METHOD RESULT', gate, mean, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'score'])
    globals()[parser.parse_args().action]()
