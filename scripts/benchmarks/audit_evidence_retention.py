"""Audit the frozen guarded/mean baseline; never alter mapping predictions.

GT signatures and RGB examples are post-hoc diagnostics. Mask and viewpoint
statistics are proxies, not calibrated quality or causal effects.
"""
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'omni_semantic_map'))
from omni_semantic_map.flat_benchmark import sha256
from omni_semantic_map.full_ovo_evaluation import BACKGROUND
from omni_semantic_map.shared_observations import select_views

BASE = ROOT / 'build/benchmarks/shared_frontend_20261110'
OUT = ROOT / 'build/benchmarks/evidence_retention_20261111'
REPORT = ROOT / 'docs/reports/evidence_retention_2026-11-11'
CACHE = Path(json.loads((BASE / 'protocol_cache.json').read_text())['cache_root'])
KEY = 'cg_guarded__all_views_mean_things'


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False))


def quantiles(values):
    a = np.asarray(values, float)
    return dict(n=len(a), median=float(np.median(a)), q25=float(np.quantile(a, .25)),
                q75=float(np.quantile(a, .75)), min=float(a.min()), max=float(a.max())) if len(a) else dict(n=0)


def audit_scene(scene, protocol):
    started = time.monotonic()
    cache = CACHE / scene
    manifest = json.loads((cache / 'manifest.json').read_text())
    expected = next(x['manifest_sha256'] for x in protocol['validated_caches'] if x['scene'] == scene)
    assert manifest['complete'] and sha256(cache / 'manifest.json') == expected
    for name, h in manifest['files_sha256'].items():
        assert sha256(cache / name) == h, name
    run_dir = BASE / 'runs/full' / scene / 'cg'
    run = json.loads((run_dir / 'run.json').read_text())
    assert run['complete'] and run['protocol_sha256'] == sha256(BASE / 'protocol_backends.json')
    for name, h in run['output_sha256'].items():
        assert sha256(run_dir / name) == h, name
    rows = json.loads((run_dir / 'guarded_classification.json').read_text())['objects']
    geom = np.load(run_dir / 'guarded_geometry.npz')
    labels, xyz = geom['instance'], geom['xyz']
    poses = np.load(cache / 'poses.npy')
    fields = defaultdict(list)
    for fi, record in enumerate(manifest['records']):
        path = cache / 'frames' / record['file']
        assert sha256(path) == record['sha256'], path
        with np.load(path) as f:
            masks = f['masks_packed']; n, h, packed_w = masks.shape
            width = manifest['config']['cam']['W']
            # np.packbits uses big-endian bits. Check only the four image edges.
            left = (masks[:, :, 0] & 128) != 0
            right = (masks[:, :, (width - 1) // 8] & (1 << (7 - (width - 1) % 8))) != 0
            truncated = left.any(1) | right.any(1) | (masks[:, 0] != 0).any(1) | (masks[:, -1] != 0).any(1)
            for name in ('features', 'areas', 'exclusive_areas'):
                fields[name].append(f[name])
            fields['supports'].append(np.diff(f['offsets']))
            fields['truncated'].append(truncated)
            fields['frames'].append(np.full(n, record['frame_id'], np.int32))
            fields['pose_indices'].append(np.full(n, fi, np.int32))
    a = {k: np.concatenate(v) for k, v in fields.items()}
    assert len(a['features']) == manifest['observations']
    text = np.load(cache / 'text_features.npy')
    text = text / np.linalg.norm(text, axis=1, keepdims=True)
    vocab = json.loads((cache / 'vocabulary.json').read_text())
    class_ids = np.array(vocab['replica_class_ids'])
    bg = np.isin(class_ids, BACKGROUND)
    scores = a['features'] @ text.T
    a['foreground_margin'] = scores[:, ~bg].max(1) - scores[:, bg].max(1)
    a['predicted_class'] = class_ids[scores.argmax(1)]
    a['exclusive_fraction'] = a['exclusive_areas'] / np.maximum(a['areas'], 1)
    # The independent feature/metadata file contains no GT; usable by stage 2.
    dest = OUT / 'observations' / scene
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / 'metadata.npz', **a, camera_positions=poses[:, :3, 3])
    dump(dest / 'provenance.json', dict(cache_manifest_sha256=expected, all_frame_hashes_verified=True,
        source_sha256=sha256(Path(__file__)), no_gt=True, metadata_sha256=sha256(dest / 'metadata.npz')))

    # Only now read saved GT-dependent diagnostic labels.
    ev = BASE / 'evaluation/full' / scene
    before = json.loads((ev / 'cg_guarded__all_regions_audit.json').read_text())
    after = json.loads((ev / (KEY + '_audit.json')).read_text())
    raw = {x['predicted_id']: x for x in before['predicted']}
    kept = {x['predicted_id']: x for x in after['predicted']}
    output = []
    for row in rows:
        oid = row['object_id']
        selected = np.array(select_views(row['observation_ids'], a['frames'], a['areas'], top_k=0))
        assert len(selected) == row['unique_frames']
        features = a['features'][selected]
        mean = features.mean(0); mean /= max(np.linalg.norm(mean), 1e-12)
        pred = row['predictions']['all_views_mean']
        assert class_ids[(text @ mean).argmax()] == pred['replica_class_id']
        center = xyz[labels == oid].mean(0)
        directions = a['camera_positions'][a['pose_indices'][selected]] - center if 'camera_positions' in a else poses[a['pose_indices'][selected], :3, 3] - center
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
        cosine = np.clip(directions @ directions.T, -1, 1)
        row_out = dict(scene=scene, object_id=oid, unique_frames=len(selected),
            prediction=pred, retained=pred['replica_class_id'] not in BACKGROUND,
            before=raw.get(oid), after=kept.get(oid), selected_observation_ids=selected.tolist(),
            center=center.tolist(),
            median_area_px=float(np.median(a['areas'][selected])),
            median_supports=float(np.median(a['supports'][selected])),
            median_exclusive_fraction=float(np.median(a['exclusive_fraction'][selected])),
            truncated_view_fraction=float(a['truncated'][selected].mean()),
            max_view_angle_deg=float(np.degrees(np.arccos(cosine.min()))),
            mean_view_cosine_distance=float((1 - cosine).mean()),
            feature_agreement=float((features @ mean).mean()),
            foreground_view_fraction=float((a['foreground_margin'][selected] > 0).mean()),
            median_foreground_margin=float(np.median(a['foreground_margin'][selected])))
        output.append(row_out)
    mapping = {r['object_id']: r for r in output}
    matched_after = {x['matched_gt'] for x in after['predicted'] if x['matched_gt'] is not None}
    losses = []
    for item in before['predicted']:
        gt = item['matched_gt']
        if gt is not None and gt not in matched_after:
            row = mapping[item['predicted_id']]
            losses.append(dict(gt_id=gt, object_id=row['object_id'],
                cause='direct_background_filter' if not row['retained'] else 'projection_reassignment',
                prediction=row['prediction'], unique_frames=row['unique_frames']))
    result = dict(scene=scene, complete=True, objects=output, filter_losses=losses,
        counts=after['predicted_counts'], ground_truth_counts=after['gt_counts'],
        classification_reproduced=True, elapsed_s=time.monotonic()-started,
        source_sha256=sha256(Path(__file__)), input_sha256={str(p.relative_to(ROOT)): sha256(p) for p in
            [run_dir/'guarded_classification.json', ev/(KEY+'_audit.json'), ev/'cg_guarded__all_regions_audit.json']})
    dump(REPORT / 'stage1' / (scene + '.json'), result)
    print('AUDITED', scene, 'objects', len(rows), 'losses', len(losses), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--scenes', nargs='+')
    args = parser.parse_args()
    protocol = json.loads((BASE / 'protocol_backends.json').read_text())
    scenes = args.scenes or protocol['scenes']
    assert set(scenes) <= set(protocol['scenes'])
    OUT.mkdir(parents=True, exist_ok=True)
    dump(OUT / 'stage1_protocol.json', dict(stage=1, scenes=scenes, baseline=KEY,
        baseline_protocol_sha256=sha256(BASE/'protocol_backends.json'), source_sha256=sha256(Path(__file__)),
        scope=__doc__, no_input_or_old_result_mutations=True,
        case_selection='per scene, lowest ID direct-filter loss, largest unmatched partial/fragment; largest matched object as control'))
    results = [audit_scene(s, protocol) for s in scenes]
    groups = defaultdict(list); counts = Counter(); losses = Counter()
    for result in results:
        counts.update(result['counts']); losses.update(x['cause'] for x in result['filter_losses'])
        for row in result['objects']:
            category = row['after']['category'] if row['after'] else 'filtered_or_unprojected'
            groups[category].append(row)
    measures = ['unique_frames', 'median_area_px', 'median_supports', 'median_exclusive_fraction',
                'truncated_view_fraction', 'max_view_angle_deg', 'feature_agreement', 'foreground_view_fraction']
    stats = {name: {k: quantiles([r[k] for r in rows]) for k in measures} for name, rows in groups.items()}
    expected = json.loads((BASE/'summary.json').read_text())['methods'][KEY]
    if scenes == protocol['scenes']:
        assert dict(counts) == expected['predicted_error_counts']
        assert sum(losses.values()) == expected['filter_lost']
    summary = dict(complete=True, scenes=scenes, baseline=KEY, baseline_pq=expected['macro_pq'],
        predicted_counts=counts, filter_losses=losses, statistics=stats,
        classification_reproduced=True, full_input_hash_verification=True,
        limitations=['Final-geometry signatures do not prove association causes.',
                    'Border contact/exclusive area are quality proxies, not mask ground truth.',
                    'All eight scenes are development scenes.', 'SAM predicted IoU/stability were not cached.'])
    dump(REPORT / 'stage1/summary.json', summary)
    print(json.dumps(dict(counts=counts, losses=losses)), flush=True)


if __name__ == '__main__':
    main()
