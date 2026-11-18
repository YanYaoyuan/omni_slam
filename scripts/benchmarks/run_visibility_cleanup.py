"""Bounded, no-training, fixed-identity multi-view cleanup pilot."""
import argparse
from dataclasses import asdict
from pathlib import Path
import resource
import time
import cv2
import numpy as np
from paper_final_common import ROOT, OLD, read, dump, sha
from omni_semantic_map.visibility_cleanup import (
    CleanupConfig, frame_evidence, rejection_mask, statistical_outliers,
)

OUT = ROOT / 'build/benchmarks/visibility_cleanup_20261112'
HISTORY = OLD / 'public_history_v1'
KEYS = ['replica_office_0', 'replica_office_1', 'scenenn_014', 'scenenn_015']
VARIANTS = ['geometric_sor', 'multiview', 'no_visibility', 'single_refutation']


def freeze():
    assert not (OUT / 'protocol.json').exists()
    parent = read(HISTORY / 'protocol.json')
    images = read(OLD / 'public_dino_v1/protocol.json')
    scenes = []
    for key in KEYS:
        old = next(s for s in parent['scenes'] if s['key'] == key)
        image_entry = next(s for s in images['scenes'] if s['key'] == key)
        source = HISTORY / 'runs' / key / 'native_bridge_2'
        scene = dict(old, geometry=str(source / 'geometry.npz'), classification=str(source / 'classification.json'))
        scene['immutable_hashes'] = dict(old['hashes'])
        for name in ['geometry', 'classification']:
            scene['immutable_hashes'][scene[name]] = sha(scene[name])
        for method in ['baseline', 'native_bridge_2']:
            for name in [method + '.npz', 'metrics.json']:
                p = HISTORY / 'evaluation' / key / name
                scene['immutable_hashes'][str(p)] = sha(p)
        scene['depths'] = []
        for r in image_entry['images']:
            rgb = Path(r['rgb'])
            depth = (rgb.with_name(rgb.name.replace('frame', 'depth')).with_suffix('.png')
                     if old['dataset'] == 'Replica' else rgb.parent.parent / 'depth' / rgb.name.replace('image', 'depth'))
            assert depth.exists(), depth
            scene['depths'].append(dict(frame_id=r['frame_id'], path=str(depth), sha256=sha(depth)))
        scenes.append(scene)
    paths = [Path(__file__).resolve(), ROOT / 'scripts/benchmarks/evaluate_visibility_cleanup.py',
             ROOT / 'omni_semantic_map/omni_semantic_map/visibility_cleanup.py',
             ROOT / 'omni_semantic_map/test/test_visibility_cleanup.py']
    paths += [ROOT / 'omni_semantic_map/omni_semantic_map' / name for name in
              ['full_ovo_evaluation.py', 'mesh_instance_evaluation.py', 'instance_error_audit.py']]
    protocol = dict(created_epoch=time.time(), scenes=scenes, config=asdict(CleanupConfig()),
        variants=VARIANTS, maximum_new_geometry_outputs=16, maximum_scored_scenes=4,
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths},
        benchmark_scope='Replica office0/office1 and SceneNN014/015, previously viewed development data. Fixed observed valid domain, NOT official whole-mesh scores.',
        baseline='Shared SAM2.1/SigLIP + CG; current frozen History-2 instance geometry. Existing predictions reused and projection equivalence checked.',
        hypothesis='Repeated, depth-confirmed mask contradictions remove contamination from fixed objects without breaking their correct geometry.',
        change='Offline rejection of existing foreground points only. Same memberships, labels, identities, coordinates, pose, masks, feature network and foreground list. No new points, splits or reassignment.',
        evidence='One vote per object/point/frame. Positive inside eroded union of own masks; negative outside dilated union but inside some SAM mask. Object must have an associated mask in that frame. Missing masks, missing depth, occlusion, FOV exclusion and boundary band abstain.',
        visibility='World-to-camera from inverse cached camera pose; native intrinsics and depth units. abs(z-depth)<=0.05m+0.01*depth; z>0.2m.',
        rejection='Negative frame count>=3 and >=2*positive count. Unknown leaves geometry unchanged. No forced per-object removal budget.',
        geometric_control='Per-object 20-NN mean distance > scene-object mean+2std, fixed conventional SOR without labels.',
        ablations={'no_visibility':'Only replace depth-agreement gate by FOV. Same masks, thresholds, frame votes.',
                   'single_refutation':'Only replace minimum 3 negative frames by minimum 1. Same ratio and visibility.'},
        prediction_sealing='All four scenes and 16 derived outputs sealed before any new GT evaluation. No scores during mapping. Failed cases included, not replaced.',
        gate=dict(dataset_mean_delta_pq100_min=2.,nondecreasing_scenes_min=3,
                  tp_not_lower_each_dataset=True,new_matches_each_dataset=True,
                  positive_vs_sor_each_dataset=True,each_ablation_lower_overall=True),
        reporting='Instance PQ/SQ/RQ, TP/FP/FN, gained/lost matched GT, per-GT IoU and coverage, point deletion, emptied objects, offline latency and peak RSS. Category descriptors fixed; no semantic-category gain claim.',
        limits='Deletion-only cannot split co-fused objects or recover unseen geometry. Neighbor-label transfer can change after removal; same 5cm projector in all arms. 2 scenes/domain are feasibility evidence, not independent generalization.',
        stop='One fixed configuration; no post-score tuning or automatic dataset expansion. If failed, archive and stop this implementation.',
        new_training=False,new_frontend_calls=False,no_gt_in_prediction=True)
    dump(OUT / 'protocol.json', protocol)
    for p in paths:
        dest = OUT / 'sources' / p.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(p.read_bytes())
    print('FROZEN: four development scenes, four cleanup variants, sixteen derived maps', flush=True)


def verify():
    protocol = read(OUT / 'protocol.json')
    for p, expected in protocol['source_sha256'].items():
        assert sha(ROOT / p) == expected, p
    return protocol


def run(key):
    protocol = verify(); config = CleanupConfig(**protocol['config'])
    scene = next(s for s in protocol['scenes'] if s['key'] == key)
    target = OUT / 'runs' / key; target.mkdir(parents=True, exist_ok=False)
    status = dict(complete=False, key=key, protocol_sha256=sha(OUT / 'protocol.json'))
    dump(target / 'run.json', status); begin = time.monotonic()
    try:
        # Reference files may be hashed here, but their labels are never decoded.
        for p, h in scene['immutable_hashes'].items():
            assert sha(p) == h, p
        cache = Path(scene['cache']); manifest = read(cache / 'manifest.json')
        assert manifest['complete'] and manifest['no_association'] and manifest['no_semantic_gt']
        for name, h in manifest['files_sha256'].items():
            assert sha(cache / name) == h, name
        g = np.load(scene['geometry']); xyz = g['xyz']; labels = g['instance']
        rows = read(scene['classification'])['objects']
        retained = [r['object_id'] for r in rows if r['predictions']['mean']['retained']]
        selected = np.flatnonzero(np.isin(labels, retained))
        points = xyz[selected]; identities = labels[selected]
        owner = {int(i): r['object_id'] for r in rows for i in r['observation_ids']}
        assert len(owner) == sum(len(r['observation_ids']) for r in rows)
        poses = np.load(cache / 'poses.npy'); camera = manifest['config']['cam']
        assert len(poses) == len(manifest['records']) == len(scene['depths'])
        counts = np.zeros((4, len(points)), np.uint16); visibility_count = np.zeros(len(points), np.uint16)
        times = []; events = []
        loaded = time.monotonic() - begin
        for index, (rec, dr) in enumerate(zip(manifest['records'], scene['depths'])):
            tic = time.monotonic(); assert rec['frame_id'] == dr['frame_id']
            path = cache / 'frames' / rec['file']; assert sha(path) == rec['sha256']
            assert sha(dr['path']) == dr['sha256'], dr['path']
            with np.load(path) as z:
                masks = np.unpackbits(z['masks_packed'], axis=2)[:, :, :camera['W']].astype(bool)
            mask_owners = np.array([owner.get(rec['observation_offset'] + j, 0) for j in range(rec['masks'])])
            raw = cv2.imread(dr['path'], cv2.IMREAD_UNCHANGED); assert raw is not None
            depth = raw.astype(np.float32) / camera['depth_scale']
            p, n, pv, nv, visible = frame_evidence(points, identities, poses[index], camera, depth, masks, mask_owners, config)
            counts += np.array([p, n, pv, nv], np.uint16); visibility_count += visible
            times.append(time.monotonic() - tic)
            events.append(dict(frame_id=rec['frame_id'], positive=int(p.sum()), negative=int(n.sum()),
                               fov_positive=int(pv.sum()), fov_negative=int(nv.sum()), visible=int(visible.sum())))
            if index % 50 == 0:
                print('EVIDENCE', key, index, '/', len(poses), flush=True)
        elapsed = time.monotonic(); sor = statistical_outliers(points, identities, config); sor_s = time.monotonic() - elapsed
        rejects = dict(geometric_sor=sor, multiview=rejection_mask(counts[0], counts[1], config),
                       no_visibility=rejection_mask(counts[2], counts[3], config),
                       single_refutation=rejection_mask(counts[0], counts[1], config, single_frame=True))
        np.savez_compressed(target / 'evidence.npz', source_indices=selected, positive=counts[0], negative=counts[1],
                            fov_positive=counts[2], fov_negative=counts[3], visible=visibility_count)
        dump(target / 'events.json', events)
        method_stats = {}
        for name, reject in rejects.items():
            folder = target / name; folder.mkdir()
            keep = np.ones(len(xyz), bool); keep[selected[reject]] = False
            np.savez_compressed(folder / 'geometry.npz', xyz=xyz[keep], instance=labels[keep])
            np.save(folder / 'retained_source_indices.npy', np.flatnonzero(keep))
            # Exact byte-copy of categories and memberships. No post-cleanup filter.
            (folder / 'classification.json').write_bytes(Path(scene['classification']).read_bytes())
            per_object = [dict(object_id=int(oid), before=int((identities == oid).sum()),
                               removed=int(((identities == oid) & reject).sum())) for oid in retained]
            method_stats[name] = dict(removed_points=int(reject.sum()), foreground_points=len(points),
                removed_fraction=float(reject.mean()), emptied_objects=[r['object_id'] for r in per_object if r['before'] and r['before'] == r['removed']],
                objects=per_object, geometry_path=str(folder / 'geometry.npz'))
        status.update(complete=True, methods=method_stats, input_verification_load_s=loaded,
            evidence_s=sum(times), evidence_frame_median_s=float(np.median(times)), evidence_frame_p95_s=float(np.quantile(times, .95)),
            geometric_control_s=sor_s, total_wall_s=time.monotonic()-begin,
            cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            frames=len(times), foreground_points=len(points), source_classification_sha256=sha(scene['classification']),
            no_reassociation=True,no_geometry_jitter=True,no_gt_labels_loaded=True,
            artifacts={str(p.relative_to(target)): sha(p) for p in target.rglob('*') if p.is_file() and p.name != 'run.json'})
    except Exception as e:
        import traceback
        status.update(failed=True,error=repr(e),traceback=traceback.format_exc()); dump(target / 'run.json', status); raise
    dump(target / 'run.json', status)
    print('COMPLETE', key, {k: v['removed_points'] for k, v in method_stats.items()}, flush=True)


def seal():
    protocol = verify(); runs = {}
    for key in KEYS:
        path = OUT / 'runs' / key / 'run.json'; status = read(path)
        assert status['complete'] or status.get('failed'), key
        for name, expected in status.get('artifacts', {}).items():
            assert sha(path.parent / name) == expected, name
        runs[str(path.relative_to(OUT))] = sha(path)
    assert not (OUT / 'predictions_complete.json').exists()
    dump(OUT / 'predictions_complete.json', dict(complete=True,runs=runs,protocol_sha256=sha(OUT/'protocol.json'),created_epoch=time.time()))
    print('SEALED all four scenes before scoring', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['freeze', 'run', 'seal']); parser.add_argument('--scene', choices=KEYS)
    args = parser.parse_args()
    if args.action == 'freeze': freeze()
    elif args.action == 'seal': seal()
    else: run(args.scene)
