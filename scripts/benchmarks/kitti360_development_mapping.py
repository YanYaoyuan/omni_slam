"""Explicit outdoor adapters; development data only, no GT during mapping.

Reuse the frozen image frontend and CG/history association kernels. Replace
only input geometry and taxonomy; alternative cleanup arms share one map.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import runpy
import shutil
import sys
import time
from types import SimpleNamespace
import cv2
import numpy as np
from paper_final_common import ROOT, FROZEN, read, dump, sha
from prepare_kitti360_frontend import OUT, META
from omni_semantic_map.sparse_lidar_support import MeasuredVoxelMap, project_measured_visibility, measured_frame_evidence
from omni_semantic_map.visibility_cleanup import CleanupConfig, rejection_mask, statistical_outliers

JOB = OUT/'jobs/2013_05_28_drive_0000_sync.json'


def freeze():
    assert not (OUT/'mapping_protocol.json').exists()
    job = read(JOB)
    assert job['kind'] == 'kitti360_development'
    paths = [Path(__file__), ROOT/'omni_semantic_map/omni_semantic_map/sparse_lidar_support.py',
        ROOT/'omni_semantic_map/omni_semantic_map/visibility_cleanup.py',
        ROOT/'scripts/benchmarks/paper_final_mapping.py', ROOT/'scripts/benchmarks/run_livo_history_graph.py',
        FROZEN/'config/cg_rgbd.json', META/'labels.py']
    dump(OUT/'mapping_protocol.json', dict(created_epoch=time.time(), job_sha256=sha(JOB),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
        voxel_size_m=.05, association_surface_gate_m=.05, cleanup=asdict(CleanupConfig()),
        frontend='Unchanged SAM2.1/SigLIP learned merger, batch8 BF16. Outdoor text labels encoded on CPU FP32.',
        geometry='Causal first measured return in each 5cm voxel; exact local XYZ and source frame/raw index retained. Match accumulated representatives to current measured depth within 5cm at nearest pixel center. No filled depths.',
        taxonomy='19 canonical nonignored KITTI-360 semantic names as competing text classes. Retain building=11 and car=26 for instance evaluation; do not use Replica exclusions.',
        readout='Same distinct-frame mean descriptor, normalized cosine text scores; unchanged association/final CG configuration. CG native geometry finalization still downsamples/denoises.',
        comparison='Shared-input core adaptations, not full original CG detector pipeline. Each cleanup is one alternative after the same history2 map, never stacked with SOR.',
        scope='Development-only adapter. No official or independent benchmark claim, no validation-window prediction. Static reference evaluates static objects only.',
        status='Prediction protocol, not a frozen held-out evaluation protocol. Development diagnostics may motivate an explicitly versioned correction.'))


def verify():
    p = read(OUT/'mapping_protocol.json')
    assert sha(JOB) == p['job_sha256']
    for name, expected in p['source_sha256'].items(): assert sha(ROOT/name) == expected, name
    return p, read(JOB)


def taxonomy():
    p, job = verify()
    from split_strong_frontend import verify_job
    verify_job(job, JOB)
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0, str(ROOT/'build/benchmarks/ovo_full_20261110/upstream'))
    from ovo.utils.clip_utils import load_clip_model
    labels = runpy.run_path(str(META/'labels.py'))['labels']
    labels = sorted([r for r in labels if not r.ignoreInEval], key=lambda r:r.trainId)
    assert [r.trainId for r in labels] == list(range(19))
    target = OUT/'taxonomy'
    target.mkdir(exist_ok=False)
    model, tokenizer, _, _ = load_clip_model(job['config']['semantic']['clip']['model_card'], False)
    names = [r.name for r in labels]
    with torch.no_grad():
        tokens = torch.cat([tokenizer(name) for name in names])
        features = model.encode_text(tokens)
        features = torch.nn.functional.normalize(features, dim=-1).cpu().numpy()
    assert features.shape == (19, 1152) and np.isfinite(features).all()
    np.save(target/'text_features.npy', features)
    dump(target/'vocabulary.json', dict(names=names, semantic_ids=[r.id for r in labels],
        train_ids=[r.trainId for r in labels], target_semantic_ids=[11, 26], prompt_template='{}'))
    dump(target/'manifest.json', dict(complete=True, no_gt=True, device='CPU FP32',
        protocol_sha256=sha(OUT/'mapping_protocol.json'), files_sha256={name:sha(target/name)
            for name in ['text_features.npy', 'vocabulary.json']}))
    print('TAXONOMY', names, flush=True)


def geometry():
    p, job = verify()
    from split_strong_frontend import verify_job
    verify_job(job, JOB)
    front = Path(job['output'])
    for phase in job['phases']:
        record = read(front/(phase+'_run.json'))
        assert record['complete'] and record['job_sha256'] == sha(JOB)
        for path, expected in record['files_sha256'].items(): assert sha(front/path) == expected
    tax = read(OUT/'taxonomy/manifest.json')
    for path, expected in tax['files_sha256'].items(): assert sha(OUT/'taxonomy'/path) == expected
    cache = front/'cache'
    cache.mkdir(exist_ok=False)
    (cache/'frames').mkdir()
    voxel_map = MeasuredVoxelMap(p['voxel_size_m'])
    cam = job['config']['cam']
    gate = SimpleNamespace(depth_absolute_m=p['association_surface_gate_m'], depth_relative=0.)
    manifest = dict(complete=False, scene=job['scene'], no_association=True, no_semantic_gt=True,
        config=job['config'], records=[], job_sha256=sha(JOB),
        geometry_scope=p['geometry'], origin_world=job['origin_world'],
        mapping_protocol_sha256=sha(OUT/'mapping_protocol.json'))
    offset = 0
    started = time.monotonic()
    for record in job['frames']:
        name = f"{record['frame_id']:06d}.npz"
        measured = np.load(record['measured'])
        image = cv2.cvtColor(cv2.imread(record['rgb']), cv2.COLOR_BGR2RGB)
        px = measured['pixel_xy']
        selected = voxel_map.append(measured['xyz'], image[px[:, 1], px[:, 0]], record['frame_id'],
                                    measured['raw_indices'], px)
        points = voxel_map.arrays()['xyz']
        pose = np.asarray(record['pose'])
        x, y, _, visible = project_measured_visibility(points, pose, cam, measured['depth'], gate)
        with np.load(front/'masks'/name) as f: arrays = {k:f[k] for k in f.files}
        with np.load(front/'features'/name) as f: arrays['features'] = f['features']
        n = len(arrays['features'])
        mid = arrays['segmentation'][y, x]
        ids = np.flatnonzero(visible & (mid >= 0))
        mask_ids = mid[ids]
        assert not len(mask_ids) or mask_ids.max() < n
        order = np.argsort(mask_ids, kind='stable')
        arrays.update(point_ids=ids[order].astype(np.int32),
            offsets=np.r_[0, np.cumsum(np.bincount(mask_ids, minlength=n))].astype(np.int64))
        path = cache/'frames'/name
        np.savez_compressed(path, **arrays)
        manifest['records'].append(dict(frame_id=record['frame_id'], file=name, masks=n,
            observation_offset=offset, matched_supports=len(ids), map_points=len(points),
            added_measured_points=len(selected), sha256=sha(path)))
        offset += n
        dump(cache/'manifest.json', manifest)
    np.savez_compressed(cache/'geometry.npz', **voxel_map.arrays())
    np.save(cache/'poses.npy', np.array([r['pose'] for r in job['frames']]))
    for name in ['text_features.npy', 'vocabulary.json']: shutil.copyfile(OUT/'taxonomy'/name, cache/name)
    manifest.update(complete=True, frames=len(job['frames']), frames_cached=len(job['frames']),
        observations=offset, map_points=len(points), wall_seconds=time.monotonic()-started,
        files_sha256={path.name:sha(path) for path in cache.iterdir() if path.is_file() and path.name != 'manifest.json'})
    dump(cache/'manifest.json', manifest)
    print('GEOMETRY', len(points), 'points', offset, 'observations', flush=True)


def outdoor_finalize(engine, groups, output, table):
    # Same native geometric stages as run_livo_history_graph.finalize; explicit
    # outdoor readout avoids injecting KITTI IDs into a Replica-named field.
    from conceptgraph.slam.slam_classes import MapObjectList
    from omni_semantic_map.fusion_diagnostics import guarded_merge, membership_audit
    from omni_semantic_map.full_ovo_evaluation import cg_geometry
    from omni_semantic_map.shared_observations import select_views
    from omni_semantic_map.evidence_readout import aggregate
    originals = {min(o['observation_ids'])+1:o for o in engine._objects}
    objects = MapObjectList()
    for group in groups:
        obj = deepcopy(originals[group[0]])
        for key in group[1:]: obj = engine.utils.merge_obj2_into_obj1(engine.config, obj, deepcopy(originals[key]), run_dbscan=False)
        objects.append(obj)
    before = membership_audit(objects)
    assert before['repeated_memberships'] == 0
    assert before['unique_observations'] == sum(len(o['observation_ids']) for o in engine._objects)
    filtered = engine.utils.filter_objects(engine.config, engine.utils.denoise_objects(engine.config, deepcopy(objects)))
    objects = guarded_merge(engine.config, deepcopy(filtered))
    after = membership_audit(objects)
    assert after['repeated_memberships'] == 0
    points, labels, identity = cg_geometry([dict(object_id=min(o['observation_ids'])+1,
        points=np.asarray(o['pcd'].points)) for o in objects])
    features, frames, areas, text, vocab, pose_by_frame = table
    rows = []
    for record in identity:
        oid = record['evaluation_id']
        members = objects[record['object_index']]['observation_ids']
        selected = np.array(select_views(members, frames, areas, top_k=0))
        center = points[labels == oid].mean(0)
        cameras = np.array([pose_by_frame[int(frames[i])][:3, 3] for i in selected])
        result = aggregate(features[selected], areas[selected], cameras, center,
                           method='mean', observation_ids=selected)
        scores = text@result['descriptor']
        order = np.argsort(-scores, kind='stable')
        best = int(order[0])
        semantic = int(vocab['semantic_ids'][best])
        rows.append(dict(object_id=oid, observation_ids=sorted(set(map(int, members))),
            selected_observation_ids=selected.tolist(), center=center.tolist(), unique_frames=len(selected),
            predictions=dict(mean=dict(name=vocab['names'][best], semantic_id=semantic,
                train_id=int(vocab['train_ids'][best]), score=float(scores[best]), margin=float(scores[best]-scores[order[1]]),
                retained=semantic in vocab['target_semantic_ids'], effective_views=result['effective_views']))))
    output.mkdir(exist_ok=False)
    np.savez_compressed(output/'geometry.npz', xyz=points, instance=labels)
    dump(output/'classification.json', dict(objects=rows, taxonomy='KITTI-360', target_semantic_ids=[11, 26]))
    return dict(before_filter=before, after_final=after, objects=rows)


def cg():
    _, job = verify()
    import run_livo_history_graph as native
    from paper_final_mapping import run_cg
    original = native.finalize
    native.finalize = outdoor_finalize
    try: run_cg(Path(job['output'])/'cache', OUT/'maps/cg')
    finally: native.finalize = original


def cleanup():
    p, job = verify()
    cache = Path(job['output'])/'cache'
    source = OUT/'maps/cg/history2'
    m = read(cache/'manifest.json')
    for name, expected in m['files_sha256'].items(): assert sha(cache/name) == expected
    g = np.load(source/'geometry.npz')
    xyz, labels = g['xyz'], g['instance']
    rows = read(source/'classification.json')['objects']
    kept = [r['object_id'] for r in rows if r['predictions']['mean']['retained']]
    fg = np.flatnonzero(np.isin(labels, kept))
    points, ids = xyz[fg], labels[fg]
    owners = {int(o):r['object_id'] for r in rows for o in r['observation_ids']}
    poses = np.load(cache/'poses.npy')
    config = CleanupConfig(**p['cleanup'])
    positive = np.zeros(len(fg), np.uint16)
    negative = np.zeros(len(fg), np.uint16)
    hit = np.zeros(len(fg), np.uint16)
    seen = np.zeros(len(fg), np.uint16)
    known = np.zeros(len(fg), np.uint16)
    events = []
    for index, (frame, rec) in enumerate(zip(job['frames'], m['records'])):
        assert frame['frame_id'] == rec['frame_id']
        path = cache/'frames'/rec['file']
        assert sha(path) == rec['sha256']
        depth = np.load(frame['measured'])['depth']
        with np.load(path) as f: masks = np.unpackbits(f['masks_packed'], axis=2)[:, :, :m['config']['cam']['W']].astype(bool)
        assigned = np.array([owners.get(rec['observation_offset']+i, 0) for i in range(rec['masks'])])
        vp, vn, visible = measured_frame_evidence(points, ids, poses[index], m['config']['cam'], depth, masks, assigned, config)
        positive += vp.astype(np.uint16)
        negative += vn.astype(np.uint16)
        x, y, _, valid = project_measured_visibility(points, poses[index], m['config']['cam'], depth, config)
        any_mask = masks.any(0)
        for oid in np.intersect1d(assigned, kept):
            local = np.flatnonzero(valid & (ids == oid))
            union = masks[assigned == oid].any(0)
            seen[local] += 1
            hit[local] += union[y[local], x[local]].astype(np.uint16)
            known[local] += any_mask[y[local], x[local]].astype(np.uint16)
        events.append(dict(frame_id=rec['frame_id'], visible=int(visible.sum()), positive=int(vp.sum()), negative=int(vn.sum())))
    rejected = dict(sor=statistical_outliers(points, ids, config),
        visibility1=rejection_mask(positive, negative, config, single_frame=True),
        simple_ratio=(seen > 0) & (hit <= .5*seen), mc_point_ratio=hit <= .5*(known+1e-6))
    target = OUT/'maps/cleanup'
    target.mkdir(exist_ok=False)
    results = {}
    for name, remove in rejected.items():
        dest = target/name
        dest.mkdir()
        retain = np.ones(len(xyz), bool)
        retain[fg] = ~remove
        np.savez_compressed(dest/'geometry.npz', xyz=xyz[retain], instance=labels[retain])
        np.save(dest/'retained_source_indices.npy', np.flatnonzero(retain))
        shutil.copyfile(source/'classification.json', dest/'classification.json')
        results[name] = dict(foreground_points=len(fg), removed=int(remove.sum()), retained=int((~remove).sum()))
    np.savez_compressed(target/'evidence.npz', source_indices=fg, positive=positive, negative=negative, hit=hit, seen=seen, known=known)
    dump(target/'events.json', events)
    dump(target/'run.json', dict(complete=True, no_gt=True, results=results,
        source_geometry_sha256=sha(source/'geometry.npz'), source_classification_sha256=sha(source/'classification.json'),
        protocol_sha256=sha(OUT/'mapping_protocol.json'),
        files_sha256={str(path.relative_to(target)):sha(path) for path in target.rglob('*') if path.is_file()}))
    print('CLEANUP', results, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'taxonomy', 'geometry', 'cg', 'cleanup'])
    args = parser.parse_args()
    globals()[args.action]()
