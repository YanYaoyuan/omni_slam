"""Frozen development experiment: same-ownership filters and reversible splits.

Prediction only reads RGB-D frontend caches and the frozen source map. Scoring
is a separate action, allowed only after both scenes' predictions are sealed.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import platform
import resource
import time
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from paper_final_common import ROOT, read, dump, sha
from omni_semantic_map.reversible_split import (
    SplitConfig, EvidenceView, propose_splits, validate_split, apply_splits, rollback,
)
from omni_semantic_map.visibility_cleanup import CleanupConfig, project_visibility

OUT = ROOT / 'build/benchmarks/reversible_revision_20261113'
KEYS = ['replica_office_0', 'scenenn_014']
ARMS = ['simple_ratio', 'mc_presence_ratio', 'geometric_split', 'consensus_split', 'validated_split']


def freeze():
    assert not (OUT / 'protocol.json').exists()
    previous = read(ROOT / 'build/benchmarks/visibility_cleanup_20261112/protocol.json')
    files = [Path(__file__), ROOT/'omni_semantic_map/omni_semantic_map/reversible_split.py',
             ROOT/'omni_semantic_map/omni_semantic_map/visibility_cleanup.py',
             ROOT/'omni_semantic_map/test/test_reversible_split.py']
    for p in files:
        dest = OUT / 'sources' / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
    dump(OUT / 'protocol.json', dict(created_epoch=time.time(),
        scenes=[s for s in previous['scenes'] if s['key'] in KEYS],
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in files},
        split_config=asdict(SplitConfig()), visibility_config=asdict(CleanupConfig()),
        arms=ARMS, maximum_new_geometry_outputs=10, maximum_bugfix_replacement_outputs=2,
        development_only=True, test_scene_reuse=False, no_training=True,
        proposal_scope='One proposal per object using first 50% of frames; complete input map is offline, not a causal online reconstruction.',
        validation_scope='Last 35% of frames; recompute correspondence to raw eroded masks; 12 degree and .15m separation from proposal and previously counted cameras; no validation-ranked proposal search.',
        filter_scope='Both filters use depth-verified raw own-mask incidence in object-associated frames. Simple denominator: any depth-valid observation. MC-style denominator: depth-valid incidence in any known mask. Retain ratio > .5; zero evidence kept for simple and removed for MC-style, matching its point-rule convention. No native object retention, clustering or split in filter arms. MC-style is an adapted isolated filter, not an end-to-end reproduction.',
        geometric_control='Two largest connected components of .1m neighbor graph over .05m source voxels; at least 12 source points each; remaining source points assigned to nearest seed. No GT.',
        runtime='CPU, OpenCV one thread, BLAS one thread requested by runner invocation; no GPU calls.',
        stop='Score complete scene outputs once. No threshold changes or scene replacement after scoring. Failed acceptance is reported, not called positive evidence.'))


def verify():
    p = read(OUT / 'protocol.json')
    for f, h in p['source_sha256'].items():
        assert sha(ROOT / f) == h, f
    return p


def build_views(scene, xyz, labels, rows, cfg):
    cache = Path(scene['cache'])
    manifest = read(cache / 'manifest.json')
    for name, expected in manifest['files_sha256'].items():
        assert sha(cache / name) == expected, name
    camera = manifest['config']['cam']
    poses = np.load(cache / 'poses.npy')
    kept = [r['object_id'] for r in rows if r['predictions']['mean']['retained']]
    foreground = np.flatnonzero(np.isin(labels, kept))
    owner = {oid:r['object_id'] for r in rows for oid in r['observation_ids']}
    counts = np.zeros((3, len(xyz)), np.uint16)
    views, events = [], []
    kernel = np.ones((2*cfg.boundary_pixels+1,)*2, np.uint8)
    for i, (rec, dep) in enumerate(zip(manifest['records'], scene['depths'])):
        assert rec['frame_id'] == dep['frame_id']
        path = cache / 'frames' / rec['file']
        assert sha(path) == rec['sha256'] and sha(dep['path']) == dep['sha256']
        depth = cv2.imread(dep['path'], cv2.IMREAD_UNCHANGED).astype(np.float32) / camera['depth_scale']
        with np.load(path) as z:
            masks = np.unpackbits(z['masks_packed'], axis=2)[:, :, :camera['W']].astype(bool)
        start = time.perf_counter()
        x, y, _, valid = project_visibility(xyz[foreground], poses[i], camera, depth, cfg)
        ids = foreground[valid]
        px, py = x[valid], y[valid]
        raw = masks[:, py, px]
        known = raw.any(0)
        observation_ids = np.arange(rec['observation_offset'], rec['observation_offset'] + len(masks))
        owners = np.array([owner.get(int(o), 0) for o in observation_ids])
        for oid in np.unique(owners):
            if oid not in kept:
                continue
            local = np.flatnonzero(labels[ids] == oid)
            own = raw[owners == oid].any(0)
            counts[0, ids[local]] += own[local]
            counts[1, ids[local]] += 1
            counts[2, ids[local]] += known[local]
        eroded = np.array([cv2.erode(m.astype(np.uint8), kernel,
                          borderType=cv2.BORDER_CONSTANT, borderValue=0)[py, px] > 0 for m in masks])
        if len(masks) == 0:
            eroded = np.empty((0, len(ids)), bool)
        views.append(EvidenceView(int(rec['frame_id']), poses[i, :3, 3], ids, eroded, observation_ids))
        events.append(dict(frame_id=int(rec['frame_id']), visible_source_points=len(ids),
                           kernel_seconds=time.perf_counter()-start))
    assert len(views) == len(manifest['records']) == len(poses)
    return views, counts, foreground, events


def geometric_proposals(xyz, labels, foreground):
    proposals = []
    for oid in np.unique(labels[foreground]):
        ids = foreground[labels[foreground] == oid]
        if len(ids) < 24:
            continue
        _, chosen, reverse = np.unique(np.floor(xyz[ids] / .05).astype(np.int64), axis=0,
                                       return_index=True, return_inverse=True)
        pts = xyz[ids[chosen]]
        pairs = cKDTree(pts).query_pairs(.1, output_type='ndarray')
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(pts), len(pts)))
        _, group = connected_components(graph, directed=False)
        group = group[reverse]
        groups, sizes = np.unique(group, return_counts=True)
        order = np.argsort(-sizes, kind='stable')
        if len(order) < 2 or sizes[order[1]] < 12:
            continue
        a, b = ids[group == groups[order[0]]], ids[group == groups[order[1]]]
        da = cKDTree(xyz[a]).query(xyz[ids], workers=1)[0]
        db = cKDTree(xyz[b]).query(xyz[ids], workers=1)[0]
        proposals.append(dict(object_id=int(oid), source_indices=ids, child=(db < da).astype(np.uint8), frame_id=-1))
    return proposals


def classify_partition(rows, labels, views, lineage):
    result = deepcopy(rows)
    by_id = {r['object_id']:r for r in result}
    for event in lineage:
        parent, child = event['parent'], event['child']
        original = by_id[parent]
        new = deepcopy(original)
        new['object_id'] = child
        allowed = set(original['observation_ids'])
        child_obs = set()
        for view in views:
            for i, oid in enumerate(view.observation_ids):
                if int(oid) not in allowed:
                    continue
                ids = view.indices[view.membership[i]]
                if (labels[ids] == child).sum() > (labels[ids] == parent).sum():
                    child_obs.add(int(oid))
        original['observation_ids'] = sorted(allowed - child_obs)
        new['observation_ids'] = sorted(child_obs)
        for r in [original, new]:
            r['partition_note'] = 'Inherited class/readout, revised observation ownership by source-point incidence; old selected_observation_ids and view statistics removed.'
            for field in ['selected_observation_ids', 'unique_frames', 'center']:
                r.pop(field, None)
        result.append(new)
    return result


def predict(key):
    protocol = verify()
    scene = next(s for s in protocol['scenes'] if s['key'] == key)
    target = OUT / 'runs' / key
    target.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    cv2.setNumThreads(1)
    for field in ['geometry', 'classification']:
        assert sha(scene[field]) == scene['immutable_hashes'][scene[field]]
    source = np.load(scene['geometry'])
    xyz, labels = source['xyz'], source['instance']
    rows = read(scene['classification'])['objects']
    views, counts, foreground, events = build_views(scene, xyz, labels, rows, CleanupConfig(**protocol['visibility_config']))
    split_cfg = SplitConfig(**protocol['split_config'])
    tic = time.perf_counter()
    proposals = propose_splits(xyz, labels, views, split_cfg)
    propose_s = time.perf_counter() - tic
    tic = time.perf_counter()
    decisions = [validate_split(p, labels, views, split_cfg) for p in proposals]
    validate_s = time.perf_counter() - tic
    geometry_proposals = geometric_proposals(xyz, labels, foreground)
    log = []
    for p, d in zip(proposals, decisions):
        log.append(dict(object_id=p['object_id'], proposal_frame=p['frame_id'],
                        proposal_observations=p['observation_ids'], proposal_score=p['proposal_score'],
                        parts=[int((p['child'] == k).sum()) for k in [0, 1]], validation=d))
    dump(target / 'proposals.json', log)
    dump(target / 'frame_events.json', events)
    outputs = {}
    for name in ARMS:
        keep = np.ones(len(labels), bool)
        lineage = []
        modified = labels.copy()
        if name in ['simple_ratio', 'mc_presence_ratio']:
            denom = counts[1 if name == 'simple_ratio' else 2]
            reject = counts[0] <= .5 * (denom + (1e-6 if name == 'mc_presence_ratio' else 0))
            if name == 'simple_ratio':
                reject &= denom > 0
            keep[foreground] = ~reject[foreground]
        else:
            candidate = geometry_proposals if name == 'geometric_split' else proposals
            accept = [True]*len(candidate) if name != 'validated_split' else [d['accepted'] for d in decisions]
            modified, lineage = apply_splits(labels, candidate, accept)
            assert np.array_equal(rollback(modified, lineage), labels)
        destination = target / name
        destination.mkdir()
        np.savez_compressed(destination / 'geometry.npz', xyz=xyz[keep], instance=modified[keep])
        np.save(destination / 'retained_source_indices.npy', np.flatnonzero(keep))
        records = classify_partition(rows, modified, views, lineage) if lineage else rows
        dump(destination / 'classification.json', dict(objects=records))
        dump(destination / 'lineage.json', [dict(e, changed_source_indices=e['changed_source_indices'].tolist()) for e in lineage])
        outputs[name] = dict(removed=int((~keep).sum()), split_objects=len(lineage),
                             points=int(keep.sum()), geometry_sha256=sha(destination/'geometry.npz'))
    dump(target / 'run.json', dict(complete=True, no_gt_decoded=True, scene=key,
        protocol_sha256=sha(OUT/'protocol.json'), proposals=len(proposals),
        accepted=sum(d['accepted'] for d in decisions), methods=outputs,
        proposal_seconds=propose_s, validation_seconds=validate_s,
        total_seconds=time.perf_counter()-start,
        cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        platform=platform.platform(), processor=platform.processor(),
        artifacts={str(p.relative_to(target)):sha(p) for p in target.rglob('*') if p.is_file()}))
    print(key, 'complete', outputs, flush=True)


def seal():
    verify()
    assert not (OUT/'predictions_complete.json').exists()
    runs = {}
    for key in KEYS:
        path = OUT/'runs'/key/'run.json'
        r = read(path)
        assert r['complete']
        for name, expected in r['artifacts'].items():
            assert sha(path.parent/name) == expected
        runs[key] = sha(path)
    dump(OUT/'predictions_complete.json', dict(complete=True, runs=runs,
         protocol_sha256=sha(OUT/'protocol.json'), created_epoch=time.time()))


def evaluate():
    from omni_semantic_map.full_ovo_evaluation import voxel_metrics
    from omni_semantic_map.mesh_instance_evaluation import transfer_labels
    from omni_semantic_map.instance_error_audit import audit_instances, filter_transition
    protocol = verify()
    gate = read(OUT/'predictions_complete.json')
    assert gate['complete'] and gate['protocol_sha256'] == sha(OUT/'protocol.json')
    assert not (OUT/'evaluation').exists()
    result = {}
    for scene in protocol['scenes']:
        key = scene['key']
        assert sha(OUT/'runs'/key/'run.json') == gate['runs'][key]
        run_record = read(OUT/'runs'/key/'run.json')
        for name, expected in run_record['artifacts'].items():
            assert sha(OUT/'runs'/key/name) == expected, name
        assert sha(scene['reference']) == scene['immutable_hashes'][scene['reference']]
        ref = np.load(scene['reference'])
        valid = ref['valid'] if 'valid' in ref.files else np.ones(len(ref['xyz']), bool)
        truth = ref['instance'][valid]
        methods, before = {}, None
        paths = {'baseline':Path(scene['geometry']).parent,
                 'sor':ROOT/'build/benchmarks/visibility_cleanup_20261112/runs'/key/'geometric_sor',
                 'visibility1':ROOT/'build/benchmarks/visibility_cleanup_20261112/runs'/key/'single_refutation'}
        paths.update({a:OUT/'runs'/key/a for a in ARMS})
        for name, path in paths.items():
            g = np.load(path/'geometry.npz')
            rows = read(path/'classification.json')['objects']
            kept = [r['object_id'] for r in rows if r['predictions']['mean']['retained']]
            labels = np.where(np.isin(g['instance'], kept), g['instance'], 0)
            projected, _ = transfer_labels(g['xyz'], labels, ref['xyz'])
            audit = audit_instances(truth, projected[valid])
            if name == 'baseline':
                before = audit
            score = voxel_metrics(truth, projected[valid])
            methods[name] = dict(score=score, transition=filter_transition(before, audit),
                predicted_counts=audit['predicted_counts'], gt_counts=audit['gt_counts'])
            folder = OUT/'evaluation'/key
            dump(folder/(name+'_audit.json'), audit)
            np.savez_compressed(folder/(name+'.npz'), instance=projected)
        result[key] = methods
        dump(OUT/'evaluation'/key/'metrics.json', methods)
        print(key, {k:round(v['score']['observed_voxel_instance_pq']*100,3) for k,v in methods.items()}, flush=True)
    dump(OUT/'summary.json', dict(complete=True, development_only=True,
         protocol_sha256=sha(OUT/'protocol.json'), results=result, new_geometry_outputs=10))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'predict', 'seal', 'evaluate'])
    parser.add_argument('--scene', choices=KEYS)
    args = parser.parse_args()
    if args.action == 'predict':
        predict(args.scene)
    else:
        globals()[args.action]()
