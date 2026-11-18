"""Bounded retrospective anchor retraction, with full-map and robot controls."""
import argparse
from pathlib import Path
import shutil
import time
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from omni_semantic_map.anchor_counterevidence import counterevidence, unit_vectors

B = ROOT / 'build/benchmarks'
OUT = B / 'anchor_counterevidence_v1'
ASSETS = B / 'utonia_evidence_v1'
METHODS = ['baseline', 'direct_only', 'ours_retract_propagation', 'matched_distance']


def verify():
    p = read(OUT / 'protocol.json')
    for f, h in {**p['input_hashes'], **p['source_hashes']}.items(): assert sha(f) == h, f
    return p


def seal(name):
    s = read(OUT / (name + '.json')); assert s['complete']
    for f, h in s['files'].items(): assert sha(f) == h, f
    return s


def freeze():
    from audit_matched_transport import inputs
    assert not OUT.exists()
    scenes = {s['key']: s for s in read(B / 'discovery_measured_inputs_v1/protocol.json')['scenes']}
    rows = []
    files = {}
    for c in inputs():
        if c['key'] == 'kitti360_0006': continue
        c = dict(c); c['job'] = scenes[c['key']]['job']; c['kind'] = scenes[c['key']]['kind']
        rows.append(c)
        paths = [Path(c['job']), Path(c['field']), Path(c['sem']), Path(c['full']) / 'geometry.npz',
                 Path(c['full']) / 'old_to_full.npy', Path(c['ours']) / 'geometry.npz',
                 Path(c['ours']) / 'classification.json', Path(c['old_summary'])]
        if c['reference']: paths.append(Path(c['reference']))
        for frame in read(c['job'])['frames']:
            paths.append(Path(frame['raw_scan'] if c['kind'] == 'outdoor' else frame['cloud']))
        for path in paths: files[str(path)] = sha(path)
    paths = [ASSETS / 'assets.json', B / 'utonia_ownership_probe_v1/summary.json', B / 'utonia_ownership_probe_v1/delivery_audit.json']
    for path in paths: files[str(path)] = sha(path)
    sources = [Path(__file__), ROOT / 'omni_semantic_map/omni_semantic_map/anchor_counterevidence.py',
               ROOT / 'omni_semantic_map/test/test_anchor_counterevidence.py',
               ROOT / 'omni_semantic_map/omni_semantic_map/batch_stable_transport.py',
               ROOT / 'omni_semantic_map/omni_semantic_map/batch_stable_transport.cpp',
               ROOT / 'scripts/benchmarks/evaluate_kitti360_development.py',
               ROOT / 'omni_semantic_map/omni_semantic_map/kitti360_evaluation.py']
    dump(OUT / 'protocol.json', dict(created_epoch=time.time(), contexts=rows, methods=METHODS,
        input_hashes=files, source_hashes={str(f): sha(f) for f in sources},
        hypothesis='Retract old foreground anchors contradicted by other acquisition frames, then retract existing map assignments whose graph support changes. This repairs existing ownership; it does not expand the failed invisible-region cosine candidate.',
        seed_features='Unmodified native Utonia base, raw whole-scan coordinates, zero color/normal, outdoor scale0.2 without centering, SDPA FP32 patch1024, seed6985480, four-level1386D. Unit-normalize per-source features then average per(cell,object,acquisition_frame). One normalized per-frame prototype per predicted object or known background semantic class. No GT feature/anchor selection.',
        robot='Archived LIVO measured tracking clouds, not original LiDAR beams. Rotate tracking points by tracking pose rotation to gravity-world axes at robot origin, no translation; source_raw_index preserved. Same feature/decision settings as public data. 330 frames, no robot GT or accuracy/real-time claim.',
        decision='For each foreground anchor unit, EXCLUDE its acquisition frame. Only compare other frames with both that same object prototype and at least one known background prototype. A frame votes refutation if max background cosine exceeds own-object cosine by>1e-6. Ties abstain but count in denominator. Retract iff>=2 comparable frames and strict majority refutes. Distinct frames are not claimed independent. No other foreground ID used in semantic decision.',
        retraction='Remove rejected anchor units from the unchanged20cm/16NN/0.4m graph seeds. A cell/object is directly revoked only if every source-frame seed for that cell/object was rejected. Recompute nearest-owner graph with remaining original foreground/background seeds; do not relabel rejected anchors as background.',
        maps='Universe is EXACT existing ours_frame map coordinates,IDs/classes/scores. No new coordinates or objects. Original fused coordinates use nearest measured-source graph-cell proxy within5cm, not a claimed original observation; exact added-source ledger reused. Direct control sets directly revoked cell/object rows to unassigned0. Ours additionally sets rows to0 where graph winner changed after retraction. No reassigning a different foreground identity. Include unchanged coordinates with label0 in every method; evaluator transfers these as abstention.',
        controls='Original baseline is the no-retraction ablation. Direct-only uses SAME evidence and revoked units but no propagation cleanup. Matched-distance retains direct-only rejection and removes the SAME additional number per object as Ours, choosing shortest Euclidean distance to revoked cell centers (row-ID ties). No new count/radius fit.',
        budget='Two already exposed public windows0000/0002 plus one330-frame robot diagnostic. At most438 shared native forwards,3new maps per context=9new maps,6new public AP scores. No training, hyperparameter/checkpoint search or additional gate-driven variants.',
        evaluation='Pinned existing locally adapted KITTI360 AP50:90/AP50, same reference and20cm transfer; not official leaderboard. All old and propagated-point removals counted separately, all GT gains/losses. Robot geometry changes only. Predictions sealed before GT read.',
        development_gate='Both public AP and AP50 >= frozen baseline; no lost AP50 matchedGT; meanAP >=baseline+0.001 and strictly above both direct-only and matched-distance. This does NOT replace earlier independent-test gates or prove paper novelty.',
        next='If failed, close this exact retraction candidate, do not alter votes/minframes/prototypes. If passed, require subsequent independent public assessment and same-module robot review before promotion.',
        scope='Retrospective refinement: predicted object assignments and prototypes come from the completed window and may use future views. Not causal online identity tracking, new-object discovery or full-SLAM innovation. Robot executes once for requested diagnostic regardless of public gate, without asserting quality improvement.'
    ))
    print('ANCHOR RETRACTION PROTOCOL FROZEN', flush=True)


def prepare():
    from scipy.spatial import cKDTree
    from audit_transport_unknown_seeds import known_semantics
    p = verify(); assert not (OUT / 'prepared.json').exists()
    files, stats = {}, {}
    for c in p['contexts']:
        key = c['key']; full = np.load(Path(c['full']) / 'geometry.npz'); field = np.load(c['field'])
        sem = known_semantics(np.load(c['sem'])); mapping = np.load(Path(c['full']) / 'old_to_full.npy')
        active = (sem['owners'] > 0) | ((sem['owners'] == 0) & (sem['seen'] > 0) & ~np.isin(sem['mean_response'], [2, 13]))
        old = np.flatnonzero(active); src = mapping[old]
        label = np.where(sem['owners'][old] > 0, sem['owners'][old], -sem['mean_response'][old] - 1)
        acquisition = full['source_frame']; raw_index = full['source_raw_index']; xyz = full['xyz']
        cell = field['source_to_cell']; records = []
        for frame in read(c['job'])['frames']:
            take = acquisition[src] == frame['frame_id']; ids = src[take]
            path = OUT / 'inputs' / key / f'{frame["frame_id"]:06d}.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, source_id=ids, raw_index=raw_index[ids], cell=cell[ids], label=label[take])
            files[str(path)] = sha(path)
            records.append(dict(frame_id=frame['frame_id'], anchors=len(ids)))
            if c['kind'] == 'robot':
                raw = np.load(frame['cloud'])['points']; t = np.asarray(frame['pose'])
                np.testing.assert_allclose(raw[raw_index[ids]] @ t[:3, :3].T + t[:3, 3], xyz[ids], rtol=0, atol=1e-9)
        base = np.load(Path(c['ours']) / 'geometry.npz')
        distance, nearest = cKDTree(xyz).query(base['xyz'], k=2, workers=2)
        assert not (distance[:, 0] == distance[:, 1]).any(), 'Ambiguous map-to-source proxy'
        assert distance[:, 0].max() < .05
        proxy = nearest[:, 0]
        added = base['added_full_source_ids']; prefix = len(proxy) - len(added)
        np.testing.assert_array_equal(base['xyz'][prefix:], xyz[added])
        proxy[prefix:] = added
        path = OUT / 'inputs' / key / 'map_proxy.npz'
        np.savez_compressed(path, source_proxy=proxy, cell=cell[proxy], distance=distance[:, 0], prefix=np.array(prefix))
        files[str(path)] = sha(path)
        stats[key] = dict(frames=records, anchors=len(src), maximum_proxy_distance=float(distance[:, 0].max()), original_rows=prefix, propagated_rows=len(added))
        print('PREPARED RETRACTION', key, len(records), 'frames', len(src), 'anchors', flush=True)
    dump(OUT / 'prepared.json', dict(complete=True, files=files, stats=stats, gt_read=False))


def means(features, keys):
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind='stable')
    _, starts, counts = np.unique(inverse[order], return_index=True, return_counts=True)
    summed = np.add.reduceat(features[order], starts, axis=0)
    return unique, unit_vectors(summed / counts[:, None]), counts


def infer():
    import torch
    import utonia
    p = verify(); seal('prepared'); assert not (OUT / 'features_complete.json').exists()
    assets = read(ASSETS / 'assets.json')
    for f, h in assets['files'].items(): assert sha(f) == h
    torch.cuda.set_per_process_memory_fraction(.85)
    utonia.utils.set_seed(6985480)
    model = utonia.load(assets['checkpoint'], custom_config=dict(enable_flash=False, enc_patch_size=[1024] * 5)).cuda().eval()
    transform = utonia.transform.default(.2, apply_z_positive=False)
    files, stats = {}, {}
    for c in p['contexts']:
        key = c['key']; stats[key] = []
        for number, frame in enumerate(read(c['job'])['frames']):
            fid = frame['frame_id']; dest = OUT / 'features' / key / f'{fid:06d}.npz'
            dest.parent.mkdir(parents=True, exist_ok=True)
            assert not dest.exists()
            inp = np.load(OUT / 'inputs' / key / f'{fid:06d}.npz')
            started = time.monotonic()
            if not len(inp['source_id']):
                np.savez_compressed(dest, units=np.empty((0, 3), np.int64), features=np.empty((0, 1386), np.float32),
                                    prototype_label=np.empty(0, np.int64), prototype_features=np.empty((0, 1386), np.float32))
                files[str(dest)] = sha(dest); stats[key].append(dict(frame_id=fid, forward=False)); continue
            if c['kind'] == 'outdoor': raw = np.fromfile(frame['raw_scan'], np.float32).reshape(-1, 4)[:, :3].copy()
            else: raw = (np.load(frame['cloud'])['points'] @ np.asarray(frame['pose'])[:3, :3].T).astype(np.float32)
            assert np.isfinite(raw).all() and inp['raw_index'].min() >= 0 and inp['raw_index'].max() < len(raw)
            utonia.utils.set_seed(6985480)
            point = transform(dict(coord=raw.copy(), color=np.zeros_like(raw), normal=np.zeros_like(raw)))
            inverse = point['inverse'].numpy().copy()
            assert np.linalg.norm(point['coord'].numpy()[inverse] - raw * .2, axis=1).max() < np.sqrt(3) * .01 + 1e-5
            with torch.inference_mode():
                point = model({k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in point.items()})
                for _ in range(4):
                    parent = point.pop('pooling_parent')
                    parent.feat = torch.cat([parent.feat, point.feat[point.pop('pooling_inverse')]], -1)
                    point = parent
                feat = unit_vectors(point.feat[inverse[inp['raw_index']]].cpu().numpy())
            labels, prototype, count = means(feat, inp['label'])
            fg = inp['label'] > 0
            if fg.any():
                units, unit_feat, unit_count = means(feat[fg], np.column_stack([inp['cell'][fg], inp['label'][fg]]))
                units = np.column_stack([units, np.full(len(units), fid)])
            else:
                units, unit_feat, unit_count = np.empty((0, 3), np.int64), np.empty((0, 1386), np.float32), np.empty(0, np.int64)
            np.savez_compressed(dest, units=units, features=unit_feat, unit_count=unit_count,
                                prototype_label=labels, prototype_features=prototype, prototype_count=count)
            files[str(dest)] = sha(dest)
            stats[key].append(dict(frame_id=fid, forward=True, points=len(raw), foreground_units=len(units),
                                   seconds=time.monotonic() - started, peak_gib=torch.cuda.max_memory_allocated() / 2**30))
            del point, feat
            if number % 10 == 0: print('RETRACTION FEATURES', key, number + 1, stats[key][-1], flush=True)
    dump(OUT / 'features_complete.json', dict(complete=True, files=files, stats=stats, gt_read=False))


def decide():
    p = verify(); seal('features_complete'); assert not (OUT / 'decisions_complete.json').exists()
    files, stats = {}, {}
    for c in p['contexts']:
        key = c['key']; units, features, prototypes, labels, frames = [], [], [], [], []
        for frame in read(c['job'])['frames']:
            d = np.load(OUT / 'features' / key / f'{frame["frame_id"]:06d}.npz')
            units.append(d['units']); features.append(d['features']); prototypes.append(d['prototype_features']); labels.append(d['prototype_label'])
            frames.append(np.full(len(d['prototype_label']), frame['frame_id']))
        units = np.concatenate(units); features = np.concatenate(features)
        field = np.load(c['field']); expected = field['seeds'][field['seeds'][:, 1] > 0]
        np.testing.assert_array_equal(np.unique(units, axis=0), expected)
        started = time.monotonic()
        result = counterevidence(features, units[:, 1], units[:, 2], np.concatenate(prototypes), np.concatenate(labels), np.concatenate(frames))
        path = OUT / 'decisions' / key / 'anchors.npz'; path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, units=units, **result); files[str(path)] = sha(path)
        stats[key] = dict(units=len(units), comparable_at_least_two=int((result['eligible'] >= 2).sum()), retracted=int(result['retract'].sum()), seconds=time.monotonic() - started)
        print('RETRACTION DECISIONS', key, stats[key], flush=True)
    dump(OUT / 'decisions_complete.json', dict(complete=True, files=files, stats=stats, gt_read=False))


def pair_keys(rows):
    a = np.ascontiguousarray(rows, dtype=np.int64)
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()


def maps():
    from scipy.spatial import cKDTree
    from audit_matched_transport import graph_from
    from audit_transport_unknown_seeds import known_semantics
    from omni_semantic_map.batch_stable_transport import native_library, nearest_groups, nearest_owner
    p = verify(); seal('prepared'); seal('decisions_complete'); assert not (OUT / 'predictions_complete.json').exists()
    lib = native_library(OUT / 'native'); files, results = {}, {}
    for c in p['contexts']:
        key = c['key']; field = np.load(c['field']); graph = graph_from(field)
        full = np.load(Path(c['full']) / 'geometry.npz'); mapping = np.load(Path(c['full']) / 'old_to_full.npy')
        sem = known_semantics(np.load(c['sem']))
        active = (sem['owners'] > 0) | ((sem['owners'] == 0) & (sem['seen'] > 0) & ~np.isin(sem['mean_response'], [2, 13]))
        ids = mapping[active]
        seeds = np.unique(np.column_stack([field['source_to_cell'][ids], sem['owners'][active], full['source_frame'][ids]]), axis=0)
        decision = np.load(OUT / 'decisions' / key / 'anchors.npz'); units = decision['units']; rejected = units[decision['retract']]
        keep = ~np.isin(pair_keys(seeds), pair_keys(rejected))
        assert (~keep).sum() == len(rejected)
        remaining = seeds[keep]
        original_labels, original_dist = nearest_groups(graph, seeds[:, 0], seeds[:, 1], lib)
        original = nearest_owner(original_labels, original_dist)
        labels, dist = nearest_groups(graph, remaining[:, 0], remaining[:, 1], lib)
        revised = nearest_owner(labels, dist)
        revoked = np.unique(rejected[:, :2], axis=0)
        revoked = revoked[~np.isin(pair_keys(revoked), pair_keys(remaining[:, :2]))]
        base = np.load(Path(c['ours']) / 'geometry.npz'); proxy = np.load(OUT / 'inputs' / key / 'map_proxy.npz')
        cell = proxy['cell']; old = base['instance']; prefix = int(proxy['prefix'])
        direct = np.isin(pair_keys(np.column_stack([cell, old])), pair_keys(revoked))
        ours = direct | (original[cell] != revised[cell])
        matched = direct.copy()
        for oid in np.unique(old):
            need = int(((old == oid) & ours & ~direct).sum())
            if not need: continue
            revoked_nodes = revoked[revoked[:, 1] == oid, 0]
            # Propagation can change another object's boundary even when that
            # object has no revoked seed: use all revoked centers in that case.
            if not len(revoked_nodes): revoked_nodes = revoked[:, 0]
            assert len(revoked_nodes)
            candidates = np.flatnonzero((old == oid) & ~direct)
            distance = cKDTree(field['centers'][revoked_nodes]).query(base['xyz'][candidates], workers=1)[0]
            chosen = candidates[np.lexsort((candidates, distance))[:need]]
            assert len(chosen) == need; matched[chosen] = True
        np.testing.assert_array_equal(np.bincount(old[ours]), np.bincount(old[matched]))
        masks = dict(baseline=np.zeros(len(old), bool), direct_only=direct, ours_retract_propagation=ours, matched_distance=matched)
        rows = {}
        for method, mask in masks.items():
            dest = OUT / 'maps' / key / method; dest.mkdir(parents=True, exist_ok=False)
            labels = old.copy(); labels[mask] = 0
            np.savez_compressed(dest / 'geometry.npz', xyz=base['xyz'], instance=labels,
                                original_row_id=np.arange(len(old)), retracted_rows=np.flatnonzero(mask))
            shutil.copyfile(Path(c['ours']) / 'classification.json', dest / 'classification.json')
            rows[method] = dict(geometry=str(dest / 'geometry.npz'), classification=str(dest / 'classification.json'),
                               removed=int(mask.sum()), removed_original=int(mask[:prefix].sum()), removed_propagated=int(mask[prefix:].sum()),
                               per_object=[dict(object_id=int(oid), before=int((old == oid).sum()), removed=int(((old == oid) & mask).sum())) for oid in np.unique(old)])
            for f in dest.iterdir(): files[str(f)] = sha(f)
        path = OUT / 'decisions' / key / 'propagation.npz'
        np.savez_compressed(path, seeds=seeds, retained_seeds=remaining, original_winner=original, revised_winner=revised, revoked_cell_owner=revoked)
        files[str(path)] = sha(path)
        results[key] = dict(methods=rows, rejected_units=len(rejected), revoked_cell_owner=len(revoked), changed_graph_cells=int((original != revised).sum()))
        print('RETRACTION MAPS', key, {m: r['removed'] for m, r in rows.items()}, flush=True)
    dump(OUT / 'predictions_complete.json', dict(complete=True, files=files, results=results, new_maps=9, gt_read=False))


def score():
    from evaluate_kitti360_development import transfer, clean_json
    from omni_semantic_map.kitti360_evaluation import evaluate
    from run_discovery_query_benchmark import matched_ids
    p = verify(); predictions = seal('predictions_complete'); assert not (OUT / 'summary.json').exists()
    results, files = {}, {}; means = {m: [] for m in METHODS}; gate = True
    for c in p['contexts']:
        key = c['key']
        if not c['reference']:
            results[key] = dict(semantic_accuracy=None, reason='Robot has no independent GT', maps=predictions['results'][key]); continue
        ref = np.load(c['reference']); rows = {}
        baseline = read(c['old_summary'])['results'][key]['methods']['ours_frame']
        for method in METHODS:
            row = predictions['results'][key]['methods'][method]
            if method == 'baseline':
                rows[method] = dict(metrics=baseline['metrics'], matched_gt_ids=baseline['matched_gt_ids'], reused_frozen_score=True)
            else:
                g = np.load(row['geometry']); cls = read(row['classification'])['objects']
                pred = transfer(g['xyz'], g['instance'], ref['xyz'], .2)
                scored = evaluate(ref['train_instance'], ref['confidence'], pred,
                                  {o['object_id']: o['predictions']['mean'] for o in cls},
                                  ROOT / 'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py')
                folder = Path(row['geometry']).parent
                dump(folder / 'matches.json', clean_json(scored['matches']))
                np.savez_compressed(folder / 'reference_prediction.npz', instance=pred)
                for f in [folder / 'matches.json', folder / 'reference_prediction.npz']: files[str(f)] = sha(f)
                rows[method] = dict(metrics=clean_json(scored['averages']), matched_gt_ids=sorted(matched_ids(scored['matches'])))
            means[method].append(rows[method]['metrics']['allAp'])
            print('RETRACTION SCORE', key, method, rows[method]['metrics'], flush=True)
        ours, base = rows['ours_retract_propagation'], rows['baseline']
        gate &= ours['metrics']['allAp'] >= base['metrics']['allAp'] and ours['metrics']['allAp50%'] >= base['metrics']['allAp50%']
        gate &= set(base['matched_gt_ids']).issubset(ours['matched_gt_ids'])
        results[key] = dict(methods=rows)
    mean = {m: float(np.mean(v)) for m, v in means.items()}
    gate &= mean['ours_retract_propagation'] >= mean['baseline'] + .001
    gate &= all(mean['ours_retract_propagation'] > mean[m] for m in ['direct_only', 'matched_distance'])
    dump(OUT / 'summary.json', dict(complete=True, results=results, means=mean, development_gate=bool(gate),
                                   files=files, new_public_scores=6, predictions_sha256=sha(OUT / 'predictions_complete.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'prepare', 'infer', 'decide', 'maps', 'score'])
    globals()[parser.parse_args().action]()
