"""Finite, posthoc equal-coverage diagnosis on already exposed maps.

No new perception, fitting, thresholds, windows or candidate promotion. Fixed
geometry and identities: deletion removes acquisition-frame anchors only.
"""
import argparse
import shutil
import time
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from paper_final_common import ROOT, read, dump, sha
from audit_observation_anchor_provenance import contexts
from audit_transport_unknown_seeds import known_semantics
from omni_semantic_map.batch_stable_transport import native_library, nearest_groups, nearest_owner, certify
from omni_semantic_map.matched_transport_audit import confidence, matched_selection, summarize_changes

B = ROOT / 'build/benchmarks'
OUT = B / 'matched_transport_audit_v1'
METHODS = ['ours_frame', 'nearest_distance', 'normalized_margin']


def inputs():
    refs = {s['key']: s['reference'] for s in read(B / 'batch_stable_transport_v1/protocol.json')['scenes']}
    rows = []
    for context in contexts():
        c = {k: str(v) for k, v in context.items() if k in ['key', 'graph', 'field', 'full', 'sem']}
        key = c['key']
        if key in refs:
            c['ours'] = str(B / 'batch_stable_transport_v1/maps' / key / 'ours_frame')
            c['reference'] = refs[key]
            c['old_summary'] = str(B / 'batch_stable_transport_v1/summary.json')
        else:
            root = B / ('batch_transport_robot_v1' if key == 'rosbag2245' else 'batch_transport_assessment_v1')
            c['ours'] = str(root / 'final_maps' / key / 'ours_frame')
            c['reference'] = None if key == 'rosbag2245' else str(root / 'evaluation' / key / 'reference.npz')
            c['old_summary'] = str(root / 'summary.json')
        rows.append(c)
    return rows


def freeze():
    assert not OUT.exists(), 'Never replace an existing diagnosis'
    rows = inputs(); files = {}
    for c in rows:
        paths = [c['field'], c['sem'], c['old_summary'], Path(c['full']) / 'geometry.npz', Path(c['full']) / 'old_to_full.npy']
        for folder in ['graph', 'ours']:
            paths += [Path(c[folder]) / name for name in ['geometry.npz', 'classification.json']]
        if c['reference']: paths.append(c['reference'])
        for f in paths: files[str(f)] = sha(f)
    sources = [Path(__file__), ROOT / 'omni_semantic_map/omni_semantic_map/matched_transport_audit.py',
               ROOT / 'omni_semantic_map/test/test_matched_transport_audit.py',
               ROOT / 'omni_semantic_map/omni_semantic_map/batch_stable_transport.py',
               ROOT / 'omni_semantic_map/omni_semantic_map/batch_stable_transport.cpp',
               ROOT / 'scripts/benchmarks/audit_transport_unknown_seeds.py',
               ROOT / 'scripts/benchmarks/evaluate_kitti360_development.py',
               ROOT / 'omni_semantic_map/omni_semantic_map/kitti360_evaluation.py',
               ROOT / 'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py']
    dump(OUT / 'protocol.json', dict(created_epoch=time.time(), contexts=rows, methods=METHODS,
        input_hashes=files, source_hashes={str(f): sha(f) for f in sources},
        scope='Posthoc mechanism diagnosis, all four scenes already exposed. No independent test, new candidate or replacement of earlier failed gates. 0007 remains N/A in its original protocol, not replaced here.',
        selection='Universe: all originally unowned measured sources with a unique foreground nearest owner on the identical corrected graph. For EACH object retain exactly as many sources as frozen Ours. Two fixed controls: descending negative nearest-owner distance; descending (competitor-own)/(competitor+own), infinite competitor=1. Ties use ascending full source ID. No radius or threshold selection; partial cells permitted to satisfy exact point budget. Existing points/IDs/classscores immutable.',
        stability='For EVERY source acquisition frame that supplies any seed, remove all of that frame\'s foreground/background anchors, keep geometry and all other seeds fixed, and recompute two nearest owners. Audit fixed preselected sources; count other foreground, background and abstention separately. Certificate is not recomputed admission; no raw scan removal, pose rerun, statistical independence or semantic-correctness guarantee.',
        metric='Unique source fraction changing in at least one deletion, mean fraction per deletion, macro-object vulnerable fraction. Also actual full-map AP50:90 and AP50 under unchanged evaluation. Counts do not measure spatial coverage equivalence. Zero flips alone cannot pass a method gate.',
        validation='Unknown semantics stay unknown. Reconstruct corrected seeds and replay frozen foreground decisions. Per-object matched counts and old map prefix exact. All source-frame deletions; four temporal-quantile deletions also independently checked at fixed nodes via SciPy per-seed distances, including ties/unreachable. Seal all eight control maps before reading reference arrays.',
        budget='Four exposed contexts x two new matched controls = eight maps; four frozen Ours replayed. Six control public scores and three Ours score replays; no robot GT. No training or foundation-model calls. No additional controls or sweeps after outcomes.',
        decision='Report both stability and clean quality, all per-object gains/losses. If certificate is dominated or has no meaningful stability advantage, close this hard-certificate direction. If a tradeoff exists, report only that tradeoff, not a passed clean-data gate or proved paper contribution.'))
    print('MATCHED TRANSPORT FROZEN', flush=True)


def verify():
    p = read(OUT / 'protocol.json')
    for f, h in {**p['input_hashes'], **p['source_hashes']}.items(): assert sha(f) == h, f
    return p


def checked_seal(name):
    s = read(OUT / (name + '.json')); assert s['complete']
    for f, h in s['files'].items(): assert sha(f) == h, f
    return s


def graph_from(d):
    n = len(d['centers'])
    return csr_matrix((d['graph_weights'], d['graph_indices'], d['graph_indptr']), shape=(n, n))


def maps():
    p = verify(); assert not (OUT / 'predictions_complete.json').exists()
    lib = native_library(OUT / 'native'); files = {}; stats = {}
    for c in p['contexts']:
        key = c['key']; folder = OUT / key; folder.mkdir(parents=True, exist_ok=False)
        field = np.load(c['field']); full = np.load(Path(c['full']) / 'geometry.npz')
        mapping = np.load(Path(c['full']) / 'old_to_full.npy'); sem = known_semantics(np.load(c['sem']))
        source_cells = field['source_to_cell']; source_frames = full['source_frame']
        owners = np.zeros(len(source_cells), np.int64); owners[mapping] = sem['owners']
        active = (sem['owners'] > 0) | ((sem['owners'] == 0) & (sem['seen'] > 0) & ~np.isin(sem['mean_response'], [2, 13]))
        src = mapping[active]
        seeds = np.unique(np.column_stack([source_cells[src], owners[src], source_frames[src]]), axis=0)
        graph = graph_from(field)
        labels, distances = nearest_groups(graph, seeds[:, 0], seeds[:, 1], lib)
        winner = nearest_owner(labels, distances)
        # Unknown-seed correction never changes foreground exports in these runs.
        np.testing.assert_array_equal(np.where(winner > 0, winner, -1), np.where(field['nearest_owner'] > 0, field['nearest_owner'], -1))
        pairs, tokens = np.unique(seeds[:, [1, 2]], axis=0, return_inverse=True)
        token, td = nearest_groups(graph, seeds[:, 0], tokens, lib)
        certified = certify(labels, distances, token, td, pairs[:, 0])
        np.testing.assert_array_equal(np.where(certified > 0, certified, -1), np.where(field['frame_owner'] > 0, field['frame_owner'], -1))
        base = np.load(Path(c['graph']) / 'geometry.npz'); ours = np.load(Path(c['ours']) / 'geometry.npz')
        nbase = len(base['xyz']); target = ours['added_full_source_ids']; target_owner = ours['instance'][nbase:]
        np.testing.assert_array_equal(target, np.flatnonzero((owners == 0) & (certified[source_cells] > 0)))
        np.testing.assert_array_equal(target_owner, winner[source_cells[target]])
        np.testing.assert_array_equal(ours['xyz'][:nbase], base['xyz']); np.testing.assert_array_equal(ours['instance'][:nbase], base['instance'])
        np.testing.assert_array_equal(ours['xyz'][nbase:], full['xyz'][target])
        assert sha(Path(c['ours']) / 'classification.json') == sha(Path(c['graph']) / 'classification.json')
        candidates = np.flatnonzero((owners == 0) & (winner[source_cells] > 0))
        selections = {'ours_frame': target}; stats[key] = {}
        for mode in METHODS[1:]:
            selections[mode] = matched_selection(candidates, winner[source_cells[candidates]],
                confidence(distances[source_cells[candidates]], mode), target, target_owner)
        # Only derived distances/seeds are persisted; original graph remains sealed.
        np.savez_compressed(folder / 'readout.npz', seeds=seeds, nearest_owner=winner, certified_owner=certified,
                            label_distance=distances, **{m + '_sources': s for m, s in selections.items()})
        files[str(folder / 'readout.npz')] = sha(folder / 'readout.npz')
        target_counts = np.bincount(target_owner, minlength=int(owners.max()) + 1)
        for method, selected in selections.items():
            assigned = winner[source_cells[selected]]
            np.testing.assert_array_equal(np.bincount(assigned, minlength=len(target_counts)), target_counts)
            dest = folder / method; dest.mkdir()
            if method == 'ours_frame':
                # Reuse existing sealed map; no silent copy labelled as new output.
                map_path = Path(c['ours']) / 'geometry.npz'
                cls_path = Path(c['ours']) / 'classification.json'
            else:
                map_path = dest / 'geometry.npz'; cls_path = dest / 'classification.json'
                np.savez_compressed(map_path, xyz=np.concatenate([base['xyz'], full['xyz'][selected]]),
                                    instance=np.r_[base['instance'], assigned], added_full_source_ids=selected)
                shutil.copyfile(Path(c['graph']) / 'classification.json', cls_path)
            overlap = len(np.intersect1d(selected, target, assume_unique=True))
            row = dict(geometry=str(map_path), classification=str(cls_path),
                       added_sources=len(selected), count_by_object={str(i): int(v) for i, v in enumerate(target_counts) if v},
                       overlap_with_ours=overlap, symmetric_source_difference=2 * (len(target) - overlap),
                       candidate_sources=len(candidates), per_object_count_exact=True)
            dump(dest / 'map.json', row); stats[key][method] = row
            for path in [map_path, cls_path, dest / 'map.json']: files[str(path)] = sha(path)
        print('MATCHED MAPS', key, len(target), 'sources;', {m: r['symmetric_source_difference'] for m, r in stats[key].items()}, flush=True)
    dump(OUT / 'predictions_complete.json', dict(complete=True, files=files, results=stats,
        protocol_sha256=sha(OUT / 'protocol.json'), references_read_by_this_stage=False, new_control_maps=8, new_model_calls=0))


def stability():
    p = verify(); checked_seal('predictions_complete'); lib = native_library(OUT / 'native')
    assert not (OUT / 'stability_complete.json').exists(); results = {}; files = {}
    for c in p['contexts']:
        key = c['key']; folder = OUT / key
        field = np.load(c['field']); state = np.load(folder / 'readout.npz'); graph = graph_from(field)
        seeds = state['seeds']; before = state['nearest_owner']; frames = np.unique(seeds[:, 2])
        cells = field['source_to_cell']; failures = np.zeros(len(before), np.int32)
        first_frame = np.full(len(before), -1, np.int64); first_after = first_frame.copy()
        fg = before > 0; certified = state['certified_owner'] > 0
        selected = {m: cells[state[m + '_sources']] for m in METHODS}
        sample_frames = set(frames[np.linspace(0, len(frames) - 1, min(4, len(frames)), dtype=int)].tolist())
        sample_nodes = []
        for mask in [certified, fg & ~certified, before == 0, before < 0]:
            ids = np.flatnonzero(mask)
            if len(ids): sample_nodes.extend(ids[np.linspace(0, len(ids) - 1, min(2, len(ids)), dtype=int)].tolist())
        sample_nodes = np.unique(sample_nodes)
        sample_paths = dijkstra(graph, indices=sample_nodes)[:, seeds[:, 0]]
        frame_rows = []; start = time.monotonic(); last = start
        for j, frame in enumerate(frames):
            keep = seeds[:, 2] != frame; remaining = seeds[keep]
            labels, distance = nearest_groups(graph, remaining[:, 0], remaining[:, 1], lib)
            after = nearest_owner(labels, distance)
            changed = fg & (after != before)
            assert not changed[certified].any(), 'Certificate counterexample'
            fresh = changed & (failures == 0); first_frame[fresh] = frame; first_after[fresh] = after[fresh]
            failures[changed] += 1
            if int(frame) in sample_frames:
                unique = np.unique(remaining[:, 1]); cost = np.full((len(sample_nodes), len(unique)), np.inf)
                for k, owner in enumerate(unique): cost[:, k] = sample_paths[:, keep & (seeds[:, 1] == owner)].min(axis=1)
                order = np.argsort(cost, axis=1, kind='stable')[:, :2]
                for q, node in enumerate(sample_nodes):
                    ds = np.full(2, np.inf); ls = np.full(2, -1, np.int64)
                    for r, k in enumerate(order[q]):
                        if np.isfinite(cost[q, k]): ds[r] = cost[q, k]; ls[r] = unique[k]
                    np.testing.assert_allclose(distance[node], ds, rtol=1e-12, atol=1e-10)
                    assert after[node] == nearest_owner(ls[None], ds[None])[0]
            rows = {}
            for m, s in selected.items():
                bad = after[s] != before[s]
                rows[m] = dict(changed=int(bad.sum()), to_abstain=int((bad & (after[s] < 0)).sum()),
                               to_background=int((bad & (after[s] == 0)).sum()),
                               to_other_object=int((bad & (after[s] > 0)).sum()))
            frame_rows.append(dict(frame=int(frame), methods=rows))
            if time.monotonic() - last >= 20 or j + 1 == len(frames):
                print('DELETION AUDIT', key, j + 1, '/', len(frames), 'seconds', round(time.monotonic() - start, 1), flush=True)
                last = time.monotonic()
        np.savez_compressed(folder / 'deletion_failures.npz', failure_count=failures,
                            first_failure_frame=first_frame, first_after_owner=first_after)
        methods = {m: summarize_changes(s, before[s], failures, len(frames)) for m, s in selected.items()}
        results[key] = dict(frames=len(frames), methods=methods, per_frame=frame_rows,
                            independent_scipy_frames=sorted(sample_frames), independent_scipy_nodes=sample_nodes.tolist(),
                            seconds=time.monotonic() - start, all_certified_cells_unchanged=True)
        dump(folder / 'stability.json', results[key])
        for path in [folder / 'deletion_failures.npz', folder / 'stability.json']: files[str(path)] = sha(path)
        print('STABILITY RESULT', key, {m: r['vulnerable_fraction'] for m, r in methods.items()}, flush=True)
    dump(OUT / 'stability_complete.json', dict(complete=True, files=files, results=results,
        predictions_sha256=sha(OUT / 'predictions_complete.json'), protocol_sha256=sha(OUT / 'protocol.json')))


def score():
    from evaluate_kitti360_development import transfer, clean_json
    from omni_semantic_map.kitti360_evaluation import evaluate
    from run_discovery_query_benchmark import matched_ids
    p = verify(); sealed = checked_seal('predictions_complete'); results = {}; objects = {}; files = {}
    assert not (OUT / 'scores_complete.json').exists()
    for c in p['contexts']:
        key = c['key']
        if not c['reference']:
            results[key] = dict(semantic_accuracy=None, reason='No independent robot reference'); continue
        ref = np.load(c['reference']); truth = ref['train_instance']; weight = ref['confidence']
        mass = np.bincount(truth, weights=weight)
        gids = np.flatnonzero((mass >= 100) & np.isin(np.arange(len(mass)) // 1000, [2, 13]))
        table = {int(g): dict(gt_id=int(g), methods={}) for g in gids}; rows = {}
        for m in METHODS:
            r = sealed['results'][key][m]; g = np.load(r['geometry']); cls = read(r['classification'])['objects']
            pred = transfer(g['xyz'], g['instance'], ref['xyz'], .2)
            scored = evaluate(truth, weight, pred, {o['object_id']: o['predictions']['mean'] for o in cls},
                              ROOT / 'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py')
            folder = OUT / key / m
            dump(folder / 'matches.json', clean_json(scored['matches']))
            np.savez_compressed(folder / 'reference_prediction.npz', instance=pred)
            rows[m] = dict(metrics=clean_json(scored['averages']), matched_gt_ids=sorted(matched_ids(scored['matches'])))
            if m == 'ours_frame':
                expected = read(c['old_summary'])['results'][key]['methods'][m]['metrics']
                for metric in ['allAp', 'allAp50%']: assert abs(rows[m]['metrics'][metric] - expected[metric]) < 1e-12
            pmass = np.bincount(pred, weights=weight)
            joint = np.bincount(pred.astype(np.int64) * len(mass) + truth, weights=weight,
                                minlength=len(pmass) * len(mass)).reshape(-1, len(mass))
            for gid in gids:
                best = dict(iou=0., predicted_id=None)
                for o in cls:
                    oid = o['object_id']
                    if oid >= len(pmass) or o['predictions']['mean']['train_id'] != gid // 1000: continue
                    inter = joint[oid, gid]; iou = float(inter / max(1e-12, mass[gid] + pmass[oid] - inter))
                    if iou > best['iou']: best = dict(iou=iou, predicted_id=int(oid))
                table[int(gid)]['methods'][m] = best
            for path in [folder / 'matches.json', folder / 'reference_prediction.npz']: files[str(path)] = sha(path)
            print('MATCHED SCORE', key, m, rows[m]['metrics']['allAp'], flush=True)
        deltas = {}
        for m in METHODS[1:]:
            d = np.array([r['methods']['ours_frame']['iou'] - r['methods'][m]['iou'] for r in table.values()])
            deltas[m] = dict(ours_better=int((d > 1e-12).sum()), ours_worse=int((d < -1e-12).sum()),
                             equal=int((np.abs(d) <= 1e-12).sum()),
                             ours_only_ap50=sorted(set(rows['ours_frame']['matched_gt_ids']) - set(rows[m]['matched_gt_ids'])),
                             control_only_ap50=sorted(set(rows[m]['matched_gt_ids']) - set(rows['ours_frame']['matched_gt_ids'])))
        results[key] = dict(methods=rows, per_object_comparisons=deltas); objects[key] = list(table.values())
    dump(OUT / 'objects.json', objects); files[str(OUT / 'objects.json')] = sha(OUT / 'objects.json')
    dump(OUT / 'scores_complete.json', dict(complete=True, files=files, results=results, exposed_diagnostic_only=True,
        predictions_sha256=sha(OUT / 'predictions_complete.json'), original_gates_unchanged=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['freeze', 'maps', 'stability', 'score'])
    globals()[parser.parse_args().action]()
