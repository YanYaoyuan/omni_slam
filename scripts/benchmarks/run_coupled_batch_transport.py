"""One frozen test of coupled source-batch deletion on complete object maps."""
import argparse
import shutil
import time
from pathlib import Path
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from audit_matched_transport import OUT as PARENT, inputs, graph_from
from omni_semantic_map.coupled_batch_transport import native_library, solve
from omni_semantic_map.matched_transport_audit import confidence, matched_selection

OUT = ROOT / 'build/benchmarks/coupled_batch_transport_v1'
NEW = ['ours_coupled', 'matched_distance', 'matched_margin']
OLD = ['baseline', 'full_nearest', 'full_two_cells', 'ours_frame']
METHODS = OLD + NEW


def freeze():
    assert not OUT.exists()
    parent = read(PARENT / 'protocol.json')
    files = dict(parent['input_hashes']); files.update(parent['source_hashes'])
    for name in ['protocol', 'predictions_complete', 'stability_complete', 'scores_complete', 'posthoc_mechanism', 'delivery_audit']:
        f = PARENT / (name + '.json'); files[str(f)] = sha(f)
    for name in ['predictions_complete', 'stability_complete']:
        for f, h in read(PARENT / (name + '.json'))['files'].items():
            assert sha(f) == h; files[f] = h
    rows = inputs()
    for c in rows:
        for method in OLD:
            for name in ['geometry.npz', 'classification.json']:
                f = Path(c['ours']).parent / method / name; files[str(f)] = sha(f)
    sources = [Path(__file__), ROOT / 'omni_semantic_map/omni_semantic_map/coupled_batch_transport.py',
        ROOT / 'omni_semantic_map/omni_semantic_map/coupled_batch_transport.cpp',
        ROOT / 'omni_semantic_map/test/test_coupled_batch_transport.py']
    dump(OUT / 'protocol.json', dict(created_epoch=time.time(), contexts=rows, methods=METHODS,
        input_hashes=files, source_hashes={str(f): sha(f) for f in sources},
        hypothesis='Correct the conservative independent-competitor bound: delete all own AND competing anchors from the same acquisition frame. On fixed geometry admit precisely those previously unowned sources whose originally unique foreground owner survives EVERY single acquisition-frame anchor deletion.',
        implementation='Tie-aware two-distinct-frame propagation retaining all frames within tolerance of second-frame distance and at most two conflicting owners within each frame tolerance. Initial unique winner needs only its closest anchor frame removed: every other deletion leaves that anchor intact. Validate against all499 actual deletions and160 randomized exhaustive tests before maps.',
        constants='Identical original20cm graph,16NN/strict0.4m,Euclidean lengths, corrected known-background anchors,1e-10 uniqueness tolerance. No new perception, fitting, class score, point synthesis, radius/threshold change or object birth.',
        controls='Reuse exact original baseline,full_nearest,full_two_cells,conservative ours_frame. New nearest-distance and normalized-margin controls match Ours-coupled added source counts PER OBJECT, candidate universe and tie rules identical to parent. All original XYZ/IDs/classscores remain immutable.',
        budget='Four already exposed contexts x3new complete maps =12 maps. Nine new public scores, reuse audited frozen old scores. Same-version robot diagnostic included regardless of clean gate, no semantic accuracy without robotGT. No more variants, models, parameter or window sweeps in this protocol.',
        gate='Implementation matches all existing foreground exhaustive decisions. On all3public windows Ours-coupled AP andAP50>=conservativeOurs and no lost AP50matches relative to it. Three-window mean AP >=conservativeOurs+0.001 and greater than each new matchedcontrol mean. On0006 AP>=max(original full_nearest,full_two_cells). If false, close without enlarging benchmarks or training. This new posthoc development gate does not replace old failed gates or assert independence.',
        scope='All0000/0002/0006/robot exposed. Original0007N/A unchanged. Fixed graph anchor deletion only; neither raw sensor deletion nor semantic correctness nor repeat admission is guaranteed. New independent evaluation remains required after a successful method.'))
    print('COUPLED TRANSPORT FROZEN', flush=True)


def verify():
    p = read(OUT / 'protocol.json')
    for f, h in {**p['input_hashes'], **p['source_hashes']}.items(): assert sha(f) == h, f
    return p


def seal(name):
    s = read(OUT / (name + '.json')); assert s['complete']
    for f, h in s['files'].items(): assert sha(f) == h, f
    return s


def fields():
    p = verify(); assert not (OUT / 'fields_complete.json').exists()
    lib = native_library(OUT / 'native'); files = {}; results = {}
    for c in p['contexts']:
        key = c['key']; folder = OUT / key; folder.mkdir(parents=True, exist_ok=False)
        d = np.load(c['field']); previous = np.load(PARENT / key / 'readout.npz')
        failures = np.load(PARENT / key / 'deletion_failures.npz')['failure_count']
        graph = graph_from(d); start = time.monotonic(); result = solve(graph, previous['seeds'], lib); seconds = time.monotonic() - start
        np.testing.assert_array_equal(result['nearest_owner'], previous['nearest_owner'])
        np.testing.assert_allclose(result['nearest_distance'], previous['label_distance'][:, 0], rtol=1e-12, atol=1e-10)
        fg = previous['nearest_owner'] > 0
        assert (failures <= 1).all(), 'Unique winner can be affected by at most one batch'
        expected = np.where(fg & (failures == 0), previous['nearest_owner'], -1)
        np.testing.assert_array_equal(np.where(result['stable_owner'] > 0, result['stable_owner'], -1), expected)
        old = previous['certified_owner']; assert (result['stable_owner'][old > 0] == old[old > 0]).all()
        fail = fg & (failures > 0)
        failure_data = np.load(PARENT / key / 'deletion_failures.npz')
        np.testing.assert_array_equal(result['critical_batch'][fail], failure_data['first_failure_frame'][fail])
        np.testing.assert_array_equal(result['after_critical_owner'][fail], failure_data['first_after_owner'][fail])
        path = folder / 'field.npz'; np.savez_compressed(path, **{k: v for k, v in result.items() if isinstance(v, np.ndarray)})
        files[str(path)] = sha(path)
        results[key] = dict(nodes=len(fg), foreground_nodes=int(fg.sum()), stable_foreground_nodes=int((result['stable_owner'] > 0).sum()),
            newly_certified_foreground_nodes=int(((result['stable_owner'] > 0) & (old < 0)).sum()),
            solve_seconds=seconds, exhaustive_foreground_replay_exact=True, old_certificate_subset=True,
            failed_critical_frames_and_owner_replay_exact=True,
            **{k: int(result[k]) for k in ['popped_states', 'expanded_states', 'max_retained_frames']})
        print('COUPLED FIELD', key, results[key], flush=True)
    dump(OUT / 'fields_complete.json', dict(complete=True, files=files, results=results,
        protocol_sha256=sha(OUT / 'protocol.json'), reference_labels_read=False))


def maps():
    p = verify(); seal('fields_complete'); assert not (OUT / 'predictions_complete.json').exists()
    files = {}; results = {}
    for c in p['contexts']:
        key = c['key']; folder = OUT / key; d = np.load(c['field']); cells = d['source_to_cell']
        state = np.load(folder / 'field.npz'); previous = np.load(PARENT / key / 'readout.npz')
        full = np.load(Path(c['full']) / 'geometry.npz'); base = np.load(Path(c['graph']) / 'geometry.npz')
        sem = np.load(c['sem']); owners = np.zeros(len(cells), np.int64)
        owners[np.load(Path(c['full']) / 'old_to_full.npy')] = sem['owners']
        selected = np.flatnonzero((owners == 0) & (state['stable_owner'][cells] > 0))
        winner = state['nearest_owner'][cells]; candidates = np.flatnonzero((owners == 0) & (winner > 0))
        selections = {'ours_coupled': selected}
        for m, ranking in [('matched_distance', 'nearest_distance'), ('matched_margin', 'normalized_margin')]:
            selections[m] = matched_selection(candidates, winner[candidates],
                confidence(previous['label_distance'][cells[candidates]], ranking), selected, winner[selected])
        target_counts = np.bincount(winner[selected], minlength=int(winner.max()) + 1); rows = {}
        old_sources = np.load(Path(c['ours']) / 'geometry.npz')['added_full_source_ids']
        assert np.isin(old_sources, selected).all()
        for method in METHODS:
            dest = folder / method; dest.mkdir()
            if method in OLD:
                source = Path(c['ours']).parent / method
                gp = source / 'geometry.npz'; cp = source / 'classification.json'
                g = np.load(gp); added = g['added_full_source_ids'] if method != 'baseline' else np.empty(0, np.int64)
            else:
                added = selections[method]; assigned = winner[added]
                np.testing.assert_array_equal(np.bincount(assigned, minlength=len(target_counts)), target_counts)
                gp = dest / 'geometry.npz'; cp = dest / 'classification.json'
                np.savez_compressed(gp, xyz=np.concatenate([base['xyz'], full['xyz'][added]]),
                                    instance=np.r_[base['instance'], assigned], added_full_source_ids=added)
                shutil.copyfile(Path(c['graph']) / 'classification.json', cp)
            row = dict(geometry=str(gp), classification=str(cp), added_sources=len(added),
                additional_vs_conservative=len(added) - len(old_sources),
                same_per_object_point_count_as_new_ours=method in NEW,
                unchanged_original_geometry_ids_scores=True)
            dump(dest / 'map.json', row); rows[method] = row
            for path in [gp, cp, dest / 'map.json']: files[str(path)] = sha(path)
        results[key] = rows
        print('COUPLED MAPS', key, 'new_ours_added', len(selected), 'extra', len(selected) - len(old_sources), flush=True)
    dump(OUT / 'predictions_complete.json', dict(complete=True, files=files, results=results,
        new_maps=12, reference_labels_read=False, protocol_sha256=sha(OUT / 'protocol.json')))


def score():
    from evaluate_kitti360_development import transfer, clean_json
    from omni_semantic_map.kitti360_evaluation import evaluate
    from run_discovery_query_benchmark import matched_ids
    p = verify(); predictions = seal('predictions_complete'); assert not (OUT / 'summary.json').exists()
    results = {}; files = {}; means = {m: [] for m in METHODS}; noharm = True
    for c in p['contexts']:
        key = c['key']
        if not c['reference']:
            results[key] = dict(semantic_accuracy=None, reason='No independent robot GT'); continue
        ref = np.load(c['reference']); rows = {}
        old_summary = read(c['old_summary'])['results'][key]['methods']
        for method in METHODS:
            if method in OLD:
                row = old_summary[method]
                rows[method] = dict(metrics=row['metrics'], matched_gt_ids=row['matched_gt_ids'], reused_frozen_score=True)
            else:
                r = predictions['results'][key][method]; g = np.load(r['geometry']); cls = read(r['classification'])['objects']
                pred = transfer(g['xyz'], g['instance'], ref['xyz'], .2)
                scored = evaluate(ref['train_instance'], ref['confidence'], pred,
                    {o['object_id']: o['predictions']['mean'] for o in cls},
                    ROOT / 'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py')
                folder = OUT / key / method
                dump(folder / 'matches.json', clean_json(scored['matches']))
                np.savez_compressed(folder / 'reference_prediction.npz', instance=pred)
                for path in [folder / 'matches.json', folder / 'reference_prediction.npz']: files[str(path)] = sha(path)
                rows[method] = dict(metrics=clean_json(scored['averages']), matched_gt_ids=sorted(matched_ids(scored['matches'])))
            means[method].append(rows[method]['metrics']['allAp'])
            print('COUPLED SCORE', key, method, rows[method]['metrics']['allAp'], flush=True)
        ours = rows['ours_coupled']; old = rows['ours_frame']
        gained = sorted(set(ours['matched_gt_ids']) - set(old['matched_gt_ids']))
        lost = sorted(set(old['matched_gt_ids']) - set(ours['matched_gt_ids']))
        noharm &= all(ours['metrics'][m] >= old['metrics'][m] for m in ['allAp', 'allAp50%']) and not lost
        results[key] = dict(methods=rows, changes_vs_conservative=dict(gained=gained, lost=lost))
    mean = {m: float(np.mean(v)) for m, v in means.items()}
    public = results['kitti360_0006']['methods']
    gate = bool(noharm and mean['ours_coupled'] >= mean['ours_frame'] + .001
                and mean['ours_coupled'] > max(mean[m] for m in NEW[1:])
                and public['ours_coupled']['metrics']['allAp'] >= max(public[m]['metrics']['allAp'] for m in ['full_nearest', 'full_two_cells']))
    dump(OUT / 'summary.json', dict(complete=True, files=files, results=results, exposed_three_window_mean=mean,
        noharm=bool(noharm), development_gate=gate, old_gates_unchanged=True, independent_test=False,
        protocol_sha256=sha(OUT / 'protocol.json'), prediction_sha256=sha(OUT / 'predictions_complete.json')))
    print('COUPLED GATE', gate, mean, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['freeze', 'fields', 'maps', 'score'])
    globals()[parser.parse_args().action]()
