"""One bounded joint type/geometry experiment on frozen observations.

Classes may change; object IDs, coordinates and ranking scores cannot. The
completed-window graph makes this retrospective, not online object discovery.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from omni_semantic_map.joint_semantic_revision import evidence, revised_class

B = ROOT / 'build/benchmarks'
OUT = B / 'joint_semantic_revision_v1'
PARENT = B / 'anchor_counterevidence_v1'
METHODS = ['baseline', 'geometry_only', 'class_only', 'simple_fusion', 'ours_joint']


def verify():
    p = read(OUT / 'protocol.json')
    sources = dict(p['source_hashes'])
    amendment = OUT / 'taxonomy_export_fix.json'
    if amendment.exists():
        fix = read(amendment)
        for f, h in fix['original_source_hashes'].items(): assert sources[f] == h
        for f, h in fix['preserved_files'].items(): assert sha(f) == h, f
        sources.update(fix['corrected_source_hashes'])
    for f, h in {**p['input_hashes'], **sources}.items():
        assert sha(f) == h, f
    return p


def seal(name):
    s = read(OUT / (name + '.json')); assert s['complete']
    for f, h in s['files'].items(): assert sha(f) == h, f
    return s


def freeze():
    assert not OUT.exists() or not any(OUT.iterdir()), 'Do not replace a frozen experiment'
    parent = read(PARENT / 'protocol.json')
    inputs = {str(f): sha(f) for f in [PARENT / n for n in
              ['protocol.json', 'summary.json', 'predictions_complete.json', 'decisions_complete.json']]}
    contexts = []
    for original in parent['contexts']:
        c = dict(original)
        semantic_root = Path(c['sem']).parents[2]
        c['observations'] = str(semantic_root / 'predictions_complete.json')
        paths = [c['field'], c['sem'], c['job'], c['observations'],
                 Path(c['full']) / 'old_to_full.npy', Path(c['ours']) / 'geometry.npz',
                 Path(c['ours']) / 'classification.json',
                 PARENT / 'decisions' / c['key'] / 'anchors.npz',
                 PARENT / 'decisions' / c['key'] / 'propagation.npz',
                 PARENT / 'inputs' / c['key'] / 'map_proxy.npz']
        if c['reference']: paths.append(c['reference'])
        old = read(PARENT / 'predictions_complete.json')['results'][c['key']]['methods']
        for method in ['baseline', 'ours_retract_propagation']:
            paths.extend([old[method]['geometry'], old[method]['classification']])
        obs = read(c['observations']); assert obs['complete']
        assert len({r['frame_id'] for r in obs['results'][c['key']]}) == len(obs['results'][c['key']])
        for r in obs['results'][c['key']]:
            assert sha(r['file']) == obs['files'][r['file']]
            inputs[r['file']] = obs['files'][r['file']]
        for path in paths: inputs[str(path)] = sha(path)
        contexts.append(c)
    sources = [Path(__file__), ROOT / 'omni_semantic_map/omni_semantic_map/joint_semantic_revision.py',
               ROOT / 'omni_semantic_map/test/test_joint_semantic_revision.py',
               ROOT / 'omni_semantic_map/omni_semantic_map/batch_stable_transport.py',
               ROOT / 'omni_semantic_map/omni_semantic_map/batch_stable_transport.cpp',
               ROOT / 'omni_semantic_map/omni_semantic_map/kitti360_evaluation.py',
               ROOT / 'scripts/benchmarks/evaluate_kitti360_development.py',
               ROOT / 'build/datasets/kitti360_prepare/labels.py',
               ROOT / 'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py']
    dump(OUT / 'protocol.json', dict(created_epoch=time.time(), contexts=contexts, methods=METHODS,
        input_hashes=inputs, source_hashes={str(f): sha(f) for f in sources},
        hypothesis='Object-wide category disagreement need not imply geometric pollution. Require local semantic incompatibility with the rest of the object before executing existing geometric retraction proposals; then revise object type after withdrawing these cells.',
        observations='Cached SegFormer19-class source probabilities. Group by ACTUAL observation frame, not source acquisition frame. Deduplicate (cell,object,pixel) within each frame; mean pixels per(cell,object,frame). Mean cells per object-frame, then equal-weight observed frames per object. Missing observations abstain. No GT selection or fitting.',
        compatibility='Remove ALL query-cell observations from the object view means, including empty resulting frames. Compare cell mean to leave-cell-out object mean: ratio=19*dot(q_cell,q_other). Conflict iff other views exist and ratio<1-1e-6. Uniform reference is a modelling assumption, not calibrated independent evidence.',
        geometry='Reject frozen parent acquisition-frame anchor unit iff parent retract AND cell/object semantic conflict. Fully revoke cell/object only if no seed survives. Reuse parent original graph/seeds, background handling and export: unassign directly revoked cell/object OR changed nearest graph winner. No other foreground reassignment, no new coordinates or IDs.',
        classes='Argmax over all19 classes; preserve original class in a1e-12 tie. Class-only/simple-fusion use before posterior. Ours recomputes view means excluding fully revoked cell/object observations once. If no evidence, keep original class. Original ranking scores NEVER change; they are not recalibrated category confidences.',
        controls='Baseline and geometry-only are reused frozen results. Class-only changes type only. Simple-fusion independently combines parent geometry removal and before type update. Ours couples local geometry decisions to semantic evidence and re-estimates type once.',
        budget='Two exposed public windows0000/0002 and330-frame rosbag2245. Three new maps per context=9 maps;6 public AP scores. Zero training/encoder calls. No subsequent thresholds, variants or window substitution in this experiment.',
        evaluation='Same locally adapted KITTI360 building/car AP50:90/AP50 and20cm projection, NOT leaderboard. Other predicted classes excluded by unchanged evaluator; reference coordinates and labels unchanged. Seal every prediction before reading reference arrays. AllGT and per-object changes reported; robot has no independent GT.',
        development_gate='Both public AP/AP50 >=baseline; no lost AP50 matchedGT; mean AP>=baseline+0.001 AND strictly greater than class-only, geometry-only and simple-fusion. A passed development gate still requires independent assessment. Failed branch closes without parameter sweep.',
        scope='Retrospective fixed-identity revision; all public windows already exposed. Does not solve object discovery or cross-frame identity association. Robot is same-module diagnostic regardless gate, not accuracy, online or S100 validation. Class adaptation alone is a control, not a claimed invention.'))
    print('JOINT PROTOCOL FROZEN', flush=True)


def pair_keys(rows):
    a = np.ascontiguousarray(rows, dtype=np.int64)
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()


def prepare():
    p = verify(); assert not (OUT / 'prepared.json').exists()
    files, stats = {}, {}
    for c in p['contexts']:
        key = c['key']; field = np.load(c['field']); sem = np.load(c['sem'])
        mapping = np.load(Path(c['full']) / 'old_to_full.npy'); owners = sem['owners']
        all_pairs, all_frames, all_prob = [], [], []; count = raw_count = unique_pixels = 0
        for record in read(c['observations'])['results'][key]:
            d = np.load(record['file']); ids = d['source_ids']
            assert ids.min(initial=0) >= 0 and ids.max(initial=0) < len(owners)
            take = owners[ids] > 0; ids = ids[take]
            if not len(ids): continue
            xy = d['source_pixel_xy'][take]; q = d['source_probabilities'][take].astype(np.float64)
            assert q.shape == (len(ids), 19) and np.isfinite(q).all()
            np.testing.assert_allclose(q.sum(1), 1, atol=2e-6)
            pairs = np.column_stack([field['source_to_cell'][mapping[ids]], owners[ids]])
            pixels, first, inverse = np.unique(np.column_stack([pairs, xy]), axis=0,
                                               return_index=True, return_inverse=True)
            np.testing.assert_allclose(q, q[first][inverse], rtol=0, atol=0)
            q = q[first]; pairs, inv = np.unique(pixels[:, :2], axis=0, return_inverse=True)
            total = np.zeros((len(pairs), 19)); np.add.at(total, inv, q)
            probabilities = total / np.bincount(inv)[:, None]
            all_pairs.append(pairs); all_prob.append(probabilities)
            all_frames.append(np.full(len(pairs), record['frame_id'], np.int64))
            count += 1; raw_count += len(ids); unique_pixels += len(pixels)
        pairs, frames, probabilities = np.concatenate(all_pairs), np.concatenate(all_frames), np.concatenate(all_prob)
        result = evidence(pairs, frames, probabilities)
        folder = OUT / 'evidence' / key; folder.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(folder / 'observations.npz', cell_owner=pairs, observation_frame=frames, probabilities=probabilities)
        np.savez_compressed(folder / 'before.npz', **result)
        for f in folder.iterdir(): files[str(f)] = sha(f)
        stats[key] = dict(observed_frames=count, raw_source_observations=raw_count,
                          unique_cell_object_pixels=unique_pixels, cell_frame_records=len(pairs),
                          cells=len(result['cell_owner']), conflicts=int(result['semantic_conflict'].sum()),
                          objects=len(result['object_ids']))
        print('JOINT EVIDENCE', key, stats[key], flush=True)
    dump(OUT / 'prepared.json', dict(complete=True, files=files, stats=stats, gt_read=False))


def class_records(original, result):
    import runpy
    # Multiple raw labels can share a train ID (building11, garage34 ->2).
    # The pinned model/evaluator use the first, canonical label, not last-wins.
    names = {}
    for label in runpy.run_path(str(ROOT / 'build/datasets/kitti360_prepare/labels.py'))['labels']:
        if 0 <= label.trainId < 19: names.setdefault(label.trainId, label)
    probs = {int(o): q for o, q in zip(result['object_ids'], result['object_probability'])}
    output = deepcopy(original); changes = []
    for obj in output['objects']:
        oid = obj['object_id']; row = obj['predictions']['mean']; old = int(row['train_id'])
        if oid not in probs: continue
        new = revised_class(probs[oid], old); label = names[new]
        if new != old: row.update(train_id=new, semantic_id=label.id, name=label.name)
        obj['semantic_revision'] = dict(probability=probs[oid].tolist(), previous_train_id=old,
                                        ranking_score_fixed=True)
        if old != new: changes.append(dict(object_id=oid, before=old, after=new))
    return output, changes


def maps():
    from audit_matched_transport import graph_from
    from omni_semantic_map.batch_stable_transport import native_library, nearest_groups, nearest_owner
    p = verify(); seal('prepared'); assert not (OUT / 'predictions_complete.json').exists()
    parent = read(PARENT / 'predictions_complete.json'); lib = native_library(OUT / 'native')
    files, results = {}, {}
    for c in p['contexts']:
        key = c['key']; folder = OUT / 'evidence' / key
        before = np.load(folder / 'before.npz'); obs = np.load(folder / 'observations.npz')
        decision = np.load(PARENT / 'decisions' / key / 'anchors.npz')
        prop = np.load(PARENT / 'decisions' / key / 'propagation.npz')
        units = decision['units']; seeds = prop['seeds']
        conflicts = before['cell_owner'][before['semantic_conflict']]
        rejected = units[decision['retract'] & np.isin(pair_keys(units[:, :2]), pair_keys(conflicts))]
        remaining = seeds[~np.isin(pair_keys(seeds), pair_keys(rejected))]
        assert len(seeds) - len(remaining) == len(rejected)
        revoked = np.unique(rejected[:, :2], axis=0)
        revoked = revoked[~np.isin(pair_keys(revoked), pair_keys(remaining[:, :2]))]
        field = np.load(c['field']); graph = graph_from(field)
        labels, dist = nearest_groups(graph, seeds[:, 0], seeds[:, 1], lib)
        original = nearest_owner(labels, dist)
        np.testing.assert_array_equal(original, prop['original_winner'])
        labels, dist = nearest_groups(graph, remaining[:, 0], remaining[:, 1], lib)
        revised = nearest_owner(labels, dist)
        base = np.load(Path(c['ours']) / 'geometry.npz'); old = base['instance']
        proxy = np.load(PARENT / 'inputs' / key / 'map_proxy.npz'); cell = proxy['cell']; prefix = int(proxy['prefix'])
        direct = np.isin(pair_keys(np.column_stack([cell, old])), pair_keys(revoked))
        drop = direct | (original[cell] != revised[cell])
        keep_obs = ~np.isin(pair_keys(obs['cell_owner']), pair_keys(revoked))
        after = evidence(obs['cell_owner'][keep_obs], obs['observation_frame'][keep_obs], obs['probabilities'][keep_obs]) if keep_obs.any() else dict(object_ids=np.array([], int), object_probability=np.empty((0, 19)))
        np.savez_compressed(folder / 'after.npz', **after)
        np.savez_compressed(folder / 'decision.npz', rejected_units=rejected, revoked_cell_owner=revoked,
                            retained_seeds=remaining, revised_winner=revised, direct=direct, drop=drop)
        for f in [folder / 'after.npz', folder / 'decision.npz']: files[str(f)] = sha(f)
        original_cls = read(Path(c['ours']) / 'classification.json')
        before_cls, before_changes = class_records(original_cls, before)
        after_cls, after_changes = class_records(original_cls, after)
        parent_rows = parent['results'][key]['methods']
        parent_geo = np.load(parent_rows['ours_retract_propagation']['geometry'])
        np.testing.assert_array_equal(parent_geo['xyz'], base['xyz'])
        masks = dict(class_only=np.zeros(len(old), bool), simple_fusion=parent_geo['instance'] != old, ours_joint=drop)
        rows = {}
        for method, parent_method in [('baseline', 'baseline'), ('geometry_only', 'ours_retract_propagation')]:
            rows[method] = dict(parent_rows[parent_method], reused=True, class_changes=[])
        for method, mask in masks.items():
            dest = OUT / 'maps' / key / method; dest.mkdir(parents=True, exist_ok=False)
            assigned = old.copy(); assigned[mask] = 0
            cls, changes = (after_cls, after_changes) if method == 'ours_joint' else (before_cls, before_changes)
            np.savez_compressed(dest / 'geometry.npz', xyz=base['xyz'], instance=assigned,
                                original_row_id=np.arange(len(old)), retracted_rows=np.flatnonzero(mask))
            dump(dest / 'classification.json', cls)
            rows[method] = dict(geometry=str(dest / 'geometry.npz'), classification=str(dest / 'classification.json'),
                                removed=int(mask.sum()), removed_original=int(mask[:prefix].sum()),
                                removed_propagated=int(mask[prefix:].sum()), class_changes=changes, reused=False)
            for f in dest.iterdir(): files[str(f)] = sha(f)
        results[key] = dict(methods=rows, rejected_units=len(rejected), revoked_cells=len(revoked),
                            parent_rejected_units=int(decision['retract'].sum()),
                            observation_rows_excluded=int((~keep_obs).sum()))
        print('JOINT MAPS', key, {m: (r['removed'], len(r['class_changes'])) for m, r in rows.items()}, flush=True)
    dump(OUT / 'predictions_complete.json', dict(complete=True, results=results, files=files, new_maps=9, gt_read=False))


def score():
    from evaluate_kitti360_development import transfer, clean_json
    from omni_semantic_map.kitti360_evaluation import evaluate
    from run_discovery_query_benchmark import matched_ids
    p = verify(); predictions = seal('predictions_complete'); assert not (OUT / 'summary.json').exists()
    parent = read(PARENT / 'summary.json'); results, files = {}, {}; means = {m: [] for m in METHODS}; gate = True
    for c in p['contexts']:
        key = c['key']
        if not c['reference']:
            results[key] = dict(semantic_accuracy=None, reason='No independent robot GT'); continue
        ref = np.load(c['reference']); rows = {}
        for method in METHODS:
            row = predictions['results'][key]['methods'][method]
            if method in ['baseline', 'geometry_only']:
                pm = 'baseline' if method == 'baseline' else 'ours_retract_propagation'
                rows[method] = dict(parent['results'][key]['methods'][pm], reused_frozen_score=True)
            else:
                g = np.load(row['geometry']); cls = read(row['classification'])['objects']
                pred = transfer(g['xyz'], g['instance'], ref['xyz'], .2)
                scored = evaluate(ref['train_instance'], ref['confidence'], pred,
                                  {o['object_id']: o['predictions']['mean'] for o in cls},
                                  ROOT / 'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py')
                dest = Path(row['geometry']).parent
                dump(dest / 'matches.json', clean_json(scored['matches']))
                np.savez_compressed(dest / 'reference_prediction.npz', instance=pred)
                for f in [dest / 'matches.json', dest / 'reference_prediction.npz']: files[str(f)] = sha(f)
                rows[method] = dict(metrics=clean_json(scored['averages']), matched_gt_ids=sorted(matched_ids(scored['matches'])))
            means[method].append(rows[method]['metrics']['allAp'])
            print('JOINT SCORE', key, method, rows[method]['metrics']['allAp'], rows[method]['metrics']['allAp50%'], flush=True)
        ours, base = rows['ours_joint'], rows['baseline']
        gate &= ours['metrics']['allAp'] >= base['metrics']['allAp'] and ours['metrics']['allAp50%'] >= base['metrics']['allAp50%']
        gate &= set(base['matched_gt_ids']).issubset(ours['matched_gt_ids'])
        results[key] = dict(methods=rows)
    mean = {m: float(np.mean(v)) for m, v in means.items()}
    gate &= mean['ours_joint'] >= mean['baseline'] + .001
    gate &= all(mean['ours_joint'] > mean[m] for m in ['class_only', 'geometry_only', 'simple_fusion'])
    dump(OUT / 'summary.json', dict(complete=True, results=results, means=mean, development_gate=bool(gate),
                                   files=files, new_public_scores=6, predictions_sha256=sha(OUT / 'predictions_complete.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'prepare', 'maps', 'score'])
    globals()[parser.parse_args().action]()
