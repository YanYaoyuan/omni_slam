"""Bounded identifiability probe on spatially withheld predicted source anchors.

No benchmark GT is read. This is pseudo-identity consistency, not accuracy,
semantic background rejection, complete-map AP, or a new-method validation.
"""
import argparse
import time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from paper_final_common import ROOT, read, dump, sha
from omni_semantic_map.mask_observation_code import (
    unique_queries, fit_code, decode_code, unique_winner,
)

OUT = ROOT / 'build/benchmarks/mask_observation_code_probe_v1'
PARENT = ROOT / 'build/benchmarks/signed_query_association_v1'
BASE = ROOT / 'build/benchmarks/intrinsic_surface_ownership_v1'
OWN = ROOT / 'build/benchmarks/surface_ownership_v1'
INPUT = ROOT / 'build/benchmarks/discovery_measured_inputs_v1'
RAW = ROOT / 'build/benchmarks/prethreshold_candidates_v1'
KEYS = ['kitti360_0000', 'kitti360_0002']
METHODS = ['single_mask', 'mask_cosine', 'full_code', 'shuffled_code', 'nearest_anchor']


def freeze():
    assert not OUT.exists()
    inputs = {}
    capture = read(RAW / 'predictions_complete.json')
    for key in KEYS:
        m = read(INPUT / key / 'manifest.json')
        paths = [INPUT / key / 'manifest.json', m['geometry'],
                 INPUT / 'query_cells' / key / 'source_to_cell.npy',
                 OWN / 'features' / key / 'core_sources.json',
                 BASE / 'maps' / key / 'intrinsic_geometry' / 'geometry.npz']
        paths += [PARENT / 'maps' / key / 'geometry_graph' / n for n in
                  ['geometry.npz', 'classification.json', 'groups.json']]
        paths += [r['file'] for r in m['records']]
        paths += [r['file'] for r in capture['results'][key]]
        for f in paths:
            inputs[str(f)] = sha(f)
    inputs[str(RAW / 'predictions_complete.json')] = sha(RAW / 'predictions_complete.json')
    sources = [Path(__file__), ROOT / 'omni_semantic_map/omni_semantic_map/mask_observation_code.py',
               ROOT / 'omni_semantic_map/test/test_mask_observation_code.py']
    dump(OUT / 'protocol.json', dict(
        created_epoch=time.time(), input_hashes=inputs, source_hashes={str(f): sha(f) for f in sources},
        scenes=KEYS, methods=METHODS,
        scope='Existing108frames and frozen geometry_graph source provenance only. No GT, new inference, fitting on reference labels, new map, scoring, robot or independent scene access.',
        inherited_supervision='Strong reference includes the earlier cross-window supervised intrinsic_geometry predictor. This probe adds no reference-label access/training, but the complete pipeline must not be called training-free or annotation-free.',
        hypothesis='Do both membership and measured exclusion across raw queries distinguish anchor identities that one retrieved mask conflates?',
        anchors='Original source provenance plus intrinsic and geometry_graph appended source IDs. Remap original IDs by frozen group assignments. Sources claimed by different old IDs abstain. No nearest-map-point label substitution.',
        split='World2m spatial blocks; training if (73856093*x XOR19349663*y XOR83492791*z)&1 ==0. Other blocks withheld. Source count weighted so each observed20cm physical cell contributes one anchor per frame/object. Require>=16distinct measured training cells per candidate object and>=2candidate objects per frame. Do not change split after coverage/results.',
        observations='Exact source pixels with recorded measured visibility; missing points excluded. Raw positive native queries of both classes. Deduplicate exactly equal masks on all current measured source points before fitting; no new score threshold. Full-mask containment is not replaced by largest3Dcomponent.',
        code='For each query/object, Beta(.5,.5) smoothed membership frequency on spatial training anchors. Full score is mean Bernoulli log score, not a calibrated posterior because queries are dependent. Comparator cosine is normalized point binary vector versus anchor-frequency vector; inspired affinity control, NOT an original SAI3D/MV3DIS reproduction.',
        controls='single_mask selects same-class raw query by IoU against observed training source anchors, then gives its contained validation points one scalar core-coverage score. Geometry is nearest global training-source anchor for each currently eligible object. Shuffled code applies seed0 query permutation to test rows only; same deduplicated queries/code/candidate set. Exact/1e-10score ties abstain.',
        evaluation='Pseudo-identity consistency only. All eligible withheld source observations, no GT. Frame-source micro accuracy and per-object macro; unique source consensus by majority frame winners, tied vote abstains. Report coverage, ambiguity, confusion, source IDs, all per-frame decisions. Missing frames never count as negative votes.',
        budget='Two exposed windows; one fixed extraction/fit/readout, five analytic arms. No parameter sweep, source resampling, model fitting/checkpoint search or full-map evaluation.',
        screening='Proceed to consider a production candidate only if full_code macro unique-source consistency strictly exceeds single_mask and shuffled_code in EACH window, and is not below mask_cosine. This is necessary information screening only, never sufficient for a paper/full-map gain; nearest geometry comparison and background/unknown identifiability must also be reported.',
        novelty='Bernoulli decoding and mask affinities are established tools. This probe does not establish novelty. A production mechanism must solve unknown-point acceptance/ownership and beat same-input strong controls on full maps.'))
    print('FROZEN MASK CODE PROBE', flush=True)


def verify():
    p = read(OUT / 'protocol.json')
    for f, h in {**p['input_hashes'], **p['source_hashes']}.items():
        assert sha(f) == h, f
    return p


def source_anchors(key, n):
    core = read(OWN / 'features' / key / 'core_sources.json')
    groups = read(PARENT / 'maps' / key / 'geometry_graph' / 'groups.json')
    assignment = {int(k): int(v) for k, v in groups['old_id_assignment'].items()}
    labels = np.zeros(n, np.int64)

    def assign(ids, oid):
        old = labels[ids]
        conflict = (old != 0) & (old != oid)
        labels[ids] = oid
        labels[ids[conflict]] = -1

    for oid, ids in core.items():
        assign(np.asarray(ids, np.int64), assignment[int(oid)])
    for path, remap in [(BASE / 'maps' / key / 'intrinsic_geometry', True),
                        (PARENT / 'maps' / key / 'geometry_graph', False)]:
        g = dict(np.load(path / 'geometry.npz'))
        ids = g['added_source_ids']
        if not len(ids):
            continue
        owner = g['instance'][-len(ids):]
        if remap:
            owner = np.asarray([assignment[int(i)] for i in owner])
        for oid in np.unique(owner):
            assign(ids[owner == oid], oid)
    return labels


def summary_for(y, pred, objects):
    return dict(observations=len(y), correct=int((pred == y).sum()), abstained=int((pred == 0).sum()),
                micro=float(np.mean(pred == y)) if len(y) else None,
                macro=float(np.mean([np.mean(pred[y == o] == o) for o in objects if (y == o).any()])) if len(y) else None,
                per_object=[dict(object_id=int(o), observations=int((y == o).sum()),
                                 correct=int(((y == o) & (pred == o)).sum()),
                                 abstained=int(((y == o) & (pred == 0)).sum()))
                            for o in objects if (y == o).any()])


def run():
    p = verify()
    assert not (OUT / 'predictions_complete.json').exists()
    capture = read(RAW / 'predictions_complete.json')
    files = {}; results = {}
    for key in KEYS:
        start = time.monotonic()
        manifest = read(INPUT / key / 'manifest.json')
        xyz = np.load(manifest['geometry'])['xyz']
        cell = np.load(INPUT / 'query_cells' / key / 'source_to_cell.npy')
        labels = source_anchors(key, len(xyz))
        blocks = np.floor(xyz / 2.).astype(np.int64)
        train = ((blocks[:, 0]*73856093 ^ blocks[:, 1]*19349663 ^ blocks[:, 2]*83492791) & 1) == 0
        # Same physical20cm cell cannot straddle the chosen2m block boundary.
        seen = np.unique(cell[train]); assert not np.isin(cell[~train], seen).any()
        objects = sorted(o['object_id'] for o in read(PARENT / 'maps' / key / 'geometry_graph' / 'classification.json')['objects'])
        semantics = {o['object_id']: o['predictions']['mean']['semantic_id'] for o in
                     read(PARENT / 'maps' / key / 'geometry_graph' / 'classification.json')['objects']}
        trees = {o: cKDTree(xyz[(labels == o) & train]) for o in objects if ((labels == o) & train).any()}
        validation = np.flatnonzero((labels > 0) & ~train)
        index = np.full(len(xyz), -1, np.int64); index[validation] = np.arange(len(validation))
        oidcol = {o: j+1 for j, o in enumerate(objects)}
        votes = {m: np.zeros((len(validation), len(objects)+1), np.int16) for m in METHODS}
        frames = []; frame_y = []; frame_pred = {m: [] for m in METHODS}
        folder = OUT / key; folder.mkdir(parents=True, exist_ok=False)
        by_frame = {r['frame_id']: r for r in manifest['records']}
        rng = np.random.RandomState(0)
        for rec in capture['results'][key]:
            frame = rec['frame_id']
            v = dict(np.load(by_frame[frame]['file'])); ids = v['visible_source_ids']; xy = v['pixel_xy']
            calibration = (labels[ids] > 0) & train[ids]
            eligible = np.asarray([o for o in objects if len(np.unique(cell[ids[calibration & (labels[ids] == o)]])) >= 16], np.int64)
            test = (~train[ids]) & np.isin(labels[ids], eligible)
            if len(eligible) < 2 or not test.any():
                frames.append(dict(frame_id=frame, eligible_objects=len(eligible), tested=0, skipped=True)); continue
            d = dict(np.load(rec['file']))
            raw = np.flatnonzero(d['scores'] > 0)
            inside = ((d['masks_packed'][raw][:, xy[:, 1], xy[:, 0]//8] >> (7-xy[:, 0]%8)[None, :]) & 1).astype(bool)
            unique, first = unique_queries(inside)
            # Duplicate classes may have the same pattern; full code only uses
            # observations, while single-mask retrieval retains class metadata.
            weights = np.zeros(len(ids), float)
            for oid in eligible:
                keep = calibration & (labels[ids] == oid)
                _, inverse, counts = np.unique(cell[ids[keep]], return_inverse=True, return_counts=True)
                weights[keep] = 1./counts[inverse]
            p_code, counts = fit_code(unique, np.where(calibration, labels[ids], 0), eligible, weights)
            decoded = decode_code(unique[:, test], p_code)
            shuffled = decode_code(unique[rng.permutation(len(unique))][:, test], p_code)['full']
            scalar = np.zeros((test.sum(), len(eligible)), float)
            for j, oid in enumerate(eligible):
                own = calibration & (labels[ids] == oid)
                intersection = inside[:, own].sum(1)
                union = inside.sum(1) + own.sum() - intersection
                allowed = np.flatnonzero((d['semantic_ids'][raw] == semantics[int(oid)]) & (intersection > 0))
                if not len(allowed): continue
                iou = intersection[allowed]/union[allowed]
                q = allowed[np.lexsort((raw[allowed], -d['scores'][raw[allowed]], -iou))[0]]
                scalar[:, j] = inside[q, test] * (intersection[q]/own.sum())
            distances = np.column_stack([-trees[int(o)].query(xyz[ids[test]], workers=1)[0] for o in eligible])
            scores = dict(single_mask=scalar, mask_cosine=decoded['cosine'], full_code=decoded['full'], shuffled_code=shuffled, nearest_anchor=distances)
            pred = {m: unique_winner(s, eligible) for m, s in scores.items()}
            y = labels[ids[test]]; test_ids = ids[test]
            frame_y.append(y)
            for method in METHODS:
                frame_pred[method].append(pred[method])
                columns = np.asarray([oidcol.get(int(o), 0) for o in pred[method]])
                np.add.at(votes[method], (index[test_ids], columns), 1)
            path = folder / 'frames' / f'{frame:06d}.npz'; path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path, source_ids=test_ids, pseudo_identity=y, candidate_objects=eligible,
                                query_rows=raw[first], code=p_code, calibration_cells=counts, **pred)
            files[str(path)] = sha(path)
            frames.append(dict(frame_id=frame, eligible_objects=len(eligible), tested=len(y), raw_queries=len(raw), unique_queries=len(unique),
                               full_vs_single_changed=int((pred['full_code'] != pred['single_mask']).sum()),
                               full_vs_cosine_changed=int((pred['full_code'] != pred['mask_cosine']).sum())))
        y = np.concatenate(frame_y) if frame_y else np.empty(0, np.int64)
        summaries = {}; consensus = {}
        used = votes[METHODS[0]].sum(1) > 0
        for method in METHODS:
            predicted = np.concatenate(frame_pred[method]) if frame_pred[method] else np.empty(0, np.int64)
            chosen = unique_winner(votes[method][used], np.r_[0, objects])
            consensus[method] = chosen
            summaries[method] = dict(frame_source=summary_for(y, predicted, objects),
                                     source_consensus=summary_for(labels[validation[used]], chosen, objects))
        np.savez_compressed(folder / 'source_consensus.npz', source_ids=validation[used],
                            pseudo_identity=labels[validation[used]], **consensus)
        stats = dict(anchors=int((labels > 0).sum()), ambiguous_sources=int((labels < 0).sum()),
                     train_sources=int(((labels > 0) & train).sum()), withheld_sources=len(validation),
                     tested_sources=int(used.sum()), tested_objects=len(np.unique(labels[validation[used]])), total_objects=len(objects),
                     total_frames=len(frames), used_frames=sum(not f.get('skipped', False) for f in frames),
                     frame_source_observations=len(y), seconds=time.monotonic()-start)
        dump(folder / 'frames.json', frames)
        dump(folder / 'summary.json', dict(stats=stats, methods=summaries))
        for name in ['source_consensus.npz', 'frames.json', 'summary.json']:
            f = folder/name; files[str(f)] = sha(f)
        results[key] = dict(stats=stats, methods=summaries)
        print('MASK CODE PROBE', key, stats, {m: s['source_consensus']['macro'] for m,s in summaries.items()}, flush=True)
    gate = all(r['methods']['full_code']['source_consensus']['macro'] is not None and
               r['methods']['full_code']['source_consensus']['macro'] > max(r['methods'][m]['source_consensus']['macro'] for m in ['single_mask', 'shuffled_code']) and
               r['methods']['full_code']['source_consensus']['macro'] >= r['methods']['mask_cosine']['source_consensus']['macro'] for r in results.values())
    dump(OUT / 'predictions_complete.json', dict(complete=True, files=files, results=results, information_screen_passed=bool(gate),
         protocol_sha256=sha(OUT/'protocol.json'), gt_read=False, new_map_generated=False, semantic_accuracy_available=False,
         limitation='Predicted core anchors were used by the upstream map/masks. Spatial holdout only separates this code fit, not upstream estimation. This can screen distinguishability, never establish correct identity or unknown-point precision.'))
    print('INFORMATION SCREEN', bool(gate), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['freeze', 'run'])
    globals()[parser.parse_args().action]()
