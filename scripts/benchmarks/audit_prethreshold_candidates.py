"""GT-assisted information audit of prethreshold measured proposals, not AP."""
import argparse
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from capture_prethreshold_candidates import OUT, FRONT, INPUT, verify
from run_surface_ownership import OUT as OWNERSHIP
from omni_semantic_map.native_query_support import exclusive_measured_queries
from omni_semantic_map.sparse_support_components import largest_measured_component

AUDIT = OUT/'measured_audit'
CLASSES = {26: 13, 11: 2}


def freeze():
    assert not AUDIT.exists(); p = verify(); seal = read(OUT/'predictions_complete.json'); assert seal['old_replay_exact']
    for f, h in seal['files'].items(): assert sha(f) == h
    sources = [Path(__file__), ROOT/'omni_semantic_map/omni_semantic_map/native_query_support.py', ROOT/'omni_semantic_map/test/test_native_query_support.py']
    dump(AUDIT/'protocol.json', dict(created_epoch=time.time(), prediction_seal_sha256=sha(OUT/'predictions_complete.json'), sources={str(f): sha(f) for f in sources},
        scope='Post-seal development-only GT-assisted mechanism diagnosis. No trained model, new map/AP, threshold selection or new-window access.',
        support='Stable native descending-score exclusive arbitration at measured pixels. Lower cutoff0.1 fixed in capture protocol. High queries>0.5 must reproduce old raw/connected supports exactly. Deferred0.1<score<=0.5 cannot take pixels from high queries. Same largest-component0.4m/min16 as native baseline.',
        oracle_anchor='For diagnostic grouping ONLY, source nearest valid reference within0.2m, confidence-weighted dominant same-class GT with >0.5 of known source mass. Ignore semantic255/unmatched/nonpositive-confidence references. Unknown source counts retained. This is a GT-assisted hypothetical grouping, not deployable association.',
        coverage='For each GT, compare same-class high raw, high connected, high+deferred raw, high+deferred connected support within0.2m. This permits perfect removal of wrong points and is only a necessary coverage condition, not IoU/AP or a joint map.',
        continue_condition='In EACH development scene at least one previously unmatched AP50 GT must have >=3 distinct deferred frames with >=16 connected sources and >0.5 diagnostic same-class purity, at least100 weighted correct unique source mass outside all original high raw same-class supports, and combined same-class connected coverage>0.5. Stronger full-map gates remain necessary if a predictor is later designed.',
        novelty='Even a successful information gate would not establish contribution. Lower-confidence association/confirmation is not intrinsically new; any future delayed-admission mechanism must outperform the fixed0.1 baseline with the same backend and evidence.'))
    print('AUDIT FROZEN', flush=True)


def run():
    import runpy
    from evaluate_kitti360_development import read_static_ply, transfer
    from audit_kitti360_object_retention import covered
    p = verify(); ap = read(AUDIT/'protocol.json'); assert sha(OUT/'predictions_complete.json') == ap['prediction_seal_sha256']; assert not (AUDIT/'summary.json').exists()
    for f, h in ap['sources'].items(): assert sha(f) == h
    prediction = read(OUT/'predictions_complete.json')
    for f, h in prediction['files'].items(): assert sha(f) == h
    parent = read(OWNERSHIP/'protocol.json'); baseline = read(OWNERSHIP/'summary.json'); original = read(FRONT/'masks_complete.json'); taxonomy = runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['id2label']; results = {}; gates = {}
    for s in p['scenes']:
        key = s['key']; m = read(INPUT/key/'manifest.json'); xyz = np.load(m['geometry'])['xyz']; measured = {r['frame_id']: r for r in m['records']}; old = {r['frame_id']: r for r in original['scenes'][key]['records']}
        ref = np.load(s['reference']); info = parent['references'][key]; assert sha(info['raw']) == info['sha256']; raw = read_static_ply(info['raw']); train = np.array([taxonomy[int(v)].trainId for v in raw['semantic']], np.int64)
        expected = train*1000+raw['instance'].astype(np.int64)%1000; expected[train == 255] = 0; np.testing.assert_array_equal(expected, ref['train_instance'])
        nearest = transfer(ref['xyz'], np.arange(1, len(raw)+1), xyz, .2); ix = np.maximum(nearest-1, 0); known = (nearest>0)&(train[ix]!=255)&(ref['confidence'][ix]>0); truth = ref['train_instance'][ix]; weight = ref['confidence'][ix]
        names = ['high_raw', 'high_connected', 'combined_raw', 'combined_connected']; domains = {sem: {stage: np.zeros(len(xyz), bool) for stage in names} for sem in CLASSES}; nodes = []; source_groups = defaultdict(set); frame_groups = defaultdict(set); counts = dict(deferred=0, deferred_raw16=0, deferred_connected16=0, deferred_pure16=0); files = {}
        for index, r in enumerate(prediction['results'][key]):
            data = np.load(r['file']); f = np.load(measured[r['frame_id']]['file']); ids = f['visible_source_ids']; px = f['pixel_xy']; width = int(data['original_size'][0])
            winner = exclusive_measured_queries(data['masks_packed'], data['scores'], px, width, .1); high = data['scores'] > .5
            high_index = np.full(len(high), -1, np.int64); high_index[high] = np.arange(high.sum())
            native = np.where(winner>=0, high_index[np.maximum(winner, 0)], -1); previous = np.load(old[r['frame_id']]['file']); np.testing.assert_array_equal(native, previous['segmentation'][px[:, 1], px[:, 0]])
            old_support = np.load(FRONT/'frontend'/key/'supports'/f'{r["frame_id"]:06d}.npz'); pieces = []; query_ids = []
            for q in np.flatnonzero(data['scores'] > .1):
                sem = int(data['semantic_ids'][q]); raw_ids = ids[winner == q]; connected = raw_ids[largest_measured_component(xyz[raw_ids], .4, 16)]
                domains[sem]['combined_raw'][raw_ids] = True; domains[sem]['combined_connected'][connected] = True
                if high[q]:
                    domains[sem]['high_raw'][raw_ids] = True; domains[sem]['high_connected'][connected] = True; j = high_index[q]
                    np.testing.assert_array_equal(connected, old_support['point_ids'][old_support['offsets'][j]:old_support['offsets'][j+1]])
                    continue
                counts['deferred'] += 1; counts['deferred_raw16'] += len(raw_ids)>=16; counts['deferred_connected16'] += len(connected)>=16
                use = connected[known[connected]]; anchor = -1; purity = 0.
                if len(use):
                    mass = np.bincount(truth[use], weights=weight[use]); target = np.flatnonzero(np.arange(len(mass))//1000 == CLASSES[sem])
                    if len(target):
                        best = int(target[np.argmax(mass[target])]); purity = float(mass[best]/mass.sum())
                        if purity > .5: anchor = best
                if anchor >= 0 and len(connected)>=16:
                    counts['deferred_pure16'] += 1; frame_groups[anchor].add(r['frame_id']); source_groups[anchor].update(map(int, connected))
                nodes.append(dict(frame_id=r['frame_id'], query_id=int(q), semantic_id=sem, score=float(data['scores'][q]), class_logit=float(data['class_logits'][q]), presence_logit=float(data['presence_logits'][q]), raw_points=len(raw_ids), connected_points=len(connected), known_points=len(use), diagnostic_gt=anchor, diagnostic_purity=purity))
                pieces.append(connected); query_ids.append(q)
            folder = AUDIT/'supports'/key; folder.mkdir(parents=True, exist_ok=True); path = folder/f'{r["frame_id"]:06d}.npz'
            np.savez_compressed(path, query_ids=np.asarray(query_ids, np.int64), point_ids=np.concatenate(pieces) if pieces else np.empty(0, np.int64), offsets=np.r_[0, np.cumsum([len(x) for x in pieces])]); files[str(path)] = sha(path)
            if (index+1)%20 == 0: print('AUDIT SUPPORT', key, index+1, len(prediction['results'][key]), flush=True)
        matched = set(baseline['results'][key]['methods']['baseline']['matched_gt_ids']); objects = []
        for sem, cid in CLASSES.items():
            selected = ref['train_instance']//1000 == cid; reference_xyz = ref['xyz'][selected]; gt = ref['train_instance'][selected]; confidence = ref['confidence'][selected]
            coverage = {stage: covered(xyz, domains[sem][stage], reference_xyz) for stage in names}
            for gid in np.unique(gt):
                mask = gt == gid; mass = float(confidence[mask].sum())
                if mass < 100: continue
                values = {stage: float(confidence[mask & hit].sum()/mass) for stage, hit in coverage.items()}
                sources = np.asarray(sorted(source_groups[int(gid)]), np.int64); novel = sources[known[sources] & (truth[sources]==gid) & ~domains[sem]['high_raw'][sources]]
                novel_mass = float(weight[novel].sum()); frames = sorted(frame_groups[int(gid)])
                viable = bool(int(gid) not in matched and len(frames)>=3 and novel_mass>=100 and values['combined_connected']>.5)
                objects.append(dict(gt_id=int(gid), category='car' if sem == 26 else 'building', already_matched=int(gid) in matched, coverage=values, diagnostic_deferred_frames=frames, new_correct_source_mass=novel_mass, viable_for_hypothesis=viable))
        gates[key] = any(o['viable_for_hypothesis'] for o in objects)
        result = dict(counts={k: int(v) for k,v in counts.items()}, objects=objects, gate=gates[key], high_support_replay_exact=True, source_counts={str(sem): {stage: int(value.sum()) for stage,value in stages.items()} for sem, stages in domains.items()})
        dump(AUDIT/key/'nodes_GT_ASSISTED.json', nodes); dump(AUDIT/key/'result.json', result); dump(AUDIT/key/'supports_complete.json', dict(files=files, complete=True)); results[key] = result
        print('AUDIT RESULT', key, counts, 'VIABLE', [(o['gt_id'], len(o['diagnostic_deferred_frames']), o['new_correct_source_mass']) for o in objects if o['viable_for_hypothesis']], flush=True)
    dump(AUDIT/'summary.json', dict(complete=True, diagnostic_only=True, not_map_accuracy=True, results=results, gate_by_scene=gates, information_gate=all(gates.values()), protocol_sha256=sha(AUDIT/'protocol.json'), prediction_seal_sha256=sha(OUT/'predictions_complete.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['freeze', 'run']); globals()[parser.parse_args().action]()
