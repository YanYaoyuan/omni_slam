"""Read-only post-hoc audit. No network, fitting, SAM inference or map replay.

Compare archived support partitions and observation memberships. Memberships
are NOT per-point geometry provenance, and endpoint co-membership is NOT proof
that a useful split was destroyed. GT is used only for offline diagnostics.
"""
from collections import Counter
from itertools import combinations
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'build/benchmarks/memory_resegmentation_20261112'
DEST = ROOT / 'build/benchmarks/memory_mechanism_audit_20261112'
KEYS = ['replica_office_0', 'replica_office_1', 'scenenn_014', 'scenenn_015']


def read(p):
    return json.loads(Path(p).read_text())


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def dump(p, data):
    Path(p).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def local_score(truth, prediction):
    good = truth >= 0
    y, pred = truth[good], prediction[good]
    gids, gi = np.unique(np.r_[0, y], return_inverse=True)
    pids, pi = np.unique(np.r_[0, pred], return_inverse=True)
    table = np.bincount(gi[1:] * len(pids) + pi[1:], minlength=len(gids)*len(pids)).reshape(len(gids), len(pids))
    union = table.sum(1)[:, None] + table.sum(0)[None] - table
    iou = np.divide(table, union, out=np.zeros_like(table, float), where=union > 0)
    hit = iou > .5
    hit[0, :] = False
    hit[:, 0] = False
    matched = gids[np.flatnonzero(hit.any(1))].tolist()
    tp = len(matched)
    fp = int((table.sum(0)[1:] > 0).sum()) - tp
    fn = len(gids) - 1 - tp
    den = tp + .5*(fp+fn)
    return dict(pq=float(iou[hit].sum()/den) if den else 0., matched=matched, tp=tp, fp=fp, fn=fn)


def membership(path):
    rows = read(path)['objects']
    raw, fg = {}, {}
    for row in rows:
        for oid in row['observation_ids']:
            assert oid not in raw
            raw[oid] = row['object_id']
            if row['predictions']['mean']['retained']:
                fg[oid] = row['object_id']
    return raw, fg


def state(children, mapping):
    owners = [mapping.get(i) for i in children]
    present = [x for x in owners if x is not None]
    if len(children) < 2:
        status = 'not_split'
    elif not present:
        status = 'all_absent'
    elif len(present) < len(children):
        status = 'some_absent'
    elif len(set(present)) == 1:
        status = 'all_same_object'
    elif len(set(present)) == len(children):
        status = 'all_distinct_objects'
    else:
        status = 'some_same_object'
    return dict(status=status, owners=owners)


def partition(raw, parent, original_ids):
    children = np.flatnonzero(raw['parents'] == parent)
    order = np.argsort(original_ids)
    sorted_ids = original_ids[order]
    assert len(np.unique(original_ids)) == len(original_ids)
    pred = np.zeros(len(original_ids), np.int32)
    sizes = []
    for label, child in enumerate(children, 1):
        lo, hi = raw['offsets'][child:child+2]
        ids = raw['point_ids'][lo:hi]
        places = np.searchsorted(sorted_ids, ids)
        assert np.array_equal(sorted_ids[places], ids)
        assert not pred[order[places]].any()
        pred[order[places]] = label
        sizes.append(len(ids))
    assert (pred > 0).all()
    return pred, children, sizes


def disagreement(a, b):
    n, m = int(a.max()), int(b.max())
    table = np.bincount((a-1)*m+b-1, minlength=n*m).reshape(n, m)
    x, y = linear_sum_assignment(-table)
    return int(len(a)-table[x, y].sum())


def main():
    assert not DEST.exists(), 'Use a new audit namespace; never overwrite results'
    DEST.mkdir(parents=True)
    protocol = read(SOURCE/'protocol.json')
    seal = read(SOURCE/'predictions_complete.json')
    assert seal['complete'] and len(seal['runs']) == 12
    checked = {}
    def check(p, expected=None):
        p = Path(p)
        h = digest(p)
        if expected is not None:
            assert h == expected, str(p)
        checked[str(p)] = h
    for name in ['protocol.json', 'selection.json', 'validation.json', 'summary.json', 'predictions_complete.json']:
        check(SOURCE/name)
    for name, expected in seal['runs'].items():
        runpath = SOURCE/name
        check(runpath, expected)
        for artifact, h in read(runpath)['artifacts'].items():
            check(runpath.parent/artifact, h)
    for name, h in read(SOURCE/'delivery_audit.json')['sources'].items():
        check(ROOT/name, h)
    for name in ['omni_semantic_map/omni_semantic_map/replay_fusion.py',
                 'scripts/benchmarks/run_livo_history_graph.py',
                 'omni_semantic_map/omni_semantic_map/fusion_diagnostics.py',
                 'build/benchmarks/expanded_semantic_20261109/concept-graphs/conceptgraph/slam/utils.py',
                 'build/benchmarks/expanded_semantic_20261109/concept-graphs/conceptgraph/slam/mapping.py']:
        check(ROOT/name)
    all_rows, scenes = [], []
    for key in KEYS:
        s = next(x for x in protocol['scenes'] if x['key'] == key)
        cache = Path(s['cache'])
        cm = read(cache/'manifest.json')
        im = read(SOURCE/'inputs'/key/'manifest.json')
        labels_path = ROOT/'build/benchmarks/full_object_mainline_20261112/supervision/evaluation'/key/'point_targets.npz'
        check(labels_path)
        truth = load(labels_path)['instance']
        manifests = {arm: read(SOURCE/'regenerated'/key/arm/'manifest.json') for arm in ['B1', 'B2']}
        stages = {arm: {stage: membership(SOURCE/'maps'/key/arm/stage/'classification.json')
                       for stage in ['prefix_050', 'prefix_100', 'prefix_150', 'final']}
                  for arm in ['B1', 'B2']}
        counts = Counter()
        for i, (rec, inp) in enumerate(zip(cm['records'], im['records'])):
            assert rec['frame_id'] == inp['frame_id']
            rawpath = cache/'frames'/rec['file']
            check(rawpath)
            original = load(rawpath)
            regenerated = {}
            for arm in ['B1', 'B2']:
                mr = manifests[arm]['records'][i]
                assert mr['frame_id'] == rec['frame_id']
                path = SOURCE/'regenerated'/key/arm/mr['file']
                check(path, mr['sha256'])
                regenerated[arm] = load(path)
                # Equal point budget does not fix the jitter assigned to a point.
                ids = regenerated[arm]['point_ids']
                assert np.array_equal(np.sort(ids), np.sort(original['point_ids']))
                counts[arm+'_point_occurrences_reordered_vs_original'] += int(np.count_nonzero(ids != original['point_ids']))
            counts['point_occurrences'] += len(original['point_ids'])
            counts['B1_B2_point_occurrences_reordered'] += int(np.count_nonzero(regenerated['B1']['point_ids'] != regenerated['B2']['point_ids']))
            for e in inp['events']:
                j = e['mask']
                lo, hi = original['offsets'][j:j+2]
                ids = original['point_ids'][lo:hi]
                y = truth[ids]
                parent = rec['observation_offset']+j
                base = local_score(y, np.ones(len(ids), np.int32))
                row = dict(scene=key, frame=rec['frame_id'], frame_index=i, mask=int(j), parent=int(parent),
                           points=len(ids), valid_points=int((y>=0).sum()), foreground_points=int((y>0).sum()),
                           gt_ids=np.unique(y[y>0]).tolist(), B0_local=base)
                predictions = {}
                counts['events'] += 1
                counts['events_with_any_fg'] += int((y>0).any())
                counts['events_majority_fg'] += int((y>0).sum() > .5*(y>=0).sum())
                for arm in ['B1', 'B2']:
                    r = regenerated[arm]
                    pred, children, sizes = partition(r, parent, ids)
                    assert all(np.array_equal(r['features'][child], original['features'][j]) for child in children)
                    predictions[arm] = pred
                    child_ids = (children+manifests[arm]['records'][i]['observation_offset']).tolist()
                    score = local_score(y, pred)
                    gained = sorted(set(score['matched'])-set(base['matched']))
                    lost = sorted(set(base['matched'])-set(score['matched']))
                    endpoints = {st: state(child_ids, maps[0]) for st, maps in stages[arm].items()
                                 if st == 'final' or int(st[-3:]) >= i+1}
                    final = endpoints['final']
                    fgfinal = state(child_ids, stages[arm]['final'][1])
                    distinct_then_same = []
                    for a, b in combinations(child_ids, 2):
                        fm = stages[arm]['final'][0]
                        if a not in fm or b not in fm or fm[a] != fm[b]:
                            continue
                        separated = [st for st in endpoints if st!='final' and a in stages[arm][st][0]
                                     and b in stages[arm][st][0] and stages[arm][st][0][a] != stages[arm][st][0][b]]
                        if separated:
                            distinct_then_same.append(dict(children=[a,b],separated_at=separated))
                    row[arm] = dict(children=child_ids,sizes=sizes,local=score,local_gained=gained,local_lost=lost,
                                    endpoint_memberships=endpoints,foreground_final=fgfinal,
                                    observed_distinct_then_same=distinct_then_same)
                    counts[arm+'_split_events'] += int(len(children)>1)
                    counts[arm+'_local_gains'] += len(gained)
                    counts[arm+'_local_losses'] += len(lost)
                    counts[arm+'_local_gain_events'] += bool(gained)
                    counts[arm+'_local_gain_final_all_same'] += bool(gained) and final['status']=='all_same_object'
                    counts[arm+'_final_'+final['status']] += 1
                    counts[arm+'_fg_final_'+fgfinal['status']] += 1
                    counts[arm+'_distinct_then_same_events'] += bool(distinct_then_same)
                    counts[arm+'_children_below_16_points'] += sum(x<16 for x in sizes)
                    counts[arm+'_children'] += len(children)
                changed = disagreement(predictions['B1'], predictions['B2'])
                row['partition_disagreement_points_B1_B2'] = changed
                counts['B1_B2_different_partition_events'] += int(changed>0)
                counts['B1_B2_disagreement_point_occurrences'] += changed
                counts['event_support_occurrences'] += len(ids)
                all_rows.append(row)
            if i%50 == 0:
                print('AUDIT',key,i+1,flush=True)
        scenes.append(dict(scene=key, counts=dict(counts)))
        print('SCENE COMPLETE', key,dict(counts),flush=True)
    totals = Counter()
    for s in scenes:
        totals.update(s['counts'])
    validation = read(SOURCE/'validation.json')
    v = {k: v for k,v in validation['selected'].items()}
    check(Path(__file__))
    for name,h in checked.items():
        assert digest(name)==h, 'Input mutated during audit: '+name
    dump(DEST/'events.json',all_rows)
    dump(DEST/'summary.json',dict(complete=True,created=time.time(),scenes=scenes,totals=dict(totals),
        validation_selected=v,training_runs_added=0,map_runs_added=0,sam_calls_added=0,model_forward_calls_added=0,
        definitions=dict(local='Per original mask, same labeled measured point support; repeated observations, NOT final-map PQ.',
            partition_disagreement='Minimum mismatched support occurrences after optimal permutation of partition IDs.',
            all_same_object='All child observation IDs appear in one final postprocessed object; no per-point survival claim.',
            distinct_then_same='At least one sibling pair in distinct objects at a saved postprocessed prefix and one object at final; exact merge stage unlogged.',
            point_reordering='Different point ID at same random-stream position. CG applies 4 mm jitter by ingestion order; fixed seed is not fixed per-point perturbation.'),
        unidentifiable=['Actual archived SAM prompt coordinates were not saved.',
            'B1 versus B2 changes learning and history together; cannot isolate history causally.',
            'No pre-final online snapshots or per-point fusion ancestry; cannot uniquely assign loss to online versus final merge.',
            'A missing child membership may be rejected or filtered; exact removal stage unlogged.'],
        input_hashes=checked))
    print('AUDIT COMPLETE',dict(totals),flush=True)


if __name__ == '__main__':
    main()
