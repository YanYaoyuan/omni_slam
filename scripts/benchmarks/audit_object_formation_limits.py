"""GT-assisted diagnosis of object formation; never exports mapping predictions.

All interventions below operate on frozen reference partitions. They are
optimistic diagnostics, not implementable policies, new AP, or independent tests.
"""
import argparse
import itertools
import time

import numpy as np
from scipy.spatial import cKDTree

from paper_final_common import ROOT, read, dump, sha

OUT = ROOT / 'build/benchmarks/object_formation_limits_v1'
PARENT = ROOT / 'build/benchmarks/counterfactual_geometry_revision_v1'
SUPPORT = ROOT / 'build/benchmarks/strong_baseline_support_audit_v1'
INPUT = ROOT / 'build/benchmarks/discovery_measured_inputs_v1'


def best_union(intersection, false_mass, gt_mass):
    """Exact per-GT optimum over unions of disjoint frozen prediction regions.

    At optimum ratio r, an atom is worth including iff I-r*F>0. Therefore
    testing prefixes ordered by I/F attains the optimum, without 2**N search.
    Different GT optima may demand the same atom: no joint feasibility claim.
    """
    inter = np.asarray(intersection, dtype=float)
    false = np.asarray(false_mass, dtype=float)
    assert inter.shape == false.shape and inter.ndim == 1
    assert gt_mass > 0 and np.all(inter >= 0) and np.all(false >= -1e-7)
    false = np.maximum(false, 0)
    eligible = np.flatnonzero(inter > 0)
    if not len(eligible):
        return 0., []
    ratio = np.divide(inter, false, out=np.full_like(inter, np.inf), where=false > 0)
    order = sorted(eligible, key=lambda j: (-ratio[j], int(j)))
    values = np.cumsum(inter[order]) / (gt_mass + np.cumsum(false[order]))
    k = int(np.argmax(values))
    return float(values[k]), list(map(int, order[:k+1]))


def self_test():
    rng = np.random.default_rng(0)
    for n in range(1, 9):
        for _ in range(40):
            i = rng.integers(0, 20, n).astype(float)
            f = rng.integers(0, 20, n).astype(float)
            g = float(i.sum() + 1)
            value, selected = best_union(i, f, g)
            brute = max(sum(i[j] for j in ids)/(g+sum(f[j] for j in ids))
                        for k in range(n+1) for ids in itertools.combinations(range(n), k))
            np.testing.assert_allclose(value, brute, atol=1e-14, rtol=0)
            np.testing.assert_allclose(value, i[selected].sum()/(g+f[selected].sum()))
    assert best_union([0, 0], [0, 3], 1) == (0., [])
    # A clean small fragment and a dirty large fragment: taking everything loses.
    value, selected = best_union([6, 1], [0, 100], 10)
    assert value == .6 and selected == [0]
    print('PASS: 320 exhaustive subset checks, empty and contaminated union cases', flush=True)


def run():
    assert not OUT.exists(), 'Preserve prior diagnostics; do not overwrite.'
    p = read(PARENT / 'protocol.json')
    inputs = {}
    for scene in p['scenes']:
        key = scene['key']
        manifest = read(INPUT/key/'manifest.json')
        paths = [scene['reference'], INPUT/key/'manifest.json', manifest['geometry'],
                 SUPPORT/(key+'.json'), PARENT/'maps'/key/'baseline/reference_prediction.npz',
                 PARENT/'maps'/key/'baseline/classification.json']
        for path in paths:
            inputs[str(path)] = sha(path)
    dump(OUT/'protocol.json', dict(created_epoch=time.time(), source_sha256=sha(__file__), inputs=inputs,
        scope='All94eligible car/building GTs in two exposed outdoor development windows. No model inference, fitting, new maps, AP scoring, robot or independent data.',
        fixed_partition='Frozen baseline reference nearest-map assignments and original classes. Verify best IoU against prior all-object coverage audit. All reference confidence mass including void contributes to predicted region mass.',
        single_rejection='Maximum existing same-class object intersection / GTmass. Allows perfect removal of false reference labels without geometry retransfer: optimistic reference-space ceiling, not a physical deletion result.',
        merge='For each GT independently, exact best union IoU of any subset of same-class frozen prediction regions, without point removal. Sort I/F prefixes; validated against exhaustive subset enumeration. Different GTs can request the same object; not jointly feasible or AP.',
        merge_rejection='Perfect identity regrouping and false-reference-label rejection over all same-class assigned reference points. Optimistic coverage only, no geometry movement or new coverage.',
        residual='Partition GT mass into best-object correct, other same-class, other-class, unassigned-source-covered, unassigned-source-uncovered. Source coverage uses all measured XYZ within strict20cm. No point labels or oracle assignments are exported for prediction.',
        decision='Compare at fixed strictIoU0.50 and0.75. Report allGTs and nonexclusive diagnostic opportunities. No tuning or automatic new method claims.'))
    results = {}
    for scene in p['scenes']:
        started = time.monotonic()
        key = scene['key']
        ref = np.load(scene['reference'])
        truth, weight = ref['train_instance'], ref['confidence'].astype(float)
        pred = np.load(PARENT/'maps'/key/'baseline/reference_prediction.npz')['instance']
        objects = read(PARENT/'maps'/key/'baseline/classification.json')['objects']
        classes = {o['object_id']: o['predictions']['mean']['train_id'] for o in objects}
        gt_mass = np.bincount(truth, weights=weight)
        pred_mass = np.bincount(pred, weights=weight, minlength=max(classes)+1)
        maximum_gt = len(gt_mass)
        joint = np.bincount(pred.astype(np.int64)*maximum_gt+truth,
                            weights=weight, minlength=len(pred_mass)*maximum_gt).reshape(-1, maximum_gt)
        category = np.full(len(pred_mass), -1, int)
        for oid, tid in classes.items():
            category[oid] = tid
        support = read(SUPPORT/(key+'.json'))['objects']
        manifest = read(INPUT/key/'manifest.json')
        source = np.load(manifest['geometry'])['xyz']
        target = np.isin(truth//1000, [2, 13])
        measured = np.zeros(len(truth), bool)
        distance = cKDTree(source).query(ref['xyz'][target], distance_upper_bound=.2, workers=1)[0]
        measured[target] = distance < .2
        rows = []
        for old in support:
            gid, mass = old['gt_id'], gt_mass[old['gt_id']]
            ids = np.array([oid for oid, tid in classes.items() if tid == gid//1000], dtype=int)
            inter = joint[ids, gid]
            false = np.maximum(pred_mass[ids]-inter, 0)
            values = np.where(pred_mass[ids] >= 100, inter/(mass+false), 0.)
            best = int(np.argmax(values)) if len(values) else None
            best_iou = float(values[best]) if best is not None else 0.
            np.testing.assert_allclose(best_iou, old['best_actual']['iou'], atol=1e-10, rtol=0)
            union, selected = best_union(inter, false, mass)
            best_oid = int(ids[best]) if best is not None and inter[best] > 0 else None
            take = truth == gid
            same = category[pred[take]] == gid//1000
            correct_best = pred[take] == best_oid if best_oid is not None else np.zeros(take.sum(), bool)
            zero = pred[take] == 0
            w = weight[take]
            pieces = dict(best_object=float(w[correct_best].sum()/mass),
                other_same_class=float(w[same & ~correct_best].sum()/mass),
                other_class=float(w[~same & ~zero].sum()/mass),
                unassigned_measured=float(w[zero & measured[take]].sum()/mass),
                unassigned_unmeasured=float(w[zero & ~measured[take]].sum()/mass))
            np.testing.assert_allclose(sum(pieces.values()), 1., atol=1e-10)
            np.testing.assert_allclose(float(w[measured[take]].sum()/mass), old['coverage']['all_measured'], atol=1e-10)
            rows.append(dict(gt_id=gid, category=old['category'], mass=float(mass), best_object_id=best_oid,
                actual_best_iou=best_iou, single_rejection_ceiling=float(inter.max(initial=0)/mass),
                best_union_iou=union, union_object_ids=ids[selected].tolist(),
                merge_rejection_ceiling=float(inter.sum()/mass),
                measured_coverage=old['coverage']['all_measured'],
                weak_query_coverage=old['coverage']['any_native'],
                high_query_coverage=old['coverage']['high_native'], mass_partition=pieces))
        counts = {}
        for name in ['building', 'car']:
            rs = [r for r in rows if r['category'] == name]
            counts[name] = dict(gt_count=len(rs), thresholds={})
            for t in [.5, .75]:
                missing = [r for r in rs if r['actual_best_iou'] <= t]
                metrics = ['actual_best_iou', 'single_rejection_ceiling', 'best_union_iou',
                           'merge_rejection_ceiling', 'measured_coverage', 'weak_query_coverage', 'high_query_coverage']
                counts[name]['thresholds'][str(t)] = dict(
                    above={m:sum(r[m]>t for r in rs) for m in metrics},
                    missing=len(missing),
                    recoverable_by_rejection_only=[r['gt_id'] for r in missing if r['single_rejection_ceiling']>t],
                    recoverable_by_merge_only=[r['gt_id'] for r in missing if r['best_union_iou']>t],
                    needs_new_support_even_with_perfect_existing_identity=[r['gt_id'] for r in missing if r['merge_rejection_ceiling']<=t and r['measured_coverage']>t],
                    insufficient_raw_coverage=[r['gt_id'] for r in missing if r['measured_coverage']<=t],
                    missing_mean_partition={k:float(np.mean([r['mass_partition'][k] for r in missing])) if missing else 0 for k in pieces})
        result = dict(objects=rows, counts=counts, seconds=time.monotonic()-started)
        dump(OUT/(key+'.json'), result)
        results[key] = result
        print(key, counts, flush=True)
    for path, digest in inputs.items():
        assert sha(path) == digest
    dump(OUT/'summary.json', dict(complete=True, gt_diagnostic_only=True, new_maps=0,
        independent_test=False, results=results, inputs_unchanged=True, protocol_sha256=sha(OUT/'protocol.json')))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['self_test', 'run'])
    args = ap.parse_args()
    globals()[args.action]()
