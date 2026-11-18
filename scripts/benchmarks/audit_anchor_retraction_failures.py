"""Read-only attribution of the closed retraction candidate's quality loss.

Raw semantic IDs resolve road-vs-ignore ambiguity only for this diagnosis;
the original evaluator, predictions and failed gate are never changed.
"""
import argparse
from pathlib import Path
import runpy
import time
import numpy as np
from paper_final_common import ROOT, read, dump, sha
import run_anchor_counterevidence as experiment

OUT = ROOT / 'build/benchmarks/anchor_retraction_failure_audit_v1'
PARENT = experiment.OUT
TAXONOMY = ROOT / 'build/datasets/kitti360_prepare/labels.py'


def freeze():
    p = experiment.verify()
    assert not OUT.exists()
    assert not read(PARENT / 'summary.json')['development_gate']
    files = {}
    for name in ['prepared', 'features_complete', 'decisions_complete', 'predictions_complete', 'summary']:
        seal = experiment.seal(name)
        files.update(seal['files'])
        files[str(PARENT / (name + '.json'))] = sha(PARENT / (name + '.json'))
    refs = read(ROOT / 'build/benchmarks/provenance_object_memory_v1/protocol.json')['references']
    for key, r in refs.items():
        assert sha(r['raw']) == r['sha256']; files[r['raw']] = r['sha256']
    files[str(TAXONOMY)] = sha(TAXONOMY)
    files[str(PARENT / 'per_object.json')] = sha(PARENT / 'per_object.json')
    sources = [Path(__file__), ROOT / 'scripts/benchmarks/evaluate_kitti360_development.py',
               ROOT / 'scripts/benchmarks/probe_utonia_ownership_evidence.py']
    dump(OUT / 'protocol.json', dict(created_epoch=time.time(), parent_protocol_sha256=sha(PARENT / 'protocol.json'),
        contexts=[c for c in p['contexts'] if c['reference']], raw_references=refs, files=files,
        sources={str(f): sha(f) for f in sources},
        question='Separate mistaken anchor rejection from propagation damage, and distinguish ranking information from the closed zero-margin majority decision.',
        scope='Posthoc diagnosis on two exposed windows; no new model calls, fits, maps, AP scores, decision thresholds or candidate promotion. Failed parent remains closed.',
        reference='Recover train IDs from pinned original PLY/taxonomy and verify row geometry/confidence. Road train0 is known here; ignored255 remains unknown. This DOES NOT alter the original evaluation adapter.',
        map_diagnostic='All withdrawn reference queries partitioned into direct-only and extra propagation; report confidence mass by semantic correctness and fixed baseline GT correspondence (best same-class IoU>0.5, mass>=100). Unmatched objects remain correspondence-unknown, not wrong. Fixed correspondence is descriptive, not one-to-one AP matching.',
        anchor_diagnostic='All actual positive foreground source anchors, no oracle foreground filter. Attach nearest reference within0.2m with deterministic GT-index ties, then read known-label correctness of the declared class. Label every retained/rejected anchor unit through its actual member sources; no centroid GT proxy. Compute confidence-weighted source-level AUROC of stored mean_margin and refute/eligible; aggregate rejection mass, all classes separately. Correlated sources are not independent statistical trials.',
        outputs='Exact stage attribution and all predicted objects; no scan/threshold/radius/prototype variants. This is a mechanism diagnosis, not a new semantic-map result.'))
    print('RETRACTION FAILURE AUDIT FROZEN', flush=True)


def verify():
    p = read(OUT / 'protocol.json')
    assert sha(PARENT / 'protocol.json') == p['parent_protocol_sha256']
    experiment.verify()
    for f, h in {**p['files'], **p['sources']}.items(): assert sha(f) == h, f
    return p


def reference(c, p):
    from evaluate_kitti360_development import read_static_ply
    ref = np.load(c['reference']); raw = read_static_ply(p['raw_references'][c['key']]['raw'])
    xyz = np.column_stack([raw[k] for k in ['x', 'y', 'z']]).astype(np.float64) - np.asarray(read(c['job'])['origin_world'])
    np.testing.assert_array_equal(xyz, ref['xyz'])
    np.testing.assert_array_equal(raw['confidence'], ref['confidence'])
    taxonomy = runpy.run_path(str(TAXONOMY))['id2label']
    train = np.array([taxonomy[int(v)].trainId for v in raw['semantic']], np.int16)
    truth = train.astype(np.int64) * 1000 + raw['instance'].astype(np.int64) % 1000
    truth[train == 255] = 0
    np.testing.assert_array_equal(truth, ref['train_instance'])
    return ref, train


def mass_record(mask, weight):
    return dict(queries=int(mask.sum()), confidence_mass=float(weight[mask].sum()))


def map_changes(c, ref, train):
    key = c['key']; old = np.load(Path(c['ours']) / 'reference_prediction.npz')['instance']
    direct = np.load(PARENT / 'maps' / key / 'direct_only/reference_prediction.npz')['instance']
    ours = np.load(PARENT / 'maps' / key / 'ours_retract_propagation/reference_prediction.npz')['instance']
    assert ((direct == 0) | (direct == old)).all() and ((ours == 0) | (ours == direct)).all()
    masks = dict(direct=(old > 0) & (direct == 0), propagation_extra=(direct > 0) & (ours == 0))
    np.testing.assert_array_equal(masks['direct'] | masks['propagation_extra'], (old > 0) & (ours == 0))
    cls = read(Path(c['ours']) / 'classification.json')['objects']; n = int(old.max()) + 1
    category = np.full(n, -1, np.int16)
    for obj in cls: category[obj['object_id']] = obj['predictions']['mean']['train_id']
    truth, weight = ref['train_instance'], ref['confidence']
    known = (train != 255) & (weight > 0)
    mass = np.bincount(truth, weights=weight)
    pmass = np.bincount(old, weights=weight, minlength=n)
    joint = np.bincount(old * len(mass) + truth, weights=weight, minlength=n * len(mass)).reshape(n, len(mass))
    fixed = np.full(n, -1, np.int64); fixed_iou = np.zeros(n)
    for obj in cls:
        oid = obj['object_id']; candidates = np.flatnonzero((mass >= 100) & (np.arange(len(mass)) // 1000 == category[oid]))
        if not len(candidates): continue
        intersection = joint[oid, candidates]
        iou = intersection / np.maximum(1e-12, mass[candidates] + pmass[oid] - intersection)
        best = int(np.argmax(iou)); fixed_iou[oid] = iou[best]
        if iou[best] > .5: fixed[oid] = candidates[best]
    same = train == category[old]; has_fixed = fixed[old] >= 0
    categories = dict(wrong_semantic_class=known & ~same,
        correct_class_fixed_instance=known & same & has_fixed & (truth == fixed[old]),
        correct_class_other_instance=known & same & has_fixed & (truth != fixed[old]),
        correct_class_no_fixed_correspondence=known & same & ~has_fixed,
        unknown_reference=~known)
    np.testing.assert_array_equal(sum(x.astype(int) for x in categories.values()), np.ones(len(old), int))
    result = {}
    for stage, stage_mask in masks.items():
        result[stage] = dict(total=mass_record(stage_mask, weight),
            categories={name: mass_record(stage_mask & mask, weight) for name, mask in categories.items()},
            road_wrong_class=mass_record(stage_mask & known & (train == 0) & ~same, weight))
    objects = []
    for obj in cls:
        oid = obj['object_id']
        objects.append(dict(object_id=oid, declared_class=int(category[oid]), fixed_gt=int(fixed[oid]), baseline_best_iou=float(fixed_iou[oid]),
            stages={stage: {name: mass_record(stage_mask & (old == oid) & mask, weight) for name, mask in categories.items()}
                    for stage, stage_mask in masks.items()}))
    return result, objects


def anchor_changes(c, ref, train):
    from scipy.spatial import cKDTree
    from sklearn.metrics import roc_auc_score
    from probe_utonia_ownership_evidence import stable_neighbors
    key = c['key']; full = np.load(Path(c['full']) / 'geometry.npz')
    decision = np.load(PARENT / 'decisions' / key / 'anchors.npz'); units = decision['units']
    lookup = {tuple(row): i for i, row in enumerate(units.tolist())}
    source, unit = [], []
    for frame in read(c['job'])['frames']:
        d = np.load(PARENT / 'inputs' / key / f'{frame["frame_id"]:06d}.npz')
        fg = d['label'] > 0
        source.append(d['source_id'][fg])
        unit.extend(lookup[(int(cell), int(owner), frame['frame_id'])] for cell, owner in zip(d['cell'][fg], d['label'][fg]))
    source = np.concatenate(source); unit = np.asarray(unit)
    assert len(source) == len(unit) and len(np.unique(source)) == len(source)
    np.testing.assert_array_equal(full['source_frame'][source], units[unit, 2])
    d, ix = stable_neighbors(cKDTree(ref['xyz']), full['xyz'][source], ref['xyz'], np.arange(len(ref['xyz'])), k=1, radius=.2)
    d, ix = d[:, 0], ix[:, 0]
    cls = read(Path(c['ours']) / 'classification.json')['objects']
    category = np.full(int(units[:, 1].max()) + 1, -1, np.int16)
    for obj in cls: category[obj['object_id']] = obj['predictions']['mean']['train_id']
    declared = category[units[unit, 1]]; truth = train[ix]; weight = ref['confidence'][ix]
    known = (d <= .2) & (truth != 255) & (weight > 0)
    wrong = declared != truth
    reject = decision['retract'][unit]
    margin = decision['mean_margin'][unit]
    fraction = decision['refute'][unit] / np.maximum(1, decision['eligible'][unit])
    def group(take):
        valid = take & known
        positive = valid & wrong; negative = valid & ~wrong
        def rate(a, b): return float(weight[a].sum() / weight[b].sum()) if weight[b].sum() > 0 else None
        return dict(source_anchors=int(take.sum()), known_anchors=int(valid.sum()),
                    wrong_class=mass_record(positive, weight), correct_class=mass_record(negative, weight),
                    rejected_wrong_class=mass_record(positive & reject, weight), rejected_correct_class=mass_record(negative & reject, weight),
                    wrong_class_rejection_rate=rate(positive & reject, positive), correct_class_rejection_rate=rate(negative & reject, negative),
                    unknown_rejected=int((take & ~known & reject).sum()),
                    auroc={name: float(roc_auc_score(wrong[valid], values[valid], sample_weight=weight[valid])) if len(np.unique(wrong[valid])) == 2 else None
                           for name, values in [('mean_margin', margin), ('refute_fraction', fraction)]})
    groups = dict(all=group(np.ones(len(source), bool)), building=group(declared == 2), car=group(declared == 13))
    hist = []
    for lo, hi in [(0, .25), (.25, .5), (.5, .75), (.75, 1.000000001)]:
        t = known & (fraction >= lo) & (fraction < hi)
        hist.append(dict(lower=lo, upper=min(hi, 1), known=int(t.sum()),
                         wrong_mass=float(weight[t & wrong].sum()), correct_mass=float(weight[t & ~wrong].sum())))
    path = OUT / key / 'anchor_readout.npz'; path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, source_id=source, unit_index=unit, known=known, wrong_semantic_class=wrong,
                        declared_class=declared, reference_train_id=truth, reference_index=ix, reference_distance=d,
                        confidence=weight, retracted=reject, mean_margin=margin, refute_fraction=fraction)
    return dict(groups=groups, refute_fraction_bins=hist, source_readout=str(path), source_readout_sha256=sha(path))


def run():
    p = verify(); assert not (OUT / 'summary.json').exists()
    results = {}
    for c in p['contexts']:
        ref, train = reference(c, p)
        stages, objects = map_changes(c, ref, train)
        anchors = anchor_changes(c, ref, train)
        results[c['key']] = dict(map_stages=stages, objects=objects, anchor_diagnostic=anchors)
        print('FAILURE ATTRIBUTION', c['key'], 'map', stages, 'anchors', anchors['groups'], flush=True)
    dump(OUT / 'summary.json', dict(complete=True, results=results, protocol_sha256=sha(OUT / 'protocol.json'),
                                   parent_gate_unchanged=False, new_model_calls=0, new_maps=0, new_ap_scores=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['freeze', 'run'])
    globals()[parser.parse_args().action]()
