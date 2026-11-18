"""Posthoc attribution of new-map readout changes; never changes predictions."""
from pathlib import Path
import numpy as np
from paper_final_common import read, dump, sha
from run_coupled_batch_transport import OUT, verify, seal


def main():
    p = verify(); seal('predictions_complete'); seal('summary'); results = {}
    for c in p['contexts']:
        if not c['reference']: continue
        key = c['key']; a = np.load(Path(c['ours']) / 'reference_prediction.npz')['instance']
        b = np.load(OUT / key / 'ours_coupled/reference_prediction.npz')['instance']
        changed = a != b; new = (a == 0) & (b > 0); swap = (a > 0) & (b > 0) & changed
        assert not ((a > 0) & (b == 0)).any(); np.testing.assert_array_equal(changed, new | swap)
        ref = np.load(c['reference']); truth = ref['train_instance']; w = ref['confidence']
        cls = read(Path(c['ours']) / 'classification.json')['objects']
        size = max(int(a.max()), int(b.max())) + 1
        category = np.full(size, -1, np.int64)
        for row in cls: category[row['object_id']] = row['predictions']['mean']['train_id']
        mass = np.bincount(truth, weights=w); pmass = np.bincount(a, weights=w, minlength=size)
        joint = np.bincount(a * len(mass) + truth, weights=w, minlength=size * len(mass)).reshape(size, len(mass))
        fixed = np.full(size, -1, np.int64); fixed_iou = np.zeros(size)
        for row in cls:
            oid = row['object_id']; gids = np.flatnonzero((mass >= 100) & (np.arange(len(mass)) // 1000 == category[oid]))
            if not len(gids): continue
            inter = joint[oid, gids]; iou = inter / np.maximum(1e-12, mass[gids] + pmass[oid] - inter)
            best = int(np.argmax(iou))
            # Use the existing AP50 threshold only to make an interpretable
            # fixed old correspondence, not to tune any map or add an AP score.
            fixed_iou[oid] = iou[best]
            if iou[best] > .5: fixed[oid] = gids[best]
        portions = {}
        for name, mask in [('new_coverage', new), ('existing_identity_changed', swap)]:
            ids = np.flatnonzero(mask); labels = truth[ids]; predclass = category[b[ids]]
            # The saved adapter conflates ignored labels and road instance0 at
            # truth0; do not count these as known negative evidence.
            known = (labels >= 1000) & (labels < 19000) & (w[ids] > 0)
            same_class = known & (labels // 1000 == predclass)
            has_fixed = fixed[b[ids]] >= 0
            masks = dict(other_known_class=known & ~same_class,
                same_class_same_fixed_instance=same_class & has_fixed & (labels == fixed[b[ids]]),
                same_class_other_fixed_instance=same_class & has_fixed & (labels != fixed[b[ids]]),
                same_class_no_reliable_old_correspondence=same_class & ~has_fixed,
                unknown_or_road0_or_zero_confidence=~known)
            np.testing.assert_array_equal(sum(m.astype(int) for m in masks.values()), np.ones(len(ids), int))
            portions[name] = {k: dict(queries=int(m.sum()), confidence_mass=float(w[ids[m]].sum())) for k, m in masks.items()}
        results[key] = dict(reference_queries=len(a), different_query_assignments=int(changed.sum()),
            new_coverage=int(new.sum()), changed_existing_identity=int(swap.sum()), lost_coverage=0,
            iou_comparison_vs_old=read(OUT / 'object_comparisons.json')[key]['ours_frame'],
            changed_readout_diagnostic=portions)
        print('COUPLED READOUT CHANGE', key, results[key], flush=True)
    dump(OUT / 'reference_changes.json', dict(complete=True, results=results,
        scope='Posthoc diagnosis on already-scored sealed predictions. Exact reference-query changes; auxiliary class/correspondence breakdown is NOT instance AP or new prediction. GT never changes selections.',
        fixed_correspondence='For predicted objects with old best same-class IoU>0.5 against confidence-mass>=100 GT, freeze that GT. Others stay unassigned. No one-to-one AP matching implied.',
        unknown='Saved truth0 conflates ignored labels and road instance0; conservatively unknown. Confidence<=0 also unknown. Known means train_id1..18. Do not extrapolate known-only diagnostic to all new sources.',
        prediction_sha256=sha(OUT / 'predictions_complete.json'), source_sha256=sha(__file__)))


if __name__ == '__main__': main()
