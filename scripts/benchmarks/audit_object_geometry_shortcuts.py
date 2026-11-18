"""Fixed-weight diagnostic; never exports revised maps or selects a model.

Replace one feature family with other-window, same-predicted-class medians.
These artificial interventions break feature correlations: sensitivity is not
proof of a causal physical failure or an evaluated deployable improvement.
"""
import time
import numpy as np
import torch
from paper_final_common import ROOT, read, dump, sha
from run_provenance_object_memory import OUT as PARENT, KEYS, verify, seal
from omni_semantic_map.provenance_object_memory import ViewEvidenceAllocator
from omni_semantic_map.surface_ownership import allocate

OUT = ROOT / 'build/benchmarks/object_geometry_shortcut_audit_v1'
GROUPS = {'unchanged': [], 'shape': [7, 8, 9],
          'relative_geometry': list(range(7)), 'semantic': list(range(12, 31))}


def run():
    assert not OUT.exists(), 'Do not overwrite a completed or running audit'
    verify(); seal('trained'); seal('features_complete'); seal('predictions_complete')
    OUT.mkdir(parents=True)
    dump(OUT / 'protocol.json', dict(
        created_epoch=time.time(), source_sha256=sha(__file__), groups=GROUPS,
        parent_prediction_sha256=sha(PARENT / 'predictions_complete.json'),
        parent_training_sha256=sha(PARENT / 'trained.json'),
        scope='All eligible old-point queries, all objects, both exposed windows; no new training, map export, AP or parameter selection.',
        intervention='Frozen Ours model. Independently replace listed static channels by same predicted-class median of eligible original-owner queries from the other window. Medians never read GT. Original view memory and all other channels unchanged.',
        interpretation='Artificial feature intervention, not a physically consistent scene or deployable method. Conditional dependence remains. Retain/reject target diagnostics are candidate-relative labels, not complete-map quality.'))
    torch.set_num_threads(4)
    results = {}; files = {}
    for key in KEYS:
        other = next(k for k in KEYS if k != key)
        d = dict(np.load(PARENT / 'features' / key / 'candidates.npz'))
        train = dict(np.load(PARENT / 'features' / other / 'candidates.npz'))
        labels = dict(np.load(PARENT / 'training_only' / key / 'labels.npz'))
        base = np.load(PARENT / 'maps' / key / 'baseline/geometry.npz')
        original = base['instance'][d['old_indices']]
        n = len(original); nt = len(train['old_indices'])
        cls = np.where(d['static'][:n, 0, 10] > 0, 13, 2)
        train_cls = np.where(train['static'][:nt, 0, 10] > 0, 13, 2)
        medians = {c: np.median(train['static'][:nt, 0][train['valid'][:nt, 0] & (train_cls == c)], axis=0) for c in [2, 13]}
        # Candidate-specific semantic class, including alternative objects.
        candidate_cls = np.where(d['static'][:n, :, 10] > 0, 13, 2)
        checkpoint = torch.load(PARENT / 'models' / key / 'ours_revision/model.pt', map_location='cpu', weights_only=True)
        assert checkpoint['training'] == [other]
        model = ViewEvidenceAllocator().eval(); model.load_state_dict(checkpoint['state']); model.requires_grad_(False)
        indices = np.flatnonzero(d['valid'][:n].any(1)); decisions = {}; confidences = {}
        with torch.inference_mode():
            for method, columns in GROUPS.items():
                choice = original.copy(); confidence = np.full(n, np.nan)
                for lo in range(0, len(indices), 1024):
                    ix = indices[lo:lo+1024]; static = d['static'][ix].copy()
                    for c in [2, 13]:
                        ii, jj = np.where(candidate_cls[ix] == c)
                        for col in columns: static[ii, jj, col] = medians[c][col]
                    x = torch.from_numpy(((static - np.array(checkpoint['mean'])) / np.array(checkpoint['std'])).astype('float32'))
                    z = model(x, torch.from_numpy(d['views'][ix].astype('float32')),
                              torch.from_numpy(d['separated'][ix]), torch.from_numpy(d['valid'][ix]), 'attention').numpy()
                    choice[ix], confidence[ix] = allocate(z, d['valid'][ix], d['owners'][ix])
                decisions[method] = choice; confidences[method] = confidence
        frozen = np.load(PARENT / 'maps' / key / 'ours_revision/decisions.npz')
        np.testing.assert_array_equal(decisions['unchanged'], frozen['owners'][:n])
        np.testing.assert_allclose(confidences['unchanged'][indices], frozen['confidence'][:n][indices], rtol=2e-4, atol=2e-5)
        binding = read(PARENT / 'training_only' / key / 'binding.json')['binding']
        rows = []
        for oid in np.unique(original):
            take = (original == oid) & labels['eligible'][:n]
            positive = take & labels['positive'][:n, 0]
            negative = take & ~labels['positive'][:n].any(1)
            removed = positive & (decisions['unchanged'] != oid)
            others = take & ~positive & ~negative
            row = dict(object_id=int(oid), assigned_gt=int(binding[str(oid)]),
                       old_points=int((original == oid).sum()), eligible=int(take.sum()),
                       own_positive=int(positive.sum()), null_target=int(negative.sum()), alternative_target=int(others.sum()),
                       false_rejected_own_positive=int(removed.sum()), methods={})
            if removed.any():
                c = int(cls[removed][0]); p = d['static'][:n, 0, 12:31][removed]
                row['false_rejected_semantic'] = dict(mean_own_probability=float(p[:, c].mean()),
                    argmax_class_count=np.bincount(p.argmax(1), minlength=19).tolist())
            for method, choice in decisions.items():
                row['methods'][method] = dict(
                    retained_positive=int((positive & (choice == oid)).sum()),
                    rejected_null=int((negative & (choice == 0)).sum()),
                    changed_decisions=int((take & (choice != decisions['unchanged'])).sum()),
                    restored_previous_false_rejections=int((removed & (choice == oid)).sum()))
            rows.append(row)
        folder = OUT / key; folder.mkdir()
        np.savez_compressed(folder / 'diagnostic_choices.npz', **decisions)
        dump(folder / 'objects.json', rows)
        dump(folder / 'training_medians.json', {str(k): v.tolist() for k, v in medians.items()})
        results[key] = dict(objects=len(rows), original_prediction_exact=True,
            own_positive=sum(r['own_positive'] for r in rows),
            false_rejected_own_positive=sum(r['false_rejected_own_positive'] for r in rows),
            restored_by_intervention={m: sum(r['methods'][m]['restored_previous_false_rejections'] for r in rows) for m in GROUPS})
        for f in folder.iterdir(): files[str(f)] = sha(f)
        print('SHORTCUT DIAGNOSTIC', key, results[key], flush=True)
        for r in rows:
            if (key == KEYS[0] and r['assigned_gt'] == 13009) or (key == KEYS[1] and r['assigned_gt'] == 2011):
                print('LOST OBJECT', key, r, flush=True)
    dump(OUT / 'summary.json', dict(complete=True, results=results, files=files,
         protocol_sha256=sha(OUT / 'protocol.json'), new_training_runs=0, new_maps=0,
         scope='Fixed-weight artificial-input sensitivity; not an AP improvement or causal physical explanation.'))


if __name__ == '__main__': run()
