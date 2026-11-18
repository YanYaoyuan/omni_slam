"""Independent artifact checks, without selecting models or adding map scores."""
import numpy as np
from paper_final_common import read, dump, sha
from run_point_visual_recovery import OUT, GEOM, PARENT, PRIOR, INPUT, METHODS, TRAINED, verify, seal, inference


def main():
    import torch
    torch.set_num_threads(1)
    protocol = verify(True)
    for name in ['tokens_complete', 'features_complete', 'trained', 'predictions_complete']: seal(name)
    summary = read(OUT/'summary.json'); checks = {}
    for key in summary['results']:
        d = np.load(OUT/'features'/key/'candidates.npz'); core = np.load(PRIOR/'maps'/key/'largest/geometry.npz'); n = len(core['xyz'])
        xyz = np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz']; valid = d['valid']; group, slot = np.where(valid)
        old = np.load(GEOM/'features'/key/'candidates.npz'); rotated = np.load(GEOM/'features'/key/'rigid_features.npz')['features']
        for field in ['source_ids', 'owners', 'valid']: np.testing.assert_array_equal(d[field], old[field])
        for m in TRAINED: np.testing.assert_array_equal(d[m][:, :, :12], old['features'][:, :, :12])
        for col in [12, 22, 23]: np.testing.assert_array_equal(d['rgb'][group, slot, col], d['ours_dino'][group, slot, col])
        one = d['ours_dino'][group, slot, 12] <= np.log(2)+1e-6
        np.testing.assert_array_equal(d['ours_dino'][group[one], slot[one]], d['dino_first'][group[one], slot[one]])
        # Each category's shuffled rows must be an exact permutation.
        category = old['features'][group, slot, 10:12].argmax(1)
        for cat in np.unique(category):
            use = category == cat
            a = d['ours_dino'][group[use], slot[use], 12:]
            b = d['shuffled_dino'][group[use], slot[use], 12:]
            assert sorted(row.tobytes() for row in a) == sorted(row.tobytes() for row in b)
        per_method = {}; decisions = {}
        for m in METHODS:
            folder = OUT/'maps'/key/m; g = np.load(folder/'geometry.npz'); z = np.load(folder/'decisions.npz'); accepted = z['owners'] > 0
            np.testing.assert_array_equal(z['source_ids'], d['source_ids'])
            np.testing.assert_array_equal(g['xyz'][:n], core['xyz']); np.testing.assert_array_equal(g['instance'][:n], core['instance'])
            np.testing.assert_array_equal(g['added_source_ids'], d['source_ids'][accepted])
            np.testing.assert_array_equal(g['xyz'][n:], xyz[d['source_ids'][accepted]])
            np.testing.assert_array_equal(g['instance'][n:], z['owners'][accepted])
            assert len(np.unique(g['added_source_ids'])) == len(g['added_source_ids'])
            assert np.all(~accepted | ((z['owners'][:, None] == d['owners']) & valid).any(1))
            assert read(folder/'classification.json') == read(PRIOR/'maps'/key/'largest/classification.json')
            checks_m = dict(core_exact=True, measured_additions_exact=True, identity_classification_fixed=True)
            if m not in TRAINED:
                for name in ['geometry.npz', 'decisions.npz', 'classification.json']: assert sha(folder/name) == sha(GEOM/'maps'/key/m/name)
                checks_m['copied_control_exact'] = True
            else:
                model = torch.load(OUT/'models'/key/m/'model.pt', map_location='cpu', weights_only=False)
                assert key not in model['training']
                train_rows = []
                for k in model['training']:
                    td = np.load(OUT/'features'/k/'candidates.npz'); y = np.load(PARENT/'training_labels'/k/'TRAINING_ONLY.npz'); mask = td['valid'] & y['eligible'][:, None]; train_rows.append(td[m][mask])
                training = np.concatenate(train_rows)
                np.testing.assert_array_equal(training.mean(0), model['mean']); np.testing.assert_array_equal(np.maximum(training.std(0), 1e-6), model['std'])
                changes = []
                for rotation in rotated:
                    x = d[m].copy(); x[:, :, :12] = rotation[:, :, :12]
                    alternate, probability = inference(model, x, valid, d['owners'])
                    np.testing.assert_array_equal(alternate, z['owners']); np.testing.assert_array_equal(probability, z['confidence']); changes.append(0)
                checks_m.update(standardization_train_only=True, rigid_changed_decisions=changes)
            decisions[m] = z['owners']; per_method[m] = checks_m
        checks[key] = dict(methods=per_method, same_single_view_features_exact=True, rgb_dino_visibility_equal=True, shuffle_is_within_class_permutation=True,
            changed_vs_geometry={m: int((a != decisions['intrinsic_geometry']).sum()) for m, a in decisions.items()},
            changed_multiview_vs_first=int((decisions['ours_dino'] != decisions['dino_first']).sum()))
    dump(OUT/'artifact_audit.json', dict(complete=True, checks=checks, prediction_seal_sha256=sha(OUT/'predictions_complete.json'),
        invariance_scope='Same physical supports/pixel evidence, rigid transforms of source geometry only; upstream SLAM/masks/grids not rerun.'))
    print('AUDITED', checks, flush=True)


if __name__ == '__main__': main()
