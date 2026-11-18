"""Validate completed benchmark exports and arithmetic without altering scores."""
import numpy as np
from paper_final_common import read, dump, sha
import run_hypersim_benchmark as bench


def main():
    protocol, gate = bench.gate()
    summary = read(bench.OUT/'summary.json'); assert summary['complete']
    assert summary['protocol_sha256'] == sha(bench.OUT/'protocol.json')
    checks = {}
    for scene in protocol['scenes']:
        paths = bench.paths_for(scene)
        source = np.load(paths['history2']/'geometry.npz')
        records = read(paths['history2']/'classification.json')['objects']
        foreground = [r['object_id'] for r in records if r['predictions']['mean']['retained']]
        cleanup = read(bench.OUT/'maps'/scene/'cleanup/run.json')
        ref_path = bench.OUT/'reference'/scene/'reference.npz'
        manifest = read(ref_path.parent/'manifest.json')
        assert manifest['prediction_gate_sha256'] == sha(bench.OUT/'predictions_complete.json')
        assert manifest['reference_sha256'] == sha(ref_path)
        ref = np.load(ref_path); truth = ref['instance']
        assert np.isfinite(ref['xyz']).all() and (truth >= 0).all()
        assert len(np.unique(ref['voxel'], axis=0)) == len(truth)
        methods = {}
        for method in protocol['methods']:
            folder = paths[method]; geometry = np.load(folder/'geometry.npz')
            assert geometry['xyz'].shape == (len(geometry['instance']), 3)
            assert np.isfinite(geometry['xyz']).all() and (geometry['instance'] >= 0).all()
            subset = None
            if method in cleanup['methods']:
                idx = np.load(folder/'retained_source_indices.npy')
                assert np.issubdtype(idx.dtype, np.integer)
                assert ((idx >= 0) & (idx < len(source['xyz']))).all()
                assert (np.diff(idx) > 0).all()
                np.testing.assert_array_equal(geometry['xyz'], source['xyz'][idx])
                np.testing.assert_array_equal(geometry['instance'], source['instance'][idx])
                assert sha(folder/'classification.json') == sha(paths['history2']/'classification.json')
                remaining = int(np.isin(geometry['instance'], foreground).sum())
                assert remaining == cleanup['methods'][method]['remaining_foreground_points']
                subset = True
            score = summary['results'][scene]['methods'][method]['primary']
            predicted = np.load(bench.OUT/'evaluation'/scene/(method+'.npz'))['instance']
            assert predicted.shape == truth.shape and (predicted >= 0).all()
            assert score['gt_visible_instances'] == len(np.unique(truth[truth > 0]))
            assert score['predicted_instances'] == len(np.unique(predicted[predicted > 0]))
            matches = score['matches']; tp = score['tp']
            assert tp == len(matches)
            assert len({r['gt_id'] for r in matches}) == tp
            assert len({r['predicted_id'] for r in matches}) == tp
            assert all(.5 < r['iou'] <= 1 for r in matches)
            assert score['fp'] == score['predicted_instances'] - tp
            assert score['fn'] == score['gt_visible_instances'] - tp
            denom = tp + .5*(score['fp'] + score['fn'])
            if denom:
                np.testing.assert_allclose(score['observed_voxel_instance_pq'], sum(r['iou'] for r in matches)/denom, atol=1e-12)
                np.testing.assert_allclose(score['rq'], tp/denom, atol=1e-12)
            if tp:
                np.testing.assert_allclose(score['sq'], sum(r['iou'] for r in matches)/tp, atol=1e-12)
            else: assert score['sq'] is None
            audit = read(bench.OUT/'evaluation'/scene/(method+'_audit.json'))
            assert {r['matched_gt'] for r in audit['predicted'] if r['matched_gt'] is not None} == {r['gt_id'] for r in matches}
            methods[method] = dict(points=len(geometry['xyz']), exact_history2_subset=subset, metrics_consistent=True)
        checks[scene] = dict(reference_voxels=len(truth), methods=methods)
    for method in protocol['methods']:
        mean = np.mean([100*summary['results'][s]['methods'][method]['primary']['observed_voxel_instance_pq'] for s in protocol['scenes']])
        np.testing.assert_allclose(summary['scene_mean_pq100'][method], mean, atol=1e-12)
    dump(bench.OUT/'verification.json', dict(complete=True, checks=checks,
         summary_sha256=sha(bench.OUT/'summary.json'), prediction_gate_sha256=sha(bench.OUT/'predictions_complete.json'),
         verifier_sha256=sha(__file__), scope='Sealed-file integrity, common reference, exact source subsets and unchanged classification, metric arithmetic and audit agreement. Does not independently validate every GT annotation or establish generalization.'))
    print('Verified', len(protocol['scenes']), 'scenes x', len(protocol['methods']), 'methods')


if __name__ == '__main__': main()
