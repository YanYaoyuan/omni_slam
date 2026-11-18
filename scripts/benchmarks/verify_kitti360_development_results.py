"""Check measured provenance, immutable cleanup subsets and AP arithmetic."""
import argparse
from pathlib import Path
import numpy as np
from paper_final_common import ROOT, read, dump, sha
import evaluate_kitti360_development as evaluation
from omni_semantic_map.kitti360_evaluation import kernel


def main(connected=False):
    if connected:
        from evaluate_kitti360_connected import configure
        configure()
    p = evaluation.verify()
    summary_path = evaluation.OUT/'summary.json'
    summary = read(summary_path)
    assert summary['complete'] and summary['protocol_sha256'] == sha(evaluation.OUT/'protocol.json')
    for path, expected in summary['files_sha256'].items(): assert sha(evaluation.OUT/path) == expected
    input_verification = ROOT/'output/kitti360_interface/reserved_windows_verification.json'
    assert read(input_verification)['complete'] and read(input_verification)['input_frames'] == 161
    job = read(p['job'])
    cache = Path(p['cache'])
    manifest = read(cache/'manifest.json')
    for name, expected in manifest['files_sha256'].items(): assert sha(cache/name) == expected
    g = np.load(cache/'geometry.npz')
    total = 0
    for record in job['frames']:
        selected = np.flatnonzero(g['source_frame'] == record['frame_id'])
        measured = np.load(record['measured'])
        lookup = np.full(int(measured['raw_indices'].max())+1, -1, np.int64)
        assert len(np.unique(measured['raw_indices'])) == len(measured['raw_indices'])
        lookup[measured['raw_indices']] = np.arange(len(measured['raw_indices']))
        indices = lookup[g['source_raw_index'][selected]]
        assert (indices >= 0).all()
        np.testing.assert_array_equal(g['xyz'][selected], measured['xyz'][indices])
        np.testing.assert_array_equal(g['source_pixel_xy'][selected], measured['pixel_xy'][indices])
        total += len(selected)
    assert total == len(g['xyz'])
    assert len(np.unique(np.floor(g['xyz']/.05).astype(np.int64), axis=0)) == total
    full = Path(p['methods']['history2'])
    source = np.load(full/'geometry.npz')
    checks = {}
    for name in ['sor','ours_visibility1','simple_ratio','mc_point_ratio']:
        folder = Path(p['methods'][name])
        selected = np.load(folder/'retained_source_indices.npy')
        output = np.load(folder/'geometry.npz')
        assert (np.diff(selected) > 0).all() and (selected >= 0).all() and (selected < len(source['xyz'])).all()
        for key in ['xyz','instance']: np.testing.assert_array_equal(output[key], source[key][selected])
        assert sha(folder/'classification.json') == sha(full/'classification.json')
        checks[name] = dict(source_subset_exact=True, unchanged_classification=True, retained_points=len(selected))
    args, ns = kernel(evaluation.EVALUATOR)
    for name in p['methods']:
        for domain in ['full_window', 'observed_support']:
            matches = read(evaluation.OUT/(name+'_'+domain+'_matches.json'))
            ap = ns['evaluateMatches'](matches, args)
            computed = evaluation.clean_json(ns['computeAverages'](ap, args))
            assert computed == summary['methods'][name][domain]['averages'], (name, domain)
    result = dict(complete=True, summary_sha256=sha(summary_path), script_sha256=sha(__file__),
        measured_points_verified=total, first_measurement_per_voxel=True, cleanup_checks=checks,
        score_arithmetic_checked=len(p['methods'])*2, sensor_verification_sha256=sha(input_verification),
        scope='Source-point retention and recorded-match AP arithmetic. Does not independently establish correctness of public annotations, source calibration, statistical generalization or the annotation compatibility adapter.')
    dump(evaluation.OUT/'verification.json', result)
    print({k:result[k] for k in ['complete','measured_points_verified','score_arithmetic_checked']}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--connected', action='store_true')
    args = parser.parse_args()
    main(args.connected)
