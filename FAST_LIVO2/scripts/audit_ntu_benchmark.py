#!/usr/bin/env python3
"""Check paired configurations, full input consumption and independent ATE fit."""
import argparse
import json
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from evaluate_ntu_benchmark import interpolate_reference, prism_positions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--ground-truth', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args()
    run = json.loads((args.run/'run.json').read_text())
    metrics = json.loads((args.report/'metrics.json').read_text())
    configs = []
    tracks = {}
    for mode in ('lio', 'livo'):
        config = yaml.safe_load((args.run/(mode+'.yaml')).read_text())['/**']['ros__parameters']
        assert config['common'].pop('img_en') == int(mode == 'livo')
        config['runtime'].pop('output_directory')
        configs.append(config)
        transport = run['modes'][mode]['transport']
        assert transport['completed'] and transport['exit_code'] == 0
        for number, name in [('1','imu'), ('2','lidar'), ('3','image')]:
            assert transport['input_counts'][name] == run['input_audit']['input_counts'][number]
        tracks[mode] = np.loadtxt(args.run/mode/'trajectory_final.tum', ndmin=2)
        assert np.isfinite(tracks[mode]).all() and np.all(np.diff(tracks[mode][:,0]) > 0)
    assert configs[0] == configs[1], 'paired configuration mismatch'
    common, ia, ib = np.intersect1d(tracks['lio'][:,0], tracks['livo'][:,0], return_indices=True)
    raw = np.loadtxt(args.ground_truth, delimiter=',', skiprows=1)
    result = dict(configs_equal_except_image_flag_and_output=True, full_input_consumption=True,
                  timestamp_conventions={})
    for column in (0, 2):
        gt = np.column_stack([raw[:,column]*1e-9, raw[:,3:6]])
        assert np.all(np.diff(gt[:,0]) > 0)
        reference, valid = interpolate_reference(common, gt)
        reference = reference[valid]
        row = dict(evaluated_epochs=int(valid.sum()), modes={})
        for mode, indices in [('lio',ia), ('livo',ib)]:
            estimated = prism_positions(tracks[mode][indices][valid])
            # Independent implementation: SciPy Wahba fit, centered vectors,
            # no scale fitting. Primary evaluator uses its own SVD/Kabsch code.
            ec = estimated-estimated.mean(axis=0)
            rc = reference-reference.mean(axis=0)
            rotation, _ = Rotation.align_vectors(rc, ec)
            residual = rotation.apply(ec)-rc
            rmse = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
            row['modes'][mode] = dict(ate_rmse_m=rmse)
            if column == 0:
                assert valid.sum() == metrics['evaluated_epochs']
                assert abs(rmse-metrics['modes'][mode]['ate']['rmse_m']) < 1e-8
        result['timestamp_conventions'][str(column)] = row
    delta = (raw[:,0]-raw[:,2])*1e-9
    result['csv_time_minus_header_stamp_s'] = dict(median=float(np.median(delta)),
                                                  max_abs=float(np.max(abs(delta))))
    result['note'] = 'Column 0: original MATLAB tutorial, headline protocol. Column 2: later official notebook; sensitivity only.'
    (args.report/'audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
