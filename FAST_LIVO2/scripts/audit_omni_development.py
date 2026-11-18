#!/usr/bin/env python3
"""Evaluate development ablations on an earlier comparison's locked GT epochs.

Incomplete coverage is reported without an ATE; it never changes the common
evaluation interval or drops difficult timestamps to improve a variant's score.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from evaluate_ntu_benchmark import prism_positions, interpolate_reference, rigid_align, error_statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline-report', type=Path, required=True)
    ap.add_argument('--runs', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    grid_path = args.baseline_report/'evaluation/FAST-LIO2_aligned.csv'
    grid = np.loadtxt(grid_path, delimiter=',')
    stamps, gt = grid[:, 0], grid[:, 4:7]
    initial = stamps - stamps[0] < 10
    files = {p.stem: p for p in sorted((args.baseline_report/'inputs').glob('*.tum'))}
    records = {}
    for directory in sorted(args.runs.iterdir()):
        if not directory.is_dir():
            continue
        name = directory.name
        run = json.loads((directory/'run.json').read_text())
        record = dict(run=run)
        dest = args.output/'provenance'/name
        dest.mkdir(parents=True)
        for filename in ('run.json', 'params.yaml'):
            shutil.copy2(directory/filename, dest/filename)
        record['config_sha256'] = digest(directory/'params.yaml')
        if run['exit_code'] != 0 or not run['transport'].get('completed'):
            record['status'] = 'execution_failed'
            shutil.copy2(directory/'console.log', dest/'console.log')
        else:
            files[name] = directory/'trajectory.tum'
            record['status'] = 'completed_not_yet_evaluated'
        records[name] = record
    errors = {}
    for name, path in files.items():
        track = np.loadtxt(path, ndmin=2)
        if track.shape[1] != 8 or not np.isfinite(track).all() or np.any(np.diff(track[:, 0]) < 0):
            raise ValueError(f'invalid trajectory: {path}')
        track = track[np.r_[np.diff(track[:, 0]) != 0, True]]
        sampled, valid = interpolate_reference(stamps, np.c_[track[:, 0], prism_positions(track)], .100001)
        record = records.setdefault(name, {})
        record.update(trajectory=str(path.resolve()), trajectory_sha256=digest(path),
                      output_poses=len(track), locked_epochs=len(stamps), valid_epochs=int(valid.sum()))
        dest = args.output/'inputs'
        dest.mkdir(exist_ok=True)
        shutil.copy2(path, dest/(name+'.tum'))
        if not valid.all():
            record['status'] = 'incomplete_locked_coverage_no_score'
            continue
        fitted, rotation, translation, _ = rigid_align(sampled, gt)
        residual = fitted - gt
        errors[name] = np.linalg.norm(residual, axis=1)
        record.update(status='scored_full_locked_grid', ate=error_statistics(residual),
                      first_10_evaluation_seconds=error_statistics(residual[initial]),
                      remaining_evaluation_seconds=error_statistics(residual[~initial]),
                      rotation=rotation.tolist(), translation=translation.tolist())
        np.savetxt(args.output/(name+'_aligned.csv'), np.c_[stamps, fitted, gt, errors[name]],
                   delimiter=',', header='stamp,est_x,est_y,est_z,gt_x,gt_y,gt_z,error_m')
    result = dict(role='development only; these sequences have been inspected/tuned',
                  locked_grid_source=str(grid_path.resolve()), locked_grid_sha256=digest(grid_path),
                  locked_epochs=len(stamps), alignment='fixed-scale global rigid SE3; prism lever arm',
                  interpolation='estimated prism at locked GT times; bracket <=0.100001s; no extrapolation',
                  time_offset_fit=False, early_window='first 10 evaluated seconds, NOT first 10 input seconds',
                  methods=records)
    (args.output/'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    shutil.copy2(grid_path, args.output/'locked_reference.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    names = list(errors)
    axes[0].barh(names, [records[n]['ate']['rmse_m'] * 100 for n in names])
    axes[0].set(xscale='log', xlabel='ATE RMSE [cm], log scale', title='All completed variants; locked evaluation epochs')
    for name in ('fastlio2', 'fastlivo2', 'omni_slam_lio', 'omni_slam_livo', 'exposure_on', 'norm_and_exposure'):
        if name in errors:
            axes[1].plot(stamps - stamps[0], errors[name] * 100, label=name, lw=1)
    axes[1].set(xlabel='Time from first evaluated GT epoch [s]', ylabel='Position error [cm]',
                title='One global alignment; initial errors retained')
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
    fig.savefig(args.output/'development_audit.png', dpi=160)
    plt.close(fig)
    for name, record in records.items():
        print(name, record['status'], record.get('ate', {}).get('rmse_m'))


if __name__ == '__main__':
    main()
