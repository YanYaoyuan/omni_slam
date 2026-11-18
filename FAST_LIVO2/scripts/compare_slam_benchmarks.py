#!/usr/bin/env python3
"""Compare IMU-frame TUM trajectories at identical Leica ground-truth epochs.

Manifest: {"ground_truth": "...csv", "methods": {"name": "...tum"}}.
Paths are relative to the manifest. No extrapolation, scale or time fitting.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from evaluate_ntu_benchmark import prism_positions, interpolate_reference, rigid_align, error_statistics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    config = json.loads(args.manifest.read_text())
    resolve = lambda p: args.manifest.parent / p
    gt_raw = np.loadtxt(resolve(config['ground_truth']), delimiter=',', skiprows=1)
    gt = np.column_stack([gt_raw[:,0]*1e-9, gt_raw[:,3:6]])
    assert np.isfinite(gt).all() and np.all(np.diff(gt[:,0]) > 0)
    sampled = {}; masks = {}; raw_counts = {}; spans = {}
    for name, path in config['methods'].items():
        track = np.loadtxt(resolve(path), ndmin=2)
        assert track.shape[1] == 8 and np.isfinite(track).all(), name
        assert np.all(np.diff(track[:,0]) >= 0), name
        # Keep the last state if an algorithm publishes repeated same-time updates.
        track = track[np.r_[np.diff(track[:,0]) != 0, True]]
        assert len(track) > 10, name
        prism = prism_positions(track)
        # Epoch timestamps near 1.6e9 have sub-microsecond float quantization.
        # Tolerate 1 us at the 0.1 s boundary; do not discard exact 10 Hz spans.
        values, valid = interpolate_reference(gt[:,0], np.column_stack([track[:,0], prism]), tolerance=.1+1e-6)
        sampled[name] = values; masks[name] = valid; raw_counts[name] = len(track)
        spans[name] = [float(track[0,0]), float(track[-1,0])]
    shared = np.logical_and.reduce(list(masks.values()))
    assert shared.sum() > 10, 'insufficient shared reference epochs'
    reference = gt[shared,1:4]; stamps = gt[shared,0]
    result = dict(protocol=dict(epochs='original Leica CSV column 0 timestamps; identical across all methods',
        estimated_position_interpolation='linear interpolation of estimated prism positions; both brackets within 0.1s (+1us numerical tolerance)',
        alignment='independent fixed-scale rigid SE3 on shared epochs',
        prism_lever_arm_m=[-.293656,-.012288,-.273095], time_offset_fit=False,
        note='Different sampling from the initial two-mode report: all methods are recomputed.'),
        reference_epochs=len(gt), shared_epochs=int(shared.sum()),
        evaluated_span_s=float(stamps[-1]-stamps[0]), methods={})
    args.output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    axes[0,0].plot(reference[:,0],reference[:,1],color='black',lw=2,label='Leica')
    for name in sampled:
        fitted, rotation, translation, _ = rigid_align(sampled[name][shared], reference)
        residual = fitted-reference; err = np.linalg.norm(residual,axis=1)
        result['methods'][name] = dict(ate=error_statistics(residual), output_poses=raw_counts[name],
            output_span=spans[name], individually_associated_epochs=int(masks[name].sum()),
            shared_reference_fraction=float(shared.mean()), rotation=rotation.tolist(), translation=translation.tolist())
        np.savetxt(args.output/(name+'_aligned.csv'), np.column_stack([stamps,fitted,reference,err]),
                   delimiter=',',header='stamp,est_x,est_y,est_z,gt_x,gt_y,gt_z,error_m')
        axes[0,0].plot(fitted[:,0],fitted[:,1],label=name,alpha=.8)
        axes[0,1].plot(stamps-stamps[0],err,label=name,alpha=.8)
        axes[1,0].plot(stamps-stamps[0],residual[:,2],label=name,alpha=.8)
    names=list(sampled)
    axes[1,1].barh(names,[result['methods'][n]['ate']['rmse_m']*100 for n in names])
    axes[0,0].set(title='Prism trajectory: SE3, fixed scale',xlabel='X [m]',ylabel='Y [m]');axes[0,0].axis('equal')
    axes[0,1].set(title='Position error: identical reference epochs',xlabel='Time [s]',ylabel='Error [m]')
    axes[1,0].set(title='Leica-frame Z residual',xlabel='Time [s]',ylabel='Residual [m]')
    axes[1,1].set(title='Position ATE RMSE',xlabel='ATE [cm]')
    for ax in axes.ravel():ax.grid(alpha=.2)
    for ax in [axes[0,0],axes[0,1],axes[1,0]]:ax.legend(fontsize=8)
    fig.savefig(args.output/'comparison.png',dpi=160);plt.close(fig)
    (args.output/'metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
