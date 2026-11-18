#!/usr/bin/env python3
"""Position-only NTU evaluation with prism lever arm and fixed-scale SE(3) fit.

No ground-truth orientation exists. Relative displacement errors below are NOT
full-pose RPE, and no scale or time-offset fitting is performed.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

PRISM_IN_IMU = np.array([-.293656, -.012288, -.273095])


def prism_positions(trajectory):
    return trajectory[:, 1:4] + Rotation.from_quat(trajectory[:, 4:8]).apply(PRISM_IN_IMU)


def interpolate_reference(stamps, gt, tolerance=.1):
    right = np.searchsorted(gt[:, 0], stamps, side='right')
    right = np.clip(right, 1, len(gt)-1); left = right-1
    before = stamps-gt[left,0]; after = gt[right,0]-stamps
    valid = (before>=0) & (after>=0) & (before<=tolerance) & (after<=tolerance)
    ratio = before/(gt[right,0]-gt[left,0])
    points = gt[left,1:4]+ratio[:,None]*(gt[right,1:4]-gt[left,1:4])
    return points, valid


def rigid_align(estimated, reference):
    a = estimated-estimated.mean(axis=0); b = reference-reference.mean(axis=0)
    u, singular, vt = np.linalg.svd(a.T@b)
    fix = np.eye(3); fix[-1,-1] = np.linalg.det(vt.T@u.T)
    rotation = vt.T@fix@u.T
    translation = reference.mean(axis=0)-rotation@estimated.mean(axis=0)
    return estimated@rotation.T+translation, rotation, translation, singular


def error_statistics(residual):
    norm = np.linalg.norm(residual, axis=1)
    return dict(rmse_m=float(np.sqrt(np.mean(norm**2))), median_m=float(np.median(norm)),
        p95_m=float(np.quantile(norm,.95)), max_m=float(norm.max()),
        xyz_rmse_m=np.sqrt(np.mean(residual**2,axis=0)).tolist())


def relative_displacement(stamps, estimated, reference, delta):
    target = stamps+delta
    j = np.searchsorted(stamps,target).clip(1,len(stamps)-1)
    closer = abs(stamps[j-1]-target)<abs(stamps[j]-target)
    j[closer]-=1
    good = (abs(stamps[j]-target)<=.05) & (j>np.arange(len(stamps)))
    i = np.flatnonzero(good); j = j[good]
    if not len(i):
        return dict(pairs=0)
    residual = (estimated[j]-estimated[i])-(reference[j]-reference[i])
    return dict(pairs=len(i), **error_statistics(residual))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--ground-truth', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    raw_gt = np.loadtxt(args.ground_truth, delimiter=',', skiprows=1)
    # Column 0 follows the original MATLAB tutorial (%time). The later official
    # notebook uses column 2 (header.stamp); keep this protocol fixed and report
    # that distinction rather than silently switching timestamp conventions.
    gt = np.column_stack([raw_gt[:,0]*1e-9,raw_gt[:,3:6]])
    assert np.isfinite(gt).all() and np.all(np.diff(gt[:,0])>0)
    tracks = {m:np.loadtxt(args.run/m/'trajectory_final.tum',ndmin=2) for m in ('lio','livo')}
    common, a, b = np.intersect1d(tracks['lio'][:,0],tracks['livo'][:,0], return_indices=True)
    if len(common)<10:
        raise ValueError('insufficient shared estimator epochs')
    reference, valid = interpolate_reference(common,gt)
    stamps=common[valid]; reference=reference[valid]
    if len(stamps)<10:
        raise ValueError('insufficient reference associations')
    result = dict(protocol=dict(reference='Leica prism position only', gt_csv_time_column=0,
        lever_arm_imu_m=PRISM_IN_IMU.tolist(), association_max_each_side_s=.1,
        alignment='independent rigid SE(3) fit per mode on identical timestamps, scale fixed to 1',
        fitted_time_offset_s=0, orientation_error_available=False,
        relative_metric='world-aligned position displacement residual, NOT full-pose RPE'),
        shared_epochs=len(common), evaluated_epochs=len(stamps),
        evaluated_start_s=float(stamps[0]),evaluated_end_s=float(stamps[-1]),
        evaluated_span_s=float(stamps[-1]-stamps[0]),
        reference_path_length_m=float(np.linalg.norm(np.diff(reference,axis=0),axis=1).sum()),
        modes={})
    aligned={}; errors={}
    for mode, indexes in [('lio',a),('livo',b)]:
        trajectory=tracks[mode][indexes][valid]
        estimated=prism_positions(trajectory)
        fitted,rotation,translation,singular=rigid_align(estimated,reference)
        aligned[mode]=fitted; errors[mode]=np.linalg.norm(fitted-reference,axis=1)
        timing=np.atleast_1d(np.genfromtxt(args.run/mode/'timing.csv',delimiter=',',names=True))
        visual=timing[timing['update_type']==1]
        result['modes'][mode]=dict(ate=error_statistics(fitted-reference),
            total_output_poses=len(tracks[mode]), evaluated_pose_fraction=len(stamps)/len(tracks[mode]),
            rigid_rotation=rotation.tolist(),rigid_translation=translation.tolist(),singular_values=singular.tolist(),
            relative_displacement_1s=relative_displacement(stamps,fitted,reference,1.),
            relative_displacement_10s=relative_displacement(stamps,fitted,reference,10.),
            visual_updates=len(visual),visual_updates_with_support=int((visual['visual_points']>0).sum()),
            processing_ms_p50=float(np.median(timing['processing_ms'])),
            processing_ms_p95=float(np.quantile(timing['processing_ms'],.95)),
            processing_ms_total=float(timing['processing_ms'].sum()))
        np.savetxt(args.output/(mode+'_aligned_prism.csv'),np.column_stack([stamps,fitted,reference,errors[mode]]),
            delimiter=',',header='stamp,estimate_x,estimate_y,estimate_z,gt_x,gt_y,gt_z,error_m')
    (args.output/'metrics.json').write_text(json.dumps(result,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    colors={'lio':'#2463b0','livo':'#d87716'}
    axes[0,0].plot(reference[:,0],reference[:,1],color='black',lw=2,label='Leica reference')
    t=stamps-stamps[0]
    axes[1,0].plot(t,reference[:,2],color='black',lw=2,label='Leica reference')
    for mode in aligned:
        rmse=result['modes'][mode]['ate']['rmse_m']
        label=f'omni_slam ({mode.upper()})'
        axes[0,0].plot(aligned[mode][:,0],aligned[mode][:,1],color=colors[mode],label=f'{label} ({rmse:.3f} m ATE)')
        axes[0,1].plot(t,errors[mode],color=colors[mode],label=label)
        axes[1,0].plot(t,aligned[mode][:,2],color=colors[mode],label=label)
        axes[1,1].plot(t,aligned[mode][:,2]-reference[:,2],color=colors[mode],label=label)
    axes[0,0].set(title='Prism trajectory: rigid alignment, no scale fit',xlabel='X [m]',ylabel='Y [m]');axes[0,0].axis('equal')
    axes[0,1].set(title='Position error on shared timestamps',xlabel='Time [s]',ylabel='Error [m]')
    axes[1,0].set(title='Reference-frame Z coordinate',xlabel='Time [s]',ylabel='Z [m]')
    axes[1,1].set(title='Reference-frame Z residual',xlabel='Time [s]',ylabel='Residual [m]')
    for ax in axes.ravel():
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(args.output/'trajectory_errors.png',dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
