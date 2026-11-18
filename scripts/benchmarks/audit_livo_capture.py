"""Verify recording coverage and atomic LIVO evidence against native archives."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import yaml
from audit_evidence_retention import ROOT,REPORT,dump
from omni_semantic_map.flat_benchmark import sha256


def audit(capture_path,output,config):
    captured=json.loads(capture_path.read_text());assert captured['complete']
    assert sha256(config)==captured['config_sha256'],'changed native LIVO configuration'
    params=yaml.safe_load(config.read_text())['/**']['ros__parameters'];cam=params['camera'];max_age=params['vio']['max_image_age_s']
    expected_K=np.array([[cam['fx'],0,cam['cx']],[0,cam['fy'],cam['cy']],[0,0,1.]])
    root=capture_path.parent;frames=captured['frames'];assert frames
    archive=list(csv.DictReader((root/'livo/keyframes.csv').open()));by_id={int(r['id']):r for r in archive}
    timing=list(csv.DictReader((root/'livo/timing.csv').open()));assert timing
    published=captured['native_keyframes'];pub_by_id={int(r['frame_id']):r for r in published}
    assert len(by_id)==len(archive) and len(pub_by_id)==len(published)
    received={r['frame_id'] for r in frames};assert len(received)==len(frames)
    assert received<=set(by_id) and set(pub_by_id)<=set(by_id)
    records=[]
    for frame in frames:
        for name,h in frame['sha256'].items():assert sha256(Path(name))==h,name
        native=by_id[frame['frame_id']];pose=np.array(frame['pose']);camera=np.array(frame['camera_from_tracking']);camera_pose=np.array(frame['camera_pose'])
        assert np.isfinite(pose).all() and np.isfinite(camera).all()
        np.testing.assert_allclose(camera_pose@camera,pose,atol=1e-9)
        np.testing.assert_allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-8)
        np.testing.assert_allclose(pose[:3,3],[float(native[k]) for k in ['x','y','z']],atol=1e-9)
        assert abs(frame['stamp_ns']/1e9-float(native['stamp']))<1e-6
        age=(frame['stamp_ns']-frame['image_stamp_ns'])/1e9
        assert -.001<=age<=max_age+.001,'packet violates configured image age'
        np.testing.assert_allclose(frame['intrinsic'],expected_K,atol=1e-9)
        assert frame['width']==cam['width'] and frame['height']==cam['height']
        points=np.load(frame['cloud'])['points'];assert points.shape==(frame['points'],3) and np.isfinite(points).all()
        records.append(dict(frame_id=frame['frame_id'],points=len(points),image_age_ms=age*1000))
    stamps=np.array([r['stamp_ns'] for r in frames],dtype=np.int64)/1e9;assert np.all(np.diff(stamps)>0)
    all_times=np.array([float(r['stamp']) for r in timing]);types=np.array([int(r['update_type']) for r in timing]);is_visual=types==1
    edges=np.linspace(all_times.min(),all_times.max(),11)
    bins=np.histogram(stamps,bins=edges)[0];visual_stamps=[r['stamp_ns']/1e9 for r in published if r['visual_update']]
    visual_hist=np.histogram(visual_stamps,bins=edges)[0]
    # Last update at a timestamp describes its final native pose for path length.
    trajectory=np.loadtxt(root/'livo/trajectory.txt');_,last_rev=np.unique(trajectory[::-1,0],return_index=True)
    indices=np.sort(len(trajectory)-1-last_rev);poses=trajectory[indices,1:4]
    native_visual={r['frame_id'] for r in published if r['visual_update']}
    updates=dict(lio=int((types==2).sum()),vio=int(is_visual.sum()),other=int((~np.isin(types,[1,2])).sum()))
    updates['visual_processing_ms_median']=float(np.median([float(r['processing_ms']) for r in timing if int(r['update_type'])==1])) if is_visual.any() else None
    result=dict(complete=True,source_sha256=sha256(Path(__file__)),capture_sha256=sha256(capture_path),scope=__doc__,
        archive_keyframes=len(archive),received_native_keyframe_messages=len(published),missing_native_message_ids=sorted(set(by_id)-set(pub_by_id)),
        native_visual_keyframes=len(native_visual),captured_semantic_frames=len(frames),missing_visual_packet_ids=sorted(native_visual-received),
        source_duration_s=captured['source_duration_s'],captured_visual_span_s=float(stamps[-1]-stamps[0]),
        captured_visual_span_fraction=float((stamps[-1]-stamps[0])/captured['source_duration_s']),
        longest_visual_packet_gap_s=float(np.diff(stamps).max()) if len(stamps)>1 else None,
        timeline=dict(bin_edges_sensor_epoch_s=edges.tolist(),captured_frames=bins.tolist(),published_visual_frames=visual_hist.tolist()),
        native_updates=updates,
        native_trajectory=dict(path_length_m=float(np.linalg.norm(np.diff(poses,axis=0),axis=1).sum()),
            start_end_distance_m=float(np.linalg.norm(poses[-1]-poses[0])),z_min_m=float(poses[:,2].min()),z_max_m=float(poses[:,2].max()),
            scope='Native odometry summary; endpoint distance and Z span are not trajectory drift without a known closed path or external GT.'),
        packet_image_age_ms_median=float(np.median([r['image_age_ms'] for r in records])),
        packet_image_age_ms_max=float(max(r['image_age_ms'] for r in records)),
        atomic_files_and_native_positions_verified=True,calibration_matches_config=True,configured_image_age_limit_s=max_age,frames=records,
        gate=dict(full_recording_visual_coverage=bool((stamps[-1]-stamps[0])/captured['source_duration_s']>=.95),
            all_ten_time_bins_have_visual_frames=bool(np.all(bins>0)),all_native_keyframe_messages_accounted=len(archive)==len(published)),
        limitations='Packet transport loss is explicit. This verifies epochs, native pose consistency and coverage, not SLAM accuracy or semantic correctness.')
    dump(output,result);print(json.dumps({k:result[k] for k in ['captured_semantic_frames','captured_visual_span_fraction','longest_visual_packet_gap_s','gate']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,default=ROOT/'build/experiments/semantic_20261106/managed_config/livo.yaml');a=p.parse_args();audit(a.capture,a.output,a.config)
