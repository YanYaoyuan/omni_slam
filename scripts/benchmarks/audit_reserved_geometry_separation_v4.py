"""Input-space separation of reserved raw scans; never reads RGB or GT labels."""
import numpy as np
from pathlib import Path
from paper_final_common import ROOT, read, dump, sha
from omni_semantic_map.kitti360_input import load_left_calibration, load_poses

DEST=ROOT/'build/datasets/kitti360_reserved_v4'
META=ROOT/'build/datasets/kitti360_prepare'


def run():
    assert not (DEST/'geometry_separation_audit.json').exists()
    assert read(DEST/'download_provenance.json')['complete']
    selection=read(DEST/'window_selection.json');calib=load_left_calibration(META/'extracted/calibration')
    c2l=np.linalg.inv(calib['rectification'])@calib['rectified_from_velodyne']
    lidar_from_pose=calib['pose_from_cam']@c2l
    jobs=[ROOT/'build/benchmarks/kitti360_development_v1/jobs/2013_05_28_drive_0000_sync.json']
    jobs += [ROOT/f'build/benchmarks/kitti360_reserved_v1/2013_05_28_drive_{i:04d}_sync/jobs/2013_05_28_drive_{i:04d}_sync.json' for i in [2,3]]
    jobs += [ROOT/f'build/benchmarks/intrinsic_public_holdout_v1/prepared/kitti360_{i:04d}/jobs/2013_05_28_drive_{i:04d}_sync.json' for i in [4,5]]
    jobs += [ROOT/f'build/benchmarks/batch_transport_assessment_v1/prepared/kitti360_{i:04d}/jobs/2013_05_28_drive_{i:04d}_sync.json' for i in [6,7]]
    windows={};files={}
    for jobfile in jobs:
        job=read(jobfile);files[str(jobfile)]=sha(jobfile)
        windows[job['scene']]=dict(role='exposed',scans={f['frame_id']:Path(f['raw_scan']) for f in job['frames']})
    for w in selection['windows']:
        manifest=read(DEST/'data'/(w['scene']+'_velodyne_download.json'));assert manifest['complete']
        scans={int(Path(f['path']).stem):ROOT/f['path'] for f in manifest['files']};assert sorted(scans)==w['input_frames']
        windows[w['scene']]=dict(role='reserved',scans=scans)
    boxes={}
    for scene,row in windows.items():
        posefile=META/'extracted'/scene/'poses.txt';poses=load_poses(posefile);files[str(posefile)]=sha(posefile)
        lo=np.full(3,np.inf);hi=np.full(3,-np.inf);count=0
        for frame,path in row['scans'].items():
            raw=np.fromfile(path,dtype='<f4').reshape(-1,4)[:,:3].astype(float);raw=raw[np.isfinite(raw).all(1)&(np.linalg.norm(raw,axis=1)>0)]
            transform=poses[frame]@lidar_from_pose;world=raw@transform[:3,:3].T+transform[:3,3]
            lo=np.minimum(lo,world.min(0));hi=np.maximum(hi,world.max(0));count+=len(raw);files[str(path)]=sha(path)
        boxes[scene]=dict(role=row['role'],frames=len(row['scans']),raw_returns=count,aabb=[lo.tolist(),hi.tolist()])
    rows=[]
    for a,aa in boxes.items():
        if aa['role']!='reserved':continue
        for b,bb in boxes.items():
            if a==b or (bb['role']=='reserved' and b<a):continue
            x,y=np.array(aa['aabb']),np.array(bb['aabb'])
            distance=float(np.linalg.norm(np.maximum(np.maximum(x[0]-y[1],y[0]-x[1]),0)))
            rows.append(dict(reserved=a,other=b,raw_geometry_aabb_lower_bound_m=distance,separated_over20cm=distance>.2))
    proven=all(r['separated_over20cm'] for r in rows)
    dump(DEST/'geometry_separation_audit.json',dict(complete=True,separation_proven=proven,boxes=boxes,pairs=rows,
        input_sha256=files,source_sha256=sha(__file__),selection_sha256=sha(DEST/'window_selection.json'),
        viewed_rgb=False,decoded_reference_labels=False,
        scope='AABB Euclidean separation is a conservative lower bound on every pair of actual scanned points. Positive>20cm proves no measured-surface overlap at evaluation radius. Zero is inconclusive, not proof of overlap; requires further exact check before independence claim. Does not prove domain generalization or eliminate geographic correlation.'))
    print('RESERVED RAW GEOMETRY SEPARATION',proven,rows,flush=True)


if __name__=='__main__':run()
