"""Verify new reserved inputs and observed geometry, without reading GT labels."""
from pathlib import Path
import zlib
import numpy as np
from PIL import Image
from paper_final_common import ROOT,read,dump,sha
from prepare_kitti360_unseen_v4 import DEST,META
from omni_semantic_map.kitti360_input import load_left_calibration,load_poses,load_times,project_lidar


def main():
    assert not (DEST/'verification.json').exists()
    selection=read(DEST/'window_selection.json');data=DEST/'data';status=read(data/'download_status.json')
    assert status['complete'] and not status['failed'];assert status['selection_sha256']==sha(DEST/'window_selection.json')
    for f,h in selection['source_sha256'].items():assert sha(ROOT/f)==h,f
    files={};count=0
    for path in data.glob('*_download.json'):
        seal=read(path);assert seal['complete'] and seal['selection_sha256']==sha(DEST/'window_selection.json')
        for item in seal['files']:
            f=ROOT/item['path'];assert sha(f)==item['sha256'];assert f.stat().st_size==item['bytes'];crc=0
            with f.open('rb') as stream:
                for block in iter(lambda:stream.read(1024**2),b''):crc=zlib.crc32(block,crc)
            assert crc&0xffffffff==item['crc32'];files[str(f)]=item['sha256'];count+=1
    assert count==2*sum(len(w['input_frames']) for w in selection['windows'])+2
    calibration=load_left_calibration(META/'extracted/calibration');rect_inverse=np.linalg.inv(calibration['rectification'])
    bounds={};scenes={}
    for w in selection['windows']:
        scene=w['scene'];poses=load_poses(META/'extracted'/scene/'poses.txt');times=load_times(META/'extracted'/scene/'image_00/timestamps.txt');scans=load_times(META/'extracted'/scene/'velodyne_points/timestamps.txt')
        low=np.full(3,np.inf);high=np.full(3,-np.inf);raw_count=0;projected_count=0;offsets=[]
        for frame in w['input_frames']:
            image=data/'data_2d_raw'/scene/'image_00/data_rect'/f'{frame:010d}.png'
            with Image.open(image) as im:assert im.size==(calibration['width'],calibration['height']);im.verify()
            scan=data/'data_3d_raw'/scene/'velodyne_points/data'/f'{frame:010d}.bin'
            assert scan.stat().st_size%16==0;raw=np.fromfile(scan,dtype='<f4').reshape(-1,4);assert np.isfinite(raw).all()
            projection=project_lidar(raw[:,:3],calibration);ids=projection['selected_indices'];assert len(ids)>0
            pose=poses[frame]@calibration['pose_from_cam']@rect_inverse
            xyz=projection['camera_xyz'][ids]@pose[:3,:3].T+pose[:3,3]
            low=np.minimum(low,xyz.min(0));high=np.maximum(high,xyz.max(0));raw_count+=len(raw);projected_count+=len(ids)
            offsets.append(float((scans[frame]-times[frame])/1e6))
        bounds[scene]=[low.tolist(),high.tolist()];scenes[scene]=dict(frames=len(w['input_frames']),raw_points=raw_count,projected_points=projected_count,
            observed_world_aabb=bounds[scene],scan_minus_image_ms=dict(min=min(offsets),max=max(offsets),mean=float(np.mean(offsets))))
    oldbounds={}
    paths=[ROOT/'build/benchmarks/kitti360_development_v1/jobs/2013_05_28_drive_0000_sync.json']
    paths += [ROOT/'build/benchmarks/kitti360_reserved_v1'/s/'jobs'/(s+'.json') for s in ['2013_05_28_drive_0002_sync','2013_05_28_drive_0003_sync']]
    paths += [ROOT/f'build/benchmarks/intrinsic_public_holdout_v1/prepared/kitti360_{i:04d}/jobs/2013_05_28_drive_{i:04d}_sync.json' for i in [4,5]]
    paths += [ROOT/f'build/benchmarks/batch_transport_assessment_v1/prepared/kitti360_{i:04d}/jobs/2013_05_28_drive_{i:04d}_sync.json' for i in [6,7]]
    for path in paths:
        job=read(path);low=np.full(3,np.inf);high=np.full(3,-np.inf);origin=np.asarray(job['origin_world'])
        for frame in job['frames']:
            xyz=np.load(frame['measured'])['xyz']+origin;low=np.minimum(low,xyz.min(0));high=np.maximum(high,xyz.max(0))
        oldbounds[job['scene']]=[low.tolist(),high.tolist()]
    distances=[];prior=dict(oldbounds)
    for scene,box in bounds.items():
        x=np.asarray(box)
        for other,b in prior.items():
            y=np.asarray(b);distance=float(np.linalg.norm(np.maximum(np.maximum(x[0]-y[1],y[0]-x[1]),0)))
            distances.append(dict(a=scene,b=other,observed_aabb_distance_m=distance))
        prior[scene]=box
    dump(DEST/'verification.json',dict(complete=True,files=count,files_sha256=files,scenes=scenes,old_observed_aabbs=oldbounds,
        observed_aabb_distances=distances,observed_geometry_disjoint=all(r['observed_aabb_distance_m']>0 for r in distances),
        reference_labels_decoded=False,rgb_content_viewed=False,source_sha256=sha(__file__),selection_sha256=sha(DEST/'window_selection.json'),
        scope='Verifies file SHA256/CRC, PNG structure and dimensions, finite raw LiDAR, actual projected-source geometry separation, timestamps. References copied/hashed but never parsed. Does not assert reference-domain disjointness or provide independent algorithm results.'))
    print('VERIFIED',count,'files',[(k,v['frames']) for k,v in scenes.items()],'min observed separation',min(r['observed_aabb_distance_m'] for r in distances),flush=True)


if __name__=='__main__':main()
