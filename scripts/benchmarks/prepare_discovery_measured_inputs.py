"""Prepare real measured query inputs, without masks, GT, or method scores."""
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from omni_semantic_map.sparse_lidar_support import MeasuredVoxelMap,project_measured_visibility
from omni_semantic_map.discovery_measured_input import robot_measured_visibility

OUT=ROOT/'build/benchmarks/discovery_measured_inputs_v1'


def main():
    assert not OUT.exists()
    specs=[dict(key='kitti360_0000',kind='outdoor',job=str(ROOT/'build/benchmarks/kitti360_development_v1/jobs/2013_05_28_drive_0000_sync.json'),cache=str(ROOT/'build/benchmarks/kitti360_connected_development_v1/cache')),
        dict(key='kitti360_0002',kind='outdoor',job=str(ROOT/'build/benchmarks/kitti360_reserved_v1/2013_05_28_drive_0002_sync/jobs/2013_05_28_drive_0002_sync.json'),cache=str(ROOT/'build/benchmarks/kitti360_reserved_v1/2013_05_28_drive_0002_sync/connected/cache')),
        dict(key='rosbag2245',kind='robot',job=str(ROOT/'build/benchmarks/evidence_retention_20261111/jobs/livo_full_v2.json'))]
    # Fail before preparing anything if any declared job path is wrong.
    for s in specs:assert Path(s['job']).is_file(),s['job']
    sources=[Path(__file__),ROOT/'omni_semantic_map/omni_semantic_map/discovery_measured_input.py',ROOT/'omni_semantic_map/omni_semantic_map/sparse_lidar_support.py']
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),scenes=specs,source_hashes={str(f):sha(f) for f in sources},
        purpose='Input preparation only; not a frozen prediction/evaluation experiment.',voxel_m=.05,depth_gate_m=.05,
        causality='Points exist only after their first measured return; per-frame visibility uses current measured scan. No future masks or GT. Buffer policy may query only frames already received.',
        robot='Persistent first-measurement voxel IDs replace concatenated scan-local IDs for query memory. No centroid/filled-depth geometry. This changes live representation; future controls must all share these inputs.',
        scope='Two exposed outdoor development windows and330-frame robot cache. No new independent test.'))
    scene_records=[];files={}
    for s in specs:
        job=read(s['job']);folder=OUT/s['key'];(folder/'frames').mkdir(parents=True);records=[];input_hashes={s['job']:sha(s['job'])};start=time.monotonic()
        if s['kind']=='outdoor':
            cache=Path(s['cache']);manifest=read(cache/'manifest.json');g=np.load(cache/'geometry.npz');xyz=g['xyz'];source_frames=g['source_frame']
            input_hashes[str(cache/'geometry.npz')]=sha(cache/'geometry.npz');input_hashes[str(cache/'manifest.json')]=sha(cache/'manifest.json')
            assert len(manifest['records'])==len(job['frames'])
        else:voxel=MeasuredVoxelMap(.05)
        for rank,frame in enumerate(job['frames']):
            input_hashes[frame['rgb']]=sha(frame['rgb'])
            if s['kind']=='outdoor':
                rec=manifest['records'][rank];assert rec['frame_id']==frame['frame_id'];n=rec['map_points'];assert np.all(source_frames[:n]<=frame['frame_id'])
                input_hashes[frame['measured']]=sha(frame['measured']);depth=np.load(frame['measured'])['depth']
                x,y,_,visible=project_measured_visibility(xyz[:n],np.asarray(frame['pose']),job['config']['cam'],depth,SimpleNamespace(depth_absolute_m=.05,depth_relative=0.))
                ids=np.flatnonzero(visible);pixels=np.column_stack([x[ids],y[ids]]);pose=frame['pose'];extra={}
            else:
                input_hashes[frame['cloud']]=sha(frame['cloud']);points=np.load(frame['cloud'])['points'];t=np.asarray(frame['pose']);world=points@t[:3,:3].T+t[:3,3]
                voxel.append(world,np.zeros((len(points),3),np.uint8),frame['frame_id'],np.arange(len(points)),np.zeros((len(points),2),np.int32))
                xyz=voxel.arrays()['xyz'];n=len(xyz);ids,pixels,extra=robot_measured_visibility(xyz,points,frame);pose=frame['camera_pose']
            dest=folder/'frames'/f'{frame["frame_id"]:06d}.npz';np.savez_compressed(dest,visible_source_ids=ids,pixel_xy=pixels)
            record=dict(frame_id=frame['frame_id'],rgb=frame['rgb'],file=str(dest),sha256=sha(dest),map_points=n,visible_points=len(ids),camera_pose=pose,**extra)
            records.append(record);files[str(dest)]=record['sha256']
            if rank%50==0:print('PREPARED',s['key'],rank,'points',n,'visible',len(ids),flush=True)
        if s['kind']=='robot':
            geometry=folder/'geometry.npz';np.savez_compressed(geometry,**voxel.arrays());files[str(geometry)]=sha(geometry)
        else:geometry=Path(s['cache'])/'geometry.npz'
        result=dict(complete=True,key=s['key'],kind=s['kind'],records=records,geometry=str(geometry),geometry_sha256=sha(geometry),
            input_hashes=input_hashes,wall_seconds=time.monotonic()-start,source_points=len(xyz),frames=len(records),no_masks_or_gt=True)
        dump(folder/'manifest.json',result);files[str(folder/'manifest.json')]=sha(folder/'manifest.json');scene_records.append(dict(key=s['key'],frames=len(records),source_points=len(xyz)))
    dump(OUT/'complete.json',dict(complete=True,files=files,scenes=scene_records,protocol_sha256=sha(OUT/'protocol.json')))
    print('COMPLETE',scene_records,flush=True)


if __name__=='__main__':main()
