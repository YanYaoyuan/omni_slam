"""Acquire frozen common validation views without decoding any reference labels."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import zipfile
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from prepare_partition_revision import OUT,META,PARENT,INPUT
from prepare_hypersim_subset import RangeFile
from download_kitti360_windows import URL,extract
from omni_semantic_map.kitti360_input import load_left_calibration,load_poses,load_times,project_lidar
from omni_semantic_map.sparse_lidar_support import project_measured_visibility


def main():
    seal=read(OUT/'candidates_complete.json');scope=read(OUT/'common_view_execution_scope.json')
    assert seal['complete'] and not (OUT/'verification_inputs.json').exists()
    for f,h in seal['files'].items():assert sha(f)==h,f
    scenes={s['key']:s for s in read(PARENT/'protocol.json')['scenes']}
    jobs={key:read(scenes[key]['job']) for key in scope['scenes']}
    tasks=[(key,jobs[key]['scene'],r['verification_frames'],kind,suffix,extension)
           for key,r in scope['scenes'].items()
           for kind,suffix,extension in [('data_2d_raw','image_00','.png'),('data_3d_raw','velodyne','.bin')]]
    dump(OUT/'download_execution.json',dict(candidate_seal_sha256=sha(OUT/'candidates_complete.json'),
        common_scope_sha256=sha(OUT/'common_view_execution_scope.json'),source_sha256=sha(__file__),
        per_archive_limit_bytes=64*1024**2,workers=2,reference_labels_requested=False))
    def worker(task):
        key,seq,frames,kind,suffix,ext=task;receipt=OUT/(seq+'_'+suffix+'_download.json')
        if receipt.exists():
            old=read(receipt);assert old['frames']==frames
            for f in old['files']:assert sha(ROOT/f['path'])==f['sha256']
            return old
        stream=RangeFile(URL+kind+'/'+seq+'_'+suffix+'.zip',64*1024**2)
        with zipfile.ZipFile(stream) as archive:
            names={f'{f:010d}'+ext for f in frames}
            members=[i.filename for i in archive.infolist() if Path(i.filename).name in names]
            assert len(members)==len(names)
            files=[extract(archive,m,OUT/'raw'/kind) for m in members]
        record=dict(key=key,url=stream.url,frames=frames,files=files,downloaded_bytes=stream.used)
        dump(receipt,record);print('DOWNLOAD',key,suffix,len(files),stream.used,flush=True);return record
    with ThreadPoolExecutor(max_workers=2) as pool:downloads=list(pool.map(worker,tasks))
    calibration=load_left_calibration(META/'calibration');files={str(f):sha(f) for f in (META/'calibration').glob('*.txt')};records=[]
    earlier=read(ROOT/'build/benchmarks/partition_query_probe_v1/verification_inputs.json')['records']
    for key,entry in scope['scenes'].items():
        job=jobs[key];seq=job['scene'];old=read(INPUT/key/'manifest.json');points=np.load(old['geometry'])['xyz']
        poses=load_poses(META/seq/'poses.txt');images=load_times(META/seq/'image_00/timestamps.txt');scans=load_times(META/seq/'velodyne_points/timestamps.txt')
        excluded={r['frame_id'] for r in old['records']}|{r['frame_id'] for r in earlier if r['key']==key}
        for frame in entry['verification_frames']:
            assert frame not in excluded
            rgb=OUT/'raw/data_2d_raw'/seq/'image_00/data_rect'/f'{frame:010d}.png'
            scan=OUT/'raw/data_3d_raw'/seq/'velodyne_points/data'/f'{frame:010d}.bin'
            raw=np.fromfile(scan,dtype='<f4').reshape(-1,4)[:,:3];projection=project_lidar(raw,calibration)
            pose=poses[frame]@calibration['pose_from_cam']@np.linalg.inv(calibration['rectification']);pose[:3,3]-=np.asarray(job['origin_world'])
            x,y,_,visible=project_measured_visibility(points,pose,job['config']['cam'],projection['depth'],SimpleNamespace(depth_absolute_m=.05,depth_relative=0.))
            ids=np.flatnonzero(visible);path=OUT/'views'/key/f'{frame:06d}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,visible_source_ids=ids,pixel_xy=np.column_stack([x[ids],y[ids]]),camera_pose=pose)
            records.append(dict(key=key,frame_id=frame,rgb=str(rgb),raw_scan=str(scan),file=str(path),visible_points=len(ids),
                                scan_minus_image_ms=float((scans[frame]-images[frame])/1e6)))
            for f in [rgb,scan,path]:files[str(f)]=sha(f)
    dump(OUT/'verification_inputs.json',dict(complete=True,candidate_seal_sha256=sha(OUT/'candidates_complete.json'),
        common_scope_sha256=sha(OUT/'common_view_execution_scope.json'),source_sha256=sha(__file__),files=files,records=records,downloads=downloads,
        scope='Unused class-query validation inputs in exposed development windows. Exact published camera poses, current real LiDAR depth gating; offline unchanged measured source map. No new point geometry or labels introduced.'))
    print('INPUTS COMPLETE',len(records),flush=True)


if __name__=='__main__':main()
