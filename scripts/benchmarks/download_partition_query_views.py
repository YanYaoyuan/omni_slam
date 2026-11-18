"""Download only frozen unused RGB/LiDAR frames, then verify measured visibility."""
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from prepare_partition_query_probe import OUT, META
from prepare_hypersim_subset import RangeFile
from download_kitti360_windows import URL, extract
from omni_semantic_map.kitti360_input import load_left_calibration, load_poses, load_times, project_lidar
from omni_semantic_map.sparse_lidar_support import project_measured_visibility


def main():
    protocol = read(OUT/'protocol.json')
    assert protocol['generator_sha256'] == sha(ROOT/'scripts/benchmarks/prepare_partition_query_probe.py')
    for f, h in protocol['source_hashes'].items(): assert sha(f) == h, f
    assert not (OUT/'verification_inputs.json').exists()
    tasks = [(seq, frames, kind, suffix, extension) for seq, frames in protocol['downloads'].items()
             for kind, suffix, extension in [('data_2d_raw','image_00','.png'), ('data_3d_raw','velodyne','.bin')]]
    def worker(task):
        seq, frames, kind, suffix, extension = task
        url = URL+kind+'/'+seq+'_'+suffix+'.zip'
        stream = RangeFile(url, 64*1024**2)
        with zipfile.ZipFile(stream) as archive:
            wanted = {f'{f:010d}'+extension for f in frames}
            members = [i.filename for i in archive.infolist() if Path(i.filename).name in wanted]
            assert len(members) == len(wanted)
            files = [extract(archive, member, OUT/'raw'/kind) for member in members]
        result = dict(url=url, frames=frames, files=files, downloaded_bytes=stream.used)
        dump(OUT/(seq+'_'+suffix+'_download.json'), result)
        print('DOWNLOADED', seq, suffix, len(files), stream.used, flush=True)
        return result
    with ThreadPoolExecutor(max_workers=2) as pool: downloads = list(pool.map(worker, tasks))
    calib = load_left_calibration(META/'calibration')
    source_files = {str(p):sha(p) for p in (META/'calibration').glob('*.txt')}
    records = []
    for seq, frames in protocol['downloads'].items():
        key = next(p['key'] for p in protocol['pairs'] if p['sequence'] == seq)
        old = read(ROOT/'build/benchmarks/discovery_measured_inputs_v1'/key/'manifest.json')
        scene = next(s for s in read(ROOT/'build/benchmarks/signed_query_association_v1/protocol.json')['scenes'] if s['key'] == key)
        job = read(scene['job']); origin = np.asarray(job['origin_world'])
        poses = load_poses(META/seq/'poses.txt')
        images = load_times(META/seq/'image_00/timestamps.txt'); scans = load_times(META/seq/'velodyne_points/timestamps.txt')
        source = np.load(old['geometry'])['xyz']
        old_frames = {r['frame_id'] for r in old['records']}
        for frame in frames:
            assert frame not in old_frames
            rgb = OUT/'raw/data_2d_raw'/seq/'image_00/data_rect'/f'{frame:010d}.png'
            scan = OUT/'raw/data_3d_raw'/seq/'velodyne_points/data'/f'{frame:010d}.bin'
            xyz = np.fromfile(scan, dtype='<f4').reshape(-1,4)[:,:3]
            projection = project_lidar(xyz, calib)
            pose = poses[frame]@calib['pose_from_cam']@np.linalg.inv(calib['rectification'])
            pose[:3,3] -= origin
            x,y,_,visible = project_measured_visibility(source, pose, job['config']['cam'], projection['depth'],
                                                       SimpleNamespace(depth_absolute_m=.05,depth_relative=0.))
            ids = np.flatnonzero(visible)
            path = OUT/'views'/key/f'{frame:06d}.npz'; path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path, visible_source_ids=ids, pixel_xy=np.column_stack([x[ids],y[ids]]), camera_pose=pose)
            records.append(dict(key=key, frame_id=frame, rgb=str(rgb), raw_scan=str(scan), file=str(path),
                                visible_points=len(ids), scan_minus_image_ms=float((scans[frame]-images[frame])/1e6)))
            for f in [rgb, scan, path]: source_files[str(f)] = sha(f)
    dump(OUT/'verification_inputs.json', dict(complete=True, protocol_sha256=sha(OUT/'protocol.json'),
         script_sha256=sha(__file__), files=source_files, records=records, downloads=downloads,
         scope='Window-end projection of unchanged old measured source map into unused observed images, checked against new actual LiDAR depth. No future-point causality claim, interpolation or GT.'))
    print('INPUTS COMPLETE', [(r['key'],r['frame_id'],r['visible_points']) for r in records], flush=True)


if __name__=='__main__':main()
