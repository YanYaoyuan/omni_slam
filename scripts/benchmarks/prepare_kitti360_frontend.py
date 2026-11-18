"""Prepare development-only KITTI-360 inputs for the existing staged image encoder.

This intentionally does not launch reserved-validation inference. Geometry uses
measured LiDAR points, never a backprojection of rounded sparse depth pixels.
"""
from pathlib import Path
import json
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from prepare_strong_jobs import sources
from omni_semantic_map.kitti360_input import load_left_calibration, load_poses, load_times, project_lidar

META = ROOT/'build/datasets/kitti360_prepare'
DATA = ROOT/'build/datasets/kitti360_windows_v1'
OUT = ROOT/'build/benchmarks/kitti360_development_v1'


def main():
    selection = read(META/'window_selection.json')
    window = selection['windows'][0]
    # Fixed first official training window; never silently advance to a test row.
    assert window['scene'] == '2013_05_28_drive_0000_sync' and window['start'] == 2 and window['end'] == 385
    scene = window['scene']
    job_path = OUT/'jobs'/(scene+'.json')
    assert not job_path.exists(), 'Do not overwrite an executed input specification'
    hashes = {str(META/'window_selection.json'): sha(META/'window_selection.json')}
    for kind in ['image_00', 'velodyne']:
        path = DATA/(scene+'_'+kind+'_download.json')
        manifest = read(path)
        assert manifest['complete'] and manifest['selection_sha256'] == sha(META/'window_selection.json')
        hashes[str(path)] = sha(path)
        for item in manifest['files']:
            path = ROOT/item['path']
            assert sha(path) == item['sha256']
            hashes[str(path)] = item['sha256']
    calibration_dir = META/'extracted/calibration'
    calib = load_left_calibration(calibration_dir)
    pose_path = META/'extracted'/scene/'poses.txt'
    poses = load_poses(pose_path)
    for path in [pose_path, *calibration_dir.glob('*.txt')]:
        hashes[str(path)] = sha(path)
    image_time_path = META/'extracted'/scene/'image_00/timestamps.txt'
    scan_time_path = META/'extracted'/scene/'velodyne_points/timestamps.txt'
    images, scans = load_times(image_time_path), load_times(scan_time_path)
    for path in [image_time_path, scan_time_path]: hashes[str(path)] = sha(path)
    rect_inverse = np.linalg.inv(calib['rectification'])
    camera_poses = {f: poses[f]@calib['pose_from_cam']@rect_inverse for f in window['input_frames']}
    origin = camera_poses[window['input_frames'][0]][:3, 3].copy()
    reference_job = ROOT/'build/benchmarks/hypersim_shared_v1/jobs/ai_001_001.json'
    cfg = read(reference_job)['config']
    hashes[str(reference_job)] = sha(reference_job)
    cfg['dataset_name'] = 'kitti360_measured_lidar'
    cfg['semantic']['depth_filter'] = False
    cfg['cam'] = dict(H=calib['height'], W=calib['width'], fx=float(calib['K'][0, 0]),
        fy=float(calib['K'][1, 1]), cx=float(calib['K'][0, 2]), cy=float(calib['K'][1, 2]), depth_scale=1.)
    folder = OUT/'inputs'/scene
    folder.mkdir(parents=True, exist_ok=False)
    records = []
    for frame in window['input_frames']:
        scan = DATA/'data_3d_raw'/scene/'velodyne_points/data'/f'{frame:010d}.bin'
        image = DATA/'data_2d_raw'/scene/'image_00/data_rect'/f'{frame:010d}.png'
        raw = np.fromfile(scan, dtype='<f4').reshape(-1, 4)
        projection = project_lidar(raw[:, :3], calib)
        ids = projection['selected_indices']
        yy, xx = np.nonzero(projection['source_point_index'] >= 0)
        np.testing.assert_array_equal(projection['source_point_index'][yy, xx], ids)
        pose = camera_poses[frame].copy()
        pose[:3, 3] -= origin
        xyz = projection['camera_xyz'][ids]@pose[:3, :3].T+pose[:3, 3]
        dest = folder/f'{frame:06d}_measured.npz'
        np.savez_compressed(dest, depth=projection['depth'], xyz=xyz, raw_indices=ids,
            pixel_xy=np.column_stack([xx, yy]).astype(np.int32), origin_world=origin)
        hashes[str(dest)] = sha(dest)
        records.append(dict(frame_id=frame, rgb=str(image), raw_scan=str(scan),
            measured=str(dest), pose=pose.tolist(), semantic=True,
            image_timestamp_ns=int(images[frame]), scan_timestamp_ns=int(scans[frame]),
            scan_minus_image_ms=float((scans[frame]-images[frame])/1e6)))
    source_hashes = sources()
    for path in [Path(__file__), ROOT/'omni_semantic_map/omni_semantic_map/kitti360_input.py']:
        source_hashes[str(path.relative_to(ROOT))] = sha(path)
    job = dict(kind='kitti360_development', scene=scene, frames=records, config=cfg,
        output=str(OUT/'frontend'/scene), input_sha256=hashes, source_sha256=source_hashes,
        reference_cache=None, origin_world=origin.tolist(), no_gt_in_inference=True,
        phases=['masks', 'triplets', 'features'], peak_cuda_allocator_limit_gib=6,
        geometry='Separate measured-source adapter required; native RGB-D geometry phase forbidden.',
        scope='One exposed development window only. Reserved windows require a separate frozen evaluation protocol.',
        timing_policy='Full-precision published system pose composed with calibration at paired image epoch. No per-point scan deskew or temporal interpolation. Signed offsets retained.',
        pixel_policy='Nearest pixel center floor(uv+0.5); zero depth is unknown. Source geometry retains actual transformed scan coordinates.',
        coordinate_policy='Subtract fixed first camera translation in float64 before any downstream float32 operations; apply the identical translation to evaluation references.')
    dump(job_path, job)
    print(json.dumps(dict(job=str(job_path), frames=len(records), origin_world=origin.tolist()), indent=2))


if __name__ == '__main__': main()
