"""Verify real KITTI-360 sample geometry and show measured laser projections."""
import json
from decimal import Decimal
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper_final_common import ROOT, read, dump, sha
from omni_semantic_map.kitti360_input import load_left_calibration, load_poses, load_times, project_lidar


def main():
    base = ROOT/'build/datasets/kitti360_prepare'; sample = read(base/'sample_manifest.json')
    for source in sample['sources'].values():
        for record in source['files']: assert sha(ROOT/record['path']) == record['sha256']
    for name, record in read(base/'downloads.json').items():
        assert sha(base/name) == record['sha256']
        for f in record.get('files', []): assert sha(ROOT/f['path']) == f['sha256']
    scene = sample['scene']; extracted = base/'extracted'; calib = load_left_calibration(extracted/'calibration')
    poses = load_poses(extracted/scene/'poses.txt'); camera_poses = load_poses(extracted/scene/'cam0_to_world.txt')
    # The published camera file uses fewer decimal digits (e.g. millimetres in
    # world Y). Check each component against its encoded rounding interval.
    rounding = {}
    for line in (extracted/scene/'cam0_to_world.txt').read_text().splitlines():
        tokens = line.split()
        rounding[int(tokens[0])] = np.array([.5*10.**Decimal(t).as_tuple().exponent for t in tokens[1:]]).reshape(4, 4)
    images = load_times(extracted/scene/'image_00/timestamps.txt'); scans = load_times(extracted/scene/'velodyne_points/timestamps.txt')
    out = ROOT/'output/kitti360_interface'; out.mkdir(parents=True, exist_ok=True)
    frame_records = []; fig, axes = plt.subplots(len(sample['frames']), 1, figsize=(15, 12), layout='constrained')
    for ax, frame in zip(axes, sample['frames']):
        image_path = base/'sample/data_2d_raw'/scene/'image_00/data_rect'/f'{frame:010d}.png'
        cloud_path = base/'sample/data_3d_raw'/scene/'velodyne_points/data'/f'{frame:010d}.bin'
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        assert image.shape[:2] == (calib['height'], calib['width'])
        cloud = np.fromfile(cloud_path, dtype='<f4').reshape(-1, 4)[:, :3]
        projection = project_lidar(cloud, calib)
        composed = poses[frame]@calib['pose_from_cam']@np.linalg.inv(calib['rectification'])
        published = camera_poses[frame]
        pose_error = float(np.abs(composed-published).max())
        allowed = rounding[frame]+1e-8
        assert (np.abs(composed-published) <= allowed).all(), ('Published pose exceeds encoded rounding bound', frame)
        delta_ms = float((scans[frame]-images[frame])/1e6)
        indices = projection['selected_indices']; camera_xyz = projection['camera_xyz'][indices]
        # Preserve exact measured source coordinates; never reconstruct XYZ from a filled depth map.
        world_xyz = camera_xyz@published[:3, :3].T+published[:3, 3]
        pixel_indices = projection['source_point_index']; yy, xx = np.nonzero(pixel_indices >= 0)
        ax.imshow(image); ax.scatter(xx, yy, c=projection['depth'][yy, xx], cmap='turbo', vmin=0, vmax=60, s=.45, alpha=.7, linewidths=0)
        ax.set_title(f'{scene} | frame {frame} | measured LiDAR depth colors | scan-image offset {delta_ms:+.3f} ms')
        ax.axis('off')
        np.savez_compressed(out/f'{frame:010d}_input.npz', measured_depth=projection['depth'],
                            source_point_index=pixel_indices, selected_source_indices=indices,
                            xyz_world_at_image_pose=world_xyz, K=calib['K'], world_from_camera=published)
        frame_records.append(dict(frame=frame, raw_points=len(cloud), visible_pixel_returns=len(indices),
             depth_coverage=float(len(indices)/(calib['width']*calib['height'])),
             scan_minus_image_ms=delta_ms, published_composed_pose_max_absolute_error=pose_error,
             maximum_fraction_of_published_rounding_bound=float(np.max(np.abs(composed-published)/allowed)),
             image_timestamp_ns=int(images[frame]), scan_timestamp_ns=int(scans[frame])))
    fig.suptitle('KITTI-360 real sensor interface check | original image resolution | missing depth stays unknown', fontsize=14)
    fig.savefig(out/'measured_projection.png', dpi=150); plt.close(fig)
    dump(out/'verification.json', dict(complete=True, scene=scene, frames=frame_records,
        sample_manifest_sha256=sha(base/'sample_manifest.json'), metadata_manifest_sha256=sha(base/'downloads.json'),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__), ROOT/'omni_semantic_map/omni_semantic_map/kitti360_input.py']},
        files_sha256={p.name:sha(p) for p in out.glob('*') if p.suffix in ['.npz', '.png']},
        scope='Interface fixture only, no model inference or semantic accuracy score. Optimized provided poses, not LIVO-estimated poses.',
        timing_limitation='Per-point scan deskew and image/scan epoch compensation are not implemented. XYZ uses the paired image pose; signed timestamp offsets are reported. Must resolve the timing policy before sequence evaluation.'))
    print(json.dumps(frame_records, indent=2))


if __name__ == '__main__': main()
