"""Audit reserved raw sensor windows without reading semantic reference labels."""
from decimal import Decimal
from pathlib import Path
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper_final_common import ROOT, read, dump, sha
from omni_semantic_map.kitti360_input import load_left_calibration, load_poses, load_times, project_lidar


def main():
    base = ROOT/'build/datasets/kitti360_prepare'
    data = ROOT/'build/datasets/kitti360_windows_v1'
    state = read(data/'download_status.json')
    if not state['complete'] or state['failed']:
        raise RuntimeError('Download must finish before auditing the reserved inputs')
    assert sha(base/'window_selection.json') == state['selection_sha256']
    selection = read(base/'window_selection.json')
    for path, expected in selection['source_sha256'].items():
        assert sha(ROOT/path) == expected
    for name, record in read(base/'downloads.json').items():
        assert sha(base/name) == record['sha256']
        for item in record.get('files', []):
            assert sha(ROOT/item['path']) == item['sha256']
    verified_files = 0
    for job in state['completed']:
        manifest = read(data/(job+'_download.json'))
        assert manifest['complete'] and manifest['selection_sha256'] == state['selection_sha256']
        for item in manifest['files']:
            # Reference PLYs are hashed as opaque bytes, not parsed.
            assert sha(ROOT/item['path']) == item['sha256']
            verified_files += 1
    extracted = base/'extracted'
    calib = load_left_calibration(extracted/'calibration')
    out = ROOT/'output/kitti360_interface'
    out.mkdir(parents=True, exist_ok=True)
    records = []
    fig, axes = plt.subplots(len(selection['windows']), 3, figsize=(18, 9), layout='constrained')
    for row_index, window in enumerate(selection['windows']):
        scene = window['scene']
        poses = load_poses(extracted/scene/'poses.txt')
        camera = load_poses(extracted/scene/'cam0_to_world.txt')
        rounding = {}
        for line in (extracted/scene/'cam0_to_world.txt').read_text().splitlines():
            tokens = line.split()
            rounding[int(tokens[0])] = np.array([
                .5*10.**Decimal(t).as_tuple().exponent for t in tokens[1:]
            ]).reshape(4, 4)
        images = load_times(extracted/scene/'image_00/timestamps.txt')
        scans = load_times(extracted/scene/'velodyne_points/timestamps.txt')
        chosen = window['input_frames']
        displayed = {chosen[i]: j for j, i in enumerate([0, len(chosen)//2, len(chosen)-1])}
        frames = []
        for frame in chosen:
            image_path = data/'data_2d_raw'/scene/'image_00/data_rect'/f'{frame:010d}.png'
            scan_path = data/'data_3d_raw'/scene/'velodyne_points/data'/f'{frame:010d}.bin'
            image = cv2.imread(str(image_path))
            assert image is not None and image.shape == (calib['height'], calib['width'], 3)
            assert scan_path.stat().st_size % 16 == 0
            raw = np.fromfile(scan_path, dtype='<f4').reshape(-1, 4)
            assert np.isfinite(raw).all()
            projected = project_lidar(raw[:, :3], calib)
            composed = poses[frame]@calib['pose_from_cam']@np.linalg.inv(calib['rectification'])
            error = np.abs(composed-camera[frame])
            allowed = rounding[frame]+1e-8
            assert (error <= allowed).all(), ('Pose rounding bound failed', scene, frame)
            count = len(projected['selected_indices'])
            frames.append(dict(frame=frame, raw_points=len(raw), visible_pixel_returns=count,
                depth_coverage=count/(calib['height']*calib['width']),
                scan_minus_image_ms=float((scans[frame]-images[frame])/1e6),
                image_timestamp_ns=int(images[frame]), scan_timestamp_ns=int(scans[frame]),
                maximum_fraction_of_published_rounding_bound=float(np.max(error/allowed))))
            if frame in displayed:
                ax = axes[row_index, displayed[frame]]
                yy, xx = np.nonzero(projected['source_point_index'] >= 0)
                ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                ax.scatter(xx, yy, c=projected['depth'][yy, xx], cmap='turbo', vmin=0, vmax=60,
                           s=.3, alpha=.65, linewidths=0)
                ax.set_title(f'{scene[-9:-5]} | frame {frame} | measured returns {100*frames[-1]["depth_coverage"]:.2f}%', fontsize=10)
                ax.axis('off')
        records.append(dict(scene=scene, start=window['start'], end=window['end'], frames=frames,
                            displayed_frames=list(displayed)))
    fig.suptitle('KITTI-360 reserved sensor windows: fixed first / middle / last inputs\n'
                 'Measured LiDAR projection only; no semantic predictions or reference labels', fontsize=14)
    figure = out/'reserved_windows_projection.png'
    fig.savefig(figure, dpi=150)
    plt.close(fig)
    result = dict(complete=True, verified_download_files=verified_files,
        input_frames=sum(len(r['frames']) for r in records), windows=records,
        selection_sha256=state['selection_sha256'], download_status_sha256=sha(data/'download_status.json'),
        metadata_manifest_sha256=sha(base/'downloads.json'),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),
            ROOT/'omni_semantic_map/omni_semantic_map/kitti360_input.py']},
        figure_sha256=sha(figure), semantic_reference_decoded=False,
        scope='Raw input validation; no semantic predictions, scores, or sequence-level SLAM evaluation.',
        timing_policy='Paired published image pose only; no scan deskew or scan/image epoch compensation.',
        reference_coverage='Not audited: forward-camera observations do not cover the full accumulated 3D reference.')
    dump(out/'reserved_windows_verification.json', result)
    print(json.dumps({k: result[k] for k in ['complete', 'verified_download_files', 'input_frames', 'semantic_reference_decoded']}, indent=2))


if __name__ == '__main__':
    main()
