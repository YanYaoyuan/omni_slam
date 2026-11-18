#!/usr/bin/env python3
"""Sequential LIO/LIVO pair; fixed NTU rig calibration and frozen trial policy."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import yaml
from prepare_ntu_benchmark import digest


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--executable', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--overrides', type=Path, default=root/'config/ntu_fullscan_overrides.yaml',
                    help='Explicit NTU trial policy; defaults to frozen development baseline')
    args = ap.parse_args(); args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    p = yaml.safe_load((root/'config/upstream_ros1/NTU_VIRAL.yaml').read_text())
    camera = yaml.safe_load((root/'config/upstream_ros1/camera_NTU_VIRAL.yaml').read_text())
    p['camera'] = dict(width=camera['cam_width'], height=camera['cam_height'], scale=1.,
        fx=camera['cam_fx'], fy=camera['cam_fy'], cx=camera['cam_cx'], cy=camera['cam_cy'], distortion=[0.]*5)
    p['common'].update(odom_frame='benchmark_odom', tracking_frame='benchmark_imu', camera_frame='benchmark_camera')
    # Isolate benchmark policy from robot deployment settings. Keep the original
    # numeric baseline; any ablation must provide and archive an explicit file.
    overrides = yaml.safe_load(args.overrides.read_text())
    (args.output/'trial_overrides.yaml').write_text(args.overrides.read_text())
    for group, values in overrides.items():
        p.setdefault(group, {}).update(values)
    p['publish']['tf_en'] = False; p['publish']['dense_map_en'] = False
    p['vio'].update(grid_size=20, grid_n_height=17)
    # ROS1 accepted integer spellings for floating parameters; ROS2 is strict.
    # This changes YAML types only, not calibration or tuning values.
    p['vio']['outlier_threshold'] = float(p['vio']['outlier_threshold'])
    p['local_map']['sliding_thresh'] = float(p['local_map']['sliding_thresh'])
    for name in ('extrinsic_T','extrinsic_R','Pcl','Rcl'):
        p['extrin_calib'][name] = [float(v) for v in p['extrin_calib'][name]]
    p['use_sim_time'] = False
    audit = json.loads((args.input/'input_audit.json').read_text())
    source_files = [root/'src/LIVMapper.cpp',root/'src/vio.cpp',root/'src/IMU_Processing.cpp',
        root/'src/preprocess.cpp',root/'src/offline_main.cpp',Path(__file__),
        root/'scripts/prepare_ntu_benchmark.py',args.overrides.resolve(),
        root/'config/upstream_ros1/NTU_VIRAL.yaml',root/'config/upstream_ros1/camera_NTU_VIRAL.yaml']
    manifest = dict(executable=str(args.executable.resolve()), executable_sha256=digest(args.executable),
        source_sha256={str(p.relative_to(root) if p.is_relative_to(root) else p):digest(p) for p in source_files},
        input_audit=audit, modes={}, policy='NTU rig defaults + explicitly archived trial_overrides.yaml; development experiment, not upstream paper reproduction',
        host=subprocess.check_output(['uname','-a'], text=True).strip(),
        cpu=subprocess.check_output(['lscpu'], text=True), ground_truth_used=False)
    for mode in ('lio','livo'):
        out = args.output/mode; out.mkdir()
        params = copy.deepcopy(p)
        params['common']['img_en'] = int(mode=='livo')
        params['runtime']['output_directory'] = str(out)
        config = args.output/(mode+'.yaml')
        config.write_text(yaml.safe_dump({'/**': {'ros__parameters': params}}, sort_keys=False))
        began = time.monotonic()
        environment = os.environ.copy(); environment.update(OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='1')
        with (args.input/'sensors.bin').open('rb') as data, (out/'console.log').open('w') as log:
            result = subprocess.run([str(args.executable.resolve()), '--ros-args', '--params-file', str(config)],
                stdin=data, stdout=log, stderr=subprocess.STDOUT, env=environment)
        elapsed = time.monotonic()-began
        transport = json.loads((out/'offline_transport.json').read_text()) if (out/'offline_transport.json').exists() else {}
        record = dict(exit_code=result.returncode, wall_s=elapsed, transport=transport)
        if result.returncode == 0 and transport.get('completed'):
            raw = np.loadtxt(out/'trajectory.txt', ndmin=2)
            final = raw[np.r_[np.diff(raw[:,0]) != 0, True], :8]
            assert len(final)>2 and np.isfinite(final).all() and np.all(np.diff(final[:,0])>0)
            np.savetxt(out/'trajectory_final.tum', final, fmt='%.12f')
            timing = np.atleast_1d(np.genfromtxt(out/'timing.csv', delimiter=',', names=True))
            record.update(final_poses=len(final), duration_s=float(final[-1,0]-final[0,0]),
                z_range_m=float(np.ptp(final[:,3])), path_length_m=float(np.linalg.norm(np.diff(final[:,1:4],axis=0),axis=1).sum()),
                timing_columns=list(timing.dtype.names))
        manifest['modes'][mode] = record
        (args.output/'run.json').write_text(json.dumps(manifest,indent=2))
        print(mode, json.dumps(record), flush=True)
        if result.returncode != 0:
            raise RuntimeError(f'{mode} failed; retained logs')


if __name__ == '__main__':
    main()
