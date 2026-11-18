"""Small official KITTI-360 interface fixture; never downloads a full sequence.

This is development data, not a reserved independent test. No semantic labels
are fetched. Source URLs come from the official download page and its scripts.
"""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile
import numpy as np
from prepare_hypersim_subset import RangeFile

OUT = Path('build/datasets/kitti360_prepare')
BASE = 'https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/'
GIT = 'https://raw.githubusercontent.com/autonomousvision/kitti360Scripts/master/'
URLS = {
    'calibration.zip': BASE+'384509ed5413ccc81328cf8c55cc6af078b8c444/calibration.zip',
    'data_poses.zip': BASE+'89a6bae3c8a6f789e12de4807fc1e8fdcf182cf4/data_poses.zip',
    'download_2d_perspective.zip': BASE+'a1d81d9f7fc7195c937f9ad12e2a2c66441ecb4e/download_2d_perspective.zip',
    'download_3d_velodyne.zip': BASE+'a1d81d9f7fc7195c937f9ad12e2a2c66441ecb4e/download_3d_velodyne.zip',
    'labels.py': GIT+'kitti360scripts/helpers/labels.py',
    'project.py': GIT+'kitti360scripts/helpers/project.py',
    'loadCalibration.py': GIT+'kitti360scripts/devkits/commons/loadCalibration.py',
    'evalInstanceLevelSemanticLabeling.py': GIT+'kitti360scripts/evaluation/semantic_3d/evalInstanceLevelSemanticLabeling.py',
    'LICENSE': GIT+'LICENSE',
    'data_timestamps_perspective.zip': BASE+'data_2d_raw/data_timestamps_perspective.zip',
    'data_timestamps_velodyne.zip': BASE+'data_3d_raw/data_timestamps_velodyne.zip',
}


def digest(data): return hashlib.sha256(data).hexdigest()
def dump(path, data):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n'); temporary.replace(path)


def extract(z, info, root):
    dest = root/info.filename
    if not dest.resolve().is_relative_to(root.resolve()) or info.file_size > 32*1024**2:
        raise ValueError('Unsafe or oversized ZIP member')
    if dest.exists():
        data = dest.read_bytes()
        if len(data) != info.file_size or zipfile.crc32(data) & 0xffffffff != info.CRC:
            raise ValueError('Existing file differs from ZIP record')
    else:
        data = z.read(info)  # zipfile verifies CRC, including remote ranges.
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(data)
    return dict(path=str(dest), bytes=len(data), sha256=digest(data))


def main():
    OUT.mkdir(parents=True, exist_ok=True); manifest = OUT/'downloads.json'
    records = json.loads(manifest.read_text()) if manifest.exists() else {}
    for name, url in URLS.items():
        dest = OUT/name
        if dest.exists(): data = dest.read_bytes()
        else:
            with urllib.request.urlopen(url, timeout=40) as r: data = r.read(16*1024**2+1)
            if len(data) > 16*1024**2: raise ValueError('Metadata size budget exceeded')
            dest.write_bytes(data)
        record = dict(url=url, bytes=len(data), sha256=digest(data))
        if name in records: assert record['sha256'] == records[name]['sha256']
        if name.endswith('.zip'):
            with zipfile.ZipFile(dest) as z:
                record['files'] = [extract(z, i, OUT/'extracted') for i in z.infolist() if not i.is_dir()]
        records[name] = record; dump(manifest, records)
    scene = '2013_05_28_drive_0000_sync'
    poses = np.loadtxt(OUT/'extracted'/scene/'cam0_to_world.txt')
    frames = poses[[0, 10, 20], 0].astype(int).tolist()
    selection = dict(scene=scene, frames=frames, purpose='Interface development only; first, eleventh and twenty-first released rectified-camera poses. Not held-out evaluation.', sources={})
    for kind, suffix, extension in [('data_2d_raw', 'image_00', '.png'), ('data_3d_raw', 'velodyne', '.bin')]:
        url = BASE+kind+'/'+scene+'_'+suffix+'.zip'
        stream = RangeFile(url, 128*1024**2)
        with zipfile.ZipFile(stream) as z:
            selected = [i for i in z.infolist() if Path(i.filename).name in {f'{f:010d}{extension}' for f in frames}]
            if len(selected) != len(frames): raise ValueError('Expected exactly one file per selected frame')
            files = [extract(z, i, OUT/'sample'/kind) for i in selected]
        selection['sources'][kind] = dict(url=url, archive_bytes=stream.size, downloaded_range_bytes=stream.used, files=files)
        dump(OUT/'sample_manifest.json', selection)
    print(json.dumps(selection, indent=2))


if __name__ == '__main__': main()
