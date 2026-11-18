"""Explicit left-rectified-camera KITTI-360 input geometry.

Missing laser returns stay unknown. Source indices preserve exact LiDAR points;
depth images are for visibility and must not replace those source coordinates.
This module does not deskew a spinning scan or interpolate missing poses.
"""
from pathlib import Path
import numpy as np


def matrix34(values):
    result = np.eye(4); result[:3] = np.asarray(values, float).reshape(3, 4)
    if not np.isfinite(result).all(): raise ValueError('Non-finite transform')
    return result


def keyed(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        if ':' not in line: continue
        key, value = line.split(':', 1)
        try: result[key] = np.asarray([float(x) for x in value.split()])
        except ValueError: continue
    return result


def load_left_calibration(folder):
    folder = Path(folder); p = keyed(folder/'perspective.txt')
    projection = p['P_rect_00'].reshape(3, 4)
    if not np.allclose(projection[:, 3], 0): raise ValueError('Expected left-camera projection')
    rectification = np.eye(4); rectification[:3, :3] = p['R_rect_00'].reshape(3, 3)
    velo_from_cam = matrix34(np.loadtxt(folder/'calib_cam_to_velo.txt'))
    pose_from_cam = matrix34(keyed(folder/'calib_cam_to_pose.txt')['image_00'])
    width, height = p['S_rect_00'].astype(int)
    return dict(K=projection[:, :3], width=int(width), height=int(height),
                rectification=rectification, pose_from_cam=pose_from_cam,
                rectified_from_velodyne=rectification@np.linalg.inv(velo_from_cam))


def load_poses(path):
    rows = np.atleast_2d(np.loadtxt(path)); result = {}
    if rows.shape[1] not in [13, 17]: raise ValueError('Unexpected pose layout')
    for row in rows:
        frame = int(row[0])
        if row[0] != frame or frame in result: raise ValueError('Invalid or duplicate frame ID')
        pose = matrix34(row[1:]) if len(row) == 13 else row[1:].reshape(4, 4)
        if not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
            raise ValueError('Invalid pose')
        result[frame] = pose
    return result


def load_times(path):
    rows = Path(path).read_text().splitlines()
    values = np.asarray([np.datetime64(r.strip().replace(' ', 'T'), 'ns').astype(np.int64) for r in rows])
    if (np.diff(values) <= 0).any(): raise ValueError('Nonmonotonic timestamps')
    return values


def project_lidar(points, calibration):
    """Nearest-pixel-center z-buffer, deterministic ties, exact source indices."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3: raise ValueError('Expected Nx3 points')
    transform = calibration['rectified_from_velodyne']
    xyz = points@transform[:3, :3].T+transform[:3, 3]
    candidates = np.flatnonzero(np.isfinite(xyz).all(1) & (xyz[:, 2] > .2))
    projected = xyz[candidates]@calibration['K'].T
    uv = projected[:, :2]/projected[:, 2:3]
    width, height = calibration['width'], calibration['height']
    valid = (uv[:, 0] >= -.5) & (uv[:, 0] < width-.5) & (uv[:, 1] >= -.5) & (uv[:, 1] < height-.5)
    candidates = candidates[valid]; uv = uv[valid]
    pixels = np.floor(uv+.5).astype(np.int64)
    flat = pixels[:, 1]*width+pixels[:, 0]
    order = np.lexsort((candidates, xyz[candidates, 2], flat))
    flat, candidates = flat[order], candidates[order]
    first = np.r_[True, flat[1:] != flat[:-1]] if len(flat) else np.empty(0, bool)
    flat, candidates = flat[first], candidates[first]
    depth = np.zeros(width*height, np.float32); indices = np.full(width*height, -1, np.int64)
    depth[flat] = xyz[candidates, 2]; indices[flat] = candidates
    return dict(depth=depth.reshape(height, width), source_point_index=indices.reshape(height, width),
                camera_xyz=xyz, selected_indices=candidates)
