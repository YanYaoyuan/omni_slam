"""Sparse measured depth at an atomic camera epoch; never fill missing pixels."""
import numpy as np


def camera_from_record(record):
    k = np.asarray(record['intrinsic'], float)
    if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError('invalid measured camera calibration')
    return dict(W=int(record['width']), H=int(record['height']), fx=float(k[0, 0]),
                fy=float(k[1, 1]), cx=float(k[0, 2]), cy=float(k[1, 2]))


def measured_depth(points, camera_from_tracking, camera):
    points = np.asarray(points)
    transform = np.asarray(camera_from_tracking)
    if points.ndim != 2 or points.shape[1] != 3 or transform.shape != (4, 4):
        raise ValueError('invalid measured cloud or transform')
    xyz = points @ transform[:3, :3].T + transform[:3, 3]
    xyz = xyz[np.isfinite(xyz).all(1) & (xyz[:, 2] > .2)]
    u = xyz[:, 0] * camera['fx'] / xyz[:, 2] + camera['cx']
    v = xyz[:, 1] * camera['fy'] / xyz[:, 2] + camera['cy']
    valid = (u >= 0) & (u < camera['W']) & (v >= 0) & (v < camera['H'])
    pixels = np.floor(v[valid]).astype(int)*camera['W'] + np.floor(u[valid]).astype(int)
    depth = np.full(camera['H']*camera['W'], np.inf, np.float32)
    np.minimum.at(depth, pixels, xyz[valid, 2])
    depth[~np.isfinite(depth)] = 0
    return depth.reshape(camera['H'], camera['W'])
