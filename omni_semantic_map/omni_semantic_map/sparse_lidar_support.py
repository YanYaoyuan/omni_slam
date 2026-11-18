"""Measured-source voxel support and consistent sparse-depth pixel indexing.

Voxel representatives are actual first-seen measurements, not centroids or
backprojected rounded pixels. Coordinates must use a common metric local frame.
"""
import numpy as np


def measured_frame_evidence(xyz, labels, pose, camera, depth, masks, owners, config):
    """Existing one-vote-per-frame rule with the measured-depth pixel convention."""
    import cv2
    masks, owners = np.asarray(masks), np.asarray(owners)
    if masks.shape != (len(owners), *depth.shape): raise ValueError('Mask/owner/depth mismatch')
    x, y, fov, visible = project_measured_visibility(xyz, pose, camera, depth, config)
    positive = np.zeros(len(xyz), bool)
    negative = np.zeros(len(xyz), bool)
    known = masks.any(axis=0)
    kernel = np.ones((2*config.boundary_pixels+1,)*2, np.uint8)
    for owner in np.unique(owners):
        if owner <= 0: continue
        ids = np.flatnonzero((labels == owner) & fov)
        if not len(ids): continue
        union = masks[owners == owner].any(0).astype(np.uint8)
        interior = cv2.erode(union, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        expanded = cv2.dilate(union, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        positive[ids] = interior[y[ids], x[ids]] > 0
        negative[ids] = (expanded[y[ids], x[ids]] == 0) & known[y[ids], x[ids]]
    assert not np.any(positive & negative)
    return positive & visible, negative & visible, visible


def project_measured_visibility(xyz, camera_to_world, camera, depth, config):
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('Finite Nx3 points required')
    if depth.shape != (camera['H'], camera['W']):
        raise ValueError('Depth/calibration shape mismatch')
    inverse = np.linalg.inv(camera_to_world)
    points = xyz@inverse[:3, :3].T+inverse[:3, 3]
    z = points[:, 2]
    safe = np.maximum(z, 1e-9)
    u = points[:, 0]*camera['fx']/safe+camera['cx']
    v = points[:, 1]*camera['fy']/safe+camera['cy']
    fov = (z > .2) & (u >= -.5) & (u < camera['W']-.5) & (v >= -.5) & (v < camera['H']-.5)
    x = np.floor(np.clip(u+.5, 0, camera['W']-1)).astype(np.int32)
    y = np.floor(np.clip(v+.5, 0, camera['H']-1)).astype(np.int32)
    measured = depth[y, x]
    visible = fov & np.isfinite(measured) & (measured > 0)
    visible &= np.abs(z-measured) <= config.depth_absolute_m+config.depth_relative*measured
    return x, y, fov, visible


class MeasuredVoxelMap:
    def __init__(self, voxel_size_m=.05):
        if not np.isfinite(voxel_size_m) or voxel_size_m <= 0:
            raise ValueError('Positive finite voxel size required')
        self.voxel_size_m = float(voxel_size_m)
        self._keys = set()
        self._xyz = []
        self._colors = []
        self._frames = []
        self._raw = []
        self._pixels = []

    def append(self, xyz, colors, frame_id, raw_indices, pixel_xy):
        xyz = np.asarray(xyz, dtype=np.float64)
        colors, raw_indices, pixel_xy = map(np.asarray, [colors, raw_indices, pixel_xy])
        n = len(xyz)
        if xyz.shape != (n, 3) or colors.shape != (n, 3) or raw_indices.shape != (n,) or pixel_xy.shape != (n, 2):
            raise ValueError('Misaligned measured-source arrays')
        if not np.isfinite(xyz).all(): raise ValueError('Nonfinite measurement')
        keys = np.floor(xyz/self.voxel_size_m).astype(np.int64)
        selected = []
        for index, key in enumerate(keys):
            value = tuple(key)
            if value in self._keys: continue
            self._keys.add(value)
            selected.append(index)
        selected = np.asarray(selected, dtype=np.int64)
        self._xyz.append(xyz[selected].copy())
        self._colors.append(colors[selected].copy())
        self._frames.append(np.full(len(selected), frame_id, dtype=np.int64))
        self._raw.append(raw_indices[selected].copy())
        self._pixels.append(pixel_xy[selected].copy())
        return selected

    def arrays(self):
        if not self._xyz:
            return dict(xyz=np.empty((0, 3), dtype=np.float64), colors=np.empty((0, 3), dtype=np.uint8),
                source_frame=np.empty(0, dtype=np.int64), source_raw_index=np.empty(0, dtype=np.int64),
                source_pixel_xy=np.empty((0, 2), dtype=np.int32))
        return dict(xyz=np.concatenate(self._xyz), colors=np.concatenate(self._colors),
            source_frame=np.concatenate(self._frames), source_raw_index=np.concatenate(self._raw),
            source_pixel_xy=np.concatenate(self._pixels))
