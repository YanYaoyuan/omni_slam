"""Offline point rejection on fixed object identities; no labels or fitting.

Only known segmented pixels can refute an object's union of masks. Occluded,
unmeasured, out-of-view and mask-boundary points abstain. Counts are per frame,
not per mask, so overlapping masks do not multiply evidence.
"""
from dataclasses import dataclass
import cv2
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class CleanupConfig:
    depth_absolute_m: float = .05
    depth_relative: float = .01
    boundary_pixels: int = 2
    minimum_negative_frames: int = 3
    negative_positive_ratio: float = 2.
    sor_neighbors: int = 20
    sor_std_ratio: float = 2.


def project_visibility(xyz, camera_to_world, camera, depth, config):
    """Return integer pixels, FOV and surface-depth agreement, in input order."""
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('finite Nx3 points required')
    if depth.shape != (camera['H'], camera['W']):
        raise ValueError('depth/image calibration shape mismatch')
    inverse = np.linalg.inv(camera_to_world)
    points = xyz @ inverse[:3, :3].T + inverse[:3, 3]
    z = points[:, 2]
    safe_z = np.maximum(z, 1e-9)
    u = points[:, 0] * camera['fx'] / safe_z + camera['cx']
    v = points[:, 1] * camera['fy'] / safe_z + camera['cy']
    fov = (z > .2) & (u >= 0) & (u < camera['W']) & (v >= 0) & (v < camera['H'])
    # Clamp only to permit indexing; the FOV mask retains the original decision.
    x = np.floor(np.clip(u, 0, camera['W'] - 1)).astype(np.int32)
    y = np.floor(np.clip(v, 0, camera['H'] - 1)).astype(np.int32)
    measured = depth[y, x]
    visible = fov & np.isfinite(measured) & (measured > 0)
    visible &= np.abs(z - measured) <= config.depth_absolute_m + config.depth_relative * measured
    return x, y, fov, visible


def frame_evidence(xyz, labels, camera_to_world, camera, depth, masks,
                   mask_owners, config):
    """Two identical votes, with and without the depth-visibility gate."""
    if len(masks) != len(mask_owners):
        raise ValueError('mask owner count differs')
    if masks.shape[1:] != depth.shape:
        raise ValueError('mask/depth shape mismatch')
    x, y, fov, visible = project_visibility(xyz, camera_to_world, camera, depth, config)
    positive = np.zeros(len(xyz), bool)
    negative = np.zeros(len(xyz), bool)
    known = masks.any(axis=0)
    kernel = np.ones((2 * config.boundary_pixels + 1,) * 2, np.uint8)
    for owner in np.unique(mask_owners):
        if owner <= 0:
            continue
        indices = np.flatnonzero((labels == owner) & fov)
        if not len(indices):
            continue
        union = masks[np.asarray(mask_owners) == owner].any(axis=0).astype(np.uint8)
        interior = cv2.erode(union, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        expanded = cv2.dilate(union, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        px, py = x[indices], y[indices]
        positive[indices] = interior[py, px] > 0
        negative[indices] = (expanded[py, px] == 0) & known[py, px]
    assert not np.any(positive & negative)
    return positive & visible, negative & visible, positive, negative, visible


def rejection_mask(positive, negative, config, *, single_frame=False):
    positive, negative = np.asarray(positive), np.asarray(negative)
    if positive.shape != negative.shape or np.any(positive < 0) or np.any(negative < 0):
        raise ValueError('invalid vote counts')
    required = 1 if single_frame else config.minimum_negative_frames
    return (negative >= required) & (negative >= config.negative_positive_ratio * positive)


def statistical_outliers(xyz, labels, config):
    """Ordinary per-object mean-kNN-distance / mean+2sigma control."""
    reject = np.zeros(len(xyz), bool)
    for owner in np.unique(labels):
        ids = np.flatnonzero(labels == owner)
        if owner <= 0 or len(ids) <= config.sor_neighbors:
            continue
        distance, _ = cKDTree(xyz[ids]).query(xyz[ids], k=config.sor_neighbors + 1, workers=1)
        mean = distance[:, 1:].mean(axis=1)
        reject[ids] = mean > mean.mean() + config.sor_std_ratio * mean.std()
    return reject
