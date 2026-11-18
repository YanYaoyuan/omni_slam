"""Measured-source input and dense readout for the joint-memory prototype.

The attention context is bounded; the provenance ledger and final readout are
not. A fixed slot is never silently recycled. This is a finite-window research
model, not an unlimited-life object tracker or a demonstrated improvement.
"""
import numpy as np
import torch
from .joint_object_memory import JointObjectMemory


class PersistentSourceMemory(JointObjectMemory):
    def decode_sources(self, features, memory):
        """Decode arbitrary registered sources without changing slot state.

        No query-to-query attention: chunking must leave every output unchanged.
        The very same readout is used for training and complete-map inference.
        """
        if features.ndim != 2 or memory.shape != self.new_slots.shape:
            raise ValueError('Invalid query or memory shape')
        h = self.project(features)
        h = self.token_norm(h + self.token_read(
            h[None], memory[None], memory[None], need_weights=False)[0][0])
        masks = self.mask_projection(h) @ memory.T / memory.shape[-1]**.5
        return torch.cat([self.background(h), masks], dim=-1)


def sample_sources(ids, probabilities, count, seed):
    """GT-free sampling balanced across predicted building, car and other.

    Unused quotas are redistributed. The selected sources remain unique;
    probabilities must be observations already available at the current time.
    """
    ids = np.asarray(ids, np.int64)
    if len(np.unique(ids)) != len(ids) or probabilities.shape != (len(ids), 19):
        raise ValueError('Invalid sampling inputs')
    if count < 1 or not np.isfinite(probabilities).all():
        raise ValueError('Invalid budget or probabilities')
    rng = np.random.default_rng(seed)
    cls = probabilities.argmax(1)
    groups = [np.flatnonzero(cls == 2), np.flatnonzero(cls == 13),
              np.flatnonzero(~np.isin(cls, [2, 13]))]
    selected = []
    for g in groups:
        selected.extend(rng.permutation(g)[:count//3].tolist())
    remainder = np.setdiff1d(np.arange(len(ids)), selected)
    selected.extend(rng.permutation(remainder)[:max(0, count-len(selected))].tolist())
    return ids[np.asarray(selected, int)]


def sample_patch_features(grid, pixel_xy, image_hw):
    """Bilinear pixel-center readout, border padding, align_corners=False."""
    grid = np.asarray(grid)
    xy = np.asarray(pixel_xy, float)
    h, w = image_hw
    if grid.ndim != 3 or xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError('Invalid grid/pixel dimensions')
    if not np.isfinite(xy).all() or np.any(xy < 0) or np.any(xy >= [w, h]):
        raise ValueError('Pixel outside original image')
    gh, gw, dim = grid.shape
    uv = (xy + .5) * np.array([gw/w, gh/h]) - .5
    lo = np.floor(uv).astype(int); frac = uv - lo
    result = np.zeros((len(xy), dim), np.float32)
    for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        weight = (frac[:, 0] if dx else 1-frac[:, 0]) * (frac[:, 1] if dy else 1-frac[:, 1])
        result += grid[np.clip(lo[:, 1]+dy, 0, gh-1),
                       np.clip(lo[:, 0]+dx, 0, gw-1)].astype(np.float32) * weight[:, None]
    return result


def retrieve_context(previous, current, old_probabilities, current_probabilities, count, seed):
    """Retrieve exact reobservations plus other history, before ledger update.

    Up to half each context budget is paired by physical source identity. This
    is a retrieval index, not an instance label. Remaining history includes
    sources currently out of view. Neither IDs nor labels from a final map enter.
    """
    previous = np.asarray(previous, np.int64); current = np.asarray(current, np.int64)
    repeated = np.intersect1d(previous, current)
    order = np.argsort(previous)
    repeat_prob = old_probabilities[order[np.searchsorted(previous[order], repeated)]]
    match = sample_sources(repeated, repeat_prob, max(1,count//2), seed)
    old_remaining = ~np.isin(previous, match)
    new_remaining = ~np.isin(current, match)
    old = sample_sources(previous[old_remaining], old_probabilities[old_remaining], count-len(match), seed)
    now = sample_sources(current[new_remaining], current_probabilities[new_remaining], count-len(match), seed)
    return np.r_[match, old], np.r_[match, now]


class SourceFeatureLedger:
    """Causal observation pointers and online class evidence for real sources.

    No object labels, final-map anchors or future observations are accepted.
    Available XYZ is immutable. Reobservation replaces the appearance pointer;
    semantic probabilities average one observation per frame and source.
    """
    def __init__(self, xyz):
        self.xyz = np.asarray(xyz, np.float32)
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3 or not np.isfinite(self.xyz).all():
            raise ValueError('Invalid measured XYZ')
        n = len(self.xyz)
        self.latest = np.full(n, -1, np.int32)
        self.pixel = np.zeros((n, 2), np.int32)
        self.probability = np.zeros((n, 19), np.float32)
        self.count = np.zeros(n, np.int32)
        self.step = -1

    def observe(self, step, source_ids, pixel_xy, probabilities):
        ids = np.asarray(source_ids, np.int64)
        xy = np.asarray(pixel_xy)
        p = np.asarray(probabilities, np.float32)
        if step != self.step+1 or ids.ndim != 1 or len(np.unique(ids)) != len(ids):
            raise ValueError('Chronological unique-source observations required')
        if np.any(ids < 0) or np.any(ids >= len(self.xyz)) or xy.shape != (len(ids), 2) or p.shape != (len(ids), 19):
            raise ValueError('Misaligned registered source observations')
        if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=2e-3):
            raise ValueError('Invalid semantic probabilities')
        if not np.isfinite(xy).all() or np.any(xy < 0) or not np.array_equal(xy, xy.astype(int)):
            raise ValueError('Integer measured pixels required')
        self.probability[ids] += (p-self.probability[ids])/(self.count[ids, None]+1)
        self.count[ids] += 1
        self.latest[ids] = step
        self.pixel[ids] = xy
        self.step = step

    def packet(self, ids, grids, image_hw):
        ids = np.asarray(ids, np.int64)
        if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= len(self.xyz)) or np.any(self.latest[ids] < 0):
            raise ValueError('Cannot retrieve an unseen source')
        appearance = np.zeros((len(ids), grids[0].shape[-1]), np.float32)
        frames = self.latest[ids].copy()
        for frame in np.unique(frames):
            take = frames == frame
            appearance[take] = sample_patch_features(grids[frame], self.pixel[ids[take]], image_hw)
        appearance /= np.maximum(np.linalg.norm(appearance, axis=1, keepdims=True), 1e-12)
        appearance *= appearance.shape[1]**.5
        features = np.concatenate([appearance, self.xyz[ids]/20., self.probability[ids]], axis=1)
        return dict(features=features, source_ids=ids.copy(), frame_ids=frames,
                    pixel_xy=self.pixel[ids].copy())


def measured_relations(packet, xyz, camera_poses, camera):
    """Soft relation: exact repeated source +1; conflicting measured ray -1.

    A ray conflict is NOT an instance inequality. Each target token is an
    actually measured, visible point. Project another token into that camera:
    same rounded pixel but >5cm axial separation supplies a depth-conflict cue.
    No filled depth, semantic label, distance-radius same-object assumption, or
    missing-depth negative is used. Relations are symmetric, unknown is zero.
    """
    ids = packet['source_ids']; frames = packet['frame_ids']; pixels = packet['pixel_xy']
    n = len(ids); relation = np.zeros((n, n), np.int8)
    points = np.asarray(xyz)[ids]
    for frame in np.unique(frames):
        columns = np.flatnonzero(frames == frame)
        pose = np.asarray(camera_poses[int(frame)])
        cam = (points-pose[:3, 3]) @ pose[:3, :3]
        z = cam[:, 2]; valid = z > .2
        uv = np.zeros((n, 2), np.int64)
        uv[valid] = np.floor(np.column_stack([
            camera['fx']*cam[valid, 0]/z[valid]+camera['cx'],
            camera['fy']*cam[valid, 1]/z[valid]+camera['cy']])+.5).astype(np.int64)
        same_ray = np.all(uv[:, None, :] == pixels[columns][None], axis=2)
        conflict = same_ray & valid[:, None] & (np.abs(z[:, None]-z[columns][None]) > .05)
        relation[:, columns] = np.where(conflict, -1, 0)
    relation = np.minimum(relation, relation.T)
    relation[ids[:, None] == ids[None]] = 1
    np.fill_diagonal(relation, 0)
    return relation
