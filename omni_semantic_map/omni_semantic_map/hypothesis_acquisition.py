"""Select real measurements that distinguish competing object partitions.

No semantic labels, reference instances or learned scores enter this module.
Partition comparisons are invariant to arbitrary instance-ID permutations.
This is a disagreement heuristic, not a calibrated information-gain bound.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AcquisitionConfig:
    seed: int = 17
    views_per_event: int = 2
    min_visible_per_fragment: int = 6
    camera_separation_m: float = .5
    samples_per_fragment: int = 128
    max_events: int = 4
    min_fragment_points: int = 16
    max_fragment_gap_m: float = .5
    min_descriptor_cosine: float = .65
    depth_absolute_m: float = .05
    depth_relative: float = .01


def partition_disagreement(partitions, weights=None):
    """Return pairwise variance of same-instance predictions across hypotheses.

    -1 means unknown; 0 means background (not a single shared object).
    A pair is usable only when all compared hypotheses assign both points.
    """
    p = np.asarray(partitions)
    if p.ndim != 2 or p.shape[0] < 2 or not np.issubdtype(p.dtype, np.integer):
        raise ValueError('At least two integer partitions required')
    w = np.ones(len(p)) if weights is None else np.asarray(weights, float)
    if w.shape != (len(p),) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError('Invalid hypothesis weights')
    w = w/w.sum()
    relation = (p[:, :, None] == p[:, None, :]) & (p[:, :, None] > 0)
    valid = (p >= 0).all(0)
    mean = np.einsum('k,kij->ij', w, relation)
    variance = mean*(1-mean)
    variance[~(valid[:, None] & valid[None, :])] = 0
    np.fill_diagonal(variance, 0)
    return variance


def choose_views(partitions, visibility, camera_centers, roles, fragments,
                 policy='disagreement', config=AcquisitionConfig()):
    """Greedy fixed-cost selection; validation frames cannot be selected.

    All policies share visibility eligibility and camera diversity. Disagreement
    discounts source-point pairs already inspected by a selected view; the
    point-uncertainty control discounts individually inspected points.
    """
    v = np.asarray(visibility, bool)
    centers = np.asarray(camera_centers, float)
    f = np.asarray(fragments)
    if v.ndim != 2 or centers.shape != (len(v), 3) or f.shape != (v.shape[1],):
        raise ValueError('Misaligned view inputs')
    if len(roles) != len(v) or len(np.unique(f)) != 2:
        raise ValueError('Two fragments and explicit frame roles required')
    if np.asarray(partitions).shape[1] != v.shape[1] or not np.isfinite(centers).all():
        raise ValueError('Invalid partitions or camera centers')
    policies = {'disagreement', 'area', 'uncertainty', 'random', 'uniform'}
    if policy not in policies: raise ValueError(policy)
    eligible = np.array([r == 'acquire' for r in roles])
    for k in np.unique(f):
        eligible &= v[:, f == k].sum(1) >= config.min_visible_per_fragment
    d = partition_disagreement(partitions)
    covered_pairs = np.zeros_like(d, bool)
    covered_points = np.zeros(v.shape[1], bool)
    rng = np.random.default_rng(config.seed)
    random_order = rng.random(len(v))
    available = np.flatnonzero(eligible)
    uniform_rank = np.zeros(len(v))
    if len(available):
        targets = np.linspace(0, len(available)-1, config.views_per_event).round().astype(int)
        for rank, j in enumerate(dict.fromkeys(targets.tolist())):
            uniform_rank[available[j]] = len(available)+1-rank
    selected, history = [], []
    for _ in range(config.views_per_event):
        scores = np.full(len(v), -np.inf)
        for i in np.flatnonzero(eligible):
            if i in selected or any(np.linalg.norm(centers[i]-centers[j]) < config.camera_separation_m for j in selected):
                continue
            if policy == 'disagreement':
                joint = v[i, :, None] & v[i, None, :] & ~covered_pairs
                scores[i] = (d*joint).sum()/max(d.sum(), 1e-12)
            elif policy == 'uncertainty':
                scores[i] = (d.sum(1)*v[i]*~covered_points).sum()/max(d.sum(), 1e-12)
            elif policy == 'area': scores[i] = v[i].sum()
            elif policy == 'random': scores[i] = random_order[i]
            else: scores[i] = uniform_rank[i]
        if not np.isfinite(scores).any(): break
        best = int(np.argmax(scores))
        selected.append(best)
        covered_pairs |= v[best, :, None] & v[best, None, :]
        covered_points |= v[best]
        history.append(dict(frame_index=best, score=float(scores[best]),
                            visible_points=int(v[best].sum())))
    return dict(selected=selected, decisions=history, eligible=np.flatnonzero(eligible).tolist(),
                pair_coverage=float((d*covered_pairs).sum()/max(d.sum(), 1e-12)))


def symmetric_prompts(pixels, visible, fragments):
    """Matched four-point prompts for joint, A-only and B-only explanations."""
    uv, valid, f = np.asarray(pixels), np.asarray(visible, bool), np.asarray(fragments)
    chosen = []
    for k in (0, 1):
        ids = np.flatnonzero(valid & (f == k))
        if len(ids) < 2: raise ValueError('Insufficient visible anchors')
        # Avoid outlying extremal prompts: start nearest median, then median-radius.
        center = np.median(uv[ids], axis=0)
        first = ids[np.argmin(np.linalg.norm(uv[ids]-center, axis=1))]
        remaining = ids[np.any(uv[ids] != uv[first], axis=1)]
        if not len(remaining): raise ValueError('Coincident anchors')
        distance = np.linalg.norm(uv[remaining]-uv[first], axis=1)
        second = remaining[np.argsort(distance, kind='stable')[len(remaining)//2]]
        chosen.extend([int(first), int(second)])
    coords = uv[chosen].astype(np.float32)
    if len(np.unique(coords, axis=0)) != 4: raise ValueError('Conflicting pixel anchors')
    labels = np.array([[1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1]], np.int32)
    return np.repeat(coords[None], 3, axis=0), labels, np.array(chosen)


def source_partition_from_logits(logits, pixels):
    """Create competing source-point partitions; no invented 3D coordinates.

    Split overlap is resolved by the larger SAM logit; ties deterministically
    favor A. A point below both mask thresholds is unknown, not deleted.
    """
    a = np.asarray(logits)
    uv = np.asarray(pixels, int)
    if a.ndim != 3 or a.shape[0] != 3: raise ValueError('Three mask logit images required')
    if uv.ndim != 2 or uv.shape[1] != 2: raise ValueError('Nx2 pixels required')
    if len(uv) and ((uv < 0).any() or (uv[:, 0] >= a.shape[2]).any() or (uv[:, 1] >= a.shape[1]).any()):
        raise ValueError('Pixel outside mask')
    values = a[:, uv[:, 1], uv[:, 0]]
    joined = np.where(values[0] > 0, 1, -1)
    split = np.where(values[1:].max(0) > 0, values[1:].argmax(0)+1, -1)
    return np.stack([joined, split]).astype(np.int8)
