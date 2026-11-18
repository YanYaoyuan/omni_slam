"""Visibility-linked appearance evidence for existing measured-point owners.

No reference labels, synthetic geometry, or new identities enter this module.
Evidence is conditional on the caller's immutable predicted object cores.
"""
import numpy as np


VISUAL_FEATURES = (
    'log_own_views', 'own_mean', 'own_min', 'own_max', 'own_std',
    'other_mean', 'margin_mean', 'margin_min', 'margin_max',
    'positive_margin_fraction', 'log_competing_views', 'own_view_fraction',
)


def patch_indices(pixel_xy, width, height, grid_width, grid_height):
    """Pixel centers in an uncropped resize, mapped to patch cells."""
    xy = np.asarray(pixel_xy)
    if xy.ndim != 2 or xy.shape[1] != 2 or min(width, height, grid_width, grid_height) <= 0:
        raise ValueError('Invalid image/grid coordinates')
    if not np.isfinite(xy).all() or np.any(xy < 0) or np.any(xy[:, 0] >= width) or np.any(xy[:, 1] >= height):
        raise ValueError('Pixels must be inside the original image')
    x = np.floor((xy[:, 0] + .5) * grid_width / width).astype(np.int64)
    y = np.floor((xy[:, 1] + .5) * grid_height / height).astype(np.int64)
    return np.minimum(y, grid_height - 1) * grid_width + np.minimum(x, grid_width - 1)


def similarity_matrix(tokens):
    x = np.asarray(tokens, np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Finite patch descriptor matrix required')
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return np.clip(x @ x.T, -1., 1.)


def view_evidence(similarity, query_patch, own_patch, other_patch):
    """Top-three distinct core-patch cosine mean, excluding self/shared patches.

    A patch containing cores from both sides is ambiguous and contributes to
    neither anchor set. A query cannot match its own image patch. Repeated
    measured points in one patch cannot create repeated visual evidence.
    """
    sim = np.asarray(similarity)
    q = np.asarray(query_patch, np.int64)
    own = np.unique(np.asarray(own_patch, np.int64))
    other = np.unique(np.asarray(other_patch, np.int64))
    shared = np.intersect1d(own, other)
    own = np.setdiff1d(own, shared)
    other = np.setdiff1d(other, shared)

    def score(anchors):
        if not len(anchors):
            return np.zeros(len(q), np.float32), np.zeros(len(q), bool)
        a = sim[np.ix_(q, anchors)].copy()
        a[q[:, None] == anchors[None, :]] = -np.inf
        k = min(3, len(anchors))
        top = np.partition(a, a.shape[1] - k, axis=1)[:, -k:]
        valid = np.isfinite(top)
        n = valid.sum(1)
        return np.where(valid, top, 0.).sum(1) / np.maximum(n, 1), n > 0

    a, valid = score(own)
    b, competition = score(other)
    return a, b, valid, competition


def summarize_evidence(pair, frame, own, other, competing, visible_views, *, first_only=False):
    """Frame-balanced evidence, one row per source/owner/frame; no labels."""
    pair = np.asarray(pair, np.int64)
    frame = np.asarray(frame, np.int64)
    own = np.asarray(own, np.float64)
    other = np.asarray(other, np.float64)
    competing = np.asarray(competing, bool)
    visible_views = np.asarray(visible_views, np.int64)
    n = len(visible_views)
    if not (pair.shape == frame.shape == own.shape == other.shape == competing.shape) or pair.ndim != 1:
        raise ValueError('Evidence rows must align')
    if np.any(pair < 0) or np.any(pair >= n) or not np.isfinite(own).all() or not np.isfinite(other).all():
        raise ValueError('Invalid evidence')
    if len(np.unique(np.column_stack((pair, frame)), axis=0)) != len(pair):
        raise ValueError('A frame cannot vote twice for one source/owner')
    if first_only and len(pair):
        order = np.lexsort((frame, pair))
        keep = order[np.r_[True, np.diff(pair[order]) != 0]]
        pair, own, other, competing = (a[keep] for a in (pair, own, other, competing))

    counts = np.bincount(pair, minlength=n)
    if np.any(counts > visible_views):
        raise ValueError('Evidence cannot exceed measured visibility')
    total = np.bincount(pair, weights=own, minlength=n)
    sq = np.bincount(pair, weights=own * own, minlength=n)
    lo, hi = np.full(n, np.inf), np.full(n, -np.inf)
    np.minimum.at(lo, pair, own); np.maximum.at(hi, pair, own)
    active = counts > 0
    lo[~active] = hi[~active] = 0.
    mean = total / np.maximum(counts, 1)
    std = np.sqrt(np.maximum(0., sq / np.maximum(counts, 1) - mean * mean))
    cp = pair[competing]; margin = (own - other)[competing]
    cc = np.bincount(cp, minlength=n)
    cm = np.bincount(cp, weights=margin, minlength=n) / np.maximum(cc, 1)
    co = np.bincount(cp, weights=other[competing], minlength=n) / np.maximum(cc, 1)
    cl, ch = np.full(n, np.inf), np.full(n, -np.inf)
    np.minimum.at(cl, cp, margin); np.maximum.at(ch, cp, margin)
    cl[cc == 0] = ch[cc == 0] = 0.
    positive = np.bincount(cp, weights=margin > 0., minlength=n) / np.maximum(cc, 1)
    return np.column_stack((np.log1p(counts), mean, lo, hi, std, co, cm, cl, ch,
                            positive, np.log1p(cc), counts / np.maximum(visible_views, 1))).astype(np.float32)
