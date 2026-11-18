"""Decode measured mask-membership patterns against visible object anchors.

This is a mechanism probe, not a calibrated semantic classifier. Queries are
correlated; averaged Bernoulli log scores are NOT posterior probabilities.
Missing visibility must be removed by the caller, never encoded as mask zero.
"""
import numpy as np


def unique_queries(membership):
    x = np.asarray(membership)
    if x.ndim != 2 or x.dtype != np.bool_:
        raise ValueError('Boolean query-by-measured-point membership required')
    if not len(x):
        return x.copy(), np.empty(0, np.int64)
    _, first = np.unique(np.packbits(x, axis=1), axis=0, return_index=True)
    first.sort()
    return x[first], first


def fit_code(membership, labels, object_ids, weights=None):
    """Beta(1/2,1/2) smoothed frequencies on calibration anchors only.

    Labels outside object_ids are ignored. Weights give each physical cell
    equal influence despite variable numbers of raw measurements in it.
    """
    x = np.asarray(membership)
    labels = np.asarray(labels)
    objects = np.asarray(object_ids)
    w = np.ones(len(labels)) if weights is None else np.asarray(weights, float)
    if x.ndim != 2 or x.dtype != np.bool_ or x.shape[1] != len(labels):
        raise ValueError('Aligned boolean memberships and labels required')
    if objects.ndim != 1 or len(np.unique(objects)) != len(objects) or np.any(objects <= 0):
        raise ValueError('Distinct positive object IDs required')
    if w.shape != labels.shape or np.any(w < 0) or not np.isfinite(w).all():
        raise ValueError('Finite nonnegative anchor weights required')
    p = np.empty((len(objects), len(x)), float)
    counts = []
    for j, oid in enumerate(objects):
        take = labels == oid
        count = float(w[take].sum())
        if count <= 0:
            raise ValueError('Every object requires measured calibration anchors')
        p[j] = (x[:, take] @ w[take] + .5) / (count + 1.)
        counts.append(count)
    return p, np.asarray(counts)


def decode_code(membership, probabilities):
    x = np.asarray(membership)
    p = np.asarray(probabilities, float)
    if x.ndim != 2 or x.dtype != np.bool_ or p.ndim != 2 or p.shape[1] != x.shape[0]:
        raise ValueError('Aligned query dimensions required')
    if not x.shape[0] or not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1)):
        raise ValueError('Nonempty queries and strictly interior probabilities required')
    # Includes both shared membership and measured exclusion. A common mask
    # does not distinguish equal anchor codes, regardless of its native score.
    log_not = np.log1p(-p)
    full = (x.T @ (np.log(p) - log_not).T + log_not.sum(1)) / len(x)
    dot = x.T @ p.T
    denom = np.sqrt(x.sum(0)[:, None] * np.square(p).sum(1)[None, :])
    cosine = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)
    return dict(full=full, cosine=cosine)


def unique_winner(scores, object_ids, tolerance=1e-10):
    s = np.asarray(scores, float)
    ids = np.asarray(object_ids)
    if s.ndim != 2 or s.shape[1] != len(ids) or not len(ids) or not np.isfinite(s).all():
        raise ValueError('Finite scores and nonempty aligned object IDs required')
    best = s.argmax(1)
    maximum = s[np.arange(len(s)), best]
    tie = (np.abs(s - maximum[:, None]) <= tolerance).sum(1) != 1
    result = ids[best].copy()
    result[tie] = 0
    return result
