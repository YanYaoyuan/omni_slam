"""Leave-acquisition-frame-out object/background counterevidence.

Negative labels identify known background semantic groups; positive labels are
predicted object IDs. Distinct frames are voting units, not independent trials.
This produces a reversible rejection proposal, not semantic correctness proof.
"""
import numpy as np


def unit_vectors(x):
    x = np.asarray(x, np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Finite feature matrix required')
    norm = np.linalg.norm(x, axis=1)
    if (norm <= 0).any():
        raise ValueError('Nonzero features required')
    return x / norm[:, None]


def counterevidence(query_features, query_owner, query_frame,
                    prototype_features, prototype_label, prototype_frame,
                    chunk_size=1024):
    """Strict majority of >=2 other comparable frames can revoke an anchor.

    Each comparable frame has the query's object prototype and at least one
    known background prototype. Its vote compares own-object cosine against
    the maximum known-background cosine in that SAME frame. Numerical ties
    abstain but remain in the denominator, so they cannot create a majority.
    No competing foreground ID is used to claim same-class disambiguation.
    """
    q, p = unit_vectors(query_features), unit_vectors(prototype_features)
    owner, frame = np.asarray(query_owner), np.asarray(query_frame)
    label, pf = np.asarray(prototype_label), np.asarray(prototype_frame)
    if q.shape[1] != p.shape[1] or owner.shape != (len(q),) or frame.shape != (len(q),):
        raise ValueError('Query shape mismatch')
    if label.shape != (len(p),) or pf.shape != (len(p),) or (owner <= 0).any() or (label == 0).any():
        raise ValueError('Positive object IDs and negative background labels required')
    if len(np.unique(np.column_stack([pf, label]), axis=0)) != len(p):
        raise ValueError('One prototype per frame and label required')
    if chunk_size < 1:
        raise ValueError('Positive chunk size required')
    eligible = np.zeros(len(q), np.int32)
    refute = np.zeros(len(q), np.int32)
    support = np.zeros(len(q), np.int32)
    mean_margin = np.zeros(len(q), np.float64)
    bg = np.flatnonzero(label < 0)
    bg = bg[np.argsort(pf[bg], kind='stable')]
    frames, starts = np.unique(pf[bg], return_index=True)
    if not len(bg):
        return dict(eligible=eligible, refute=refute, support=support,
                    mean_margin=mean_margin, retract=np.zeros(len(q), bool))
    for oid in np.unique(owner):
        own = np.flatnonzero(label == oid)
        comparable, bi, oi = np.intersect1d(frames, pf[own], return_indices=True)
        if not len(comparable): continue
        rows = np.flatnonzero(owner == oid)
        for start in range(0, len(rows), chunk_size):
            ix = rows[start:start + chunk_size]
            background = np.maximum.reduceat(q[ix] @ p[bg].T, starts, axis=1)[:, bi]
            positive = q[ix] @ p[own[oi]].T
            delta = background - positive
            usable = frame[ix, None] != comparable[None, :]
            eligible[ix] = usable.sum(1)
            refute[ix] = ((delta > 1e-6) & usable).sum(1)
            support[ix] = ((delta < -1e-6) & usable).sum(1)
            mean_margin[ix] = np.sum(np.where(usable, delta, 0), axis=1) / np.maximum(1, eligible[ix])
    retract = (eligible >= 2) & (2 * refute > eligible)
    return dict(eligible=eligible, refute=refute, support=support,
                mean_margin=mean_margin, retract=retract)
