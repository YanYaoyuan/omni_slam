"""Read raw-query evidence on measured-visible cells of a 3D hypothesis.

No detector acceptance threshold and no GT enter retrieval. Unseen hypothesis
cells abstain. Returned query scores remain evidence, not an object admission.
"""
import numpy as np
from scipy.sparse import csr_matrix


def membership(sets, cell_count):
    offsets = [0]; columns = []
    for values in sets:
        ids = np.unique(np.asarray(values, np.int64))
        if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= cell_count): raise ValueError('Invalid cell IDs')
        columns.extend(ids); offsets.append(len(columns))
    return csr_matrix((np.ones(len(columns), np.float32), np.asarray(columns, np.int64), np.asarray(offsets, np.int64)), shape=(len(sets), cell_count))


def retrieve(hypotheses, hypothesis_classes, queries, query_classes, query_scores, visible_cells):
    """Best same-class measured IoU query per hypothesis, conditional on visibility.

    Tie order is overlap, native score, query row. Explicitly unobserved rows
    return -1 and zero agreement. Zero-overlap rows also have no correspondence;
    they can still be measured-visible, which is recorded separately.
    """
    h = hypotheses.tocsr(); q = queries.tocsr(); hc = np.asarray(hypothesis_classes); qc = np.asarray(query_classes); scores = np.asarray(query_scores, float)
    if h.shape[1] != q.shape[1] or hc.shape != (h.shape[0],) or qc.shape != (q.shape[0],) or scores.shape != qc.shape: raise ValueError('Aligned hypotheses/queries required')
    if not np.isfinite(scores).all() or np.any(scores < 0): raise ValueError('Finite nonnegative native scores required')
    visible = np.unique(np.asarray(visible_cells, np.int64))
    if np.any(visible < 0) or np.any(visible >= h.shape[1]): raise ValueError('Invalid visibility cells')
    # Candidate masks must only contain current measured-visible cells. Keeping
    # hidden cells here would silently turn lack of measurement into refutation.
    if not np.isin(q.indices, visible).all(): raise ValueError('Queries contain unmeasured cells')
    hv = np.asarray(h[:, visible].sum(1)).ravel(); qm = np.asarray(q.sum(1)).ravel()
    intersections = (h@q.T).toarray()
    union = hv[:, None]+qm[None, :]-intersections
    agreement = np.divide(intersections, union, out=np.zeros_like(intersections), where=union>0)
    allowed = (hc[:, None] == qc[None, :]) & (scores[None, :] > 0) & (intersections>0)
    agreement[~allowed] = -1.
    best = np.full(h.shape[0], -1, np.int64); value = np.zeros(h.shape[0], np.float32); confidence = np.zeros(h.shape[0], np.float32)
    if q.shape[0]:
        # Stable descending native score order makes argmax's first tie exact.
        order = np.argsort(-scores, kind='stable'); selected = order[np.argmax(agreement[:, order], axis=1)]
        positive = agreement[np.arange(len(selected)), selected] > 0
        best[positive] = selected[positive]; value[positive] = agreement[np.arange(len(selected))[positive], selected[positive]]; confidence[positive] = scores[selected[positive]]
    return dict(query=best, agreement=value, score=confidence, visible_cells=hv.astype(np.int64), hypothesis_cells=np.diff(h.indptr), measured=hv>0)
