"""Measured semantic-query evidence; native scores are not class posteriors."""
import numpy as np


def reduce_support(masks, scores):
    """Max native query score per measured point, with an explicit witness.

    A point with no positive-score mask remains unknown (query=-1, score=0).
    Repeated queries do not accumulate support. Exact ties use first query.
    """
    masks = np.asarray(masks); scores = np.asarray(scores)
    if masks.ndim != 2 or masks.dtype != np.bool_ or scores.shape != (len(masks),):
        raise ValueError('Boolean query/point masks and aligned scores required')
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Native probabilities required')
    if not len(masks):
        return np.zeros(masks.shape[1], np.float32), np.full(masks.shape[1], -1, np.int16)
    values = np.where(masks, scores[:, None], 0.)
    winner = values.argmax(0)
    best = values[winner, np.arange(masks.shape[1])]
    winner = winner.astype(np.int16)
    winner[best <= 0] = -1
    return best.astype(np.float32), winner


def equal_cell_mean(features, cells):
    """Each occupied physical cell has equal weight; no unseen cell is added."""
    x = np.asarray(features, float); cells = np.asarray(cells)
    if x.ndim != 2 or cells.shape != (len(x),) or not len(x) or not np.isfinite(x).all():
        raise ValueError('Nonempty aligned finite observed features required')
    _, inverse, counts = np.unique(cells, return_inverse=True, return_counts=True)
    return (x/counts[inverse, None]).sum(0)/len(counts)


def context_relative(features, object_mean):
    x = np.asarray(features, float); mu = np.asarray(object_mean, float)
    if x.ndim != 2 or mu.shape != (x.shape[1],) or not np.isfinite(x).all() or not np.isfinite(mu).all():
        raise ValueError('Aligned observed semantic evidence required')
    return np.concatenate([x, x-mu], axis=1).astype(np.float32)
