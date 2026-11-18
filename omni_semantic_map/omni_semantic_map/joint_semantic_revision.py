"""View-balanced object type and leave-cell-out semantic compatibility.

The uniform-reference compatibility ratio is a modelling score, not a calibrated
probability or independent-observation guarantee. It never consumes GT labels.
"""
import numpy as np


def evidence(cell_owner, observation_frame, probabilities):
    pairs = np.asarray(cell_owner, dtype=np.int64)
    frames = np.asarray(observation_frame, dtype=np.int64)
    q = np.asarray(probabilities, dtype=np.float64)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or frames.shape != (len(pairs),):
        raise ValueError('Cell/object pairs and observation frames required')
    if q.ndim != 2 or len(q) != len(pairs) or q.shape[1] < 2 or not np.isfinite(q).all() or (q < 0).any():
        raise ValueError('Finite nonnegative categorical probabilities required')
    if not len(pairs):
        raise ValueError('At least one observed cell required')
    if (pairs[:, 0] < 0).any() or (pairs[:, 1] <= 0).any() or (q.sum(1) <= 0).any():
        raise ValueError('Valid cells, positive object IDs and nonempty probabilities required')
    if len(np.unique(np.column_stack([pairs, frames]), axis=0)) != len(pairs):
        raise ValueError('Deduplicate each cell/object within each observation frame first')
    q = q / q.sum(1)[:, None]
    cells, ci = np.unique(pairs, axis=0, return_inverse=True)
    view_keys, vi = np.unique(np.column_stack([pairs[:, 1], frames]), axis=0, return_inverse=True)
    objects, oi = np.unique(view_keys[:, 0], return_inverse=True)
    counts = np.bincount(vi)
    view_sum = np.zeros((len(view_keys), q.shape[1])); np.add.at(view_sum, vi, q)
    view_mean = view_sum / counts[:, None]
    object_sum = np.zeros((len(objects), q.shape[1])); np.add.at(object_sum, oi, view_mean)
    object_frames = np.bincount(oi)
    object_probability = object_sum / object_frames[:, None]
    observed = np.bincount(ci)
    cell_sum = np.zeros((len(cells), q.shape[1])); np.add.at(cell_sum, ci, q)
    cell_probability = cell_sum / observed[:, None]
    # Remove every observation of the query cell, not just its acquisition frame.
    many = counts[vi] > 1
    delta = -view_mean[vi].copy()
    delta[many] += (view_sum[vi[many]] - q[many]) / (counts[vi[many]] - 1)[:, None]
    cell_delta = np.zeros_like(cell_sum); np.add.at(cell_delta, ci, delta)
    removed_views = np.bincount(ci, weights=~many, minlength=len(cells)).astype(np.int64)
    co = np.searchsorted(objects, cells[:, 1])
    other_views = object_frames[co] - removed_views
    other = (object_sum[co] + cell_delta) / np.maximum(1, other_views)[:, None]
    other[other_views == 0] = 0
    # Tiny roundoff from subtraction is not negative evidence.
    assert other.min() > -1e-10
    other = np.maximum(other, 0)
    valid = other_views > 0
    other[valid] /= other[valid].sum(1)[:, None]
    ratio = np.ones(len(cells))
    ratio[valid] = q.shape[1] * np.einsum('ij,ij->i', cell_probability[valid], other[valid])
    conflict = valid & (ratio < 1. - 1e-6)
    return dict(cell_owner=cells, cell_probability=cell_probability,
                cell_observation_frames=observed, other_observation_frames=other_views,
                leave_cell_out_probability=other, compatibility_ratio=ratio,
                semantic_conflict=conflict, object_ids=objects,
                object_probability=object_probability, object_observation_frames=object_frames)


def revised_class(probability, previous):
    p = np.asarray(probability, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
        raise ValueError('Nonempty class probability vector required')
    best = np.flatnonzero(p >= p.max() - 1e-12)
    return int(previous) if previous in best else int(best[0])
