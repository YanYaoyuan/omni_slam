"""Equal per-object source-count controls; no semantic reference access."""
import numpy as np


def confidence(distance, mode):
    distance = np.asarray(distance, dtype=np.float64)
    if distance.ndim != 2 or distance.shape[1] != 2:
        raise ValueError('Expected two owner distances per candidate')
    a, b = distance.T
    if not np.isfinite(a).all() or (a < 0).any() or np.isnan(b).any() or (b < a).any():
        raise ValueError('Candidates require finite, ordered, nonnegative distances')
    if mode == 'nearest_distance':
        return -a
    if mode != 'normalized_margin':
        raise ValueError(mode)
    # An unreachable competitor is maximal confidence; exact ties are zero.
    value = np.ones(len(a), dtype=np.float64)
    finite = np.isfinite(b)
    den = a[finite] + b[finite]
    value[finite] = np.divide(b[finite] - a[finite], den,
                              out=np.zeros_like(den), where=den > 0)
    return value


def matched_selection(source_ids, owners, scores, target_ids, target_owners):
    """Return source IDs, respecting each target object's exact count.

    Higher confidence first; source ID breaks all ties. The target supplies
    counts only. Candidate sources may differ from the target's cell set.
    """
    ids = np.asarray(source_ids, dtype=np.int64)
    owners = np.asarray(owners, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    target = np.asarray(target_ids, dtype=np.int64)
    target_owners = np.asarray(target_owners, dtype=np.int64)
    if ids.ndim != 1 or owners.shape != ids.shape or scores.shape != ids.shape:
        raise ValueError('Mismatched candidates')
    if target.ndim != 1 or target.shape != target_owners.shape:
        raise ValueError('Mismatched target')
    if len(np.unique(ids)) != len(ids) or len(np.unique(target)) != len(target):
        raise ValueError('Duplicate source IDs')
    if (owners <= 0).any() or not np.isfinite(scores).all():
        raise ValueError('Expected positive owners and finite confidence')
    index = np.argsort(ids)
    positions = np.searchsorted(ids[index], target)
    if (positions >= len(ids)).any() or not np.array_equal(ids[index][positions], target):
        raise ValueError('Target not in candidate universe')
    if not np.array_equal(owners[index][positions], target_owners):
        raise ValueError('Target owner differs from candidate owner')
    selected = []
    for owner, count in zip(*np.unique(target_owners, return_counts=True)):
        loc = np.flatnonzero(owners == owner)
        if len(loc) < count:
            raise ValueError('Insufficient candidates')
        order = np.lexsort((ids[loc], -scores[loc]))
        selected.append(ids[loc[order[:count]]])
    return np.sort(np.concatenate(selected)) if selected else np.empty(0, np.int64)


def summarize_changes(cells, owners, failures, frames):
    count = failures[cells]
    per_object = []
    for oid in np.unique(owners):
        x = count[owners == oid]
        per_object.append(dict(object_id=int(oid), added_sources=len(x),
                               vulnerable_sources=int((x > 0).sum()),
                               vulnerable_fraction=float((x > 0).mean()),
                               mean_failure_fraction=float(x.mean() / frames)))
    return dict(added_sources=len(cells), vulnerable_sources=int((count > 0).sum()),
                vulnerable_fraction=float((count > 0).mean()) if len(count) else None,
                mean_failure_fraction=float(count.mean() / frames) if len(count) else None,
                macro_object_vulnerable_fraction=float(np.mean([r['vulnerable_fraction'] for r in per_object])) if per_object else None,
                per_object=per_object)
