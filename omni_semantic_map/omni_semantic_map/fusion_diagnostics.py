"""Development-only fusion diagnostics; not used by the frozen benchmark.

The guarded control changes only the native final merge's live-source check.
It retains the original overlap matrix, ordering, thresholds and merge kernel.
No ground truth is consumed by this module.
"""
from collections import Counter
from copy import deepcopy

import numpy as np


def membership_audit(objects):
    counts = Counter(int(i) for obj in objects for i in obj['observation_ids'])
    across = Counter(int(i) for obj in objects for i in set(obj['observation_ids']))
    return dict(objects=len(objects), unique_observations=len(counts),
                repeated_memberships=sum(n - 1 for n in counts.values()),
                cross_object_observations=sum(n > 1 for n in across.values()))


def guarded_merge(cfg, objects, overlap_matrix=None):
    """Consume each source once; reject already ambiguous input partitions.

    Like the native routine, mutates objects that it consumes. Callers must
    deepcopy if they need to retain the input. This is a correctness control,
    not a proposed learned association method.
    """
    import torch.nn.functional as F
    from conceptgraph.slam import utils
    from conceptgraph.slam.slam_classes import MapObjectList
    before = membership_audit(objects)
    if before['repeated_memberships']:
        raise ValueError('guarded merge requires disjoint input memberships')
    if cfg.merge_overlap_thresh <= 0 or not objects:
        return objects
    if overlap_matrix is None:
        overlap_matrix = utils.compute_overlap_matrix(cfg, objects)
    x, y = overlap_matrix.nonzero()
    ratios = overlap_matrix[x, y]
    kept = np.ones(len(objects), dtype=bool)
    for k in np.argsort(ratios)[::-1]:
        i, j, ratio = x[k], y[k], ratios[k]
        if ratio <= cfg.merge_overlap_thresh:
            break
        # Native checks only kept[j], allowing an already consumed i to recur.
        if i == j or not kept[i] or not kept[j]:
            continue
        visual = F.cosine_similarity(utils.to_tensor(objects[i]['clip_ft']),
                                     utils.to_tensor(objects[j]['clip_ft']), dim=0)
        text = F.cosine_similarity(utils.to_tensor(objects[i]['text_ft']),
                                   utils.to_tensor(objects[j]['text_ft']), dim=0)
        if visual > cfg.merge_visual_sim_thresh and text > cfg.merge_text_sim_thresh:
            objects[j] = utils.merge_obj2_into_obj1(cfg, objects[j], objects[i], run_dbscan=True)
            kept[i] = False
    result = MapObjectList([obj for obj, keep in zip(objects, kept) if keep])
    after = membership_audit(result)
    if after['repeated_memberships'] or after['unique_observations'] != before['unique_observations']:
        raise AssertionError('merge violated observation conservation')
    return result


def stages(backend):
    """Branch final postprocessors from the identical, unmodified online map."""
    cfg, utils = backend.config, backend.utils
    online = deepcopy(backend._objects)
    denoised = utils.denoise_objects(cfg, deepcopy(online))
    filtered = utils.filter_objects(cfg, deepcopy(denoised))
    return dict(online=online, denoised=denoised, filtered=filtered,
                native_final=utils.merge_objects(cfg, deepcopy(filtered)),
                guarded_final=guarded_merge(cfg, deepcopy(filtered)),
                guarded_without_filter=guarded_merge(cfg, deepcopy(denoised)))


def assignment(objects, n):
    """Retain native last-wins semantics only for the audited native control."""
    result = np.zeros(n, np.int32)
    for obj in objects:
        result[obj['observation_ids']] = min(obj['observation_ids']) + 1
    return result
