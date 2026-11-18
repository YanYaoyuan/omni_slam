"""Post-hoc instance error signatures on a fixed evaluation domain.

Signatures describe final geometry, not proven causes in detection/association.
GT is never consumed by a mapping backend. Thresholds are diagnostic only.
"""
from collections import Counter
import numpy as np

THRESHOLDS = dict(foreground_share=.5, dominant_purity=.7,
                  mixed_foreground_share=.2, fragment_gt_share=.1,
                  low_gt_coverage=.1, minimum_support=5)


def audit_instances(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.ndim != 1 or prediction.shape != truth.shape:
        raise ValueError('incompatible labels')
    if any(not np.issubdtype(a.dtype, np.integer) or (a < 0).any()
           for a in (truth, prediction)):
        raise ValueError('IDs must be nonnegative integers')
    gids = np.unique(np.r_[0, truth]); pids = np.unique(np.r_[0, prediction])
    gi, pi = np.searchsorted(gids, truth), np.searchsorted(pids, prediction)
    table = np.bincount(gi*len(pids)+pi, minlength=len(gids)*len(pids)).reshape(len(gids), len(pids))
    ng, npred = table.sum(1), table.sum(0)
    union = ng[:, None]+npred[None, :]-table
    iou = np.divide(table, union, out=np.zeros_like(table, float), where=union>0)
    matched = (iou > .5); matched[0, :] = False; matched[:, 0] = False
    assert (matched.sum(0) <= 1).all() and (matched.sum(1) <= 1).all()
    pure_parts = {}
    for j in range(1, len(pids)):
        if len(gids) < 2:
            continue
        g = 1+int(table[1:, j].argmax()); intersection=table[g,j]
        if intersection >= THRESHOLDS['minimum_support'] and intersection/npred[j] >= .7 and intersection/ng[g] >= .1:
            pure_parts.setdefault(g, []).append(j)
    objects=[]
    for j in range(1,len(pids)):
        fg=int(table[1:,j].sum()); size=int(npred[j]); g=1+int(table[1:,j].argmax()) if len(gids)>1 else 0
        n=int(table[g,j]) if g else 0; purity=n/size; coverage=n/ng[g] if g else 0.
        significant=[k for k in range(1,len(gids)) if table[k,j]>=5 and table[k,j]/max(fg,1)>=.2]
        if matched[:,j].any():category='matched'
        elif fg/size < .5:category='background_dominated'
        elif len(significant)>=2:category='mixed_foreground'
        elif j in pure_parts.get(g,[]) and len(pure_parts[g])>=2:category='fragmentation_consistent'
        elif purity>=.7 and coverage<=.5:category='partial_coverage'
        elif coverage>.5:category='excess_support'
        else:category='ambiguous_overlap'
        objects.append(dict(predicted_id=int(pids[j]),category=category,voxels=size,
            background_voxels=int(table[0,j]),foreground_share=fg/size,
            dominant_gt=int(gids[g]) if n else None,dominant_purity=purity,
            dominant_gt_coverage=float(coverage),best_iou=float(iou[1:,j].max()) if len(gids)>1 else 0.,
            mixed_gt_ids=[int(gids[k]) for k in significant],
            matched_gt=int(gids[np.flatnonzero(matched[:,j])[0]]) if matched[:,j].any() else None))
    ground_truth=[]
    for g in range(1,len(gids)):
        coverage=float(table[g,1:].sum()/ng[g]);parts=pure_parts.get(g,[])
        if matched[g].any():category='matched'
        elif coverage<.1:category='very_low_prediction_coverage'
        elif len(parts)>=2 and table[g,parts].sum()/ng[g]>=.5:category='fragmentation_consistent'
        elif any(o['category']=='mixed_foreground' and int(gids[g]) in o['mixed_gt_ids'] for o in objects):category='mixed_prediction'
        else:category='insufficient_iou'
        ground_truth.append(dict(gt_id=int(gids[g]),category=category,voxels=int(ng[g]),
            prediction_coverage=coverage,best_iou=float(iou[g,1:].max()) if len(pids)>1 else 0.,
            pure_fragment_ids=[int(pids[j]) for j in parts]))
    return dict(thresholds=THRESHOLDS,predicted=objects,ground_truth=ground_truth,
                predicted_counts=dict(Counter(o['category'] for o in objects)),
                gt_counts=dict(Counter(o['category'] for o in ground_truth)))


def filter_transition(before, after):
    """Compare matched GT sets: count lost and gained, not just net TP change."""
    a={o['matched_gt'] for o in before['predicted'] if o['matched_gt'] is not None}
    b={o['matched_gt'] for o in after['predicted'] if o['matched_gt'] is not None}
    return dict(lost_matched_gt=sorted(a-b),gained_matched_gt=sorted(b-a),retained_matched_gt=sorted(a&b))
