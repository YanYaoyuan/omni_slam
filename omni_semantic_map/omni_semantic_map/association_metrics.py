"""Observation-pair scores with explicit unassigned/missed prediction labels."""
from collections import Counter


def pair_score_with_misses(gt, pred):
    """GT is the fixed eligible observation subset; predicted 0 means no object.

    Missing observations remain in the GT recall denominator. They create no
    predicted positive pairs, and MUST NOT form a single artificial object.
    """
    if len(gt) != len(pred):
        raise ValueError('label length mismatch')
    c2 = lambda n: n*(n-1)//2
    assigned = [(int(g), int(p)) for g,p in zip(gt,pred) if p > 0]
    tp = sum(c2(n) for n in Counter(assigned).values())
    pp = sum(c2(n) for n in Counter(p for _,p in assigned).values())
    gp = sum(c2(n) for n in Counter(map(int,gt)).values())
    return dict(precision=tp/pp if pp else None, recall=tp/gp if gp else None,
                f1=2*tp/(pp+gp) if pp+gp else None, observations=len(gt),
                gt_instances=len(set(gt)), unassigned_observations=len(gt)-len(assigned))
