"""Observed identity compatibility before changing a persistent object's ID.

Scores compare shared and separate native-mask explanations, not calibrated
identity probabilities. A positive margin must survive deletion of any one
observed view. Complete-link groups prevent transitive incompatible merges.
"""
import numpy as np


def margin_after_view_deletion(frame_ids, shared, separate):
    frame_ids=np.asarray(frame_ids);shared=np.asarray(shared,float);separate=np.asarray(separate,float)
    if frame_ids.ndim!=1 or shared.shape!=frame_ids.shape or separate.shape!=frame_ids.shape:
        raise ValueError('Aligned one-dimensional observed-view evidence required')
    if len(np.unique(frame_ids))!=len(frame_ids):raise ValueError('Duplicate views cannot multiply evidence')
    if not np.isfinite(shared).all() or not np.isfinite(separate).all() or np.any(shared<0) or np.any(separate<0):
        raise ValueError('Finite nonnegative explanation scores required')
    delta=shared-separate;total=float(delta.sum())
    # Include removal of no frame; absent views never become negative evidence.
    worst=total-max(0.,float(delta.max(initial=0.)))
    return dict(views=len(delta),margin=total,worst_after_one_deletion=worst,
                approved=bool(worst>1e-10))


def compatible_donors(recipient, donors, evidence):
    """Return donors whose entire accepted group remains pairwise compatible.

    Evidence maps canonical (minID,maxID) pairs to margin results. No evidence
    means abstention. The receiver is fixed by the previously frozen proposal,
    not selected by this test. Larger worst-case support wins deterministic ties.
    """
    recipient=int(recipient);donors=sorted(set(map(int,donors))-{recipient})
    def result(a,b):return evidence.get(tuple(sorted((a,b))))
    def value(a):
        r=result(a,recipient)
        return r['worst_after_one_deletion'] if r is not None else -float('inf')
    members=[recipient];accepted=[];rejected=[]
    for donor in sorted(donors,key=lambda a:(-value(a),a)):
        checks=[result(donor,b) for b in members]
        if all(r is not None and r['approved'] for r in checks):
            members.append(donor);accepted.append(donor)
        else:rejected.append(donor)
    return dict(recipient=recipient,accepted=accepted,rejected=rejected,members=members)
