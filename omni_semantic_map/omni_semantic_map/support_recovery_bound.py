"""Posthoc optimistic IoU bounds for append-only, fixed-identity maps.

Reference labels are diagnostic inputs, never inputs to a predictor. Each
reference point may independently choose any reachable owner, relaxing shared
source decisions. Hence bounds need not be jointly attainable. No AP bound is
claimed: confidence ranking, FP counts and mutually consistent masks remain.
"""
import numpy as np
from scipy.spatial import cKDTree


def optimistic_iou(core_xyz, core_owner, candidate_xyz, candidate_owner,
                   reference_xyz, reference_gt, confidence, radius=.2):
    core=np.asarray(core_xyz,float);labels=np.asarray(core_owner,np.int64)
    candidates=np.asarray(candidate_xyz,float);owners=np.asarray(candidate_owner,np.int64)
    ref=np.asarray(reference_xyz,float);truth=np.asarray(reference_gt,np.int64);w=np.asarray(confidence,float)
    if core.shape!=(len(labels),3) or candidates.shape!=(len(owners),3) or ref.shape!=(len(truth),3) or w.shape!=truth.shape:
        raise ValueError('Inconsistent geometry/label shapes')
    if not all(np.isfinite(a).all() for a in [core,candidates,ref,w]) or np.any(w<0) or radius<=0:
        raise ValueError('Invalid finite geometry/weights/radius')
    if np.any(labels<=0) or np.any(owners<=0) or not np.isin(owners,labels).all():raise ValueError('Fixed positive owner IDs required')
    # Exact old transfer with original-index tie breaking, also used by maps.
    distance,_=cKDTree(core).query(ref,workers=1,distance_upper_bound=radius)
    # Conservatively allow any tied old owner to retain the point. For ordinary
    # unique distances only one is possible. This keeps the bound optimistic.
    object_ids=np.unique(labels);gt_ids=np.unique(truth[truth>0]);mass=np.bincount(truth,weights=w)
    bound=np.zeros((len(object_ids),len(gt_ids)));stats=[]
    def closer(points):
        if not len(points):return np.zeros(len(ref),bool)
        d=cKDTree(points).query(ref,workers=1,distance_upper_bound=radius)[0]
        return np.isfinite(d)&(d<=distance+1e-12)
    for row,oid in enumerate(object_ids):
        # Same-owner old candidates and other-owner old points are treated
        # symmetrically in exact-distance ties, further relaxing the partition.
        old=closer(core[labels==oid]);old_other=closer(core[labels!=oid])
        add=closer(candidates[owners==oid]);other=closer(candidates[owners!=oid])
        possible=old|add;fixed=old&~old_other&~other
        hit=np.bincount(truth,weights=w*possible,minlength=len(mass));fixed_hit=np.bincount(truth,weights=w*fixed,minlength=len(mass));fixed_mass=float(w[fixed].sum())
        denominator=mass[gt_ids]+fixed_mass-fixed_hit[gt_ids]
        bound[row]=np.divide(hit[gt_ids],denominator,out=np.zeros(len(gt_ids)),where=denominator>0)
        stats.append(dict(owner=int(oid),possible_mass=float(w[possible].sum()),irreducible_mass=fixed_mass))
    assert np.all((bound>=0)&(bound<=1+1e-10))
    return dict(object_ids=object_ids,gt_ids=gt_ids,iou_upper=bound,owners=stats)
