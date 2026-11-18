"""Conservative ranking among already accepted association candidates."""
import numpy as np


def rank_accepted(native_scores,evidence_scores):
    native=np.asarray(native_scores,float);evidence=np.asarray(evidence_scores,float)
    if native.shape!=evidence.shape or native.ndim!=1:raise ValueError('incompatible candidate vectors')
    eligible=np.flatnonzero(np.isfinite(native))
    if not len(eligible):return None,'new_object'
    baseline=int(eligible[np.argmax(native[eligible])])
    if len(eligible)<2:return baseline,'unique_candidate'
    if not np.isfinite(evidence[eligible]).all():return baseline,'insufficient_evidence'
    maximum=evidence[eligible].max();tied=eligible[evidence[eligible]==maximum]
    # Exact ties retain the native ranking, including stable index order.
    return int(tied[np.argmax(native[tied])]),'evidence_rank'


def local_correspondence_score(xyz_a,features_a,xyz_b,features_b,radius=.4,min_pairs=3):
    a=np.asarray(features_a,float);b=np.asarray(features_b,float)
    if not len(a) or not len(b):return float('nan'),0
    a=a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)
    b=b/np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-12)
    cos=a@b.T;j=cos.argmax(1);i=np.arange(len(j));mutual=cos.argmax(0)[j]==i
    d=np.linalg.norm(np.asarray(xyz_a)-np.asarray(xyz_b)[j],axis=1)
    keep=mutual&(d<radius);count=int(keep.sum())
    return (float(np.mean(cos[i[keep],j[keep]])) if count>=min_pairs else float('nan')),count
