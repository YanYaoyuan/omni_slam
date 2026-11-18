"""Relative instance evidence, without fitting scores to GT identities."""
import numpy as np


def local_similarity(a,b,min_pairs=3):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if not len(a) or not len(b):return float('nan')
    a=a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)
    b=b/np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-12)
    sim=a@b.T;j=sim.argmax(1);i=np.arange(len(a));keep=sim.argmax(0)[j]==i
    return float(sim[i[keep],j[keep]].mean()) if keep.sum()>=min_pairs else float('nan')


def relative_vote(native,alternate,competitors):
    """Missing evidence retains the original native vote, never a fake score.

    Valid history maxima must both beat every same-frame competitor score.
    A competitor with unknown local correspondence triggers whole-vote fallback.
    """
    n=np.asarray(native,float);a=np.asarray(alternate,float);c=np.asarray(competitors,float)
    n=n[np.isfinite(n)];a=a[np.isfinite(a)]
    if not len(n) or not len(a) or not len(c) or not np.isfinite(c).all():
        return dict(pass_vote=True,reason='unknown_native_fallback',native=None,alternate=None,competitor=None)
    x,y,z=float(n.max()),float(a.max()),float(c.max())
    return dict(pass_vote=min(x,y)>z,reason='supported' if min(x,y)>z else 'appearance_veto',
                native=x,alternate=y,competitor=z)
