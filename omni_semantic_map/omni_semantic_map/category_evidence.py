"""Fixed-identity category readout with exclusive-mask quality and consensus.

Mask exclusivity is a proposal-competition proxy, not an occlusion estimate.
Consistency is not correctness: a coherent wrong descriptor remains wrong.
"""
import numpy as np

METHODS=('mean','area','exclusive_quality','quality_siglip','quality_dino','score_median')


def readout(features,areas,exclusive_areas,text,*,method='mean',dino=None):
    f=np.asarray(features,np.float64);a=np.asarray(areas,np.float64)
    x=np.asarray(exclusive_areas,np.float64);t=np.asarray(text,np.float64)
    if method not in METHODS or f.ndim!=2 or not len(f) or t.ndim!=2 or t.shape[1]!=f.shape[1]:
        raise ValueError('invalid method or feature dimensions')
    if a.shape!=(len(f),) or x.shape!=a.shape or any(not np.isfinite(z).all() for z in [f,a,x,t]):
        raise ValueError('invalid observation arrays')
    if (a<=0).any() or (x<0).any() or (x>a).any():raise ValueError('invalid mask areas')
    quality=np.sqrt(a)*(x/a);reason='none';selected=np.ones(len(f),bool);centrality=None
    if quality.sum()<=1e-12:quality=np.ones(len(f));reason='no_exclusive_pixels'
    weights=np.ones(len(f)) if method in ['mean','score_median'] else a.copy() if method=='area' else quality.copy()
    if method in ['quality_siglip','quality_dino']:
        z=f if method=='quality_siglip' else np.asarray(dino,np.float64)
        if z.ndim!=2 or len(z)!=len(f) or not np.isfinite(z).all():raise ValueError('invalid consensus features')
        norms=np.linalg.norm(z,axis=1)
        if len(f)<3:reason='fewer_than_three_views'
        elif (norms<=1e-12).any():reason='missing_consensus_feature'
        else:
            z=z/norms[:,None];similarity=np.clip(z@z.T,-1,1)
            # Exclude diagonal self-matches. Keep the more consistent half,
            # including every tie; a constant signal never arbitrarily drops views.
            centrality=np.median(similarity[~np.eye(len(f),dtype=bool)].reshape(len(f),-1),axis=1)
            selected=centrality>=np.median(centrality)
            if (weights*selected).sum()<=1e-12:reason='selected_views_have_no_quality'
            else:weights*=selected
    weights/=weights.sum()
    if method=='score_median':scores=np.median(f@t.T,axis=0)
    else:
        descriptor=weights@f;norm=np.linalg.norm(descriptor)
        if norm<=1e-12:raise ValueError('degenerate descriptor')
        scores=t@(descriptor/norm)
    return dict(scores=scores,weights=weights,selected=selected,
                centrality=centrality,fallback=reason,effective_sample_size=float(1/(weights@weights)))
