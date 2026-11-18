"""Empirical geometry intervals for deciding whether a depth cue is precise.

Calibrate from separate measured samples. No IID/conformal guarantee is made
for correlated sequential scans. Passing an interval is not proof of visibility.
"""
import numpy as np


def fit_intervals(predicted, measured, anchor_distance, bins=3):
    p,z,d=map(lambda a:np.asarray(a,float),(predicted,measured,anchor_distance))
    if p.ndim!=1 or not(p.shape==z.shape==d.shape) or not len(p):raise ValueError('Aligned nonempty vectors required')
    if not(np.isfinite(p).all() and np.isfinite(z).all() and np.isfinite(d).all()) or np.any(p<=0) or np.any(z<=0) or np.any(d<0):raise ValueError('Valid positive depths and nonnegative distances required')
    edges=np.unique(np.quantile(d,np.arange(1,bins)/bins));label=np.searchsorted(edges,d,side='right')
    residual=(z-p)/(.05+.01*p);ranges=[]
    for j in range(len(edges)+1):
        values=residual[label==j]
        if len(values)<32:ranges.append(None)
        else:ranges.append([float(np.quantile(values,.05)),float(np.quantile(values,.95))])
    return dict(distance_edges=edges.tolist(),residual_quantiles=ranges,calibration_count=len(p),absolute_m=.05,relative=.01)


def intervals(predicted, anchor_distance, model):
    p,d=map(lambda a:np.asarray(a,float),(predicted,anchor_distance))
    if p.shape!=d.shape:raise ValueError('Aligned predictions and distances required')
    bins=np.searchsorted(model['distance_edges'],d,side='right');tol=model['absolute_m']+model['relative']*p
    lower=np.full(p.shape,np.nan);upper=np.full(p.shape,np.nan)
    for j,limits in enumerate(model['residual_quantiles']):
        if limits is None:continue
        take=bins==j;lower[take]=p[take]+limits[0]*tol[take];upper[take]=p[take]+limits[1]*tol[take]
    valid=np.isfinite(p)&(p>0)&np.isfinite(d)&(d>=0)&np.isfinite(lower)&(lower>0)&(upper>=lower)
    precise=valid&((upper-lower)<=2*tol)
    return lower,upper,precise


def candidate_depth_agreement(depth, lower, upper, precise):
    z=np.asarray(depth,float)
    return np.asarray(precise,bool)&np.isfinite(z)&(z>0)&(z>=lower)&(z<=upper)
