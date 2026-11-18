"""Rigid-frame-invariant point-to-object geometry, with no PCA frame choice.

Only scalar distances, covariance eigenvalues and covariance contractions are
used. Repeated eigenvalues do not require selecting unstable eigenvectors.
The caller retains physical evidence memberships under coordinate changes.
"""
import numpy as np
from scipy.spatial import cKDTree


GEOMETRY_FEATURES = ('log_nearest_core','log_nearest_other_core','core_distance_ratio',
    'log_center_distance_over_rms','log_regularized_mahalanobis',
    'log_parallel_std','log_perpendicular_std',
    'log_principal_std_0','log_principal_std_1','log_principal_std_2','is_car','is_building')


def geometry_features(points, core, other_core, semantic, *, measurement_scale=.05):
    p=np.asarray(points,np.float64);c=np.asarray(core,np.float64);other=np.asarray(other_core,np.float64)
    if any(a.ndim!=2 or a.shape[1]!=3 or not np.isfinite(a).all() for a in [p,c,other]):
        raise ValueError('Finite Nx3 measured coordinates required')
    if not len(c) or not np.isfinite(measurement_scale) or measurement_scale<=0:
        raise ValueError('Nonempty core and positive measurement scale required')
    d=cKDTree(c).query(p,workers=1)[0]
    od=cKDTree(other).query(p,workers=1)[0] if len(other) else np.full(len(p),100.)
    center=c.mean(0);centered=c-center;cov=centered.T@centered/len(c)
    eigen=np.maximum(np.linalg.eigvalsh(cov),0.)
    delta=p-center;r2=np.einsum('ij,ij->i',delta,delta);trace=float(np.trace(cov));ridge=measurement_scale**2
    projected=np.einsum('ij,jk,ik->i',delta,cov,delta)
    parallel=np.divide(projected,r2,out=np.full(len(p),trace/3),where=r2>1e-24)
    parallel=np.clip(parallel,0.,max(trace,0.));perpendicular=np.maximum((trace-parallel)/2,0.)
    inverse=np.linalg.inv(cov+ridge*np.eye(3));mahalanobis=np.maximum(np.einsum('ij,jk,ik->i',delta,inverse,delta),0.)
    features=np.column_stack([np.log1p(d),np.log1p(od),d/(d+od+1e-6),
        np.log1p(np.sqrt(r2/(trace+ridge))),np.log1p(np.sqrt(mahalanobis)),
        np.log1p(np.sqrt(parallel)),np.log1p(np.sqrt(perpendicular)),
        np.broadcast_to(np.log1p(np.sqrt(eigen)),(len(p),3)),
        np.full(len(p),semantic==26),np.full(len(p),semantic==11)])
    if not np.isfinite(features).all():raise ValueError('Nonfinite intrinsic feature')
    return features.astype(np.float32)
