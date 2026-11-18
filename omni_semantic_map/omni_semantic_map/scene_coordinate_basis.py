"""Canonical metric basis for provided poses with one fixed world affine map.

If raw poses are G @ T_i with rigid T_i and a fixed positive-determinant world
linear map G, W=(R_0 R_0.T)^(-1/2) makes W @ G orthogonal. Apply W to BOTH raw
translations and reference geometry. W uses the first pose only, never a mesh
or alignment objective. Reject varying affine deformation rather than fitting
individual frames. This is an input adapter, not SLAM or a mapping innovation.
"""
import numpy as np


def canonicalize_world(poses, tolerance=1e-3):
    raw=np.asarray(poses,np.float64)
    if raw.ndim!=3 or raw.shape[1:]!=(4,4) or not len(raw) or not np.isfinite(raw).all():
        raise ValueError('expected finite nonempty Nx4x4 provided poses')
    if not np.allclose(raw[:,3],[0,0,0,1],atol=1e-7,rtol=0):
        raise ValueError('invalid homogeneous row')
    rotations=raw[:,:3,:3]
    if (np.linalg.det(rotations)<=0).any():raise ValueError('reflected or singular frame')
    gram=rotations[0]@rotations[0].T
    eigenvalues,eigenvectors=np.linalg.eigh(gram)
    if eigenvalues.min()<=1e-8:raise ValueError('degenerate world coordinate basis')
    W=(eigenvectors*(1/np.sqrt(eigenvalues)))@eigenvectors.T
    transform=np.eye(4);transform[:3,:3]=W
    normalized=transform[None]@raw
    u,_,vh=np.linalg.svd(normalized[:,:3,:3]);rigid=u@vh
    residual=np.linalg.norm(rigid-normalized[:,:3,:3],axis=(1,2))
    if (residual>tolerance).any():raise ValueError('deformation is not a single fixed world basis within tolerance')
    normalized[:,:3,:3]=rigid
    return dict(poses=normalized,canonical_from_raw=transform,
        first_pose_singular_values=np.linalg.svd(rotations[0],compute_uv=False),
        max_remaining_rotation_projection=float(residual.max()),
        max_world_gram_change=float(np.linalg.norm(rotations@rotations.transpose(0,2,1)-gram,axis=(1,2)).max()),
        tolerance=tolerance)
