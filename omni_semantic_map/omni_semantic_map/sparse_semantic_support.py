"""Measured LiDAR supports at the image epoch, without synthetic depth."""
import numpy as np


def sparse_support(points, camera_from_tracking, intrinsic, segmentation, *, depth_tolerance=.05):
    """Return visible point indices and exclusive mask IDs.

    A per-pixel z buffer rejects measured points more than 5 cm behind the
    nearest measured surface. It cannot detect occlusion by unmeasured surfaces.
    Coordinates and K come from the same atomic LIVO packet as the image.
    """
    points=np.asarray(points,dtype=np.float64);T=np.asarray(camera_from_tracking,dtype=np.float64)
    K=np.asarray(intrinsic,dtype=np.float64);seg=np.asarray(segmentation)
    if points.ndim!=2 or points.shape[1]!=3 or T.shape!=(4,4) or K.shape!=(3,3) or seg.ndim!=2:
        raise ValueError('invalid projection shapes')
    if not np.isfinite(T).all() or not np.isfinite(K).all() or not np.allclose(T[3],[0,0,0,1]) or depth_tolerance<0:
        raise ValueError('invalid camera transform/intrinsics/tolerance')
    camera=points@T[:3,:3].T+T[:3,3]
    ids=np.flatnonzero(np.isfinite(camera).all(1)&(camera[:,2]>.2))
    image=camera[ids]@K.T
    uv=image[:,:2]/image[:,2,None]
    h,w=seg.shape;ok=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    ids=ids[ok];pixels=np.floor(uv[ok]).astype(np.int64)
    linear=pixels[:,1]*w+pixels[:,0]
    nearest=np.full(h*w,np.inf);np.minimum.at(nearest,linear,camera[ids,2])
    visible=camera[ids,2]<=nearest[linear]+depth_tolerance
    ids=ids[visible];pixels=pixels[visible]
    masks=seg[pixels[:,1],pixels[:,0]].astype(np.int64)
    assigned=masks>=0
    return dict(visible_indices=ids,visible_pixels=pixels,
                point_indices=ids[assigned],mask_ids=masks[assigned],
                rejected_occluded=int((~visible).sum()))
