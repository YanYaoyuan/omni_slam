"""Sensor-independent projection helpers; no assumption of Flat intrinsics."""
import numpy as np
from .dino_features import normalize


def camera_pose_from_anchor(anchor_pose, camera_from_anchor):
    """T_world_camera = T_world_anchor * inverse(T_camera_anchor)."""
    return np.asarray(anchor_pose) @ np.linalg.inv(camera_from_anchor)


def pool_measured_tokens(tokens, pixels, local_points, image_shape, max_tokens=24):
    height, width = image_shape[:2]
    pixels=np.asarray(pixels);local_points=np.asarray(local_points)
    if len(pixels)==0 or len(pixels)!=len(local_points) or not np.isfinite(pixels).all() or not np.isfinite(local_points).all():
        raise ValueError('invalid measured supports')
    if np.any(pixels<0) or np.any(pixels[:,0]>=width) or np.any(pixels[:,1]>=height):
        raise ValueError('support outside image')
    h,w,dim=tokens.shape
    indices=(pixels[:,1]*h/height).astype(int)*w+(pixels[:,0]*w/width).astype(int)
    ids,counts=np.unique(indices,return_counts=True)
    features=normalize(tokens.reshape(-1,dim)[ids])
    descriptor=normalize(np.average(features,axis=0,weights=counts))
    selected=np.unique(np.linspace(0,len(ids)-1,min(max_tokens,len(ids))).astype(int))
    anchors=np.array([np.median(local_points[indices==ids[i]],axis=0) for i in selected])
    return descriptor,anchors.astype(np.float32),features[selected]
