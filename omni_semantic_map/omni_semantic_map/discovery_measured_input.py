"""Measured, causal support for discovery queries; unknown pixels never vote."""
import numpy as np


def robot_measured_visibility(world_xyz,raw_tracking_xyz,frame,depth_tolerance=.05):
    """Preserve the existing LIVO floor-pixel convention and measured z buffer.

Only historical *measured coordinates* are projected; no depth backprojection
creates new geometry. The current scan, not the historical map, defines depth.
"""
    world=np.asarray(world_xyz,dtype=np.float64);raw=np.asarray(raw_tracking_xyz,dtype=np.float64)
    if world.ndim!=2 or world.shape[1]!=3 or raw.ndim!=2 or raw.shape[1]!=3 or depth_tolerance<0:raise ValueError('Invalid geometry')
    h,w=int(frame['height']),int(frame['width']);k=np.asarray(frame['intrinsic'],float)
    extrinsic=np.asarray(frame['camera_from_tracking'],float)
    measured=raw@extrinsic[:3,:3].T+extrinsic[:3,3]
    def pixels(points):
        valid=np.isfinite(points).all(1)&(points[:,2]>.2);uv=points[:,:2]/np.maximum(points[:,2,None],1e-9)
        u=uv[:,0]*k[0,0]+k[0,2];v=uv[:,1]*k[1,1]+k[1,2]
        valid&=(u>=0)&(u<w)&(v>=0)&(v<h);ids=np.flatnonzero(valid)
        return ids,np.column_stack([np.floor(u[ids]),np.floor(v[ids])]).astype(np.int32)
    ids,px=pixels(measured);depth=np.full(h*w,np.inf);np.minimum.at(depth,px[:,1]*w+px[:,0],measured[ids,2])
    inverse=np.linalg.inv(np.asarray(frame['camera_pose'],float));camera=world@inverse[:3,:3].T+inverse[:3,3]
    ids,px=pixels(camera);z=depth[px[:,1]*w+px[:,0]];known=np.isfinite(z)&(np.abs(camera[ids,2]-z)<=depth_tolerance)
    return ids[known],px[known],dict(current_measured_pixels=int(np.isfinite(depth).sum()),unknown_or_depth_rejected=int((~known).sum()))
