"""Measured support boxes and local semantic/geometric ground verification."""
import itertools
import numpy as np


def bounds_corners(lower, upper):
    return np.asarray(list(itertools.product(*zip(lower, upper))), dtype=float)


BOX_EDGES = [(i, j) for i in range(8) for j in range(i+1, 8)
             if bin(i ^ j).count('1') == 1]


def observed_bounds(points):
    p=np.asarray(points,dtype=float)
    p=p[np.isfinite(p).all(axis=1)]
    if len(p)<8:return None
    low,high=np.quantile(p,[.02,.98],axis=0)
    return {'min':low.tolist(),'max':high.tolist(),'definition':'central 96% of observed support points; not amodal object dimensions'}


def object_box(observations, poses, center):
    corners=[];used=[]
    for ob in observations:
        if 'local_support_bounds' not in ob:continue
        pose=np.asarray(poses[int(ob['keyframe_id'])])
        position=np.asarray(ob['local_position'])@pose[:3,:3].T+pose[:3,3]
        if np.linalg.norm(position-center)>1.2:continue
        b=ob['local_support_bounds'];c=bounds_corners(b['min'],b['max'])
        corners.append(c@pose[:3,:3].T+pose[:3,3]);used.append(int(ob['keyframe_id']))
    if not corners:return None
    points=np.concatenate(corners);low=points.min(axis=0);high=points.max(axis=0)
    return {'min':low.tolist(),'max':high.tolist(),'center':((low+high)/2).tolist(),
            'extent':(high-low).tolist(),'corners':bounds_corners(low,high).tolist(),
            'keyframe_ids':used,'definition':'map-axis aligned envelope of measured surface supports; not a learned amodal 3D detection'}


def local_ground(points_world, semantic_candidates, sensor_position, seed=0):
    """Verify visible semantic ground in several local patches, without flattening.

    Ground points must have image support, be within 18m of the sensor, fit a
    locally estimated near-horizontal plane and belong to a populated grid cell.
    Disjoint patches retain their original elevations, so slopes and different
    levels are not snapped onto a global z plane. Returns original point indices.
    """
    points=np.asarray(points_world,dtype=float)
    rel=points-np.asarray(sensor_position)
    eligible=np.asarray(semantic_candidates,dtype=bool)&np.isfinite(points).all(axis=1)
    eligible&=(np.linalg.norm(rel[:,:2],axis=1)<18)&(rel[:,2]>-2.5)&(rel[:,2]<1.0)
    candidates=np.flatnonzero(eligible)
    accepted=np.zeros(len(points),bool);planes=[]
    if len(candidates)<30:return accepted,planes
    rng=np.random.default_rng(seed)
    # Small overlapping patches allow locally varying slope/elevation.
    for cx in [-12,-6,0,6,12]:
        for cy in [-12,-6,0,6,12]:
            idx=candidates[(abs(rel[candidates,0]-cx)<5)&(abs(rel[candidates,1]-cy)<5)]
            if len(idx)<30:continue
            p=points[idx];origin=np.median(p,axis=0);q=p-origin
            best=None;best_count=0
            for _ in range(48):
                sample=q[rng.choice(len(q),3,replace=False)]
                normal=np.cross(sample[1]-sample[0],sample[2]-sample[0]);norm=np.linalg.norm(normal)
                if norm<1e-6:continue
                normal/=norm
                if abs(normal[2])<np.cos(np.deg2rad(25)):continue
                residual=abs((q-sample[0])@normal);inside=residual<.10
                count=int(inside.sum())
                if count>best_count:best_count=count;best=inside
            if best is None or best_count<25 or best_count/len(q)<.35:continue
            fit=q[best];mean=fit.mean(axis=0)
            if np.linalg.eigvalsh(np.cov(fit[:,:2].T))[0]<.04:continue
            _,_,v=np.linalg.svd(fit-mean,full_matrices=False);normal=v[-1]
            if normal[2]<0:normal=-normal
            if normal[2]<np.cos(np.deg2rad(25)):continue
            if np.linalg.norm(np.ptp(fit[:,:2],axis=0))<1.2:continue
            inside=abs((q-mean)@normal)<.12
            # Sparse isolated projections are not enough to turn a cell green.
            cells=np.floor(rel[idx,:2]/.75).astype(int)
            _,inverse,counts=np.unique(cells[inside],axis=0,return_inverse=True,return_counts=True)
            chosen=np.flatnonzero(inside)[counts[inverse]>=3]
            accepted[idx[chosen]]=True
            planes.append({'point':(origin+mean).tolist(),'normal':normal.tolist(),'inliers':len(chosen)})
    return accepted,planes
