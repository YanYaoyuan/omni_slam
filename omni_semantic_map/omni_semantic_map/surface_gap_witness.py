"""Measured-ray evidence for surface-continuation links, not point deletion.

Only a supported-endpoints/free-midpoint observation witnesses a gap. Missing
or occluded measurements cannot certify a gap. Rays are finite returns from
one scan; this module neither invents depth nor claims calibrated probability.
"""
import numpy as np
from scipy.spatial import cKDTree

UNKNOWN, SUPPORTED, OCCLUDED, FREE = range(4)


class MeasuredRayTubes:
    def __init__(self, origin, returns, radius=.05, depth_tolerance=.05):
        self.origin = np.asarray(origin, float)
        points = np.asarray(returns, float)
        if self.origin.shape != (3,) or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('One origin and Nx3 returns required')
        if not np.isfinite(self.origin).all() or not np.isfinite(points).all() or radius<=0 or depth_tolerance<=0 or not np.isfinite([radius,depth_tolerance]).all():
            raise ValueError('Finite metric evidence and positive scales required')
        delta=points-self.origin; ranges=np.linalg.norm(delta,axis=1);valid=ranges>0
        self.ranges=ranges[valid];self.directions=delta[valid]/ranges[valid,None]
        self.tree=cKDTree(self.directions) if valid.any() else None
        self.radius=float(radius);self.tolerance=float(depth_tolerance)

    def classify(self, points):
        p=np.asarray(points,float)
        if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():raise ValueError('Finite Nx3 query required')
        status=np.full(len(p),UNKNOWN,np.uint8)
        if self.tree is None:return status
        q=p-self.origin;distance=np.linalg.norm(q,axis=1);valid=distance>self.radius+self.tolerance
        rows=np.flatnonzero(valid)
        if not len(rows):return status
        ratio=np.minimum(self.radius/distance[rows],1.)
        # Exact unit-sphere chord radius for a tube of fixed metric radius.
        chord=np.sqrt(2.-2.*np.sqrt(1.-ratio**2))
        neighbours=self.tree.query_ball_point(q[rows]/distance[rows,None],chord+1e-12,workers=1)
        count=np.array([len(v) for v in neighbours]);local=np.repeat(np.arange(len(rows)),count)
        if not len(local):return status
        ray=np.concatenate([np.asarray(v,np.int64) for v in neighbours if len(v)])
        along=np.einsum('ij,ij->i',q[rows[local]],self.directions[ray])
        perpendicular2=np.maximum(0,distance[rows[local]]**2-along**2)
        aligned=(along>0)&(perpendicular2<=self.radius**2+1e-10)
        residual=along-self.ranges[ray]
        support=np.zeros(len(rows),bool);occluded=support.copy();free=support.copy()
        np.logical_or.at(support,local,aligned&(np.abs(residual)<=self.tolerance))
        np.logical_or.at(occluded,local,aligned&(residual>self.tolerance))
        np.logical_or.at(free,local,aligned&(residual < -self.tolerance))
        # A nearer aligned return vetoes carving through an occluder. Actual
        # support takes precedence over contradictory nearby rays at boundaries.
        status[rows[free&~occluded&~support]]=FREE
        status[rows[occluded&~support]]=OCCLUDED
        status[rows[support]]=SUPPORTED
        return status

    def gaps(self, starts, ends):
        a,b=np.asarray(starts,float),np.asarray(ends,float)
        if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=3:raise ValueError('Aligned link endpoints required')
        values=self.classify(np.concatenate([a,(a+b)/2,b])).reshape(3,len(a))
        gap=(values[0]==SUPPORTED)&(values[1]==FREE)&(values[2]==SUPPORTED)
        return gap,values
