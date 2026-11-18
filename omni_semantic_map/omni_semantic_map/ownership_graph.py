"""Measured visibility and bidirectional ownership gate for observation edges.

Uses predicted disjoint mask owners and measured depth only. No GT labels.
Occluded/out-of-image/unknown-depth samples abstain; visible background or a
different predicted observation is contradictory evidence. A majority in both
directions prevents a mixed mask from bridging two distinct visible objects.
"""
from dataclasses import dataclass
import numpy as np
from .objects import transform_points
from .witness_graph import WitnessGraph, WitnessConfig


@dataclass(frozen=True)
class OwnershipConfig(WitnessConfig):
    # Same tolerance as the existing support contact test; no scene tuning.
    visibility_depth_tolerance: float = .08
    ownership_fraction: float = .5
    ownership_min_samples: int = 3


def ownership_evidence(points, source_pose, target_pose, target, target_id,
                       tolerance=.08):
    """Project source supports to the nearest target sampled depth/owner pixel.

    Samples in front of the measured target surface are free-space conflicts.
    Samples behind it are occluded, hence excluded. Counts include sample
    multiplicity (the same sensor sampling used to construct observations).
    """
    xyz = transform_points(points, np.linalg.inv(target_pose) @ source_pose)
    z = xyz[:, 2]
    finite = np.isfinite(xyz).all(axis=1) & (z > 0)
    xyz = xyz[finite]; z = z[finite]
    k = np.asarray(target['intrinsic'])
    uv = xyz[:, :2] / z[:, None] * [k[0,0], k[1,1]] + [k[0,2], k[1,2]]
    # Owner/depth arrays are the same regular sample grid, anchored at pixel 0.
    uv = np.floor(uv / target['stride'] + .5).astype(np.int64)
    h,w = target['depth'].shape
    inside = (uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    uv=uv[inside]; z=z[inside]
    # One vote per target depth pixel: projection magnification must not turn a
    # single measured cell into several independent supporting samples.
    flat=uv[:,1]*w+uv[:,0]
    order=np.lexsort((z,flat));flat=flat[order]
    first=np.r_[True,np.diff(flat)!=0] if len(flat) else np.zeros(0,bool)
    selected=order[first];uv=uv[selected];z=z[selected]
    measured=target['depth'][uv[:,1],uv[:,0]]
    known=target['valid'][uv[:,1],uv[:,0]] & np.isfinite(measured)
    owner=target['owner'][uv[:,1],uv[:,0]]
    visible=known & (z <= measured+tolerance)
    surface=visible & (np.abs(z-measured)<=tolerance)
    owned=surface & (owner==target_id)
    return dict(visible=int(visible.sum()), owned=int(owned.sum()),
                occluded=int((known&~visible).sum()),
                free_space=int((visible&~surface).sum()),
                fraction=float(owned.sum()/visible.sum()) if visible.any() else None)


class OwnershipGraph(WitnessGraph):
    def __init__(self, views, config=OwnershipConfig(), conflict_policy='witness'):
        super().__init__(config, conflict_policy)
        if (config.visibility_depth_tolerance <= 0 or
            not .5 <= config.ownership_fraction < 1 or config.ownership_min_samples < 1):
            raise ValueError('invalid visibility ownership configuration')
        self.views=views
        self.edge_evidence={}

    def _add_edges(self,index):
        super()._add_edges(index)
        keys=[key for key in self.edges if key[1]==index]
        q=self.observations[index]
        for a,b in keys:
            p=self.observations[a]
            left=ownership_evidence(p['support_xyz'],self.poses[p['keyframe_id']],
                self.poses[q['keyframe_id']],self.views[q['keyframe_id']],q['observation_id'],
                self.config.visibility_depth_tolerance)
            right=ownership_evidence(q['support_xyz'],self.poses[q['keyframe_id']],
                self.poses[p['keyframe_id']],self.views[p['keyframe_id']],p['observation_id'],
                self.config.visibility_depth_tolerance)
            accepted=all(x['owned']>=self.config.ownership_min_samples and
                         x['fraction'] is not None and x['fraction']>self.config.ownership_fraction
                         for x in [left,right])
            self.edge_evidence[(a,b)]=dict(forward=left,reverse=right,accepted=accepted)
            if not accepted: del self.edges[(a,b)]

    def revise_poses(self,poses):
        # Visibility itself depends on poses: never reuse stale pair evidence.
        self.edge_evidence={}
        return super().revise_poses(poses)
