"""Experimental fixed-effect context separation for persistent object evidence.

For observed feature f_oi and full-image context g_i, fit
    f_oi = mu_o + beta g_i + residual
using within-object centered vectors. beta is a shared nonnegative scalar,
bounded by one for the frozen experiment, not a trained accuracy confidence.
Updating beta revises historical object descriptors without deleting evidence.
This module does not change geometry or observation ownership automatically.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ContextObservation:
    observation_id: int
    frame_id: int
    object_id: int
    area: float
    feature: np.ndarray
    context: np.ndarray


def unit(x):
    x=np.asarray(x,dtype=np.float64)
    n=float(np.linalg.norm(x))
    if not np.isfinite(x).all() or n<1e-12:raise ValueError('Finite nonzero descriptor required')
    return x/n


def fit_context(feature_groups, context_groups):
    """Within-object fixed effect regression; no class vocabulary or GT."""
    if len(feature_groups)!=len(context_groups):raise ValueError('Misaligned groups')
    numerator=denominator=0.;groups_used=0;degrees=0
    for features,contexts in zip(feature_groups,context_groups):
        f=np.asarray(features,dtype=np.float64);g=np.asarray(contexts,dtype=np.float64)
        if f.ndim!=2 or f.shape!=g.shape or not len(f) or not np.isfinite(f).all() or not np.isfinite(g).all():
            raise ValueError('Finite aligned feature/context matrices required')
        if len(f)<2:continue
        # Equal object contribution, independent of repeated sampling frequency.
        fc=f-f.mean(0);gc=g-g.mean(0)
        numerator+=float((fc*gc).sum())/(len(f)-1)
        denominator+=float((gc*gc).sum())/(len(f)-1)
        groups_used+=1;degrees+=len(f)-1
    identifiable=denominator>1e-12
    raw=numerator/denominator if identifiable else 0.
    return dict(beta=float(np.clip(raw,0.,1.)),unconstrained_beta=float(raw),
        context_variation=denominator,groups_used=groups_used,residual_degrees=degrees,
        identifiable=identifiable)


class ContextResidualMemory:
    """Immutable observations, deterministic one-observation-per-frame readout."""
    def __init__(self):
        self._observations={};self._dimension=None

    def add(self, observation):
        o=observation
        if o.observation_id in self._observations:raise ValueError('Duplicate observation ID')
        f=np.array(o.feature,dtype=np.float64,copy=True);g=np.array(o.context,dtype=np.float64,copy=True)
        if o.object_id<=0 or o.frame_id<0 or o.observation_id<0 or not np.isfinite(o.area) or o.area<=0:
            raise ValueError('Invalid identity/frame/area')
        if f.ndim!=1 or f.shape!=g.shape or not len(f):raise ValueError('Invalid vector shape')
        unit(f);unit(g)
        if self._dimension is not None and len(f)!=self._dimension:raise ValueError('Changed descriptor dimension')
        self._dimension=len(f);f.flags.writeable=False;g.flags.writeable=False
        self._observations[o.observation_id]=ContextObservation(o.observation_id,o.frame_id,o.object_id,o.area,f,g)

    def remove(self, observation_id):
        """Rollback an appended observation; re-fit automatically on readout."""
        if observation_id not in self._observations:raise KeyError(observation_id)
        del self._observations[observation_id]
        if not self._observations:self._dimension=None

    def selected(self):
        groups={}
        for o in sorted(self._observations.values(),key=lambda o:(o.object_id,-o.area,o.observation_id)):
            frames=groups.setdefault(o.object_id,{})
            frames.setdefault(o.frame_id,o)
        return {oid:list(frames.values()) for oid,frames in groups.items()}

    def readout(self,mode='shared',fixed_beta=.5):
        if mode not in ('shared','local','none','fixed'):raise ValueError('Unknown context mode')
        groups=self.selected();ids=sorted(groups)
        features=[np.stack([o.feature for o in groups[oid]]) for oid in ids]
        contexts=[np.stack([o.context for o in groups[oid]]) for oid in ids]
        fit=fit_context(features,contexts);objects={}
        if not np.isfinite(fixed_beta) or not 0<=fixed_beta<=1:raise ValueError('Invalid fixed coefficient')
        for oid,f,g in zip(ids,features,contexts):
            local=fit_context([f],[g])
            beta={'shared':fit['beta'],'local':local['beta'],'none':0.,'fixed':fixed_beta}[mode]
            mean=f.mean(0);residual=mean-beta*g.mean(0)
            fallback=bool(np.linalg.norm(residual)<1e-12)
            objects[oid]=dict(descriptor=unit(mean if fallback else residual),beta=beta,
                selected_observation_ids=[o.observation_id for o in groups[oid]],
                unique_frames=len(f),residual_norm=float(np.linalg.norm(residual)),
                degenerate_fallback=fallback,local_fit=local)
        return dict(fit=fit,mode=mode,objects=objects)
