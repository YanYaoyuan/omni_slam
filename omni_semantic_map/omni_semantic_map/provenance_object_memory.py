"""Object-identified view evidence and reversible window-end map allocation.

Object identities come from a supplied predictor, never from anonymous latent
slots or reference labels. Keeping provenance is an implementation property,
not proof of accurate instances or an algorithmic novelty claim.
"""
import numpy as np
import torch
from torch import nn


def disjoint_inputs(query_frames, view_frames, window=4):
    """Two causal RGB buffers share no input image iff their rank gap >=window.

    This is input separation, NOT statistical independence or an unseen scene.
    Negative ranks denote absent observations and can never authorize evidence.
    """
    q, v = np.asarray(query_frames), np.asarray(view_frames)
    if window < 1:raise ValueError('Positive context length required')
    return (q>=0)&(v>=0)&(np.abs(q-v)>=window)


class ProvenanceObjectMemory:
    """Append measured views under explicit predicted IDs; absent IDs unchanged.

    One event per object/frame. Each event stores its actual source provenance
    and a descriptor with equal weight per physical20cm cell. Replay with final
    map IDs is window-end refinement, not a claim of online ID discovery.
    """
    def __init__(self, object_ids, dimension):
        ids = np.asarray(object_ids, np.int64)
        if ids.ndim!=1 or np.any(ids<=0) or len(np.unique(ids))!=len(ids) or dimension<1:
            raise ValueError('Unique positive predicted object IDs required')
        self.object_ids = ids.copy(); self.dimension = dimension
        self.views = {int(i):[] for i in ids}; self.last_frame = -1

    def observe(self, frame, source_ids, owners, cells, features, probability):
        sid=np.asarray(source_ids,np.int64); own=np.asarray(owners,np.int64)
        cells=np.asarray(cells,np.int64); f=np.asarray(features,np.float32); p=np.asarray(probability,np.float32)
        if frame<=self.last_frame or sid.ndim!=1 or len(np.unique(sid))!=len(sid) or np.any(sid<0):
            raise ValueError('Chronological unique measured sources required')
        if own.shape!=sid.shape or cells.shape!=sid.shape or f.shape!=(len(sid),self.dimension) or p.shape!=(len(sid),19):
            raise ValueError('Misaligned view evidence')
        if not np.isfinite(f).all() or not np.isfinite(p).all() or np.any(p<0):
            raise ValueError('Invalid evidence')
        if not np.isin(own[own>0],self.object_ids).all():raise ValueError('Unregistered object ID')
        # Validate everything before writing any object state.
        events=[]
        for oid in np.unique(own[own>0]):
            take=own==oid; unique,inverse,count=np.unique(cells[take],return_inverse=True,return_counts=True)
            weight=1/count[inverse]; weight/=weight.sum()
            descriptor=np.sum(f[take]*weight[:,None],axis=0)
            descriptor/=max(float(np.linalg.norm(descriptor)),1e-12)
            events.append((int(oid),dict(frame=int(frame),source_ids=sid[take].copy(),
                descriptor=descriptor,semantic=np.sum(p[take]*weight[:,None],axis=0),
                source_count=int(take.sum()),cell_count=len(unique))))
        for oid,event in events:self.views[oid].append(event)
        self.last_frame=int(frame)


class ViewEvidenceAllocator(nn.Module):
    """Same trainable capacity for mean, nearest-descriptor and attention arms.

    Inputs already contain raw cosine and measured-view geometry. No recurrent
    anonymous slot update exists. Every candidate competes with explicit null.
    """
    def __init__(self, static_dimensions=32, view_dimensions=5):
        super().__init__()
        self.view=nn.Sequential(nn.Linear(view_dimensions,16),nn.ReLU(),nn.Linear(16,16),nn.ReLU())
        self.quality=nn.Linear(16,1)
        self.readout=nn.Sequential(nn.Linear(static_dimensions+17,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU(),nn.Linear(32,1))

    def forward(self, static, views, view_valid, candidate_valid, mode):
        if mode not in ['mean','retrieval','attention']:raise ValueError('Unknown pooling mode')
        if static.shape[:-1]!=candidate_valid.shape or views.shape[:-2]!=candidate_valid.shape or views.shape[:-1]!=view_valid.shape:
            raise ValueError('Misaligned candidate evidence')
        if torch.any(candidate_valid & ~view_valid.any(-1)):raise ValueError('No evidence for valid candidate')
        if torch.any(view_valid & ~candidate_valid[...,None]):raise ValueError('Evidence on padding candidate')
        encoded=self.view(views);quality=self.quality(encoded).squeeze(-1)
        if mode=='mean':
            weights=view_valid.float()/view_valid.sum(-1,keepdim=True).clamp_min(1)
        elif mode=='retrieval':
            # Fixed strongest raw descriptor cosine; deterministic first-frame tie.
            scores=views[...,0].masked_fill(~view_valid,-torch.inf)
            best=scores.argmax(-1)
            weights=torch.nn.functional.one_hot(best,views.shape[-2]).to(views.dtype)*candidate_valid[...,None]
        else:
            scores=quality.masked_fill(~view_valid,-torch.inf)
            scores=torch.where(candidate_valid[...,None],scores,torch.zeros_like(scores))
            weights=scores.softmax(-1)*view_valid
        pooled=(encoded*weights[...,None]).sum(-2)
        scalar=(quality*weights).sum(-1,keepdim=True)
        logits=self.readout(torch.cat([static,pooled,scalar],-1)).squeeze(-1)
        return logits.masked_fill(~candidate_valid,-torch.inf)


def revision_map(base_xyz, base_owners, source_xyz, old_indices, old_choices,
                 new_source_ids, new_choices, known_objects, *, revise=True):
    """Atomic export: revise precisely named old points and append real sources.

    Unexamined old points remain unchanged. Zero removes an examined point;
    positive IDs must already exist. No synthetic XYZ, ID birth or merge occurs.
    """
    xyz=np.asarray(base_xyz); owners=np.asarray(base_owners,np.int64)
    oi=np.asarray(old_indices,np.int64); oc=np.asarray(old_choices,np.int64)
    si=np.asarray(new_source_ids,np.int64); sc=np.asarray(new_choices,np.int64)
    if owners.shape!=(len(xyz),) or oi.shape!=oc.shape or si.shape!=sc.shape or oi.ndim!=1 or si.ndim!=1:
        raise ValueError('Misaligned point decisions')
    if len(np.unique(oi))!=len(oi) or len(np.unique(si))!=len(si) or np.any(oi<0) or np.any(oi>=len(xyz)) or np.any(si<0) or np.any(si>=len(source_xyz)):
        raise ValueError('Invalid point provenance')
    allowed=np.r_[0,np.asarray(known_objects,np.int64)]
    if not np.isin(oc,allowed).all() or not np.isin(sc,allowed).all():raise ValueError('Unregistered owner')
    revised=owners.copy()
    if revise:revised[oi]=oc
    keep=np.flatnonzero(revised>0);add=sc>0
    return dict(xyz=np.concatenate([xyz[keep],np.asarray(source_xyz)[si[add]]]),
        instance=np.r_[revised[keep],sc[add]],retained_base_indices=keep,added_source_ids=si[add],
        revised_base_owners=revised)
