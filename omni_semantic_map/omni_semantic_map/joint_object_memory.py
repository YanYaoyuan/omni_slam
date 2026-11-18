"""Prototype for jointly revising current and retained source-token ownership.

This module is an untrained research component, not a claimed improvement.
Both frame attention and global attention update retained tokens. Object slots
then decode assignments for every selected source, including historical ones.
Geometry relations are supplied measurements with an explicit unknown state;
they are soft learned biases, never a hard same-instance label.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class JointObjectMemory(nn.Module):
    def __init__(self, input_dim=2048, width=128, heads=4, layers=2, max_objects=32):
        super().__init__()
        if width % heads or max_objects < 1 or layers < 1:
            raise ValueError('Invalid model dimensions')
        self.max_objects=max_objects
        self.project=nn.Linear(input_dim,width)
        self.frame_attention=nn.ModuleList([nn.MultiheadAttention(width,heads,batch_first=True) for _ in range(layers)])
        self.global_attention=nn.ModuleList([nn.MultiheadAttention(width,heads,batch_first=True) for _ in range(layers)])
        self.frame_norm=nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.global_norm=nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.ffn=nn.ModuleList([nn.Sequential(nn.Linear(width,width*2),nn.GELU(),nn.Linear(width*2,width)) for _ in range(layers)])
        self.ffn_norm=nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        # Indices: occluded/incompatible measurement, unknown, supported. Unknown
        # contributes exactly zero bias; other terms are learned and unrestricted.
        self.relation_bias=nn.Parameter(torch.zeros(layers,2))
        self.new_slots=nn.Parameter(torch.randn(max_objects,width)*.02)
        self.slot_read=nn.MultiheadAttention(width,heads,batch_first=True)
        self.token_read=nn.MultiheadAttention(width,heads,batch_first=True)
        self.slot_norm=nn.LayerNorm(width)
        self.token_norm=nn.LayerNorm(width)
        self.mask_projection=nn.Linear(width,width)
        self.background=nn.Linear(width,1)
        self.objectness=nn.Linear(width,1)

    def forward(self, features, frame_ids, relations, memory=None, use_geometry=True):
        """All tensors describe selected observations, not just the new frame.

        features: [N,D], frame_ids: [N], relations: [N,N] in {-1,0,+1}.
        memory: previous [K,width] object-slot state or None; K is bounded.
        Source IDs remain external and are preserved in exactly this row order.
        """
        if features.ndim!=2 or not len(features):raise ValueError('Nonempty token matrix required')
        n=len(features)
        if frame_ids.shape!=(n,) or relations.shape!=(n,n):raise ValueError('Misaligned token metadata')
        if not torch.isfinite(features).all() or not torch.isin(relations,relations.new_tensor([-1,0,1])).all():
            raise ValueError('Invalid features or measured relations')
        if not torch.equal(relations,relations.T):raise ValueError('Relations must be symmetric')
        h=self.project(features)
        for layer,(frame,global_) in enumerate(zip(self.frame_attention,self.global_attention)):
            local=torch.zeros_like(h)
            for fid in torch.unique(frame_ids):
                ids=torch.where(frame_ids==fid)[0]
                value=h[ids][None]
                local[ids]=frame(value,value,value,need_weights=False)[0][0]
            h=self.frame_norm[layer](h+local)
            bias=torch.zeros((n,n),dtype=h.dtype,device=h.device)
            if use_geometry:
                bias=torch.where(relations<0,self.relation_bias[layer,0].to(h.dtype),bias)
                bias=torch.where(relations>0,self.relation_bias[layer,1].to(h.dtype),bias)
            value=h[None]
            h=self.global_norm[layer](h+global_(value,value,value,attn_mask=bias,need_weights=False)[0][0])
            h=self.ffn_norm[layer](h+self.ffn[layer](h))
        if memory is None:
            slots=self.new_slots
        else:
            if memory.shape!=self.new_slots.shape:raise ValueError('Persistent slot shape changed')
            slots=memory
        slots=self.slot_norm(slots+self.slot_read(slots[None],h[None],h[None],need_weights=False)[0][0])
        updated=self.token_norm(h+self.token_read(h[None],slots[None],slots[None],need_weights=False)[0][0])
        masks=self.mask_projection(updated)@slots.T/(slots.shape[-1]**.5)
        # Background is a separate class, not the first instance slot.
        logits=torch.cat([self.background(updated),masks],dim=-1)
        return dict(logits=logits,memory=slots,token_features=updated,
                    objectness=self.objectness(slots).squeeze(-1))


def aggregate_source_logits(logits, source_ids, frame_ids):
    """One contribution per distinct frame per source, retaining autograd.

    Multiple token occurrences of a point in one frame cannot multiply its vote.
    No geometric nearest-neighbor lookup silently changes physical point IDs.
    """
    if logits.ndim!=2 or source_ids.shape!=(len(logits),) or frame_ids.shape!=(len(logits),):
        raise ValueError('Misaligned source predictions')
    if (source_ids<0).any():raise ValueError('Unknown source ID')
    unique=torch.unique(source_ids,sorted=True);rows=[]
    for sid in unique:
        selected=source_ids==sid
        frame_rows=[logits[selected&(frame_ids==fid)].mean(0) for fid in torch.unique(frame_ids[selected])]
        rows.append(torch.stack(frame_rows).mean(0))
    return unique,torch.stack(rows)


def source_instance_loss(logits, targets, objectness, established=None):
    """Training-only set matching with optional persistent GT-to-slot binding.

    Inputs are unique-source predictions, not repeated per-frame appearances.
    Positive targets identify instances, zero is background, negative is unknown.
    Existing bindings remain fixed; new instances match unused slots. Returning
    the binding lets consecutive training windows share slot identity. Reference
    IDs are never accepted by the inference forward method.
    """
    from scipy.optimize import linear_sum_assignment
    if logits.ndim!=2 or targets.shape!=(len(logits),) or objectness.shape!=(logits.shape[1]-1,):
        raise ValueError('Misaligned instance supervision')
    valid=targets>=0
    if not valid.any():raise ValueError('No supervised sources')
    ids=torch.unique(targets[targets>0],sorted=True).tolist()
    slots=logits.shape[1]-1
    binding=dict(established or {})
    if len(set(binding.values()))!=len(binding) or any(k<=0 or v<1 or v>slots for k,v in binding.items()):
        raise ValueError('Invalid persistent instance binding')
    new_ids=[i for i in ids if i not in binding]
    available=[i for i in range(1,slots+1) if i not in binding.values()]
    if len(new_ids)>len(available):raise ValueError('Object slot capacity exhausted; cannot silently drop GT')
    logp=F.log_softmax(logits,dim=-1)
    if new_ids:
        cost=torch.stack([-logp[targets==i][:,available].mean(0) for i in new_ids])
        rows,columns=linear_sum_assignment(cost.detach().cpu().numpy())
        binding.update({int(new_ids[r]):int(available[c]) for r,c in zip(rows,columns)})
    mapped=torch.zeros_like(targets)
    for i in ids:mapped[targets==i]=binding[i]
    # Equal weight per visible instance/background, not per surface area.
    losses=[F.nll_loss(logp[targets==i],mapped[targets==i]) for i in torch.unique(targets[valid])]
    ownership=torch.stack(losses).mean()
    present=torch.zeros_like(objectness)
    for i in ids:present[binding[i]-1]=1
    # An established object absent from this window may simply be occluded.
    observed_slots=torch.ones_like(objectness,dtype=torch.bool)
    for i,slot in binding.items():
        if i not in ids:observed_slots[slot-1]=False
    existence=F.binary_cross_entropy_with_logits(objectness[observed_slots],present[observed_slots]) if observed_slots.any() else objectness.sum()*0
    return dict(loss=ownership+existence,ownership=ownership,existence=existence,binding=binding)


class SourceOwnership:
    """Explicit historical ownership ledger with transactional rollback.

    Slot IDs are globally bound by the caller; this helper does not invent slot
    identity matching across windows. A commit updates precisely named sources.
    """
    def __init__(self, source_ids, object_ids):
        s,o=np.asarray(source_ids,np.int64),np.asarray(object_ids,np.int64)
        if s.ndim!=1 or o.shape!=s.shape or len(np.unique(s))!=len(s) or (s<0).any() or (o<0).any():
            raise ValueError('Unique nonnegative source IDs and object IDs required')
        self.owners=dict(zip(map(int,s),map(int,o)))
        self.version=0

    def commit(self, source_ids, object_ids):
        s,o=np.asarray(source_ids,np.int64),np.asarray(object_ids,np.int64)
        if s.ndim!=1 or s.shape!=o.shape or len(np.unique(s))!=len(s) or (o<0).any():
            raise ValueError('Invalid revision')
        if any(int(k) not in self.owners for k in s):raise ValueError('Unregistered source measurement')
        before=[self.owners[int(k)] for k in s]
        for k,v in zip(s,o):self.owners[int(k)]=int(v)
        self.version+=1
        return dict(version=self.version,source_ids=s.tolist(),before=before,after=o.tolist(),
                    changed=int(np.count_nonzero(np.asarray(before)!=o)))

    def rollback(self, transaction):
        if transaction['version']!=self.version:raise ValueError('Only latest revision can roll back')
        if any(self.owners[k]!=v for k,v in zip(transaction['source_ids'],transaction['after'])):
            raise ValueError('Ledger changed outside transaction')
        for k,v in zip(transaction['source_ids'],transaction['before']):self.owners[k]=v
        self.version-=1
