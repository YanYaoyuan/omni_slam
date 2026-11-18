"""Learned acceptance of whole-observation map edits; no GT at inference.

Proposal likelihood, identity coding cost, and source ownership are inherited
unchanged from learned_observation_revision. Only commit decisions differ.
"""
import numpy as np
from .learned_observation_revision import objective,rollback

OP_FEATURES=('merge','new_identity','log_moved_observations','log_source_observations',
    'log_target_observations','moved_source_observation_fraction','moved_source_mass_fraction',
    'moved_target_mass_fraction','log_moved_mass','log_source_mass','log_target_mass',
    'log_source_frames','log_source_remaining_frames','log_target_frames','log_target_after_frames',
    'moved_source_frame_fraction','shared_source_target_frame_fraction',
    'source_coherence','source_remaining_coherence','target_coherence','target_after_coherence',
    'moved_coherence','moved_source_cosine','moved_target_cosine',
    'log_likelihood_gain','old_edge_mean','new_edge_mean','old_edge_positive_fraction',
    'new_edge_positive_fraction','log_old_known_edges','log_new_known_edges')


def operation_features(weights,labels,indices,target,features,frames,mass,gain):
    labels=np.asarray(labels);indices=np.asarray(indices,dtype=int);target=int(target)
    if not len(indices):raise ValueError('Empty edit')
    source=int(labels[indices[0]])
    if not len(indices) or not np.all(labels[indices]==source) or source==target:raise ValueError('Invalid edit')
    old=np.flatnonzero(labels==source);remaining=np.setdiff1d(old,indices);dest=np.flatnonzero(labels==target)
    after=np.r_[dest,indices]
    def descriptor(ids):return features[ids].mean(0) if len(ids) else np.zeros(features.shape[1])
    def coherence(ids):return float(np.linalg.norm(descriptor(ids)))
    def cosine(a,b):
        x,y=descriptor(a),descriptor(b);return float(x@y/max(np.linalg.norm(x)*np.linalg.norm(y),1e-12))
    def edges(a,b):
        v=weights[np.ix_(a,b)].ravel();v=v[v!=0]
        return (float(v.mean()),float((v>0).mean()),float(np.log1p(len(v)))) if len(v) else (0.,0.,0.)
    oldedge=edges(indices,remaining);newedge=edges(indices,dest)
    fm=set(frames[indices]);fo=set(frames[old]);fr=set(frames[remaining]);ft=set(frames[dest]);fa=set(frames[after])
    moved_mass=mass[indices].sum();oldmass=mass[old].sum();newmass=mass[dest].sum()
    values=[float(len(indices)==len(old)),float(not len(dest)),np.log1p(len(indices)),np.log1p(len(old)),np.log1p(len(dest)),
        len(indices)/len(old),moved_mass/max(oldmass,1),moved_mass/max(newmass+moved_mass,1),
        np.log1p(moved_mass),np.log1p(oldmass),np.log1p(newmass),np.log1p(len(fo)),np.log1p(len(fr)),np.log1p(len(ft)),np.log1p(len(fa)),
        len(fm)/max(len(fo),1),len(fo&ft)/max(len(fo|ft),1),coherence(old),coherence(remaining),coherence(dest),coherence(after),
        coherence(indices),cosine(indices,remaining),cosine(indices,dest),np.log1p(max(gain,0)),oldedge[0],newedge[0],oldedge[1],newedge[1],oldedge[2],newedge[2]]
    result=np.asarray(values,np.float32);assert result.shape==(len(OP_FEATURES),) and np.isfinite(result).all()
    return result


def gated_revision(weights,initial,features,frames,mass,*,gate=None,recorder=None,allow_reassignment=True,max_sweeps=10):
    """With gate=None, exactly reproduce the previous greedy partition.

recorder observes (pre-state, moved indices, target, features, proposed gain)
before commitment; it is for training only and cannot change the decision.
"""
    w=np.asarray(weights,float);labels=np.asarray(initial,dtype=np.int64).copy();n=len(labels)
    features=np.asarray(features,float);frames=np.asarray(frames);mass=np.asarray(mass,float)
    if max_sweeps<1:raise ValueError('At least one sweep required')
    if w.shape!=(n,n) or not np.array_equal(w,w.T) or not np.isfinite(w).all() or np.any(np.diag(w)):raise ValueError('Invalid graph')
    if features.ndim!=2 or len(features)!=n or frames.shape!=(n,) or mass.shape!=(n,) or np.any(mass<0):raise ValueError('Invalid observations')
    features=features/np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-12)
    penalty=float(np.log1p(n));next_id=int(labels.max(initial=0))+1;events=[];decisions=[];start=objective(w,labels,penalty)
    def consider(indices,target,gain,kind,sweep,record=True):
        x=operation_features(w,labels,indices,target,features,frames,mass,gain)
        if recorder is not None and record:recorder(labels,indices,target,x,gain,kind)
        result={'accept':True} if gate is None else gate(x)
        accepted=bool(result['accept']);decisions.append(dict(kind=kind,sweep=sweep,indices=list(map(int,indices)),source=int(labels[indices[0]]),
            target=int(target),likelihood_gain=float(gain),**result))
        return accepted
    converged=False
    for sweep in range(max_sweeps):
        count_before=len(events)
        while True:
            ids,inv=np.unique(labels,return_inverse=True)
            if len(ids)<2:break
            membership=np.eye(len(ids))[inv];cross=membership.T@w@membership;cross[np.tril_indices(len(ids))]=-np.inf
            candidates=np.argwhere(cross>1e-10)
            if not len(candidates):break
            candidates=sorted(candidates,key=lambda ij:(-cross[tuple(ij)],int(ij[0]),int(ij[1])))
            # Training extra counterfactuals are read-only and bounded.
            if recorder is not None:
                for a,b in candidates[1:8]:
                    indices=np.flatnonzero(labels==ids[b]);gain=float(cross[a,b]+penalty)
                    x=operation_features(w,labels,indices,int(ids[a]),features,frames,mass,gain)
                    recorder(labels,indices,int(ids[a]),x,gain,'merge')
            committed=False
            for a,b in candidates:
                indices=np.flatnonzero(labels==ids[b]);gain=float(cross[a,b]+penalty);target=int(ids[a])
                if not consider(indices,target,gain,'merge',sweep):continue
                before=labels[indices].copy();labels[indices]=target
                events.append(dict(kind='merge',sweep=sweep,indices=indices.tolist(),before=before.tolist(),after=labels[indices].tolist(),gain=gain));committed=True;break
            if not committed:break
        if allow_reassignment:
            for i in range(n):
                old=int(labels[i]);ids,inv=np.unique(labels,return_inverse=True);sums=np.bincount(inv,weights=w[i],minlength=len(ids))
                own=sums[int(np.where(ids==old)[0][0])];count=int(np.count_nonzero(labels==old));choices=[]
                for pos,target in enumerate(ids):
                    if target==old or not np.any(w[i,labels==target]>0):continue
                    gain=float(sums[pos]-own+(penalty if count==1 else 0.));choices.append((gain,-int(target),int(target)))
                if count>1:choices.append((float(-own-penalty),-next_id,next_id))
                choices=sorted((c for c in choices if c[0]>1e-10),reverse=True)
                if not choices:continue
                if recorder is not None:
                    for gain,_,target in choices[1:4]:
                        x=operation_features(w,labels,[i],target,features,frames,mass,gain);recorder(labels,[i],target,x,gain,'reassign')
                gain,_,target=choices[0]
                if not consider([i],target,gain,'reassign',sweep):continue
                labels[i]=target
                if target==next_id:next_id+=1
                events.append(dict(kind='reassign',sweep=sweep,indices=[i],before=[old],after=[int(target)],gain=gain))
        if len(events)==count_before:converged=True;break
    end=objective(w,labels,penalty)
    assert abs((end-start)-sum(e['gain'] for e in events))<1e-6*max(1,abs(end),len(events))
    np.testing.assert_array_equal(rollback(labels,events),initial)
    return dict(labels=labels,events=events,decisions=decisions,converged=converged,sweeps=sweep+1,
        objective_before=start,objective_after=end,exact_rollback_verified=True)
