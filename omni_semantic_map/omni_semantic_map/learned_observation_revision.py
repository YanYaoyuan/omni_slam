"""Source-conserving observation-identity revision with learned pair evidence.

No GT or classifier training occurs in this module. Observations are indivisible.
The caller freezes physical point ownership before any partition intervention.
Greedy signed-graph optimization is approximate, not a globally optimal cut.
"""
import numpy as np
from scipy.sparse import csr_matrix

FEATURE_NAMES=('cosine','intersection_over_min','intersection_over_max','jaccard',
    'center_distance_over_min_extent','center_distance_over_max_extent',
    'box_overlap_over_min_volume','box_gap_over_min_extent','log_support_ratio',
    'log_extent_ratio','same_frame','log_frame_gap')


def pair_features(xyz,supports,features,frames,native,radius=.2):
    n=len(supports);xyz=np.asarray(xyz);features=np.asarray(features,float);frames=np.asarray(frames);native=np.asarray(native)
    if features.ndim!=2 or len(features)!=n or frames.shape!=(n,) or native.shape!=(n,):raise ValueError('Misaligned observations')
    if n<1 or radius<=0 or not np.isfinite(features).all() or not np.isfinite(xyz).all():raise ValueError('Invalid inputs')
    supports=[np.unique(np.asarray(s,dtype=np.int64)) for s in supports]
    if any(not len(s) or s.min()<0 or s.max()>=len(xyz) for s in supports):raise ValueError('Nonempty measured supports required')
    size=np.array([len(s) for s in supports]);indices=np.concatenate(supports)
    incidence=csr_matrix((np.ones(len(indices),np.float64),indices,np.r_[0,np.cumsum(size)]),shape=(n,len(xyz)))
    intersection=(incidence@incidence.T).toarray()
    lo=np.stack([xyz[s].min(0) for s in supports]);hi=np.stack([xyz[s].max(0) for s in supports])
    centers=np.stack([xyz[s].mean(0) for s in supports]);extent=np.maximum(np.linalg.norm(hi-lo,axis=1),radius)
    a,b=np.triu_indices(n,1);gap=np.maximum(np.maximum(lo[a]-hi[b],lo[b]-hi[a]),0.)
    distance=np.linalg.norm(gap,axis=1);inter=intersection[a,b]
    eligible=(distance<=radius)|(inter>0)|(native[a]==native[b]);a=a[eligible];b=b[eligible];distance=distance[eligible];inter=inter[eligible]
    f=features/np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-12)
    small=np.minimum(size[a],size[b]);large=np.maximum(size[a],size[b])
    s_extent=np.minimum(extent[a],extent[b]);l_extent=np.maximum(extent[a],extent[b])
    box_inter=np.prod(np.maximum(np.minimum(hi[a],hi[b])-np.maximum(lo[a],lo[b]),0.),axis=1)
    volume=np.prod(np.maximum(hi-lo,radius),axis=1)
    center_dist=np.linalg.norm(centers[a]-centers[b],axis=1)
    x=np.column_stack([(f[a]*f[b]).sum(1),inter/small,inter/large,inter/(size[a]+size[b]-inter),
        center_dist/s_extent,center_dist/l_extent,box_inter/np.minimum(volume[a],volume[b]),distance/s_extent,
        np.log(large/small),np.log(l_extent/s_extent),(frames[a]==frames[b]).astype(float),np.log1p(abs(frames[a]-frames[b]))])
    assert x.shape==(len(a),len(FEATURE_NAMES)) and np.isfinite(x).all()
    return np.column_stack([a,b]).astype(np.int32),x.astype(np.float32)


def signed_weights(n,pairs,probabilities,frames):
    """Reduce duplicate candidate-mask multiplicity per target frame."""
    pairs=np.asarray(pairs);probabilities=np.asarray(probabilities,float);frames=np.asarray(frames)
    if pairs.shape!=(len(probabilities),2) or frames.shape!=(n,):raise ValueError('Misaligned edges')
    if not np.isfinite(probabilities).all() or np.any((probabilities<0)|(probabilities>1)):raise ValueError('Invalid probabilities')
    if pairs.size and (pairs.min()<0 or pairs.max()>=n or np.any(pairs[:,0]>=pairs[:,1])):raise ValueError('Invalid unordered edge')
    if len(np.unique(pairs,axis=0))!=len(pairs):raise ValueError('Duplicate edges')
    groups={}
    for a,b in pairs:
        groups[a,int(frames[b])]=groups.get((a,int(frames[b])),0)+1
        groups[b,int(frames[a])]=groups.get((b,int(frames[a])),0)+1
    p=np.clip(probabilities,.01,.99);values=np.log(p/(1-p))
    values/=np.array([np.sqrt(groups[a,int(frames[b])]*groups[b,int(frames[a])]) for a,b in pairs])
    w=np.zeros((n,n),float);w[pairs[:,0],pairs[:,1]]=values;w[pairs[:,1],pairs[:,0]]=values
    return w


def objective(weights,labels,penalty):
    labels=np.asarray(labels);same=labels[:,None]==labels[None,:]
    return float(np.triu(weights*same,1).sum()-penalty*len(np.unique(labels)))


def revise(weights,initial,*,allow_reassignment,max_sweeps=10):
    """Same likelihood/cost for merge-only and whole-observation revision.

Unknown edges have weight zero; merging needs strictly positive total cross
evidence. Singleton creation pays log(1+N) identity-description cost. This is a
fixed coding-cost heuristic, not a calibrated posterior or formal BIC.
"""
    labels=np.asarray(initial,dtype=np.int64).copy();n=len(labels);w=np.asarray(weights,float)
    if w.shape!=(n,n) or not np.isfinite(w).all() or not np.array_equal(w,w.T) or np.any(np.diag(w)) or np.any(labels<=0):raise ValueError('Invalid graph/identities')
    if max_sweeps<1:raise ValueError('Positive sweep budget required')
    penalty=float(np.log1p(n));start=objective(w,labels,penalty);events=[];next_id=int(labels.max(initial=0))+1
    converged=False
    for sweep in range(max_sweeps):
        count_before=len(events)
        # Exact cross-component sum, recomputed after each committed merge.
        while True:
            ids,inv=np.unique(labels,return_inverse=True);membership=np.eye(len(ids))[inv]
            cross=membership.T@w@membership;cross[np.tril_indices(len(ids))]=-np.inf
            if len(ids)<2 or not np.any(cross>1e-10):break
            a,b=np.unravel_index(np.argmax(cross),cross.shape);old,new=int(ids[b]),int(ids[a]);members=np.flatnonzero(labels==old)
            before=labels[members].copy();gain=float(cross[a,b]+penalty);labels[members]=new
            events.append(dict(kind='merge',sweep=sweep,indices=members.tolist(),before=before.tolist(),after=labels[members].tolist(),gain=gain))
        if allow_reassignment:
            for i in range(n):
                old=int(labels[i]);ids,inv=np.unique(labels,return_inverse=True)
                sums=np.bincount(inv,weights=w[i],minlength=len(ids));oldpos=int(np.where(ids==old)[0][0]);own=sums[oldpos]
                count=int(np.count_nonzero(labels==old));choices=[]
                for pos,target in enumerate(ids):
                    if target==old or not np.any(w[i,labels==target]>0):continue
                    gain=float(sums[pos]-own+(penalty if count==1 else 0.))
                    choices.append((gain,-int(target),int(target)))
                if count>1:choices.append((float(-own-penalty),-next_id,next_id))
                if not choices:continue
                gain,_,target=max(choices)
                if gain<=1e-10:continue
                labels[i]=target
                if target==next_id:next_id+=1
                events.append(dict(kind='reassign',sweep=sweep,indices=[i],before=[old],after=[int(target)],gain=gain))
        if len(events)==count_before:converged=True;break
    end=objective(w,labels,penalty)
    assert abs((end-start)-sum(e['gain'] for e in events))<1e-6*max(1,abs(end),len(events))
    restored=rollback(labels,events);np.testing.assert_array_equal(restored,initial)
    return dict(labels=labels,events=events,objective_before=start,objective_after=end,penalty=penalty,
        converged=converged,sweeps=sweep+1,exact_rollback_verified=True)


def rollback(labels,events):
    labels=np.asarray(labels).copy()
    for event in reversed(events):
        ids=np.asarray(event['indices']);np.testing.assert_array_equal(labels[ids],event['after']);labels[ids]=event['before']
    return labels


def source_ownership(supports,observation_ids,frames,point_count):
    """Fixed earliest-frame, then original observation ID; never assignment based."""
    owners=np.full(point_count,-1,np.int32)
    for i in sorted(range(len(supports)),key=lambda i:(int(frames[i]),int(observation_ids[i]))):
        ids=np.unique(supports[i]);available=owners[ids]<0;owners[ids[available]]=i
    return owners
