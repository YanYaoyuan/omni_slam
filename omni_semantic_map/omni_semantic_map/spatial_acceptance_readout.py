"""Exact spatial readout for optional measured-point additions to a fixed map.

The old map is always present. A rejected proposal is absent, not a new spatial
null anchor. Nearest candidates are tried in deterministic append order; the
old map wins distance ties. Soft readout marginalizes independent categorical
accept/reject choices at each source, without moving/creating coordinates.
This is a training/diagnostic operator, not a new map predictor or exact E[IoU].
"""
import numpy as np
from scipy.spatial import cKDTree


def _points(x):
    a=np.asarray(x,np.float64)
    if a.ndim!=2 or a.shape[1]!=3 or not np.isfinite(a).all():
        raise ValueError('Finite Nx3 coordinates required')
    return a


def build_readout(base_xyz,base_labels,candidate_xyz,queries,radius=.2):
    base=_points(base_xyz);candidate=_points(candidate_xyz);query=_points(queries)
    labels=np.asarray(base_labels)
    if labels.shape!=(len(base),) or not np.issubdtype(labels.dtype,np.integer) or np.any(labels<0):
        raise ValueError('Aligned nonnegative old labels required')
    if not np.isfinite(radius) or radius<=0:raise ValueError('Positive radius required')
    distance=np.full(len(query),np.inf);fallback=np.zeros(len(query),np.int64)
    if len(base):
        tree=cKDTree(base)
        for start in range(0,len(query),200000):
            q=query[start:start+200000]
            d,nearest=tree.query(q,k=min(2,len(base)),workers=1,distance_upper_bound=radius)
            if len(base)==1:d,nearest=d[:,None],nearest[:,None]
            valid=np.isfinite(d[:,0]);chosen=nearest[:,0].copy()
            if len(base)>1:
                for i in np.flatnonzero(valid&(d[:,0]==d[:,1])):
                    candidates=np.asarray(tree.query_ball_point(q[i],np.nextafter(d[i,0],np.inf)))
                    d2=np.square(base[candidates]-q[i]).sum(1)
                    chosen[i]=np.min(candidates[d2==d2.min()])
            distance[start:start+len(q)]=d[:,0]
            local=fallback[start:start+len(q)];local[valid]=labels[chosen[valid]]
    if len(candidate) and len(query):
        neighbors=cKDTree(candidate).sparse_distance_matrix(cKDTree(query),radius,output_type='coo_matrix')
        keep=(neighbors.data<radius)&(neighbors.data<distance[neighbors.col])
        source=neighbors.row[keep];row=neighbors.col[keep];d=neighbors.data[keep]
        order=np.lexsort((source,d,row));source=source[order];row=row[order]
        lengths=np.bincount(row,minlength=len(query))
    else:source=np.empty(0,np.int64);lengths=np.zeros(len(query),np.int64)
    return dict(offsets=np.r_[0,np.cumsum(lengths)],sources=source.astype(np.int64),fallback=fallback,
                candidate_count=np.asarray(len(candidate),np.int64))


def validate(adjacency):
    offsets=np.asarray(adjacency['offsets']);sources=np.asarray(adjacency['sources']);fallback=np.asarray(adjacency['fallback']);n=int(adjacency['candidate_count'])
    if any(not np.issubdtype(a.dtype,np.integer) for a in [offsets,sources,fallback]):raise ValueError('Integer adjacency required')
    if offsets.ndim!=1 or sources.ndim!=1 or fallback.shape!=(len(offsets)-1,) or not len(offsets) or offsets[0]!=0 or offsets[-1]!=len(sources) or np.any(np.diff(offsets)<0):
        raise ValueError('Invalid CSR readout')
    if n<0 or np.any((sources<0)|(sources>=n)) or np.any(fallback<0):raise ValueError('Invalid readout IDs')
    return offsets,sources,fallback,n


def first_two_readouts(adjacency,accepted_owners):
    """Return label, first candidate index (-1=old), and deletion fallback label."""
    offsets,sources,fallback,n=validate(adjacency);owners=np.asarray(accepted_owners)
    if owners.shape!=(n,) or not np.issubdtype(owners.dtype,np.integer) or np.any(owners<0):
        raise ValueError('One nonnegative owner per measured candidate required')
    rows=np.repeat(np.arange(len(fallback)),np.diff(offsets));active=np.flatnonzero(owners[sources]>0)
    first=np.full(len(fallback),len(sources),np.int64);np.minimum.at(first,rows[active],active)
    second=np.full_like(first,len(sources));later=active[active>first[rows[active]]];np.minimum.at(second,rows[later],later)
    result=fallback.copy();backup=fallback.copy();winner=np.full(len(fallback),-1,np.int64)
    has=first<len(sources);winner[has]=sources[first[has]];result[has]=owners[winner[has]]
    has_second=second<len(sources);backup[has_second]=owners[sources[second[has_second]]]
    return result,winner,backup


def hard_readout(adjacency,accepted_owners):
    return first_two_readouts(adjacency,accepted_owners)[0]


def slice_readout(adjacency,start,end):
    offsets,sources,fallback,n=validate(adjacency)
    if not 0<=start<=end<=len(fallback):raise ValueError('Invalid query slice')
    lo,hi=offsets[start],offsets[end]
    return dict(offsets=offsets[start:end+1]-lo,sources=sources[lo:hi],fallback=fallback[start:end],candidate_count=np.asarray(n))


def expected_labels(adjacency,probabilities,owner_ids,label_count=None):
    """Differentiable exact label marginals under independent source choices.

    P[source,slot] is categorical object mass; 1-sum(P[source]) is rejection.
    Correlations between query locations sharing a source are not modeled by
    marginal output; sum statistics are exact expectations but their ratio is
    only a soft-IoU surrogate, not expected IoU. Call with bounded query chunks.
    """
    import torch
    offsets,sources,fallback,n=validate(adjacency);p=probabilities;owners=np.asarray(owner_ids,np.int64)
    if p.ndim!=2 or p.shape[0]!=n or owners.shape!=tuple(p.shape):raise ValueError('Aligned categorical source probabilities required')
    detached=p.detach()
    if not torch.isfinite(detached).all() or torch.any(detached<0) or torch.any(detached.sum(1)>1+1e-7):raise ValueError('Categorical masses must be finite, nonnegative and sum<=1')
    if np.any(owners<0):raise ValueError('Invalid owners')
    if torch.any(detached[torch.as_tensor(owners==0,device=p.device)]!=0):raise ValueError('Owner0 is rejection, never an accepted spatial null anchor')
    ordered=np.sort(owners,axis=1)
    if np.any((ordered[:,1:]==ordered[:,:-1])&(ordered[:,1:]>0)):raise ValueError('Each source must have distinct candidate objects')
    maximum=max(int(owners.max(initial=0)),int(fallback.max(initial=0)))
    labels=maximum+1 if label_count is None else int(label_count)
    if labels<=maximum:raise ValueError('Label dimension too small')
    lengths=np.diff(offsets);width=int(lengths.max(initial=0));nr=len(fallback)
    out=p.new_zeros((nr,labels))+p.sum()*0
    fallback_t=torch.as_tensor(fallback,device=p.device,dtype=torch.long)
    if not width:
        return out.scatter_add(1,fallback_t[:,None],p.new_ones((nr,1)))
    rows=np.repeat(np.arange(nr),lengths);slots=np.arange(len(sources))-np.repeat(offsets[:-1],lengths)
    padded=np.zeros((nr,width),np.int64);mask=np.zeros((nr,width),bool);padded[rows,slots]=sources;mask[rows,slots]=True
    indices=torch.as_tensor(padded,device=p.device);valid=torch.as_tensor(mask,device=p.device)
    mass=p[indices]*valid[:,:,None];reject=1-mass.sum(2)
    prefix=torch.cumprod(reject,dim=1);survival=torch.cat([p.new_ones((nr,1)),prefix[:,:-1]],dim=1)
    weighted=mass*survival[:,:,None]
    target=torch.as_tensor(owners,device=p.device)[indices]
    out=out.scatter_add(1,target.reshape(nr,-1),weighted.reshape(nr,-1))
    return out.scatter_add(1,fallback_t[:,None],prefix[:,-1,None])
