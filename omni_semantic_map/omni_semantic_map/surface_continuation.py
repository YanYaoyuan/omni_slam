"""Rigid-invariant local geometry for evidence-supervised surface transport.

Unknown anchor ownership is excluded from supervision, never made background.
Features contain no instance IDs, absolute coordinates, image labels or GT.
The learned boundary score is an affinity, not a calibrated probability.
"""
import numpy as np
from scipy.sparse import csr_matrix

FEATURES=('log_length_over_cell','normal_cosine','normal_step_min','normal_step_max',
          'planarity_min','planarity_max','scattering_min','scattering_max',
          'linearity_min','linearity_max','log_degree_min','log_degree_max',
          'both_normals_valid','any_normal_valid')


def pure_anchor_labels(node_count,seed_nodes,seed_owners):
    nodes=np.asarray(seed_nodes,np.int64);owners=np.asarray(seed_owners,np.int64)
    if nodes.shape!=owners.shape or nodes.ndim!=1 or (nodes<0).any() or (nodes>=node_count).any() or (owners<0).any():
        raise ValueError('Aligned valid anchor labels required')
    pairs=np.unique(np.column_stack([nodes,owners]),axis=0)
    unique,first,count=np.unique(pairs[:,0],return_index=True,return_counts=True)
    result=np.full(node_count,-1,np.int64);keep=count==1
    result[unique[keep]]=pairs[first[keep],1]
    return result


def geometry_features(centers,graph):
    xyz=np.asarray(centers,np.float64);n=len(xyz);g=graph.tocsr(copy=True)
    if xyz.shape!=(n,3) or g.shape!=(n,n) or not np.isfinite(xyz).all():raise ValueError('Finite aligned geometry required')
    g.setdiag(0);g.eliminate_zeros();g.data[:]=1;g=g.maximum(g.T);g.sort_indices()
    degree=np.diff(g.indptr);g=g+csr_matrix((np.ones(n),(np.arange(n),np.arange(n))),shape=(n,n))
    # Center before second moments to avoid world-origin-dependent cancellation.
    centered=xyz-xyz.mean(0) if n else xyz.copy();count=degree+1
    mean=(g@centered)/count[:,None]
    outer=(centered[:,:,None]*centered[:,None,:]).reshape(n,9)
    covariance=((g@outer)/count[:,None]).reshape(n,3,3)-mean[:,:,None]*mean[:,None,:]
    values,vectors=np.linalg.eigh(covariance);values=np.maximum(values,0);normal=vectors[:,:,0]
    valid=(degree>=3)&(values[:,1]>1e-12)
    normal[~valid]=0
    largest=np.maximum(values[:,2],1e-12);trace=np.maximum(values.sum(1),1e-12)
    planarity=(values[:,1]-values[:,0])/largest;scattering=values[:,0]/trace;linearity=(values[:,2]-values[:,1])/largest
    g.setdiag(0);g.eliminate_zeros();left,right=g.nonzero();keep=left<right;left,right=left[keep],right[keep]
    delta=xyz[right]-xyz[left];length=np.linalg.norm(delta,axis=1)
    if (length<=0).any():raise ValueError('Distinct geometric endpoints required')
    unit=delta/length[:,None]
    ca=np.abs(np.einsum('ij,ij->i',normal[left],unit));cb=np.abs(np.einsum('ij,ij->i',normal[right],unit))
    cosine=np.abs(np.einsum('ij,ij->i',normal[left],normal[right]))
    columns=[np.log1p(length/.2),cosine,np.minimum(ca,cb),np.maximum(ca,cb)]
    for value in [planarity,scattering,linearity,np.log1p(degree)]:columns.extend([np.minimum(value[left],value[right]),np.maximum(value[left],value[right])])
    columns.extend([valid[left]&valid[right],valid[left]|valid[right]])
    features=np.column_stack(columns).astype(np.float32)
    assert features.shape==(len(left),len(FEATURES)) and np.isfinite(features).all()
    return dict(left=left.astype(np.int64),right=right.astype(np.int64),length=length,features=features,normal_valid=valid,degree=degree)


def affinity_length(length,affinity):
    length=np.asarray(length,np.float64);a=np.asarray(affinity,np.float64)
    if length.shape!=a.shape or (length<=0).any() or not np.isfinite(length).all() or not np.isfinite(a).all() or ((a<0)|(a>1)).any():
        raise ValueError('Positive lengths and affinities in [0,1] required')
    # Fixed continuous penalty; no hard edge deletion or learned threshold.
    return length*(1.-np.log(np.maximum(a,1e-6)))
