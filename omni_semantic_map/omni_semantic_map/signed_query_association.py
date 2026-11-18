"""View-conditioned pair evidence and signed object grouping.

Positive and negative pair costs compete over complete groups. Missing common
observations contribute zero cost, never a negative observation. Standard
signed clustering is not itself claimed novel; the experiment concerns raw
query evidence changing actual object identities and measured geometry.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

GEOMETRY_FEATURES=('cell_iou','cell_overlap_min','log_min_cells','log_max_cells',
    'log_relative_center_distance','log_size_ratio','min_native_score','max_native_score',
    'same_seed_frame','old_nodes','is_car')
VIEW_FEATURES=('log_common_measured_views','log_common_queries','same_query_fraction',
    'mean_query_mask_iou','max_query_mask_iou','mean_min_agreement',
    'mean_query_iou_agreement','separate_strong_fraction','mean_min_score',
    'common_query_fraction','mean_min_visible_fraction')
FEATURES=GEOMETRY_FEATURES+VIEW_FEATURES


def pair_features(left,right,geometry,readout,query_overlap):
    """Symmetric features for pairs; query_overlap is pair-by-frame.

    Geometry contains GT-free cell overlap, counts, centers, RMS shape sizes,
    native seed score/class, seed frame (-1 for old persistent objects) and an
    old-object flag. Readout includes exact measured visibility and selections.
    """
    left=np.asarray(left,np.int64);right=np.asarray(right,np.int64)
    if left.shape!=right.shape or left.ndim!=1:raise ValueError('Aligned pairs required')
    a=geometry['cells'][left];b=geometry['cells'][right];inter=geometry['intersections']
    lo=np.minimum(a,b);hi=np.maximum(a,b);distance=np.linalg.norm(geometry['centers'][left]-geometry['centers'][right],axis=1)
    sizes=geometry['rms'][left]+geometry['rms'][right]+.05
    score=geometry['scores'];seed=geometry['seed_frames'];old=geometry['old']
    geom=np.column_stack((inter/np.maximum(1,a+b-inter),inter/np.maximum(1,lo),np.log1p(lo),np.log1p(hi),np.log1p(distance/sizes),np.log1p(hi/np.maximum(1,lo)),np.minimum(score[left],score[right]),np.maximum(score[left],score[right]),(seed[left]==seed[right])&(seed[left]>=0),old[left]+old[right],geometry['semantic_ids'][left]==26))
    frames=readout['witness_frames'];va=readout['visible_cells'][left];vb=readout['visible_cells'][right]
    common=(va>=16)&(vb>=16)&(frames[None,:]!=seed[left,None])&(frames[None,:]!=seed[right,None])
    qa=readout['query_ids'][left];qb=readout['query_ids'][right];matched=common&(qa>=0)&(qb>=0)
    overlap=np.where(matched,query_overlap,0.);agreement=np.where(matched,np.minimum(readout['agreement'][left],readout['agreement'][right]),0.)
    confidence=np.where(matched,np.minimum(readout['scores'][left],readout['scores'][right]),0.)
    denominator=np.maximum(1,common.sum(1));fraction=np.minimum(va/np.maximum(1,a[:,None]),vb/np.maximum(1,b[:,None]))
    views=np.column_stack((np.log1p(common.sum(1)),np.log1p(matched.sum(1)),((qa==qb)&matched).sum(1)/denominator,
        overlap.sum(1)/denominator,overlap.max(1,initial=0),agreement.sum(1)/denominator,
        (overlap*agreement).sum(1)/denominator,((overlap<.1)&(agreement>.5)&matched).sum(1)/denominator,
        confidence.sum(1)/denominator,matched.sum(1)/denominator,np.where(common,fraction,0).sum(1)/denominator))
    x=np.column_stack((geom,views)).astype(np.float64)
    if not np.isfinite(x).all():raise ValueError('Nonfinite pair evidence')
    return x,common.any(1)


def positive_components(count,left,right,positive):
    left=np.asarray(left)[positive];right=np.asarray(right)[positive]
    graph=csr_matrix((np.ones(len(left)),(left,right)),shape=(count,count))
    _,labels=connected_components(graph,directed=False)
    return [np.flatnonzero(labels==k).tolist() for k in range(int(labels.max())+1)] if count else []


def signed_groups(count,left,right,weights):
    """Greedy positive-gain agglomeration for sum of within-group pair costs.

    All known cross-group costs are added, so one positive bridge does not
    automatically erase negative evidence between the groups. Unlisted pairs
    have zero cost. Stable original-index ties; no fitted merge threshold.
    """
    left=np.asarray(left,np.int64);right=np.asarray(right,np.int64);weights=np.asarray(weights,float)
    if not (left.shape==right.shape==weights.shape) or not np.isfinite(weights).all():raise ValueError('Aligned finite edge costs required')
    if np.any(left<0) or np.any(right>=count) or np.any(left>=right):raise ValueError('Unique undirected i<j pairs required')
    if len(np.unique(left*count+right))!=len(left):raise ValueError('Duplicate edges')
    matrix=np.zeros((count,count),float);matrix[left,right]=weights;matrix[right,left]=weights
    active=np.ones(count,bool);groups={i:[i] for i in range(count)};merges=[]
    upper=np.triu(np.ones((count,count),bool),1)
    while active.sum()>1:
        valid=upper&active[:,None]&active[None,:];gains=np.where(valid,matrix,-np.inf)
        index=int(np.argmax(gains));i,j=divmod(index,count);gain=float(gains[i,j])
        if gain<=0:break
        merges.append(dict(left=list(groups[i]),right=list(groups[j]),gain=gain))
        groups[i].extend(groups.pop(j));groups[i].sort();matrix[i,:]+=matrix[j,:];matrix[:,i]=matrix[i,:];matrix[i,i]=0.;matrix[j,:]=0.;matrix[:,j]=0.;active[j]=False
    return [groups[i] for i in sorted(groups)],merges
