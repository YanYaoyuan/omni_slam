"""Recover discarded measured components with explicit cross-view witnesses.

This module never synthesizes points or changes masks. It computes the earliest
time at which historical support can be promoted. A caller replaying all revised
observations at window end is performing delayed refinement, not online SLAM.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def components(points,radius=.4):
    p=np.asarray(points,float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():raise ValueError('Finite Nx3 measured points required')
    if not np.isfinite(radius) or radius<=0:raise ValueError('Positive component radius required')
    if not len(p):return np.empty(0,np.int64)
    edge=cKDTree(p).query_pairs(radius,output_type='ndarray')
    graph=coo_matrix((np.ones(len(edge)),(edge[:,0],edge[:,1])),shape=(len(p),len(p)))
    return connected_components(graph,directed=False)[1]


def recover(observations,source_xyz,source_cells,features,*,radius=.4,similarity=.7,fraction=.5):
    """Return largest/raw/pointwise/component supports and promotion provenance.

    Observation: unique id, frame, semantic, raw source IDs, baseline source IDs.
    A cell is confirmed only by a different-frame, same-class observation whose
    normalized descriptor cosine is >= similarity. Repeated points/masks in one
    frame never constitute temporal confirmation. Component promotion needs
    ceil(fraction * distinct component cells) confirmed cells. Baseline points
    are never removed. Unknown observations do not supply negative evidence.

    The batch computation also returns confirmation times, so any prefix cutoff
    can be checked without treating future witnesses as available earlier.
    """
    xyz=np.asarray(source_xyz);cells=np.asarray(source_cells);f=np.asarray(features,np.float32)
    if len(cells)!=len(xyz) or cells.ndim!=1 or np.any(cells<0):raise ValueError('Invalid source-cell mapping')
    if not 0<fraction<=1 or not -1<=similarity<=1:raise ValueError('Invalid evidence parameters')
    ids=[int(o['id']) for o in observations]
    if len(ids)!=len(set(ids)) or any(i<0 or i>=len(f) for i in ids):raise ValueError('Invalid observation IDs')
    norm=np.linalg.norm(f,axis=1)
    if not np.isfinite(f).all() or np.any(norm[ids]<=0):raise ValueError('Finite nonzero descriptors required')
    unit=f/np.maximum(norm[:,None],1e-12);cosine=unit@unit.T
    by_id={int(o['id']):o for o in observations};cell_views={};prepared={}
    for o in observations:
        oid=int(o['id']);raw=np.asarray(o['raw'],np.int64);base=np.asarray(o['baseline'],np.int64)
        if len(np.unique(raw))!=len(raw) or np.any(raw<0) or np.any(raw>=len(xyz)):raise ValueError('Raw supports must be unique valid source IDs')
        if not np.isin(base,raw).all():raise ValueError('Baseline must be a subset of actual mask support')
        units=np.unique(cells[raw]);prepared[oid]=(raw,base,units)
        for cell in units:cell_views.setdefault(int(cell),[]).append(oid)
    confirmations={oid:{} for oid in ids};witnesses={oid:{} for oid in ids}
    for cell,views in cell_views.items():
        for oid in views:
            a=by_id[oid]
            candidates=[other for other in views if by_id[other]['frame']!=a['frame'] and by_id[other]['semantic']==a['semantic'] and cosine[oid,other]>=similarity]
            if candidates:
                other=min(candidates,key=lambda i:(by_id[i]['frame'],i))
                confirmations[oid][cell]=max(int(a['frame']),int(by_id[other]['frame']));witnesses[oid][cell]=other
    result={};events=[]
    for o in observations:
        oid=int(o['id']);raw,base,units=prepared[oid];labels=components(xyz[raw],radius)
        point_times=np.asarray([confirmations[oid].get(int(cells[i]),-1) for i in raw],np.int64)
        is_base=np.isin(raw,base);pointwise=is_base|(point_times>=0);component=is_base.copy();component_times=np.full(len(raw),-1,np.int64)
        for label in np.unique(labels):
            part=np.flatnonzero(labels==label);part_cells=np.unique(cells[raw[part]]);need=max(1,int(np.ceil(fraction*len(part_cells))))
            known=sorted((confirmations[oid][int(c)],int(c)) for c in part_cells if int(c) in confirmations[oid])
            if len(known)<need:continue
            time=int(known[need-1][0]);component[part]=True;component_times[part]=time
            added=raw[part[~is_base[part]]]
            if len(added):
                proof=known[:need]
                events.append(dict(observation_id=oid,source_frame=int(o['frame']),promotion_frame=time,
                    component=int(label),component_cells=len(part_cells),required_cells=need,
                    witness_cells=[c for _,c in proof],witness_observations=[witnesses[oid][c] for _,c in proof],
                    added_source_ids=added.tolist()))
        for times in [point_times,component_times]:times[is_base]=int(o['frame'])
        result[oid]=dict(largest=base.copy(),all=raw.copy(),pointwise=raw[pointwise],component=raw[component],
            pointwise_times=point_times[pointwise],component_times=component_times[component])
    return result,events
