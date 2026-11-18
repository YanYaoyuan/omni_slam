"""Deterministic prompts from actually measured, predicted-object anchors.

No GT or mask confidence input. Spatially withheld anchors are never prompts.
"""
import numpy as np


def calibration_blocks(xyz):
    b=np.floor(np.asarray(xyz)/2.).astype(np.int64)
    return ((b[:,0]*73856093 ^ b[:,1]*19349663 ^ b[:,2]*83492791)&1)==0


def spread(xy, count):
    p=np.asarray(xy,float)
    if p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all():raise ValueError('Finite Nx2 pixels required')
    if not len(p):return np.empty(0,np.int64)
    first=int(np.argmin(np.square(p-p.mean(0)).sum(1)));selected=[first];distance=np.square(p-p[first]).sum(1)
    while len(selected)<min(count,len(p)):
        distance[selected]=-1;idx=int(np.argmax(distance));selected.append(idx);distance=np.minimum(distance,np.square(p-p[idx]).sum(1))
    return np.asarray(selected)


def prompts(source_ids,pixels,owners,cells,calibration,oid):
    ids=np.asarray(source_ids);xy=np.asarray(pixels);label=owners[ids];allowed=calibration[ids]
    def unique_cells(take):
        ix=np.flatnonzero(take);_,first=np.unique(cells[ids[ix]],return_index=True)
        return ix[np.sort(first)]
    own=unique_cells(allowed&(label==oid));foreign=unique_cells(allowed&(label>0)&(label!=oid))
    if len(own)<16:return dict(eligible=False,own_cells=len(own),source_ids=[],pixels=[],labels=[])
    positive=own[spread(xy[own],3)];near=np.argsort(np.square(xy[foreign]-xy[positive].mean(0)).sum(1),kind='stable')[:32]
    foreign=foreign[near];negative=foreign[spread(xy[foreign],3)];take=np.r_[positive,negative]
    return dict(eligible=True,own_cells=len(own),source_ids=ids[take].tolist(),pixels=xy[take].tolist(),labels=[1]*len(positive)+[0]*len(negative))
