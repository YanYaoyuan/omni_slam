"""Spatial query-memory IDs; never replace, average, or label measured geometry."""
import numpy as np


def query_cell_ids(xyz,cell_m=.2):
    """Number cells by first source occurrence, preserving prefix causality.

Cell boundaries are a fixed query-resolution heuristic, not object boundaries.
This avoids equating a new LiDAR sample on an existing surface with a new region.
"""
    xyz=np.asarray(xyz)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all() or not np.isfinite(cell_m) or cell_m<=0:
        raise ValueError('Finite measured coordinates and positive cell size required')
    if not len(xyz):return np.empty(0,np.int64)
    keys=np.floor(xyz/cell_m).astype(np.int64)
    _,first,inverse=np.unique(keys,axis=0,return_index=True,return_inverse=True)
    order=np.argsort(first);causal=np.empty(len(first),np.int64);causal[order]=np.arange(len(first))
    return causal[inverse]
