"""Source-batch deletion certificates for measured-surface label transport.

This certifies a fixed graph's label stability, NOT semantic correctness,
independent views, robustness to deleting measurements, or new object birth.
"""
import ctypes
import hashlib
from pathlib import Path
import subprocess
import numpy as np


def native_library(directory):
    source=Path(__file__).with_suffix('.cpp')
    key=hashlib.sha256(source.read_bytes()).hexdigest()[:20]
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    target=directory/('transport_'+key+'.so')
    if not target.exists():
        subprocess.run(['g++','-std=c++17','-O3','-shared','-fPIC',str(source),'-o',str(target)],check=True)
    lib=ctypes.CDLL(str(target))
    ip=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    dp=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    lib.nearest_two_groups.argtypes=[ctypes.c_int64,ip,ip,dp,ctypes.c_int64,ip,ip,ip,dp]
    lib.nearest_two_groups.restype=ctypes.c_int
    return lib


def nearest_groups(graph, seed_nodes, seed_groups, library):
    graph=graph.tocsr(copy=True);graph.sum_duplicates();graph.sort_indices()
    n=graph.shape[0]
    if graph.shape!=(n,n) or not np.isfinite(graph.data).all() or (graph.data<0).any():
        raise ValueError('Finite square nonnegative graph required')
    nodes=np.asarray(seed_nodes,dtype=np.int64);groups=np.asarray(seed_groups,dtype=np.int64)
    if nodes.ndim!=1 or groups.shape!=nodes.shape or (nodes<0).any() or (nodes>=n).any() or (groups<0).any():
        raise ValueError('Valid nonnegative node/group seeds required')
    seeds=np.unique(np.column_stack([nodes,groups]),axis=0)
    ptr=np.ascontiguousarray(graph.indptr,dtype=np.int64);col=np.ascontiguousarray(graph.indices,dtype=np.int64)
    weight=np.ascontiguousarray(graph.data,dtype=np.float64)
    sn=np.ascontiguousarray(seeds[:,0]);sg=np.ascontiguousarray(seeds[:,1])
    answer=np.empty((n,2),np.int64);distance=np.empty((n,2),np.float64)
    status=library.nearest_two_groups(n,ptr,col,weight,len(sn),sn,sg,answer,distance)
    if status:raise RuntimeError('Native graph transport failed')
    return answer,distance


def nearest_owner(labels, distances, tolerance=1e-10):
    valid=(labels[:,0]>=0)&np.isfinite(distances[:,0])&(distances[:,0]<distances[:,1]-tolerance)
    return np.where(valid,labels[:,0],-1)


def certify(labels, distances, tokens, token_distances, token_owners, tolerance=1e-10):
    """Two distinct source batches of the winner beat every competing label.

    The second closest (owner,batch) distance bounds the winner's distance
    after any single batch removal. Competitors retain all seeds, giving a
    conservative certificate even when that removal also deletes their seeds.
    A finite second batch is required even in components without competitors.
    """
    owner=nearest_owner(labels,distances,tolerance)
    safe=np.maximum(tokens,0)
    tokowner=np.asarray(token_owners,dtype=np.int64)[safe]
    valid=(owner>=0)&(tokens[:,0]>=0)&(tokens[:,1]>=0)
    valid &= (tokowner[:,0]==owner)&(tokowner[:,1]==owner)
    valid &= np.isfinite(token_distances[:,1])&(token_distances[:,1]<distances[:,1]-tolerance)
    return np.where(valid,owner,-1)
