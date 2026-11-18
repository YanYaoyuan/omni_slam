"""Spatial support validation for sparsely sampled measured object surfaces."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def largest_measured_component(points, radius, minimum_points):
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
        raise ValueError('expected finite Nx3 measured supports')
    if not np.isfinite(radius) or radius<=0 or minimum_points<1:
        raise ValueError('invalid component parameters')
    n=len(points)
    if n<minimum_points:return np.empty(0,np.int64)
    edges=cKDTree(points).query_pairs(radius,output_type='ndarray')
    graph=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(n,n))
    _,labels=connected_components(graph,directed=False)
    counts=np.bincount(labels);label=int(counts.argmax())
    if counts[label]<minimum_points:return np.empty(0,np.int64)
    return np.flatnonzero(labels==label)
