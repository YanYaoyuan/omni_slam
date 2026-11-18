"""Whole-map coverage diagnostics and provenance for existing DINO tokens.

GT is permitted ONLY in offline diagnostics/supervision. Coverage ceilings are
geometric necessary conditions on the fixed observed domain, not mapper scores.
"""
import numpy as np
from scipy.spatial import cKDTree


def covered_by(points, vertices, radius=.05):
    points,vertices=np.asarray(points),np.asarray(vertices)
    if points.shape!=(len(points),3) or vertices.shape!=(len(vertices),3):
        raise ValueError('expected xyz matrices')
    if not np.isfinite(points).all() or not np.isfinite(vertices).all():
        raise ValueError('nonfinite coordinates')
    if not len(points):return np.zeros(len(vertices),bool)
    return cKDTree(points).query(vertices,workers=1)[0]<=radius


def object_coverage(truth, coverage):
    truth,coverage=np.asarray(truth),np.asarray(coverage)
    if truth.shape!=coverage.shape or truth.ndim!=1:
        raise ValueError('incompatible fixed domain')
    ids,total=np.unique(truth[truth>0],return_counts=True)
    result=[]
    for gt,n in zip(ids,total):
        covered=int(coverage[truth==gt].sum())
        result.append(dict(gt=int(gt),voxels=int(n),covered=covered,
                           fraction=covered/int(n),can_exceed_iou_half=bool(2*covered>n)))
    return result


def token_groups(world_points,world_to_camera,K,width,height,grid_width,grid_height,max_tokens=24):
    """Reproduce livo_dino_support.pool's exact selected-bin order WITHOUT features.

    Returns source support-row indices for each cached local token. No category,
    GT or learned feature participates in this spatial provenance recovery.
    """
    points=np.asarray(world_points);T=np.asarray(world_to_camera)
    cam=points@T[:3,:3].T+T[:3,3]
    ids=np.flatnonzero(np.isfinite(cam).all(1)&(cam[:,2]>.2))
    pix=cam[ids]@np.asarray(K).T;uv=pix[:,:2]/pix[:,2,None]
    inside=(uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
    ids=ids[inside];uv=uv[inside]
    if not len(ids):return np.empty(0,np.int64),[]
    ij=np.floor(uv*np.array([grid_width/width,grid_height/height])).astype(np.int64)
    bins=ij[:,1]*grid_width+ij[:,0];unique=np.unique(bins)
    selected=np.unique(np.linspace(0,len(unique)-1,min(max_tokens,len(unique))).astype(int))
    chosen=unique[selected]
    return chosen,[ids[bins==b] for b in chosen]


def label_distribution(labels):
    """Full histogram: -1 unknown; 0 known background; >0 instance. No dropping."""
    labels=np.asarray(labels)
    if labels.ndim!=1 or not np.issubdtype(labels.dtype,np.integer) or (labels < -1).any():
        raise ValueError('invalid supervision labels')
    ids,counts=np.unique(labels,return_counts=True)
    return dict(ids=ids.tolist(),counts=counts.tolist(),total=int(len(labels)),
                valid=int((labels>=0).sum()),foreground=int((labels>0).sum()),
                foreground_instances=int((ids>0).sum()))
