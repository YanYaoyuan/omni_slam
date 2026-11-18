"""Resolution-aware frozen-token pooling on measured LIVO supports."""
import numpy as np


def unit(x):
    x=np.asarray(x,np.float32)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def pool(tokens,world_points,world_to_camera,K,width,height,max_tokens=24):
    h,w,dim=tokens.shape
    points=np.asarray(world_points);T=np.asarray(world_to_camera)
    cam=points@T[:3,:3].T+T[:3,3]
    valid=np.isfinite(cam).all(1)&(cam[:,2]>.2);ids=np.flatnonzero(valid)
    pix=cam[ids]@np.asarray(K).T;uv=pix[:,:2]/pix[:,2,None]
    inside=(uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
    ids=ids[inside];uv=uv[inside]
    if not len(ids):return np.zeros(dim,np.float32),np.empty((0,3),np.float32),np.empty((0,dim),np.float32)
    ij=np.floor(uv*np.array([w/width,h/height])).astype(np.int64)
    bins=ij[:,1]*w+ij[:,0];unique,counts=np.unique(bins,return_counts=True)
    features=unit(tokens.reshape(-1,dim)[unique]);descriptor=unit(np.average(features,axis=0,weights=counts))
    selected=np.unique(np.linspace(0,len(unique)-1,min(max_tokens,len(unique))).astype(int))
    anchors=np.array([np.median(points[ids[bins==unique[i]]],axis=0) for i in selected],np.float32)
    return descriptor,anchors,features[selected]
