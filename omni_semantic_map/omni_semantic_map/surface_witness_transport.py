"""Transport current measured mask evidence along locally compatible surfaces.

This is an experimental observation operator, not an ownership updater. A
negative vote only excludes this object's mask; it does not label background
or identify a replacement object. No measurements means unknown. Geometry
support is deliberately distinct from exact-point measured-depth visibility.
"""
import numpy as np
from scipy.spatial import cKDTree


def points(value):
    a=np.asarray(value,dtype=np.float64)
    if a.ndim!=2 or a.shape[1]!=3 or not np.isfinite(a).all():
        raise ValueError('Finite Nx3 coordinates required')
    return a


def local_planes(reference, queries, radius=.2, neighbours=16, minimum=6,
                 curvature=.05, linearity=.05, plane_distance=.05):
    """Local PCA; reject insufficient, line-like, curved and off-plane support.

    Normals have arbitrary sign. No object labels, image masks, or GT enter
    this calculation. Batched neighbourhoods bound temporary memory use.
    """
    ref,q=points(reference),points(queries)
    if not 3<=minimum<=neighbours or radius<=0 or plane_distance<=0:
        raise ValueError('Positive radius/distance and valid neighbour counts required')
    normal=np.zeros_like(q); valid=np.zeros(len(q),bool)
    if not len(ref) or not len(q): return normal,valid
    tree=cKDTree(ref)
    for start in range(0,len(q),4096):
        current=q[start:start+4096]
        distance,index=tree.query(current,k=neighbours,distance_upper_bound=radius,workers=1)
        present=np.isfinite(distance); count=present.sum(1)
        xyz=ref[np.minimum(index,len(ref)-1)]; mean=np.sum(xyz*present[:,:,None],axis=1)/np.maximum(count[:,None],1)
        centered=(xyz-mean[:,None,:])*present[:,:,None]
        covariance=np.einsum('nki,nkj->nij',centered,centered)/np.maximum(count[:,None,None],1)
        eigenvalues,eigenvectors=np.linalg.eigh(covariance); total=np.maximum(eigenvalues.sum(1),1e-30)
        n=eigenvectors[:,:,0]
        ok=(count>=minimum)&(eigenvalues[:,0]/total<=curvature)&(eigenvalues[:,1]/total>=linearity)
        ok &= np.abs(np.einsum('ij,ij->i',current-mean,n))<=plane_distance
        normal[start:start+len(current)]=n; valid[start:start+len(current)]=ok
    return normal,valid


def surface_edges(target_xyz, target_normal, target_valid, witness_xyz,
                  witness_normal, witness_valid, radius=.2,
                  plane_distance=.05, normal_cosine=np.sqrt(3.)/2.):
    """Bipartite edges from old geometry to CURRENT measured source witnesses.

    Require normal agreement and both point-to-plane residuals within bounds.
    Radius only proposes neighbours; it never substitutes for surface tests.
    Output is ordered by target then original witness index, without duplicates.
    """
    target,witness=points(target_xyz),points(witness_xyz)
    tn,wn=points(target_normal),points(witness_normal)
    tv,wv=np.asarray(target_valid,bool),np.asarray(witness_valid,bool)
    if tn.shape!=target.shape or wn.shape!=witness.shape or tv.shape!=(len(target),) or wv.shape!=(len(witness),):
        raise ValueError('Aligned normals and validity required')
    if radius<=0 or plane_distance<=0 or not 0<=normal_cosine<=1:
        raise ValueError('Invalid geometric thresholds')
    if not np.allclose(np.linalg.norm(tn[tv],axis=1),1.) or not np.allclose(np.linalg.norm(wn[wv],axis=1),1.):
        raise ValueError('Valid normals must be unit length')
    ti,wi=np.flatnonzero(tv),np.flatnonzero(wv)
    empty=np.empty(0,np.int64)
    if not len(ti) or not len(wi): return empty,empty.copy()
    nearby=cKDTree(witness[wi]).query_ball_point(target[ti],radius,workers=1,return_sorted=True)
    lengths=np.array([len(a) for a in nearby]); row=np.repeat(ti,lengths)
    if not len(row): return empty,empty.copy()
    col=wi[np.concatenate([np.asarray(a,np.int64) for a in nearby])]
    delta=witness[col]-target[row]
    keep=np.abs(np.einsum('ij,ij->i',tn[row],wn[col]))>=normal_cosine
    keep &= np.abs(np.einsum('ij,ij->i',tn[row],delta))<=plane_distance
    keep &= np.abs(np.einsum('ij,ij->i',wn[col],delta))<=plane_distance
    return row[keep],col[keep]


def frame_votes(target_count, rows, witnesses, inside):
    """One ternary vote per target/frame: support(+1), exclusion(-1), unknown(0).

    Mixed compatible witnesses abstain; additional points in a frame never
    create extra temporal confirmations. The caller must deduplicate frames.
    """
    rows,witnesses=np.asarray(rows,np.int64),np.asarray(witnesses,np.int64)
    inside=np.asarray(inside,bool)
    if rows.ndim!=1 or rows.shape!=witnesses.shape or inside.ndim!=1 or target_count<0:
        raise ValueError('Aligned one-dimensional edge and mask arrays required')
    if np.any(rows<0) or np.any(rows>=target_count) or np.any(witnesses<0) or np.any(witnesses>=len(inside)):
        raise ValueError('Edge index out of bounds')
    count=np.bincount(rows,minlength=target_count); positive=np.bincount(rows,weights=inside[witnesses],minlength=target_count)
    vote=np.zeros(target_count,np.int8)
    vote[(count>0)&(positive==count)]=1
    vote[(count>0)&(positive==0)]=-1
    return vote
