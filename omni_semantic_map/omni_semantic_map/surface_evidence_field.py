"""Joint measured-surface ownership with explicit unknown observations.

This is a candidate inference primitive, not an established method result.
Graph-regularized label propagation is prior art; the experiment must isolate
whether view-normalized evidence and joint surface ownership improve maps.
No geometry, object IDs, confidence ranking or unseen surface is synthesized.
"""
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg
from scipy.spatial import cKDTree


def surface_graph(points, radius=.4, scale=.2):
    """Fixed physical neighbors, with an optional unoriented surface kernel.

    Eight nearest neighbors within radius, symmetrized; normals use up to16
    local neighbors. Eigenvalue planarity suppresses arbitrary normals on
    lines/isotropic regions. Return BOTH Euclidean and surface weights for
    a shared-graph control. Coordinates remain input actual measurements.
    """
    xyz=np.asarray(points,float)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all() or len(xyz)<2:
        raise ValueError('At least two finite physical points required')
    if not np.isfinite(radius+scale) or radius<=0 or scale<=0:raise ValueError('Positive physical scales required')
    distances,neighbors=cKDTree(xyz).query(xyz,k=min(17,len(xyz)),distance_upper_bound=radius,workers=1)
    distances,neighbors=distances[:,1:],neighbors[:,1:];valid=np.isfinite(distances)
    delta=xyz[np.minimum(neighbors,len(xyz)-1)]-xyz[:,None];delta[~valid]=0
    count=valid.sum(1);mean=delta.sum(1)/np.maximum(count[:,None],1)
    centered=(delta-mean[:,None])*valid[:,:,None];cov=np.einsum('nki,nkj->nij',centered,centered)/np.maximum(count[:,None,None],1)
    values,vectors=np.linalg.eigh(cov);normals=vectors[:,:,0]
    planarity=np.clip((values[:,1]-values[:,0])/np.maximum(values[:,2],1e-12),0,1);planarity[count<3]=0
    row,slot=np.where(valid[:,:8]);edges=np.unique(np.sort(np.column_stack([row,neighbors[row,slot]]),axis=1),axis=0)
    a,b=edges.T;degree=np.bincount(edges.ravel(),minlength=len(xyz));diff=xyz[a]-xyz[b]
    euclidean=np.exp(-np.sum(diff**2,axis=1)/(2*scale**2))/np.sqrt(degree[a]*degree[b])
    orientation=np.abs(np.einsum('ij,ij->i',normals[a],normals[b]))**2
    plane=(np.einsum('ij,ij->i',diff,normals[a])**2+np.einsum('ij,ij->i',diff,normals[b])**2)/(2*scale**2)
    reliable=planarity[a]*planarity[b];surface=euclidean*((1-reliable)+reliable*orientation*np.exp(-plane))
    return edges,euclidean,surface,dict(normals=normals,planarity=planarity)


def aggregate_views(point_ids, frame_ids, object_ids, support, visible_pairs,
                    num_points, num_objects):
    """Each physically observed point/frame contributes ONE categorical term.

    Object labels are 1..K; channel0 is null. visible_pairs explicitly lists
    (point,frame) observations with measured visibility. Unobserved pairs add
    neither positive nor negative evidence. Within each pair/object, duplicate
    or overlapping proposals contribute their maximum support, never a sum.
    Residual null support is 1-max(object support), then all channels normalize.
    Return mean categorical evidence and number of distinct measured views.
    """
    p,f,o=[np.asarray(x) for x in [point_ids,frame_ids,object_ids]]
    v=np.asarray(visible_pairs);s=np.asarray(support,float)
    if any(x.ndim!=1 or not np.issubdtype(x.dtype,np.integer) for x in [p,f,o]):
        raise ValueError('Integer point/frame/object vectors required')
    if not(p.shape==f.shape==o.shape==s.shape) or not np.isfinite(s).all() or np.any((s<0)|(s>1)):
        raise ValueError('Aligned finite support in [0,1] required')
    if v.ndim!=2 or v.shape[1]!=2 or not np.issubdtype(v.dtype,np.integer):
        raise ValueError('Explicit integer point/frame visibility pairs required')
    if num_points<0 or num_objects<1 or np.any((p<0)|(p>=num_points)) or np.any((o<1)|(o>num_objects)) or np.any(f<0):
        raise ValueError('Invalid point/frame/object domain')
    if np.any((v[:,0]<0)|(v[:,0]>=num_points)) or np.any(v[:,1]<0):raise ValueError('Invalid visibility domain')
    v=np.unique(v,axis=0);lookup={tuple(row):i for i,row in enumerate(v)}
    try:index=np.asarray([lookup[(a,b)] for a,b in zip(p,f)],np.int64)
    except KeyError as error:raise ValueError('Support without measured visibility') from error
    rows=np.zeros((len(v),num_objects+1),float)
    np.maximum.at(rows,(index,o),s)
    rows[:,0]=1-rows[:,1:].max(axis=1)
    rows/=rows.sum(axis=1,keepdims=True)
    sums=np.zeros((num_points,num_objects+1),float);np.add.at(sums,v[:,0],rows)
    count=np.bincount(v[:,0],minlength=num_points)
    means=np.divide(sums,count[:,None],out=np.zeros_like(sums),where=count[:,None]>0)
    means[count==0,0]=1
    return means,count


def solve_field(unary, reliability, edges, weights, strength=1., tolerance=1e-8):
    """Solve a convex surface field with no implicit evidence at missing nodes.

    min .5 sum_i r_i ||P_i-U_i||² + .5 strength sum_(i,j) w_ij ||P_i-P_j||².
    U is categorical (including explicit null). Evidence-free connected
    components remain null. Symmetric Laplacian + nonnegative fidelity gives
    nonnegative unit-sum P mathematically; numerical residual/simplex checks
    guard the solver. No hard threshold or object decision is chosen here.
    The caller must separately specify map export and acceptance rules.
    """
    u=np.asarray(unary,float);r=np.asarray(reliability,float);e=np.asarray(edges);w=np.asarray(weights,float)
    if u.ndim!=2 or u.shape[1]<2 or r.shape!=(len(u),) or not np.isfinite(u).all() or np.any(u<0) or not np.allclose(u.sum(1),1,atol=1e-10):
        raise ValueError('Categorical unary and aligned reliability required')
    if not np.isfinite(r).all() or np.any(r<0):raise ValueError('Nonnegative finite reliability required')
    if e.ndim!=2 or e.shape[1]!=2 or not np.issubdtype(e.dtype,np.integer) or w.shape!=(len(e),) or not np.isfinite(w).all() or np.any(w<0):
        raise ValueError('Integer edges and nonnegative finite weights required')
    if np.any((e<0)|(e>=len(u))) or np.any(e[:,0]==e[:,1]):raise ValueError('Invalid graph edge')
    if not np.isfinite(strength) or strength<0 or not np.isfinite(tolerance) or tolerance<=0:raise ValueError('Invalid solve parameters')
    # Duplicate/reversed input edges denote the same physical adjacency.
    canonical=np.sort(e,axis=1);unique,inverse=np.unique(canonical,axis=0,return_inverse=True)
    ew=np.zeros(len(unique));np.maximum.at(ew,inverse,w);positive=ew>0;unique,ew=unique[positive],ew[positive]*strength
    a,b=unique.T if len(unique) else (np.empty(0,int),np.empty(0,int))
    graph=coo_matrix((np.r_[ew,ew],(np.r_[a,b],np.r_[b,a])),shape=(len(u),len(u))).tocsr();graph.eliminate_zeros()
    _,components=connected_components(graph,directed=False)
    evidenced=np.unique(components[r>0]);active=np.isin(components,evidenced)
    out=np.zeros_like(u);out[:,0]=1
    if not active.any():return out,dict(active_points=0,residual=0.,energy=0.)
    lap=diags(np.asarray(graph.sum(1)).ravel())-graph
    matrix=(lap+diags(r))[active][:,active].tocsr();rhs=(r[:,None]*u)[active]
    solved=np.zeros_like(rhs);residual=0.;preconditioner=diags(1/matrix.diagonal())
    for j in range(u.shape[1]):
        if not rhs[:,j].any():continue
        solved[:,j],status=cg(matrix,rhs[:,j],M=preconditioner,tol=tolerance,atol=0.,maxiter=2000)
        rel=np.linalg.norm(matrix@solved[:,j]-rhs[:,j])/max(np.linalg.norm(rhs[:,j]),1e-30)
        residual=max(residual,float(rel))
        if status!=0 or rel>max(10*tolerance,1e-10):raise RuntimeError(f'Surface solve did not converge: {status}, residual={rel}')
    if solved.min(initial=0)<-max(100*tolerance,1e-6) or not np.allclose(solved.sum(1),1,atol=max(100*tolerance,1e-6)):
        raise RuntimeError('Categorical solution violated maximum principle')
    solved=np.maximum(solved,0);solved/=solved.sum(1,keepdims=True);out[active]=solved
    energy=.5*float(np.sum(r[:,None]*(out-u)**2))+.5*float(np.sum(ew[:,None]*(out[a]-out[b])**2))
    return out,dict(active_points=int(active.sum()),residual=residual,energy=energy)
