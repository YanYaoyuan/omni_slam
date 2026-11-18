"""Sparse, anchored surface partition with accumulated signed boundary costs.

This is a candidate formation operator, not a claim that signed clustering is
novel. Old anchors are must-linked; different old identities cannot merge.
Unanchored regions can become new objects in the caller's common exporter.
"""
import heapq
import numpy as np


def partition_surfaces(count, left, right, cost, anchors):
    """Greedily merge the pair with greatest positive *total* boundary cost.

    Costs on every known cross-region edge are summed, including negatives.
    Missing edges have zero cost. Stable integer node ordering resolves ties.
    Return one root per node plus merge gains; no dense N-by-N allocation.
    """
    left=np.asarray(left,np.int64);right=np.asarray(right,np.int64)
    cost=np.asarray(cost,float);anchors=np.asarray(anchors,np.int64)
    if anchors.shape!=(count,) or np.any(anchors<0):raise ValueError('Nonnegative aligned anchors required')
    if not (left.shape==right.shape==cost.shape) or left.ndim!=1 or not np.isfinite(cost).all():raise ValueError('Aligned finite edges required')
    if np.any(left<0) or np.any(right>=count) or np.any(left>=right):raise ValueError('Undirected i<j edges required')
    if len(np.unique(left*count+right))!=len(left):raise ValueError('Duplicate edges')
    parent=np.arange(count);label=anchors.copy();members=np.ones(count,np.int64)
    first={}
    for i,a in enumerate(anchors):
        if a:
            if int(a) in first:
                root=first[int(a)];parent[i]=root;members[root]+=1;members[i]=0
            else:first[int(a)]=i
    graph=[{} for _ in range(count)]
    for u,v,w in zip(parent[left],parent[right],cost):
        u,v=int(u),int(v)
        if u==v:continue
        graph[u][v]=graph[u].get(v,0.)+float(w)
        graph[v][u]=graph[v].get(u,0.)+float(w)
    heap=[];version=np.zeros(count,np.int64)
    def offer(a,b):
        if a>b:a,b=b,a
        w=graph[a].get(b,0.)
        if w>1e-12 and not(label[a] and label[b] and label[a]!=label[b]):
            heapq.heappush(heap,(-w,a,b,int(version[a]),int(version[b])))
    for a,g in enumerate(graph):
        for b in g:
            if a<b:offer(a,b)
    trace=[]
    while heap:
        neg,a,b,va,vb=heapq.heappop(heap)
        if parent[a]!=a or parent[b]!=b or version[a]!=va or version[b]!=vb:continue
        gain=graph[a].get(b,0.)
        if gain<=1e-12 or (label[a] and label[b] and label[a]!=label[b]):continue
        assert abs(gain+neg)<1e-8*max(1.,abs(gain))
        trace.append((a,b,gain,int(members[a]),int(members[b])))
        parent[b]=a;members[a]+=members[b];members[b]=0
        label[a]=max(label[a],label[b]);version[a]+=1;version[b]+=1
        graph[a].pop(b,None);graph[b].pop(a,None)
        for other,w in list(graph[b].items()):
            assert parent[other]==other
            value=graph[a].get(other,0.)+w
            graph[a][other]=value;graph[other][a]=value
            graph[other].pop(b,None)
        graph[b].clear()
        for other in graph[a]:offer(a,other)
    for i in range(count):
        j=i
        while parent[j]!=j:j=int(parent[j])
        while parent[i]!=i:
            k=int(parent[i]);parent[i]=j;i=k
    remaining=max((w for a,g in enumerate(graph) if parent[a]==a for b,w in g.items()
                   if a<b and not(label[a] and label[b] and label[a]!=label[b])),default=0.)
    assert remaining<=1e-12
    for a in np.unique(anchors[anchors>0]):assert len(np.unique(parent[anchors==a]))==1
    return parent,dict(merges=trace,objective_gain=float(sum(t[2] for t in trace)),maximum_eligible_remaining_gain=float(remaining))


def edge_view_cost(left_token,right_token,mask_iou):
    """Unknown tokens abstain. A valid pair contributes one bounded vote."""
    a=np.asarray(left_token,np.int64);b=np.asarray(right_token,np.int64)
    if a.shape!=b.shape:raise ValueError('Aligned tokens required')
    valid=(a>=0)&(b>=0)
    out=np.zeros(a.shape,float)
    out[valid]=2*np.asarray(mask_iou)[a[valid],b[valid]]-1
    return out,valid
