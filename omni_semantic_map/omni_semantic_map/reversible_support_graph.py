"""Category-free, reversible partition of persistent measured-support cells.

A mask is evidence, not an indivisible identity. Same-mask votes attract;
repeated simultaneously observed different-mask boundaries can separate.
Unknown/occluded cells never cast a negative vote. This research prototype uses
fixed-pose cached supports; it does not claim SLAM pose-revision support.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SupportGraphConfig:
    cell_size: float = .05
    min_negative_frames: int = 2
    negative_positive_ratio: float = 2.
    dominance: float = .7
    max_repair_atoms: int = 100000


def partition_signed(vertices, positive_edges, weights, negative_edges, locked_groups=()):
    """Deterministic greedy signed partition with component-wide exclusions.

    Existing identity components are locked only in the no-revocation ablation.
    Negative edges internal to a lock are recorded by the caller, not hidden.
    """
    vertices=list(map(int,vertices));parent={v:v for v in vertices}
    members={v:{v} for v in vertices};forbidden={v:set() for v in vertices}
    for a,b in negative_edges:
        a,b=int(a),int(b);forbidden[a].add(b);forbidden[b].add(a)
    def root(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    def join(a,b,force=False):
        a,b=root(int(a)),root(int(b))
        if a==b:return
        if not force and (not forbidden[a].isdisjoint(members[b]) or not forbidden[b].isdisjoint(members[a])):return
        # Small-to-large membership sets, but public IDs are minimum vertex IDs.
        if len(members[a])<len(members[b]) or (len(members[a])==len(members[b]) and a>b):a,b=b,a
        parent[b]=a;members[a].update(members.pop(b));forbidden[a].update(forbidden.pop(b))
    for group in locked_groups:
        group=list(group)
        for v in group[1:]:join(group[0],v,True)
    edges=np.asarray(positive_edges,dtype=np.int64).reshape(-1,2)
    weights=np.asarray(weights)
    order=np.lexsort((edges[:,1],edges[:,0],-weights)) if len(edges) else []
    for i in order:join(*edges[i])
    aliases={r:min(v)+1 for r,v in members.items()}
    return np.array([aliases[root(v)] for v in vertices],dtype=np.int64)


def atomize(xyz,cell_size=.05):
    """A fixed grid defines atoms; no GT or data-dependent resolution fitting."""
    keys,inverse=np.unique(np.floor(np.asarray(xyz)/cell_size).astype(np.int64),axis=0,return_inverse=True)
    # Fixed 26-neighborhood. No edge is active until its endpoints are observed.
    packed=np.ascontiguousarray(keys).view([('x','<i8'),('y','<i8'),('z','<i8')]).reshape(-1)
    edges=[]
    for dx in [-1,0,1]:
        for dy in [-1,0,1]:
            for dz in [-1,0,1]:
                if (dx,dy,dz)<=(0,0,0):continue
                target=np.ascontiguousarray(keys+[dx,dy,dz]).view(packed.dtype).reshape(-1)
                idx=np.searchsorted(packed,target);safe=np.minimum(idx,len(keys)-1)
                found=(idx<len(keys))&(packed[safe]==target)
                edges.append(np.column_stack([np.flatnonzero(found),idx[found]]))
    return keys,inverse,np.concatenate(edges).astype(np.int32) if edges else np.empty((0,2),np.int32)


def frame_atom_labels(point_to_atom,point_ids,observation_ids,n_atoms,dominance=.7):
    """Exclusive point evidence; ambiguous cell ownership is unknown (-1)."""
    point_ids=np.asarray(point_ids,dtype=np.int64);observation_ids=np.asarray(observation_ids,dtype=np.int64)
    if point_ids.shape!=observation_ids.shape:raise ValueError('point/observation length mismatch')
    if len(np.unique(point_ids))!=len(point_ids):raise ValueError('a point was consumed twice in one frame')
    output=np.full(n_atoms,-1,np.int64)
    if not len(point_ids):return output
    pairs,counts=np.unique(np.column_stack([point_to_atom[point_ids],observation_ids]),axis=0,return_counts=True)
    totals=np.bincount(pairs[:,0],weights=counts,minlength=n_atoms)
    # Highest support count, then smallest observation ID; >=.7 makes winner unique.
    order=np.lexsort((pairs[:,1],-counts,pairs[:,0]));p=pairs[order];c=counts[order]
    first=np.r_[True,p[1:,0]!=p[:-1,0]];p=p[first];c=c[first]
    keep=c>=dominance*totals[p[:,0]];output[p[keep,0]]=p[keep,1]
    return output


class ReversibleSupportGraph:
    def __init__(self,n_atoms,edges,config=None,allow_revoke=True):
        self.config=config or SupportGraphConfig();self.n=n_atoms
        self.edges=np.asarray(edges,dtype=np.int32).reshape(-1,2)
        if len(self.edges) and ((self.edges<0).any() or (self.edges>=n_atoms).any()):raise ValueError('bad endpoint')
        self.positive=np.zeros(len(self.edges),np.int32);self.negative=np.zeros(len(self.edges),np.int32)
        self.labels=np.zeros(n_atoms,np.int64);self.seen=np.zeros(n_atoms,bool)
        self.allow_revoke=allow_revoke;self.frame=-1;self.revision=0;self.events=[]

    def blocked(self):
        return (self.negative>=self.config.min_negative_frames)&(self.negative>=self.config.negative_positive_ratio*self.positive)

    def update(self,frame,observation_by_atom):
        if frame<=self.frame:raise ValueError('frame IDs must strictly increase')
        obs=np.asarray(observation_by_atom,dtype=np.int64)
        if obs.shape!=(self.n,):raise ValueError('bad atom table')
        a,b=self.edges.T;known=(obs[a]>=0)&(obs[b]>=0);same=known&(obs[a]==obs[b]);different=known&~same
        positive=self.positive+same;negative=self.negative+different
        # Select all old components touched by actual evidence; unobserved cells do
        # not create a negative witness. Work outside these components is retained.
        touched=np.unique(np.r_[np.flatnonzero(obs>=0),self.edges[known].reshape(-1)])
        old_ids=np.unique(self.labels[touched]);old_ids=old_ids[old_ids>0]
        selected=np.isin(self.labels,old_ids);selected[touched]=True
        vertices=np.flatnonzero(selected)
        if len(vertices)>self.config.max_repair_atoms:raise RuntimeError('predeclared repair atom budget exceeded')
        active=(positive>0);blocked=(negative>=self.config.min_negative_frames)&(negative>=self.config.negative_positive_ratio*positive)
        inside=selected[a]&selected[b]
        positive_ids=np.flatnonzero(inside&active&~blocked);negative_ids=np.flatnonzero(inside&blocked)
        old=self.labels[vertices].copy();locks=[]
        if not self.allow_revoke:
            order=np.argsort(old,kind='stable');sorted_ids=old[order]
            for group in np.split(order,np.flatnonzero(np.diff(sorted_ids))+1):
                if len(group) and old[group[0]]>0:locks.append(vertices[group])
        weights=(positive[positive_ids]+.01)/(positive[positive_ids]+negative[positive_ids]+.01)
        labels=partition_signed(vertices,self.edges[positive_ids],weights,self.edges[negative_ids],locks)
        candidate=self.labels.copy();candidate[vertices]=labels
        violations=int(np.sum(blocked&(candidate[a]>0)&(candidate[a]==candidate[b])))
        if self.allow_revoke and violations:raise AssertionError('confirmed exclusion bypassed')
        transitions={}
        for previous in np.unique(old[old>0]):
            descendants=np.unique(labels[old==previous]);
            if len(descendants)>1:transitions[str(int(previous))]=descendants.tolist()
        # Commit only after every invariant has passed.
        self.positive=positive;self.negative=negative;self.labels=candidate
        self.seen|=obs>=0;self.frame=int(frame);self.revision+=1
        event=dict(frame=int(frame),revision=self.revision,repair_atoms=len(vertices),
            positive_witnesses=int(same.sum()),negative_witnesses=int(different.sum()),
            confirmed_exclusion_edges=int(blocked.sum()),remaining_exclusion_violations=violations,
            changed_atoms=int(np.sum(old!=labels)),split_lineage=transitions)
        self.events.append(event);return event

    def check_export(self,labels=None):
        labels=self.labels if labels is None else np.asarray(labels)
        if labels.shape!=(self.n,):raise ValueError('bad export labels')
        a,b=self.edges.T;violations=int(np.sum(self.blocked()&(labels[a]>0)&(labels[a]==labels[b])))
        if self.allow_revoke and violations:raise ValueError('export bypassed a confirmed exclusion')
        return violations
