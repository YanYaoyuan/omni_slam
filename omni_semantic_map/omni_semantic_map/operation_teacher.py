"""TRAINING-ONLY exact counterfactual object-PQ rewards on a fixed support.

Never imported by operation_revision, and never given target-scene GT in fit.
"""
import numpy as np


class OperationTeacher:
    def __init__(self,atom_counts,gt_counts):
        self.atoms=np.asarray(atom_counts,dtype=np.int64);self.gt=np.asarray(gt_counts,dtype=np.int64)
        if self.atoms.ndim!=2 or self.atoms.shape[1]!=len(self.gt) or np.any(self.atoms<0) or np.any(self.gt<0):raise ValueError('Invalid contingency')
        if np.any(self.atoms.sum(0)>self.gt):raise ValueError('Overlapping atom ownership')
        self.ng=int(np.count_nonzero(self.gt[1:]));self.cached_labels=None

    def contribution(self,vector):
        size=int(vector.sum())
        if not size:return 0.,0,0
        union=self.gt[1:]+size-vector[1:];iou=np.divide(vector[1:],union,out=np.zeros(len(union),float),where=union>0)
        match=iou>.5
        if match.sum()>1:raise ValueError('Exclusive prediction cannot match multiple objects')
        return float(iou[match].sum()),int(match.sum()),1

    def state(self,labels):
        labels=np.asarray(labels,dtype=np.int64)
        if labels.shape!=(len(self.atoms),) or np.any(labels<=0):raise ValueError('Invalid atom identities')
        if self.cached_labels is not None and np.array_equal(labels,self.cached_labels):return
        self.vectors={int(oid):self.atoms[labels==oid].sum(0) for oid in np.unique(labels)}
        c=[self.contribution(v) for v in self.vectors.values()];self.iou=sum(v[0] for v in c);self.tp=sum(v[1] for v in c);self.np=sum(v[2] for v in c)
        self.cached_labels=labels.copy()

    def metrics(self,labels):
        self.state(labels);den=.5*(self.ng+self.np)
        return dict(pq=self.iou/den if den else 0.,tp=self.tp,fp=self.np-self.tp,fn=self.ng-self.tp,predicted=self.np)

    def reward(self,labels,indices,target):
        self.state(labels);indices=np.asarray(indices,dtype=int)
        if not len(indices):raise ValueError('Empty training operation')
        source=int(labels[indices[0]])
        if not len(indices) or not np.all(labels[indices]==source) or source==target:raise ValueError('Invalid training operation')
        old=self.vectors[source];dest=self.vectors.get(int(target),np.zeros_like(old));move=self.atoms[indices].sum(0)
        before=np.sum([self.contribution(old),self.contribution(dest)],axis=0)
        after=np.sum([self.contribution(old-move),self.contribution(dest+move)],axis=0)
        diou,dtp,dn=after-before;den=.5*(self.ng+self.np);newden=.5*(self.ng+self.np+dn)
        pq=self.iou/den if den else 0.;newpq=(self.iou+diou)/newden if newden else 0.
        return dict(delta_pq=float(newpq-pq),normalized_gain=float((newpq-pq)*den),delta_tp=int(dtp),delta_predicted=int(dn))
