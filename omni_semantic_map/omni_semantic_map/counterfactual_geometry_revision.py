"""Reference-free counterfactual risk of deleting historical geometry regions.

The observation domain is actual measured surfaces with semantic probabilities.
Deleting geometry can expose another object or no object; it does not insert a
background anchor. Evaluate all affected nearest-map assignments, rather than
assuming each removed point has an independent or local effect.
"""
import numpy as np
from .spatial_acceptance_readout import build_readout


def nearest_indices(xyz, queries, radius=.2):
    return build_readout(xyz,np.arange(1,len(xyz)+1,dtype=np.int64),
                         np.empty((0,3)),queries,radius)['fallback']-1


def connected_cells(keys):
    """Deterministic26-neighbour components; caller separates object IDs."""
    keys=np.asarray(keys,np.int64)
    if keys.ndim!=2 or keys.shape[1]!=3 or len(np.unique(keys,axis=0))!=len(keys):
        raise ValueError('Distinct integer3Dcellkeys required')
    lookup={tuple(k):i for i,k in enumerate(keys)};unseen=set(lookup);result=[]
    offsets=[(a,b,c) for a in [-1,0,1] for b in [-1,0,1] for c in [-1,0,1] if (a,b,c)!=(0,0,0)]
    while unseen:
        start=min(unseen);unseen.remove(start);pending=[start];members=[]
        while pending:
            key=pending.pop();members.append(lookup[key])
            for delta in offsets:
                other=tuple(a+b for a,b in zip(key,delta))
                if other in unseen:unseen.remove(other);pending.append(other)
        result.append(np.array(sorted(members),np.int64))
    return result


class CounterfactualRevision:
    def __init__(self,xyz,classes,queries,probabilities,weights,radius=.2):
        self.xyz=np.asarray(xyz,np.float64);self.classes=np.asarray(classes,np.int64)
        self.queries=np.asarray(queries,np.float64);self.prob=np.asarray(probabilities,np.float64);self.weights=np.asarray(weights,np.float64);self.radius=radius
        if self.classes.shape!=(len(self.xyz),) or self.prob.ndim!=2 or self.prob.shape[0]!=len(self.queries) or self.weights.shape!=(len(self.queries),):
            raise ValueError('Aligned map classes,query probabilities and weights required')
        self.null=self.prob.shape[1]-1
        if self.null<1 or np.any(self.classes<0) or np.any(self.classes>=self.null):raise ValueError('Last probability column is no tracked class,never a map point')
        if not np.isfinite(self.prob).all() or np.any(self.prob<0) or not np.allclose(self.prob.sum(1),1.,rtol=0,atol=1e-6) or not np.isfinite(self.weights).all() or np.any(self.weights<0):
            raise ValueError('Normalized finite probabilities and nonnegative weights required')
        self.active=np.ones(len(self.xyz),bool);self.nearest=nearest_indices(self.xyz,self.queries,radius)
        self.loss=np.sum(self.cost(np.arange(len(self.queries)),self.nearest));self.initial_loss=float(self.loss)

    def labels(self,indices):
        result=np.full(len(indices),self.null,np.int64);valid=indices>=0;result[valid]=self.classes[indices[valid]];return result

    def cost(self,query_ids,indices):
        return -np.log(np.maximum(self.prob[query_ids,self.labels(indices)],1e-8))*self.weights[query_ids]

    def trial(self,remove,local_queries=None):
        """Try one region against current map; strict gain commits atomically.

        local_queries is the explicit restricted-domain control. Full-domain
        mode considers every affected measured sample and guarantees descent of
        THIS model-consistency objective, not GT accuracy/AP/expected utility.
        """
        remove=np.unique(np.asarray(remove,np.int64))
        if remove.ndim!=1 or np.any(remove<0) or np.any(remove>=len(self.xyz)):raise ValueError('Invalid map indices')
        remove=remove[self.active[remove]];mark=np.zeros(len(self.xyz),bool);mark[remove]=True
        affected=np.flatnonzero((self.nearest>=0)&mark[np.maximum(self.nearest,0)]) if len(mark) else np.empty(0,np.int64)
        if not len(affected):return dict(accepted=False,reason='no_affected_measurement',points=len(remove),affected=0,gain=0.,full_gain=0.,class_changes=0,nearest_point_changes=0)
        surviving=np.flatnonzero(self.active&~mark);local=nearest_indices(self.xyz[surviving],self.queries[affected],self.radius);after=np.full(len(local),-1,np.int64);has=local>=0;after[has]=surviving[local[has]]
        before=self.nearest[affected];delta=self.cost(affected,before)-self.cost(affected,after)
        if local_queries is None:selected=np.ones(len(affected),bool)
        else:
            local_queries=np.asarray(local_queries,bool)
            if local_queries.shape!=(len(self.queries),):raise ValueError('Aligned local observation scope required')
            selected=local_queries[affected]
        gain=float(delta[selected].sum());full_gain=float(delta.sum());accepted=gain>1e-9
        if accepted:self.active[remove]=False;self.nearest[affected]=after;self.loss-=full_gain
        return dict(accepted=accepted,reason='strict_model_consistency_gain' if accepted else 'no_gain',points=len(remove),affected=len(affected),tested_measurements=int(selected.sum()),gain=gain,full_gain=full_gain,class_changes=int((self.labels(before)!=self.labels(after)).sum()),nearest_point_changes=int((before!=after).sum()))

    def audit(self):
        ids=np.flatnonzero(self.active);indices=nearest_indices(self.xyz[ids],self.queries,self.radius);actual=np.full(len(indices),-1,np.int64);valid=indices>=0;actual[valid]=ids[indices[valid]]
        np.testing.assert_array_equal(actual,self.nearest)
        loss=float(self.cost(np.arange(len(actual)),actual).sum());np.testing.assert_allclose(loss,self.loss,rtol=1e-10,atol=1e-8)
        return dict(initial_loss=self.initial_loss,final_loss=loss,retained_points=int(self.active.sum()),exact_final_replay=True)
