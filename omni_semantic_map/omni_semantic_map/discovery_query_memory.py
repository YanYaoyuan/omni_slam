"""Budgeted selection of segmentation queries from measured map evidence.

No GT, class labels, descriptors or future masks enter query selection.
Unexplained is an inference state, not a claim that a point is a true object.
Previously unsuccessful queries reduce priority but never delete map points.
IDs name measured query-memory units (20cm cells in the prepared experiment),
not object identities. The geometry and per-point mask support remain separate.
"""
import numpy as np


def _ids(values):
    values=np.asarray(values)
    if values.ndim!=1 or not np.issubdtype(values.dtype,np.integer) or np.any(values<0):
        raise ValueError('Measured unit IDs must be a nonnegative integer vector')
    return np.unique(values)


class DiscoveryQueryMemory:
    """One decision per completed causal buffer; caller controls query budget.

Residual score = sum((1 - explained) / (1 + previous queries)) on visible
measured source IDs. Geometric control ignores explained; uniform picks the
latest buffered frame. These are fixed heuristics, not calibrated utilities.
"""
    def __init__(self,policy='residual'):
        if policy not in ['uniform','geometric','residual']:raise ValueError('Unknown policy')
        self.policy=policy;self.queries=np.zeros(0,np.int64);self.explained=np.zeros(0,bool)
        self.last_decision=-1;self.pending=None;self.events=[]

    def _grow(self,ids):
        required=int(ids.max(initial=-1))+1
        if required>len(self.queries):
            pad=required-len(self.queries);self.queries=np.pad(self.queries,(0,pad));self.explained=np.pad(self.explained,(0,pad))

    def choose(self,frames,decision_frame):
        if self.pending is not None:raise ValueError('Commit current query before next decision')
        if decision_frame<=self.last_decision or not frames:raise ValueError('Nonempty new causal buffer required')
        identifiers=[int(f['frame_id']) for f in frames]
        if len(set(identifiers))!=len(identifiers) or any(i>decision_frame or i<=self.last_decision for i in identifiers):
            raise ValueError('Buffer contains future or already consumed frames')
        candidates=[]
        for f in frames:
            ids=_ids(f['visible_unit_ids']);self._grow(ids)
            weight=1./(1+self.queries[ids])
            if self.policy=='residual':weight=weight*(~self.explained[ids])
            score=float(weight.sum());candidates.append((int(f['frame_id']),ids,score))
        selected=max(candidates,key=lambda x:x[0] if self.policy=='uniform' else (x[2],x[0]))
        self.pending=dict(decision_frame=int(decision_frame),selected_frame=selected[0],visible=selected[1],
            candidates=[dict(frame_id=i,visible_units=len(ids),score=score) for i,ids,score in candidates])
        return selected[0]

    def commit(self,frame_id,positive_unit_ids):
        if self.pending is None or frame_id!=self.pending['selected_frame']:raise ValueError('Unexpected query result')
        ids=self.pending['visible'];positive=_ids(positive_unit_ids)
        if not np.isin(positive,ids).all():raise ValueError('Mask support must be current measured visible support')
        new=int(np.count_nonzero(~self.explained[positive]));self.explained[positive]=True;self.queries[ids]+=1
        event={k:v for k,v in self.pending.items() if k!='visible'}
        event.update(positive_units=len(positive),newly_explained_units=new,queried_units=len(ids),policy=self.policy)
        self.events.append(event);self.last_decision=self.pending['decision_frame'];self.pending=None
        return event
