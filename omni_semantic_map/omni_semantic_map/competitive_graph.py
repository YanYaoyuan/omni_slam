"""Revisable fragment witnesses checked against measured mask ownership.

Unlike a strict reciprocal-majority gate, partial observations remain eligible.
Positive witnesses must cover the fragments jointly. Independent views that
instead assign the fragments to distinct masks veto the merge. No GT inputs.
"""
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from .ownership_graph import OwnershipConfig, ownership_evidence
from .witness_graph import WitnessGraph


@dataclass(frozen=True)
class CompetitiveConfig(OwnershipConfig):
    rank_by_ownership: bool = True
    separation_veto: bool = True


class CompetitiveGraph(WitnessGraph):
    def __init__(self, views, config=CompetitiveConfig()):
        super().__init__(config,'witness')
        self.views=views
        self.directional={}
        self.separation_proofs={}

    def _evidence(self,a,b):
        key=(a,b)
        if key not in self.directional:
            x,y=self.observations[a],self.observations[b]
            self.directional[key]=ownership_evidence(x['support_xyz'],self.poses[x['keyframe_id']],
                self.poses[y['keyframe_id']],self.views[y['keyframe_id']],y['observation_id'],
                self.config.visibility_depth_tolerance)
        return self.directional[key]

    def _majority(self,a,b):
        d=self._evidence(a,b)
        return d['owned']>=self.config.ownership_min_samples and d['fraction'] is not None and d['fraction']>self.config.ownership_fraction

    def _independent(self,frames):
        selected=[]
        for f in sorted(set(frames)):
            if all(np.linalg.norm(self.poses[f,:3,3]-self.poses[j,:3,3])>=self.config.witness_translation for j in selected):
                selected.append(f)
        return selected

    def _add_edges(self,index):
        super()._add_edges(index)
        if self.config.rank_by_ownership:
            for a,b in [key for key in self.edges if key[1]==index]:
                evidence=[self._evidence(a,b),self._evidence(b,a)]
                # Unknown visibility is not a contradiction, but is weak ranking
                # evidence. Candidate acceptance still uses the original cost.
                confidence=min(x['fraction'] if x['fraction'] is not None else 0. for x in evidence)
                cost,support=self.edges[(a,b)]
                self.edges[(a,b)]=(cost+self.config.graph_max_cost*(1-confidence),support)

    def _witnesses(self,a,b,adjacency):
        if not self._touch(a,b):return []
        original=self.observations[a]['keyframe_id']
        positive=[]
        for r in sorted(adjacency[a]&adjacency[b]):
            if self.observations[r]['keyframe_id']==original:continue
            if not (self._majority(a,r) and self._majority(b,r)):continue
            ra,rb=self._evidence(r,a),self._evidence(r,b)
            # a and b share one image; the visibility denominator is identical.
            assert ra['visible']==rb['visible']
            if ra['visible'] and (ra['owned']+rb['owned'])/ra['visible']>self.config.ownership_fraction:
                positive.append(self.observations[r]['keyframe_id'])
        selected=self._independent(positive)
        if len(selected)<self.config.witness_views:return []
        if self.config.separation_veto:
            left,right=defaultdict(list),defaultdict(list)
            for src,target in [(a,left),(b,right)]:
                for r in sorted(adjacency[src]):
                    f=self.observations[r]['keyframe_id']
                    if f!=original and self._majority(src,r):target[f].append(r)
            distinct={f:next(((c,d) for c in left[f] for d in right[f] if c!=d),None)
                      for f in left.keys()&right.keys()}
            independent=self._independent([f for f,pair in distinct.items() if pair is not None])
            if len(independent)>=self.config.witness_views:
                self.separation_proofs[(min(a,b),max(a,b))]=[
                    dict(frame=f,observations=list(distinct[f])) for f in independent]
                return []
        return selected

    def _resolve(self,reason):
        self.separation_proofs={}
        super()._resolve(reason)
        self.events[-1]['separation_vetoes']=[dict(pair=list(pair),views=proof)
            for pair,proof in sorted(self.separation_proofs.items())]

    def revise_poses(self,poses):
        self.directional={}
        return super().revise_poses(poses)
