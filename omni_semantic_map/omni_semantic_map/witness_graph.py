"""Development hypothesis: revisable co-visibility conflicts with independent witnesses.

A same-frame mask pair remains a cannot-link unless its measured surfaces touch
and >=2 other, spatially separated camera views each supply a common observation
linked to BOTH masks. No category-specific exception, GT or retrospective input.
All constraints are re-derived after pose updates; fragment evidence survives.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from .evidence_association import Config, RevisableGraph


@dataclass(frozen=True)
class WitnessConfig(Config):
    witness_views: int = 2
    witness_translation: float = .10
    contact_radius: float = .08
    contact_min_points: int = 3


class WitnessGraph(RevisableGraph):
    def __init__(self, config=WitnessConfig(), conflict_policy='witness'):
        super().__init__(config)
        if conflict_policy not in ['hard', 'off', 'witness']:
            raise ValueError(conflict_policy)
        if config.witness_views < 2 or config.witness_translation <= 0 or config.contact_radius <= 0 or config.contact_min_points < 1:
            raise ValueError('invalid witness/contact configuration')
        self.conflict_policy = conflict_policy
        self.poses = None
        self.contact_cache = {}

    def add_frame(self, observations, poses):
        self.poses = np.asarray(poses)
        return super().add_frame(observations, poses)

    def revise_poses(self, poses):
        self.poses = np.asarray(poses)
        return super().revise_poses(poses)

    def _touch(self, a, b):
        key = (min(a,b), max(a,b))
        if key not in self.contact_cache:
            # Both observations share a camera frame, hence this test is invariant
            # to that frame's pose update. Use all valid measured support points.
            x = np.asarray(self.observations[a]['support_xyz'])
            y = np.asarray(self.observations[b]['support_xyz'])
            if not len(x) or not len(y):
                self.contact_cache[key] = False
            else:
                dx = cKDTree(y).query(x)[0]; dy = cKDTree(x).query(y)[0]
                self.contact_cache[key] = bool(min(np.count_nonzero(dx <= self.config.contact_radius),
                                                  np.count_nonzero(dy <= self.config.contact_radius)) >= self.config.contact_min_points)
        return self.contact_cache[key]

    def _witnesses(self, a, b, adjacency):
        if not self._touch(a,b):
            return []
        original_frame = self.observations[a]['keyframe_id']
        frames = sorted({self.observations[i]['keyframe_id'] for i in adjacency[a] & adjacency[b]
                         if self.observations[i]['keyframe_id'] != original_frame})
        selected = []
        for frame in frames:
            center = self.poses[frame, :3, 3]
            # Distinct frame numbers alone are not independent viewpoints.
            if all(np.linalg.norm(center-self.poses[f,:3,3]) >= self.config.witness_translation for f in selected):
                selected.append(frame)
        return selected if len(selected) >= self.config.witness_views else []

    def _resolve(self, reason):
        if self.conflict_policy == 'hard':
            return super()._resolve(reason)
        n = len(self.observations)
        adjacency = [set() for _ in range(n)]
        for a,b in self.edges:
            adjacency[a].add(b); adjacency[b].add(a)
        parent = list(range(n))
        members = [{o['keyframe_id']:[i]} for i,o in enumerate(self.observations)]
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]; i = parent[i]
            return i
        accepted, conflicts, relaxed = 0, 0, []
        witness_cache = {}
        for (a,b),_ in sorted(self.edges.items(), key=lambda item:(item[1][0],item[0])):
            a,b = find(a),find(b)
            if a == b:
                continue
            common_frames = members[a].keys() & members[b].keys()
            allowed, proof = True, []
            if self.conflict_policy == 'witness':
                for frame in sorted(common_frames):
                    for x in members[a][frame]:
                        for y in members[b][frame]:
                            key=(min(x,y),max(x,y))
                            if key not in witness_cache:
                                witness_cache[key]=self._witnesses(x,y,adjacency)
                            witnesses=witness_cache[key]
                            if not witnesses:
                                allowed=False;break
                            proof.append(dict(observations=list(key),witness_frames=witnesses))
                        if not allowed:break
                    if not allowed:break
            if not allowed:
                conflicts += 1;continue
            if common_frames:
                relaxed.append(dict(components=[a+1,b+1],policy=self.conflict_policy,proof=proof))
            a,b=min(a,b),max(a,b);parent[b]=a
            for frame,indices in members[b].items():members[a].setdefault(frame,[]).extend(indices)
            accepted += 1
        labels=np.array([find(i)+1 for i in range(n)],np.int32)
        old=self.assignment;changed=np.flatnonzero(labels[:len(old)]!=old)
        self.revision += 1
        self.events.append(dict(revision=self.revision,reason=reason,observations=n,candidate_edges=len(self.edges),
            accepted_edges=accepted,co_visibility_conflicts=conflicts,relaxed_conflicts=relaxed,
            changed=[dict(observation_id=int(i),old=int(old[i]),new=int(labels[i])) for i in changed]))
        self.assignment=labels
