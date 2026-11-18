"""Experimental association with immutable observation evidence, not deployment defaults.

DINO is frozen. Graph revisions are explicit and never rewrite causal outputs.
Co-visible detection masks impose a one-instance-per-frame constraint; this is a
model assumption and can prevent correction of detector over-segmentation.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from .dino_features import normalize
from .objects import transform_points


@dataclass(frozen=True)
class Config:
    radius: float = 1.
    appearance_weight: float = .6
    support_radius: float = .20
    min_local_pairs: int = 3
    memory_views: int = 8
    graph_neighbors: int = 12
    graph_max_cost: float = .55


def world_observation(obs, pose):
    result = dict(obs)
    result['world_position'] = transform_points([obs['local_position']], pose)[0]
    result['world_tokens'] = transform_points(obs['local_xyz'], pose)
    return result


def appearance_distance(a, b):
    return float(np.clip((1. - np.dot(normalize(a), normalize(b))) * .5, 0., 1.))


def local_distance(query, xyz, features, config):
    """Compare overlapping measured supports only; missing overlap is unknown."""
    if len(xyz) == 0 or len(query['world_tokens']) == 0:
        return None, 0
    distances = np.linalg.norm(query['world_tokens'][:, None] - xyz[None], axis=2)
    nearest = distances.argmin(axis=1)
    selected = distances[np.arange(len(nearest)), nearest] <= config.support_radius
    if int(selected.sum()) < config.min_local_pairs:
        return None, int(selected.sum())
    q = normalize(query['local_features'][selected])
    p = normalize(features[nearest[selected]])
    return float(np.mean(np.clip((1. - np.sum(q * p, axis=1)) * .5, 0., 1.))), int(selected.sum())


class MemoryAssociator:
    """Same class/radius/Hungarian policy across geometry, HSV and DINO ablations."""
    def __init__(self, method, config=Config()):
        if method not in ['geometry', 'hsv', 'dino_mean', 'dino_local']:
            raise ValueError(method)
        self.method, self.config = method, config
        self.observations, self.objects = [], []
        self.local_comparisons = self.local_fallbacks = 0

    def add_frame(self, observations, poses):
        current = [world_observation(o, poses[o['keyframe_id']]) for o in observations]
        costs = np.full((len(current), len(self.objects)), 1e6)
        for j, members in enumerate(self.objects):
            history = [self.observations[i] for i in members]
            center = np.median([o['world_position'] for o in history], axis=0)
            mean = normalize(np.mean([o['descriptor'] for o in history], axis=0))
            histogram = np.mean([o['appearance'] for o in history], axis=0)
            histogram /= max(histogram.sum(), 1e-12)
            views = [history[i] for i in np.unique(np.linspace(0, len(history)-1, min(self.config.memory_views, len(history))).astype(int))]
            xyz = np.concatenate([o['world_tokens'] for o in views])
            features = np.concatenate([o['local_features'] for o in views])
            for i, obs in enumerate(current):
                if obs['label'] != history[0]['label']:
                    continue
                distance = np.linalg.norm(obs['world_position'] - center)
                if distance > self.config.radius:
                    continue
                appearance = 0.
                if self.method == 'hsv':
                    appearance = float(np.sqrt(max(0., 1. - np.sqrt(histogram * obs['appearance']).sum())))
                elif self.method.startswith('dino'):
                    appearance = appearance_distance(obs['descriptor'], mean)
                    if self.method == 'dino_local':
                        self.local_comparisons += 1
                        local, _ = local_distance(obs, xyz, features, self.config)
                        if local is not None:
                            appearance = local
                        else:
                            self.local_fallbacks += 1
                costs[i, j] = distance + self.config.appearance_weight * appearance
        assignments = {}
        if costs.size:
            rows, cols = linear_sum_assignment(costs)
            assignments = {int(i): int(j) for i, j in zip(rows, cols) if costs[i, j] < 1e5}
        labels = []
        for i, obs in enumerate(current):
            j = assignments.get(i)
            if j is None:
                j = len(self.objects); self.objects.append([])
            self.objects[j].append(len(self.observations))
            self.observations.append(obs); labels.append(j + 1)
        return labels

    def labels(self):
        result = np.zeros(len(self.observations), np.int32)
        for j, members in enumerate(self.objects):
            result[members] = j + 1
        return result


class RevisableGraph:
    """Evidence graph: new evidence can merge/split components; poses can be revised.

    Bounded previous-observation candidate search, geometry-conditioned DINO
    edges, then greedy cost-ordered union with co-visibility cannot-links.
    All raw observations remain individually addressable. No GT is accessed.
    """
    def __init__(self, config=Config()):
        self.config = config
        self.observations, self.world, self.edges, self.events = [], [], {}, []
        self.assignment = np.zeros(0, np.int32)
        self.revision = 0

    def _add_edges(self, index):
        q = self.world[index]
        candidates = []
        for j in range(index):
            p = self.world[j]
            if p['label'] != q['label'] or p['keyframe_id'] == q['keyframe_id']:
                continue
            d = float(np.linalg.norm(q['world_position'] - p['world_position']))
            if d <= self.config.radius:
                candidates.append((d, j))
        # Favor spatially relevant past views; no same-frame edges are allowed.
        for d, j in sorted(candidates)[:self.config.graph_neighbors]:
            p = self.world[j]
            local, support = local_distance(q, p['world_tokens'], p['local_features'], self.config)
            if local is None:
                continue
            cost = d + self.config.appearance_weight * local
            if cost <= self.config.graph_max_cost:
                self.edges[(j, index)] = (cost, support)

    def _resolve(self, reason):
        n = len(self.observations)
        parent = list(range(n))
        frames = [{o['keyframe_id']} for o in self.observations]
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]; i = parent[i]
            return i
        accepted, conflicts = 0, 0
        for (a, b), _ in sorted(self.edges.items(), key=lambda item: (item[1][0], item[0])):
            a, b = find(a), find(b)
            if a == b:
                continue
            if frames[a] & frames[b]:
                conflicts += 1; continue
            a, b = min(a, b), max(a, b)
            parent[b] = a; frames[a] |= frames[b]; accepted += 1
        labels = np.array([find(i) + 1 for i in range(n)], np.int32)
        old = self.assignment
        changed = np.flatnonzero(labels[:len(old)] != old)
        self.revision += 1
        self.events.append(dict(revision=self.revision, reason=reason, observations=n,
            candidate_edges=len(self.edges), accepted_edges=accepted, co_visibility_conflicts=conflicts,
            changed=[dict(observation_id=int(i), old=int(old[i]), new=int(labels[i])) for i in changed]))
        self.assignment = labels

    def add_frame(self, observations, poses):
        start = len(self.observations)
        for obs in observations:
            self.observations.append(obs)
            self.world.append(world_observation(obs, poses[obs['keyframe_id']]))
            self._add_edges(len(self.observations) - 1)
        self._resolve('new_observations')
        return self.assignment[start:].tolist()

    def revise_poses(self, poses):
        # Recompute candidate discovery as well as edge weights: old geometry
        # must not survive a pose correction, including previously absent edges.
        self.world = [world_observation(o, poses[o['keyframe_id']]) for o in self.observations]
        self.edges = {}
        for i in range(len(self.observations)):
            self._add_edges(i)
        self._resolve('pose_revision')
        return self.labels()

    def labels(self):
        return self.assignment.copy()
