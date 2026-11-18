"""Conservative offline history proposals with reversible component decisions.

Inputs are fixed baseline fragments and observed bridge events. No class labels,
ground truth, generated points, or connected-component scores enter this module.
"""
from collections import defaultdict

import numpy as np


def contrastive_bridge(query, native_history, alternate_history, competitors):
    """Both historical hypotheses must beat all valid same-frame alternatives.

    Same-frame observations are operational competitors, not identity GT.
    Missing competitors or descriptors mean unknown, never a passing vote.
    Vectors may have arbitrary scale; zero/nonfinite descriptors are invalid.
    """
    def unit(rows):
        a = np.asarray(rows, float)
        if a.ndim != 2:
            raise ValueError('descriptors must be matrices')
        norm = np.linalg.norm(a, axis=1)
        valid = np.isfinite(a).all(1) & (norm > 1e-12)
        return a[valid] / norm[valid, None]

    q = unit(np.asarray(query)[None])
    n, a, c = [unit(x) for x in (native_history, alternate_history, competitors)]
    if not all(len(x) for x in (q, n, a, c)):
        return dict(pass_vote=False, reason='unknown', native=None, alternate=None, competitor=None)
    native, alternate, competitor = [float((x @ q[0]).max()) for x in (n, a, c)]
    passed = min(native, alternate) > competitor
    return dict(pass_vote=passed, reason='pass' if passed else 'not_above_competitor',
                native=native, alternate=alternate, competitor=competitor)


def build_history_graph(node_frames, proposals, min_votes=1, use_dino=False):
    """Join fixed edges in common order with component-wide coframe exclusion.

    Every proposal contains endpoint node IDs, a witness frame, and dino_pass.
    Repeated proposals in a frame count once. Edge order is fixed by geometry
    votes for ALL variants. A fresh call recomputes components from untouched
    inputs, so disabling an edge fully reverses its downstream connectivity.
    """
    if min_votes < 1:
        raise ValueError('min_votes must be positive')
    parents = {int(k): int(k) for k in node_frames}
    frames = {int(k): set(map(int, v)) for k, v in node_frames.items()}
    edges = defaultdict(lambda: {'geometry_frames': set(), 'dino_frames': set()})
    for p in proposals:
        a, b = sorted([int(p['a']), int(p['b'])])
        if a not in parents or b not in parents:
            raise ValueError('unknown graph endpoint')
        if a == b:
            continue
        edges[a, b]['geometry_frames'].add(int(p['frame']))
        if p['dino_pass']:
            edges[a, b]['dino_frames'].add(int(p['frame']))

    def root(k):
        while k != parents[k]:
            k = parents[k]
        return k

    decisions = []
    for (a, b), votes in sorted(edges.items(), key=lambda e: (-len(e[1]['geometry_frames']), e[0])):
        x, y = root(a), root(b)
        supported = votes['dino_frames' if use_dino else 'geometry_frames']
        conflict = sorted(frames[x] & frames[y]) if x != y else []
        if len(supported) < min_votes:
            reason = 'insufficient_votes'
        elif x == y:
            reason = 'already_connected'
        elif conflict:
            reason = 'coframe_conflict'
        else:
            reason = 'joined'
            lo, hi = sorted([x, y])
            parents[hi] = lo
            frames[lo] |= frames[hi]
        decisions.append(dict(a=a, b=b, geometry_frames=sorted(votes['geometry_frames']),
                              dino_frames=sorted(votes['dino_frames']), reason=reason,
                              conflict_frames=conflict))
    groups = defaultdict(list)
    for node in sorted(parents):
        groups[root(node)].append(node)
    return dict(groups=list(groups.values()), decisions=decisions)
