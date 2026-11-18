"""Whole-region mask agreement with missing-view censoring.

Regions carry sparse per-view categorical mask counts. The score is a
Dirichlet-integrated categorical likelihood (unit total prior mass), used as
a compatibility objective, NOT a calibrated posterior on physical identity.
Correlated surface cells/views do not become independent by this formula.

Old object IDs are deliberately not constraints. Partitioning measured atoms
can form new objects and split/rejoin historical identities in the caller.
"""
import heapq
import math
import numpy as np


def view_score(counts, categories):
    """Sequence likelihood: no multinomial combinatorial coefficient."""
    if categories < 2:
        raise ValueError('Include the unobserved escape category; K must be >=2')
    values = np.asarray(list(counts.values()), dtype=float)
    if any(int(k) != k or k < 0 or k >= categories for k in counts):
        raise ValueError('Category outside frame vocabulary')
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Finite nonnegative counts required')
    alpha = 1. / categories
    return -math.lgamma(1.+float(values.sum())) + sum(
        math.lgamma(alpha+float(v))-math.lgamma(alpha) for v in values if v > 0)


def join_histogram(a, b):
    result = {f:dict(v) for f,v in a.items()}
    for frame, hist in b.items():
        dst = result.setdefault(frame,{})
        for token, mass in hist.items():
            dst[token] = dst.get(token,0.)+mass
    return result


def region_score(histogram, categories):
    return sum(view_score(v,int(categories[f])) for f,v in histogram.items())


def merge_gain(a, b, categories):
    """Unknown/noncommon views contribute exactly zero, not negative evidence."""
    gain = 0.
    for frame in sorted(a.keys() & b.keys()):
        merged = dict(a[frame])
        for token,mass in b[frame].items():
            merged[token] = merged.get(token,0.)+mass
        k = int(categories[frame])
        gain += view_score(merged,k)-view_score(a[frame],k)-view_score(b[frame],k)
    return gain


def make_histograms(count, atom, frame, token, mass, categories):
    atom,frame,token = [np.asarray(v) for v in (atom,frame,token)]
    mass = np.asarray(mass,float); categories = np.asarray(categories)
    if count < 0 or atom.ndim != 1 or not(atom.shape==frame.shape==token.shape==mass.shape):
        raise ValueError('Aligned sparse observations required')
    if not all(np.issubdtype(v.dtype,np.integer) for v in (atom,frame,token,categories)):
        raise ValueError('Integer index arrays required')
    if categories.ndim != 1 or np.any(categories<2):
        raise ValueError('K>=2 for every frame')
    if np.any((atom<0)|(atom>=count)) or np.any((frame<0)|(frame>=len(categories))):
        raise ValueError('Observation index out of range')
    if np.any((token<0)|(token>=categories[frame])) or not np.isfinite(mass).all() or np.any(mass<=0):
        raise ValueError('Valid categories and positive finite mass required')
    result = [{} for _ in range(count)]
    for a,f,t,w in zip(atom,frame,token,mass):
        h = result[int(a)].setdefault(int(f),{})
        h[int(t)] = h.get(int(t),0.)+float(w)
    return result


def partition(histograms, categories, left, right):
    """Greedy positive whole-region gain on a fixed undirected proposal graph.

    A graph edge only proposes a merge; it has no permanent scalar affinity.
    Costs are recomputed against ALL observations of both current regions.
    Stop at a graph-local optimum; no global-optimum/online/realtime claim.
    """
    n = len(histograms)
    left,right = np.asarray(left),np.asarray(right)
    if left.ndim != 1 or left.shape != right.shape:
        raise ValueError('Aligned graph edges required')
    if not all(np.issubdtype(v.dtype,np.integer) for v in (left,right)):
        raise ValueError('Integer graph indices required')
    if np.any(left<0) or np.any(right>=n) or np.any(left>=right):
        raise ValueError('Undirected i<j graph required')
    if len(np.unique(left*n+right)) != len(left):
        raise ValueError('Duplicate graph edges')
    stats = [{f:dict(v) for f,v in h.items()} for h in histograms]
    initial_score = sum(region_score(h,categories) for h in stats)
    parent = np.arange(n); version = np.zeros(n,np.int64)
    graph = [set() for _ in range(n)]
    for a,b in zip(left,right):
        graph[int(a)].add(int(b));graph[int(b)].add(int(a))
    heap = []; trace = []; evaluations = 0
    def offer(a,b):
        nonlocal evaluations
        a,b = sorted((a,b)); gain = merge_gain(stats[a],stats[b],categories)
        evaluations += 1
        if gain > 1e-10:
            heapq.heappush(heap,(-gain,a,b,int(version[a]),int(version[b])))
    for a,b in zip(left,right):offer(int(a),int(b))
    while heap:
        neg,a,b,va,vb = heapq.heappop(heap)
        if parent[a]!=a or parent[b]!=b or version[a]!=va or version[b]!=vb:
            continue
        gain = merge_gain(stats[a],stats[b],categories)
        if gain <= 1e-10:continue
        assert abs(gain+neg)<1e-8*max(1.,abs(gain))
        trace.append((a,b,float(gain)))
        parent[b] = a;version[a] += 1;version[b] += 1
        stats[a] = join_histogram(stats[a],stats[b]);stats[b] = {}
        neighbors = (graph[a]|graph[b])-{a,b}
        for other in graph[b]:
            graph[other].discard(b)
            if other != a:graph[other].add(a)
        graph[a] = neighbors;graph[b] = set()
        for other in sorted(neighbors):offer(a,other)
    for i in range(n):
        j = i
        while parent[j]!=j:j=int(parent[j])
        parent[i] = j
    roots = np.flatnonzero(parent==np.arange(n))
    final_score = sum(region_score(stats[i],categories) for i in roots)
    gain_sum = sum(v[2] for v in trace)
    assert abs(final_score-initial_score-gain_sum) < 1e-7*max(1.,abs(initial_score))
    max_gain = max((merge_gain(stats[a],stats[b],categories) for a in roots for b in graph[a] if a<b),default=0.)
    assert max_gain <= 1e-10
    return parent, dict(merges=trace, initial_score=float(initial_score),final_score=float(final_score),
                        score_gain=float(gain_sum),maximum_remaining_gain=float(max_gain),
                        gain_evaluations=evaluations,regions=len(roots))
