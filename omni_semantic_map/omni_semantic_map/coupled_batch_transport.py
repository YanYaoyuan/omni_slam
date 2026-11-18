"""Exact single-batch anchor-deletion identity stability on a fixed graph.

If the initial closest owner is unique, deleting a batch other than one chosen
closest anchor's batch retains that anchor and cannot change the winner. Only
that critical batch needs checking per node. A tie-aware two-frame propagation
computes this counterfactual for all nodes in one multi-source traversal.

This is not semantic correctness, raw-scan deletion, or repeat admission after
deletion. The geometry and all surviving anchor labels are held fixed.
"""
import ctypes
import hashlib
from pathlib import Path
import subprocess
import numpy as np


def native_library(directory):
    source = Path(__file__).with_suffix('.cpp')
    key = hashlib.sha256(source.read_bytes()).hexdigest()[:20]
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    target = directory / ('coupled_' + key + '.so')
    if not target.exists():
        subprocess.run(['g++', '-std=c++17', '-O3', '-shared', '-fPIC', str(source), '-o', str(target)], check=True)
    lib = ctypes.CDLL(str(target))
    ip = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
    dp = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    lib.coupled_batch_transport.argtypes = [ctypes.c_int64, ip, ip, dp, ctypes.c_int64,
        ip, ip, ip, ctypes.c_double, ip, ip, ip, dp, dp, ip]
    lib.coupled_batch_transport.restype = ctypes.c_int
    return lib


def solve(graph, seeds, library, tolerance=1e-10):
    graph = graph.tocsr(copy=True); graph.sum_duplicates(); graph.sort_indices()
    n = graph.shape[0]
    if graph.shape != (n, n) or not np.isfinite(graph.data).all() or (graph.data < 0).any():
        raise ValueError('Finite nonnegative square graph required')
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Finite nonnegative tolerance required')
    raw = np.asarray(seeds)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError('Seeds must have node, owner, acquisition batch columns')
    seeds = raw.astype(np.int64)
    if not np.array_equal(seeds, raw) or (seeds < 0).any() or (seeds[:, 0] >= n).any():
        raise ValueError('Integral valid nonnegative seeds required')
    seeds = np.unique(seeds, axis=0)
    ip = np.ascontiguousarray(graph.indptr, dtype=np.int64)
    col = np.ascontiguousarray(graph.indices, dtype=np.int64)
    w = np.ascontiguousarray(graph.data, dtype=np.float64)
    before = np.empty(n, np.int64); after = before.copy(); critical = before.copy()
    d = np.empty(n, np.float64); ad = d.copy(); stats = np.empty(3, np.int64)
    status = library.coupled_batch_transport(n, ip, col, w, len(seeds),
        *[np.ascontiguousarray(seeds[:, i]) for i in range(3)], tolerance,
        before, after, critical, d, ad, stats)
    if status: raise RuntimeError('Coupled batch propagation failed')
    return dict(nearest_owner=before, critical_batch=critical, after_critical_owner=after,
                nearest_distance=d, after_critical_distance=ad,
                stable_owner=np.where((before >= 0) & (after == before), before, -1),
                popped_states=int(stats[0]), expanded_states=int(stats[1]), max_retained_frames=int(stats[2]))
