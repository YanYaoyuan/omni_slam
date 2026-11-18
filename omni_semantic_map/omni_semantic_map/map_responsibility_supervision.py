"""Training-only losses for categorical measured-point acceptance.

The spatial operator returns label marginals, not E[IoU] or a physical renderer.
No reference coordinates/labels are required by the inference MLP.
"""
import numpy as np
import torch
from .spatial_acceptance_readout import validate


def categorical(logits, valid):
    if logits.shape != valid.shape or not torch.all(valid.any(1)):
        raise ValueError('Every source requires a valid owner candidate')
    z = torch.cat([torch.zeros_like(logits[:, :1]), logits.masked_fill(~valid, -torch.inf)], 1)
    return torch.softmax(z, 1)[:, 1:]


def source_labels(probabilities, owners, labels):
    out = probabilities.new_zeros((len(probabilities), labels))
    out = out.scatter_add(1, owners, probabilities)
    return out.scatter_add(1, torch.zeros((len(out), 1), dtype=torch.long, device=out.device),
                           (1-probabilities.sum(1))[:, None])


def marginal_nll(probabilities, target, weight):
    """Categorical NLL; unreachable labels contribute a constant, no gradient.

    Unlike squared probability error, this does not reward spreading mass
    among wrong owners when the correct owner is absent from the candidate set.
    The 1e-8 floor is numerical and fixed, never an inference threshold.
    """
    correct = probabilities.gather(1, target[:, None])[:, 0]
    return -(correct.clamp_min(1e-8).log()*weight).sum()


def normalized_weights(target, weight, balanced):
    target = np.asarray(target, np.int64); w = np.asarray(weight, np.float64).copy()
    if target.shape != w.shape or np.any(target < 0) or np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError('Nonempty finite known-label mass required')
    if balanced:
        mass = np.bincount(target, weights=w)
        w /= np.maximum(mass[target], 1e-30)
    return w/w.sum()


class CompiledReadout:
    """Bounded, precompiled query slice equivalent to expected_labels()."""
    def __init__(self, adjacency, owners, labels, device='cpu'):
        offsets, sources, fallback, n = validate(adjacency)
        owners = np.asarray(owners, np.int64)
        if owners.ndim != 2 or len(owners) != n or np.any(owners < 0):
            raise ValueError('Aligned owner slots required')
        if max(owners.max(initial=0), fallback.max(initial=0)) >= labels:
            raise ValueError('Label count too small')
        lengths = np.diff(offsets); self.rows = len(fallback); self.width = int(lengths.max(initial=0)); self.labels = labels
        padded = np.zeros((self.rows, self.width), np.int64); mask = np.zeros_like(padded, bool)
        rows = np.repeat(np.arange(self.rows), lengths); slots = np.arange(len(sources))-np.repeat(offsets[:-1], lengths)
        padded[rows, slots] = sources; mask[rows, slots] = True
        self.indices = torch.as_tensor(padded, device=device)
        self.valid = torch.as_tensor(mask, device=device)
        self.owners = torch.as_tensor(owners, device=device)
        self.fallback = torch.as_tensor(fallback, device=device)

    def __call__(self, p):
        out = p.new_zeros((self.rows, self.labels)) + p.sum()*0
        if not self.width:
            return out.scatter_add(1, self.fallback[:, None], p.new_ones((self.rows, 1)))
        mass = p[self.indices]*self.valid[:, :, None]
        prefix = torch.cumprod(1-mass.sum(2), 1)
        survival = torch.cat([p.new_ones((self.rows, 1)), prefix[:, :-1]], 1)
        targets = self.owners[self.indices].reshape(self.rows, -1)
        out = out.scatter_add(1, targets, (mass*survival[:, :, None]).reshape(self.rows, -1))
        return out.scatter_add(1, self.fallback[:, None], prefix[:, -1, None])


def select_queries(adjacency, indices):
    """Select arbitrary query rows while retaining source indexing and order."""
    offsets, sources, fallback, n = validate(adjacency)
    indices = np.asarray(indices, np.int64)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(fallback)):
        raise ValueError('Valid query indices required')
    lengths = offsets[indices+1]-offsets[indices]
    new_offsets = np.r_[0, np.cumsum(lengths)]
    positions = np.repeat(offsets[indices], lengths) + np.arange(lengths.sum()) - np.repeat(new_offsets[:-1], lengths)
    return dict(offsets=new_offsets, sources=sources[positions], fallback=fallback[indices], candidate_count=np.asarray(n))
