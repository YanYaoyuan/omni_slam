"""Small supervised, geometry-conditioned object-memory ranker.

DINO is frozen. Mean and attention have identical embedding, value/output
projections and residual scoring head. Only attention learns Q/K projections.
Both receive the same native eligibility gate, native score and memory views.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ObjectMemoryRanker(nn.Module):
    def __init__(self, mode='attention', input_dim=384, width=64, heads=4):
        super().__init__()
        if mode not in ('attention', 'mean') or width % heads:
            raise ValueError('invalid reader')
        self.mode, self.width, self.heads = mode, width, heads
        self.embed = nn.Sequential(nn.Linear(input_dim, width), nn.LayerNorm(width), nn.GELU())
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.score = nn.Sequential(nn.Linear(4*width+4, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.score[-1].weight); nn.init.zeros_(self.score[-1].bias)
        if mode == 'mean':
            self.query.requires_grad_(False); self.key.requires_grad_(False)

    def forward(self, query, memory, valid, native_score, return_weights=False):
        if memory.ndim != 3 or valid.shape != memory.shape[:2] or not valid.any(1).all():
            raise ValueError('each candidate requires at least one valid memory')
        q = F.normalize(query, dim=-1)
        h = F.normalize(memory, dim=-1)
        qe, he = self.embed(q), self.embed(h)
        n, views, _ = he.shape; depth = self.width // self.heads
        values = self.value(he).reshape(n, views, self.heads, depth).transpose(1, 2)
        if self.mode == 'attention':
            queries = self.query(qe).reshape(n, self.heads, depth)
            keys = self.key(he).reshape(n, views, self.heads, depth).transpose(1, 2)
            logits = (queries[:, :, None, :] * keys).sum(-1) / math.sqrt(depth)
            weights = logits.masked_fill(~valid[:, None, :], -torch.inf).softmax(-1)
        else:
            weights = (valid.float() / valid.sum(1, keepdim=True))[:, None, :].expand(-1, self.heads, -1)
        pooled = self.output((weights[..., None] * values).sum(2).reshape(n, self.width))
        cos = (q[:, None, :] * h).sum(-1)
        max_cos = cos.masked_fill(~valid, -torch.inf).max(1).values
        mean_cos = (cos * valid).sum(1) / valid.sum(1)
        count = torch.log1p(valid.sum(1).float()) / math.log(17.)
        side = torch.stack([native_score, max_cos, mean_cos, count], -1)
        pair = torch.cat([qe, pooled, (qe-pooled).abs(), qe*pooled, side], -1)
        # Epoch zero exactly reproduces native CG scores, for BOTH learned arms.
        scores = native_score + self.score(pair).squeeze(-1)
        return (scores, weights) if return_weights else scores


def positive_mass_loss(scores, offsets, positives):
    """Event-wise listwise loss allowing multiple identity-correct candidates."""
    losses=[]
    for a,b in zip(offsets[:-1],offsets[1:]):
        a,b=int(a),int(b); good=positives[a:b]
        if not good.any() or good.all():
            raise ValueError('training requires both positive and negative candidates')
        losses.append(torch.logsumexp(scores[a:b],0)-torch.logsumexp(scores[a:b][good],0))
    return torch.stack(losses)
