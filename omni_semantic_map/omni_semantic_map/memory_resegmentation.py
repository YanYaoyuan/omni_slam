"""Local evidence readers and deterministic, label-free SAM prompt proposals.

The memory contains only earlier measured observations, never final map IDs.
GT distributions belong to the offline trainer, not this module.
"""
import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
import torch.nn.functional as F


class LocalEvidenceReader(nn.Module):
    def __init__(self, mode='attention', dim=384, width=64):
        super().__init__()
        if mode not in ('current', 'mean', 'attention'):
            raise ValueError(mode)
        self.mode = mode
        self.project = nn.Linear(dim, width)
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.foreground = nn.Linear(width, 1)
        self.pair = nn.Sequential(nn.Linear(width*2+2, width), nn.GELU(), nn.Linear(width, 1))

    def encode(self, current, history, valid):
        q = self.project(current)
        if self.mode == 'current':
            return self.norm(q)
        h = self.project(history)
        # Empty histories get a zero-valued, unmasked dummy token; output then zeroed.
        any_valid = valid.any(-1)
        safe = valid.clone()
        safe[:, 0] |= ~any_valid
        h = h * valid[..., None]
        if self.mode == 'mean':
            context = h.sum(1) / valid.sum(1).clamp_min(1)[:, None]
        else:
            context = self.attention(q[:, None], h, h, key_padding_mask=~safe,
                                     need_weights=False)[0][:, 0]
        return self.norm(q + context * any_valid[:, None])

    def affinity(self, a, b, raw_cosine, distance):
        return self.pair(torch.cat([torch.abs(a-b), a*b,
            raw_cosine[..., None], distance.clamp(max=2)[..., None]], -1)).squeeze(-1)


class CausalTokenMemory:
    """Bounded FIFO raw evidence, radius-limited retrieval from earlier frames."""
    def __init__(self, max_tokens=8192, neighbors=8, radius=.20):
        self.max_tokens, self.neighbors, self.radius = max_tokens, neighbors, radius
        self.xyz = np.empty((0, 3), np.float32)
        self.features = np.empty((0, 384), np.float32)
        self.frames = np.empty(0, np.int64)
        self.ids = np.empty(0, np.int64)

    def query(self, xyz, frame):
        n = len(xyz)
        features = np.zeros((n, self.neighbors, 384), np.float32)
        valid = np.zeros((n, self.neighbors), bool)
        ids = np.full((n, self.neighbors), -1, np.int64)
        if len(self.xyz) and n:
            assert np.all(self.frames < frame), 'future/current memory leak'
            dist, idx = cKDTree(self.xyz).query(xyz, k=self.neighbors,
                                               distance_upper_bound=self.radius, workers=1)
            if self.neighbors == 1:
                dist, idx = dist[:, None], idx[:, None]
            valid = np.isfinite(dist)
            features[valid] = self.features[idx[valid]]
            ids[valid] = self.ids[idx[valid]]
        return features, valid, ids

    def add(self, xyz, features, frame, ids):
        if len(self.frames) and frame <= self.frames[-1]:
            raise ValueError('frames must increase')
        for key, value in [('xyz',xyz), ('features',features),
                           ('frames',np.full(len(xyz),frame)), ('ids',ids)]:
            setattr(self, key, np.concatenate([getattr(self,key),value])[-self.max_tokens:])


def choose_events(offsets, features, max_events=2):
    """Shared acquisition policy: largest DINO disagreement, no labels/model."""
    events=[]
    for j,(lo,hi) in enumerate(zip(offsets[:-1],offsets[1:])):
        if hi-lo < 4:
            continue
        f=features[lo:hi]
        cosine=f@f.T
        a,b=np.unravel_index(np.argmin(cosine), cosine.shape)
        events.append(dict(mask=j, a=int(lo+a), b=int(lo+b), disagreement=float(1-cosine[a,b])))
    return sorted(events,key=lambda e:(-e['disagreement'],e['mask']))[:max_events]


def prompt_sets(event, offsets, pixels, scores=None):
    """Two calls in every arm. Learned heads add one positive/negative each.

Seeds and the initial mutually-negative anchors are identical across arms.
No box restricts a candidate to the original (possibly wrong) mask.
"""
    lo,hi=offsets[event['mask']:event['mask']+2]
    result=[]
    for side,(anchor,other) in enumerate([(event['a'],event['b']),(event['b'],event['a'])]):
        indices=[anchor,other];labels=[1,0]
        if scores is not None:
            values=np.asarray(scores[side])
            for order,label in [(np.argsort(-values,kind='stable'),1),(np.argsort(values,kind='stable'),0)]:
                for k in order:
                    index=int(lo+k)
                    if index not in indices:
                        indices.append(index);labels.append(label);break
        result.append((pixels[indices].astype(np.float32),np.array(labels,np.int32)))
    return result


def partition_candidates(original, candidates, scores):
    """Keep measured footprint fixed; resolve overlap once, preserve remainder.

No geometry is hallucinated and no candidate can swallow a neighboring mask.
The output can split an original mask, but does not invent unseen support.
"""
    orig=np.asarray(original,bool)
    masks=np.array(candidates,bool)&orig[None]
    if not len(masks):return [orig.copy()]
    winner=np.argmax(np.asarray(scores))
    overlap=masks.sum(0)>1
    masks[:,overlap]=False;masks[winner,overlap]=True
    remainder=orig & ~masks.any(0)
    return [m for m in list(masks)+[remainder] if m.any()]
