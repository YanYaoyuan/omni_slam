"""Frozen-feature, causal object-memory readers; no learned weights or GT.

This is an offline decision probe, not a trained Transformer or a SLAM backend.
All readers consume the SAME observations. Attention uses identity Q/K/V and
cosine logits with a fixed temperature; its output is normalized before scoring.
"""
import numpy as np

READERS = ('mean', 'nearest', 'attention')


def unit(features):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError('features must be a matrix')
    norms = np.linalg.norm(x, axis=1)
    valid = np.isfinite(x).all(axis=1) & (norms > 1e-12)
    out = np.zeros_like(x)
    out[valid] = x[valid] / norms[valid, None]
    return out, valid


def select_memory(members, query, frames, areas):
    """One largest mask per prior frame, tie by ID; no future-frame repair."""
    ids = sorted(set(map(int, members)))
    if not ids or min(ids) < 0 or max(ids) >= len(frames):
        raise ValueError('empty or invalid memory')
    if not 0 <= query < len(frames):
        raise ValueError('invalid query')
    if any(frames[i] >= frames[query] for i in ids):
        raise ValueError('noncausal memory')
    chosen = {}
    for i in ids:
        f = int(frames[i])
        if f not in chosen or (float(areas[i]), -i) > (float(areas[chosen[f]]), -chosen[f]):
            chosen[f] = i
    return [chosen[f] for f in sorted(chosen)]


def read_memory(query, history, temperature=.1):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be positive')
    q, qvalid = unit(np.asarray(query)[None])
    h, hvalid = unit(history)
    if not len(h) or not qvalid.all() or not hvalid.all():
        return None
    similarities = h @ q[0]
    logits = similarities / temperature
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    pooled = np.stack([h.mean(0), weights @ h])
    pooled, valid = unit(pooled)
    if not valid.all():
        return None
    nearest = int(np.argmax(similarities))
    return dict(scores=dict(mean=float(pooled[0] @ q[0]),
                            nearest=float(similarities[nearest]),
                            attention=float(pooled[1] @ q[0])),
                weights=weights.tolist(), nearest_index=nearest,
                effective_views=float(1 / np.square(weights).sum()))


def rerank(event, features, frames, areas, temperature=.1):
    """Same candidate set and memory for all readers; event-wide fallback.

    Invalid descriptors do not silently remove hard candidates. All readers
    keep the native choice for that entire event and record it as unsupported.
    Native wins exact ties, then original candidate order. A changed choice
    does NOT change subsequent recorded histories in this diagnostic.
    """
    q = int(event['query'])
    if event['frame'] != int(frames[q]):
        raise ValueError('query frame mismatch')
    candidates = event['candidates']
    native = [i for i, c in enumerate(candidates) if c['is_native']]
    if len(native) != 1 or len(candidates) < 2:
        raise ValueError('requires one native and multiple candidates')
    native = native[0]
    memories = [select_memory(c['members'], q, frames, areas) for c in candidates]
    reads = [read_memory(features[q], features[m], temperature) for m in memories]
    supported = all(r is not None for r in reads)
    choices = {'native': native}
    if supported:
        for name in READERS:
            choices[name] = max(range(len(candidates)),
                                key=lambda i: (reads[i]['scores'][name], i == native, -i))
    else:
        choices.update({name: native for name in READERS})
    return dict(query=q, frame=int(frames[q]), supported=supported,
                choices=choices, memories=memories, reads=reads,
                candidate_count=len(candidates))
