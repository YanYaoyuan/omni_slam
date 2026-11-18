"""Explicit reference-identity encodings for a KITTI-360 protocol audit.

Only ``upstream_train`` reproduces the pinned public reader. The other modes
are sensitivity diagnostics, NOT replacements for official server labels.
No mode changes reference positions, weights, predictions, or ranking.
"""
import numpy as np


MODES = ('upstream_train', 'raw_identity', 'target_only')
CANONICAL = {2: 11, 13: 26}


def encode_reference(semantic, instance, taxonomy, mode='upstream_train'):
    """Return encoded labels and a reversible target-instance mapping.

    raw_identity keeps every train-2/train-13 raw object, separating aliases
    that collide after ``%1000``. Canonical building/car IDs take precedence.
    target_only voids noncanonical aliases, keeping all reference positions.
    """
    s, ins = np.asarray(semantic), np.asarray(instance)
    if mode not in MODES:
        raise ValueError('Unknown reference mode')
    if s.ndim != 1 or s.shape != ins.shape:
        raise ValueError('Aligned one-dimensional arrays required')
    if not all(np.issubdtype(v.dtype, np.integer) for v in (s, ins)):
        raise ValueError('Integer raw labels required')
    train = np.empty(s.shape, np.int64)
    for raw in np.unique(s):
        if int(raw) not in taxonomy:
            raise ValueError(f'Unknown raw semantic ID {raw}')
        train[s == raw] = taxonomy[int(raw)].trainId
    out = train * 1000 + ins.astype(np.int64) % 1000
    out[train == 255] = 0
    mapping = []
    for cls, canonical in CANONICAL.items():
        target = train == cls
        pairs = np.unique(np.column_stack((s[target], ins[target])), axis=0)
        if len(pairs) and np.any(pairs[:, 1] // 1000 != pairs[:, 0]):
            raise ValueError('Target raw instance prefix disagrees with raw semantic ID')
        pairs = sorted(map(tuple, pairs.tolist()), key=lambda v: (v[0] != canonical, v[0], v[1]))
        used = set()
        if len(pairs) > 1000 and mode == 'raw_identity':
            raise ValueError('More raw instances than this evaluator encoding can hold')
        for raw, rid in pairs:
            old = cls * 1000 + rid % 1000
            encoded = old
            if mode == 'raw_identity':
                suffix = rid % 1000
                if suffix in used:
                    suffix = next(i for i in range(1000) if i not in used)
                used.add(suffix)
                encoded = cls * 1000 + suffix
            elif mode == 'target_only' and raw != canonical:
                encoded = 0
            selected = (s == raw) & (ins == rid)
            out[selected] = encoded
            mapping.append(dict(raw_semantic=raw, raw_instance=rid,
                                upstream_id=old, encoded_id=encoded,
                                raw_name=taxonomy[raw].name,
                                ignore_in_instance=bool(taxonomy[raw].ignoreInInst)))
    if mode == 'raw_identity':
        assert len({v['encoded_id'] for v in mapping}) == len(mapping)
    return out, mapping
