"""Development shared-input controls; no ground truth or learned fitting.

One raw mask per frame (largest area, then lowest observation ID), followed by
10 largest views and an L1 medoid, provides a common semantic readout. Native
backend geometry/association is left intact. Mean of all unique views is a
predeclared semantic-fusion sensitivity control, not a separate trained model.
"""
import numpy as np


def select_views(members, frames, areas, top_k=10):
    members = sorted(set(map(int, members)), key=lambda i: (-int(areas[i]), i))
    seen, selected = set(), []
    for i in members:
        if int(frames[i]) not in seen:
            selected.append(i)
            seen.add(int(frames[i]))
    return selected[:top_k] if top_k > 0 else selected


def classify_objects(memberships, features, frames, areas, text_features, vocabulary):
    features = np.asarray(features, np.float32)
    text = np.asarray(text_features, np.float32)
    if features.ndim != 2 or text.ndim != 2 or features.shape[1] != text.shape[1]:
        raise ValueError('feature dimensions differ')
    if not np.isfinite(features).all() or not np.isfinite(text).all():
        raise ValueError('nonfinite features')
    text = text / np.linalg.norm(text, axis=1, keepdims=True)
    rows = []
    for object_id, members in memberships:
        if object_id <= 0 or not members:
            raise ValueError('output IDs must be positive and have evidence')
        selected = select_views(members, frames, areas)
        f = features[selected]
        medoid = int(np.abs(f[:, None] - f[None]).sum((1, 2)).argmin())
        all_views = select_views(members, frames, areas, top_k=0)
        mean = features[all_views].mean(0)
        descriptors = {'top10_l1_medoid': f[medoid], 'all_views_mean': mean}
        row = dict(object_id=int(object_id), observation_ids=sorted(set(map(int, members))),
                   selected_observation_ids=selected, medoid_observation_id=selected[medoid],
                   unique_frames=len(all_views), predictions={})
        for name, desc in descriptors.items():
            score = text @ (desc / max(float(np.linalg.norm(desc)), 1e-12))
            order = np.argsort(-score, kind='stable')
            best = int(order[0])
            row['predictions'][name] = dict(name=vocabulary['names'][best],
                replica_class_id=int(vocabulary['replica_class_ids'][best]),
                score=float(score[best]), margin=float(score[best] - score[order[1]]))
        rows.append(row)
    return rows


def observation_table(cache):
    """Read only cached descriptors and metadata, with immutable ingestion IDs."""
    import json
    manifest = json.loads((cache/'manifest.json').read_text())
    if not manifest['complete'] or not manifest['no_association']:
        raise ValueError('requires completed pre-association cache')
    features, frames, areas = [], [], []
    n = 0
    for rec in manifest['records']:
        if rec['observation_offset'] != n:
            raise ValueError('observation IDs are not contiguous')
        with np.load(cache/'frames'/rec['file']) as f:
            features.append(f['features'])
            areas.append(f['areas'])
        frames.extend([rec['frame_id']] * rec['masks'])
        n += rec['masks']
    return np.concatenate(features), np.array(frames), np.concatenate(areas)
