"""Frozen, provided-pose OVO versus CG geometry on common observed voxels.

This is a full-system development comparison, not an identity-module ablation
or the official OVO mesh benchmark. Predictions are prepared before GT is read.
"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import pickle

import numpy as np

from .flat_benchmark import instance_metrics, sha256
from .mesh_instance_evaluation import transfer_labels

# Same domain as the earlier Replica development benchmark. This is a fixed
# global class list, not selected using classes present in a test scene.
BACKGROUND = (0, 5, 12, 30, 31, 40, 60, 79, 92, 93, 95, 97, 98, 255)
BASE = Path('build/benchmarks/ovo_full_20261110')
TRANSFERS = ('nearest_5cm', 'knn5_unbounded')


def cg_geometry(objects):
    """Native CG list entries are distinct objects even if export aliases collide.

    The replay exporter uses min(observation_ids)+1, which can repeat when
    native final merging reuses an observation. Preserve geometry and list
    entries; rank (export alias, list index) to get unique evaluation IDs.
    Ranking preserves all old ID tie ordering when aliases were unique.
    """
    aliases = [int(o['object_id']) for o in objects]
    if any(i <= 0 for i in aliases):
        raise ValueError('invalid CG export alias')
    order = sorted(range(len(objects)), key=lambda i: (aliases[i], i))
    identity = {index: rank+1 for rank, index in enumerate(order)}
    mapping = [dict(object_index=i, export_alias=aliases[i], evaluation_id=identity[i])
               for i in range(len(objects))]
    points = np.concatenate([o['points'] for o in objects]) if objects else np.empty((0, 3))
    labels = np.concatenate([np.full(len(o['points']), identity[i], np.int64)
                             for i, o in enumerate(objects)]) if objects else np.empty(0, np.int64)
    return points, labels, mapping


def ovo_labels(native_ids, classification, *, filter_background):
    """OVO ID zero is an object; -1 is unassigned. Output zero means miss.

    Unknown semantic classes are retained, since they cannot justify removing
    an instance. Only predicted background classes are removed, never GT.
    """
    ids = np.asarray(native_ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or (ids < -1).any():
        raise ValueError('invalid native object IDs')
    rows = classification['objects']
    classes = {int(o['object_id']): int(o['replica_class_id']) for o in rows}
    if len(classes) != len(rows) or any(i < 0 for i in classes):
        raise ValueError('duplicate or negative classified IDs')
    if not set(map(int, ids[ids >= 0])) <= set(classes):
        raise ValueError('assigned ID missing from classification')
    result = ids.astype(np.int64) + 1
    if filter_background:
        excluded = [i for i, c in classes.items() if c in BACKGROUND]
        result[np.isin(ids, excluded)] = 0
    return result


def voxel_metrics(truth, predicted):
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    if truth.ndim != 1 or truth.shape != predicted.shape:
        raise ValueError('incompatible label shapes')
    if any(not np.issubdtype(x.dtype, np.integer) or (x < 0).any()
           for x in (truth, predicted)):
        raise ValueError('invalid evaluation IDs')
    pairs, counts = np.unique(np.column_stack((truth, predicted)), axis=0, return_counts=True)
    metrics = instance_metrics(Counter({tuple(map(int, p)): int(c) for p, c in zip(pairs, counts)}))
    metrics['observed_voxel_instance_pq'] = metrics.pop('sequence_pixel_instance_pq')
    return metrics


def evaluate(scene, tag='full'):
    import torch
    protocol_path = BASE/'protocol_evaluation_v2.json'
    protocol = json.loads(protocol_path.read_text())
    if scene not in protocol['scenes'] or tuple(protocol['background_class_ids']) != BACKGROUND:
        raise ValueError('protocol mismatch')
    if sha256(Path(__file__)) != protocol['evaluation_source_sha256']:
        raise ValueError('evaluation source differs from frozen protocol')
    for path, expected in protocol['frozen_references'][scene].items():
        if sha256(path) != expected:
            raise ValueError(f'changed frozen reference: {path}')
    run = BASE/'runs'/tag/scene
    status_path = run/'run.json'
    status = json.loads(status_path.read_text())
    if status['protocol_sha256'] != protocol['mapping_protocol_sha256']:
        raise ValueError('mapping protocol changed')
    if not status['complete'] or status['frames'] != 2000 or status['camera_frames_consumed'] != 2000:
        raise ValueError('requires a complete 2000-frame native run')
    checkpoint = run/'ovo_map.ckpt'
    classification_path = run/'native_classification.json'
    for path, key in ((checkpoint, 'checkpoint_sha256'), (classification_path, 'classification_sha256')):
        if sha256(path) != status[key]:
            raise ValueError(f'changed run artifact: {path}')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)['map_params']
    xyz = saved['xyz'].detach().cpu().numpy().reshape(-1, 3)
    ids = saved['obj_ids'].detach().cpu().numpy().reshape(-1)
    classification = json.loads(classification_path.read_text())
    predictions = {
        'ovo_native_predicted_things': (xyz, ovo_labels(ids, classification, filter_background=True)),
        'ovo_native_all_regions': (xyz, ovo_labels(ids, classification, filter_background=False)),
    }
    cg = Path('build/benchmarks/replay_fusion_20261109/replay_runs')/scene/'objects.pkl.gz'
    with gzip.open(cg, 'rb') as f:
        objects = pickle.load(f)['objects']
    points, labels, cg_identity_map = cg_geometry(objects)
    predictions['cg_native_geometry'] = (points, labels)
    # GT is read only after both full prediction representations are prepared.
    reference = Path('build/benchmarks/expanded_semantic_20261109/conceptgraphs_evaluation/evaluation')/scene/'reference.npz'
    with np.load(reference) as f:
        vertices, truth = f['xyz'].copy(), f['instance'].copy()
    out = BASE/'evaluation_v2'/tag/scene
    out.mkdir(parents=True, exist_ok=False)
    results = dict(scene=scene, complete=False, evaluation_version=2,
                   cg_identity_map=cg_identity_map, evaluation_voxels=len(vertices),
                   scope=__doc__, primary_transfer='nearest_5cm', methods={},
                   source_sha256={str(p): sha256(p) for p in
                       (status_path, checkpoint, classification_path, cg, reference,
                        Path(__file__), protocol_path)})
    for name, (points, labels) in predictions.items():
        scores = {}
        for transfer in TRANSFERS:
            projected, distance = transfer_labels(points, labels, vertices, mode=transfer)
            scores[transfer] = voxel_metrics(truth, projected)
            scores[transfer]['assigned_reference_voxels'] = int((projected > 0).sum())
            np.savez_compressed(out/(name+'_'+transfer+'.npz'), instance=projected)
        results['methods'][name] = scores
    results['complete'] = True
    results['prediction_sha256'] = {p.name: sha256(p) for p in out.glob('*.npz')}
    (out/'metrics.json').write_text(json.dumps(results, indent=2, allow_nan=False))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', required=True)
    p.add_argument('--tag', default='full')
    args = p.parse_args()
    print(json.dumps(evaluate(args.scene, args.tag)['methods'], indent=2))


if __name__ == '__main__':
    main()
