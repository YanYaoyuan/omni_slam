"""Separate post-hoc stage diagnosis on existing development scenes.

Run mapping before opening GT; reuse frozen observations, poses and domains.
All six branches share an online state. Does not change earlier experiments.
"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import pickle

import numpy as np
from PIL import Image

from .flat_benchmark import sha256, instance_metrics
from .flat_surface_evaluation import majority_labels
from .fusion_diagnostics import stages, membership_audit, assignment
from .replay_fusion import ConceptGraphsBackend

BASE = Path('build/benchmarks/fusion_diagnosis_20261110')
SOURCE = Path('build/benchmarks/expanded_semantic_20261109')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def replica_domain(cache, poses, scene):
    from . import replica_benchmark as r
    source = cache.parents[1] / 'data' / scene
    coordinates, truths, owners = [], [], []
    hashes = {}
    for k, fid in enumerate(r.FRAME_IDS):
        points, valid, u, v = r.projection(np.array(Image.open(source/'depth'/f'depth_{fid}.png')))
        ip = source/'semantic_instance'/f'semantic_instance_{fid}.png'
        sp = source/'semantic_class'/f'semantic_class_{fid}.png'
        hashes.update({str(p): sha256(p) for p in [ip, sp]})
        truth = r.truth_labels(np.array(Image.open(ip)), np.array(Image.open(sp)))[v, u]
        with np.load(cache/'frames'/f'{fid:04d}.npz') as f:
            np.testing.assert_array_equal(f['valid'], valid)
            owners.append(f['owner'][valid])
        coordinates.append(np.floor(r.transform_points(points[valid], poses[k])/.05).astype(np.int32))
        truths.append(truth[valid])
    voxel, inverse = np.unique(np.concatenate(coordinates), axis=0, return_inverse=True)
    truth = majority_labels(inverse, np.concatenate(truths), len(voxel))
    xyz = (voxel.astype(float)+.5)*.05
    ref = SOURCE/'conceptgraphs_evaluation/evaluation'/scene/'reference.npz'
    with np.load(ref) as f:
        np.testing.assert_array_equal(f['xyz'], xyz)
        np.testing.assert_array_equal(f['instance'], truth)
    return dict(xyz=xyz, truth=truth, inverse=inverse, owner=np.concatenate(owners)), hashes


def score(domain, labels):
    owner, inv, truth = domain['owner'], domain['inverse'], domain['truth']
    pred = np.zeros(len(owner), np.int32)
    valid = owner >= 0
    pred[valid] = labels[owner[valid]]
    vp = majority_labels(inv, pred, len(truth))
    pairs, counts = np.unique(np.column_stack([truth, vp]), axis=0, return_counts=True)
    table = Counter({tuple(map(int, p)): int(c) for p, c in zip(pairs, counts)})
    metric = instance_metrics(table)
    metric['pq'] = metric.pop('sequence_pixel_instance_pq')
    gt_sizes, pred_sizes = Counter(), Counter()
    for (g, p), c in table.items():
        gt_sizes[g] += c; pred_sizes[p] += c
    objects = []
    for g, size in sorted(gt_sizes.items()):
        if g == 0:
            continue
        pieces = [(c/(size+pred_sizes[p]-c), p, c) for (a, p), c in table.items() if a == g and p > 0]
        best = max(pieces, default=(0., 0, 0))
        objects.append(dict(gt_id=g, voxels=size, best_iou=best[0], best_prediction=best[1],
                            predicted_coverage=sum(x[2] for x in pieces)/size,
                            fragments_over_10pct=sum(x[2]/size >= .1 for x in pieces)))
    return dict(voxel=metric, gt_objects=objects), vp


def run(dataset, scene):
    import torch
    from omegaconf import OmegaConf
    from . import scenenn_benchmark as s
    torch.set_num_threads(1)
    out = BASE/dataset/scene
    out.mkdir(parents=True, exist_ok=False)
    if dataset == 'scenenn':
        cache, records, poses, n = s.load(scene)
        reference = s.BASE/'reference'/scene/'objects.pkl.gz'
    else:
        from .fusion_benchmark import load
        from .local_fusion_benchmark import records_from_cache
        cache, obs, frames, poses, features = load(SOURCE, scene)
        records = records_from_cache(cache, obs, frames, features); n = len(obs)
        reference = Path('build/benchmarks/replay_fusion_20261109/replay_runs')/scene/'objects.pkl.gz'
    cfg = OmegaConf.create(json.loads(s.CFG.read_text()))
    backend = ConceptGraphsBackend(cfg)
    for k, group in enumerate(records):
        backend.integrate(group, poses[k])
    branches = stages(backend)
    # Check membership and all exported numeric fields against previous native run.
    with gzip.open(reference, 'rb') as f:
        saved = pickle.load(f)
    if isinstance(saved, dict):
        saved = saved['objects']
    assert len(saved) == len(branches['native_final'])
    for a, b in zip(saved, branches['native_final']):
        assert a['observation_ids'] == b['observation_ids']
        for key, value in [('points', np.asarray(b['pcd'].points)), ('colors', np.asarray(b['pcd'].colors)),
                           ('box', np.asarray(b['bbox'].get_box_points())),
                           ('image_feature', b['clip_ft'].cpu().numpy()), ('text_feature', b['text_ft'].cpu().numpy())]:
            np.testing.assert_allclose(a[key], value, rtol=0, atol=1e-10)
    labels = {name: assignment(objects, n) for name, objects in branches.items()}
    np.savez_compressed(out/'assignments.npz', **labels)
    dump(out/'memberships.json', {name: [o['observation_ids'] for o in objects] for name, objects in branches.items()})
    dump(out/'mapping.json', dict(native_reference_exact=True, observations=n,
         inputs={str(cache/'manifest.json'): sha256(cache/'manifest.json'), str(s.CFG): sha256(s.CFG), str(reference): sha256(reference)},
         stages={name: membership_audit(objects) for name, objects in branches.items()},
         source_hashes={str(p):sha256(p) for p in [Path(__file__), Path(__file__).with_name('fusion_diagnostics.py')]}))
    # Only now open GT. Nothing below is fed back to branches.
    if dataset == 'scenenn':
        path = s.BASE/'evaluation'/scene/'domain.npz'
        with np.load(path) as f:
            domain = {k:f[k].copy() for k in f.files}
        hashes = {str(path):sha256(path)}
    else:
        domain, hashes = replica_domain(cache, poses, scene)
    results, predictions = {}, {}
    for name, lab in labels.items():
        results[name], predictions[name] = score(domain, lab)
    np.savez_compressed(out/'voxel_predictions.npz', xyz=domain['xyz'], truth=domain['truth'], **predictions)
    # Union coverage is an optimistic geometric ceiling, not an algorithm score.
    owner, inv, truth = domain['owner'], domain['inverse'], domain['truth']
    covered = np.zeros(len(truth), bool); covered[inv[owner >= 0]] = True
    coverage = [dict(gt_id=int(g), union_coverage=float(covered[truth == g].mean())) for g in np.unique(truth) if g > 0]
    dump(out/'evaluation.json', dict(dataset=dataset, scene=scene, complete=True,
         scope='post-hoc development diagnosis; shared observed 5cm voxel class-agnostic PQ; not official mesh AP',
         gt_domain_sha256=hashes, stages=results, mask_union_coverage=coverage))
    print('COMPLETE', dataset, scene, {k:round(v['voxel']['pq']*100, 3) for k,v in results.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['replica', 'scenenn'])
    parser.add_argument('scene')
    args = parser.parse_args()
    run(args.dataset, args.scene)
