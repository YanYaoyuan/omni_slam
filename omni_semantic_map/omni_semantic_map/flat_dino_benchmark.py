"""Frozen-feature Flat development experiments. No training or GT-driven mapping."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
from .flat_benchmark import frame_index, read_pose, sha256, evaluate
from .dino_features import normalize
from .evidence_association import Config, MemoryAssociator, RevisableGraph


def load_cache(cache, sequence):
    meta = json.loads((cache / 'cache.json').read_text())
    if not meta['complete'] or Path(meta['sequence']).resolve() != sequence.resolve():
        raise ValueError('incomplete cache or wrong sequence')
    hashes = json.loads((cache / 'input_sha256.json').read_text())
    hashes[str((cache / 'cache.json').resolve())] = sha256(cache / 'cache.json')
    for name, digest in meta['files_sha256'].items():
        hashes[str((cache / name).resolve())] = digest
    for path, digest in hashes.items():
        if sha256(path) != digest:
            raise ValueError('changed input/cache: ' + path)
    observations = json.loads((cache / 'observations.json').read_text())
    ids, _ = frame_index(sequence)
    grouped = [[] for _ in ids]
    for i, obs in enumerate(observations):
        if obs['observation_id'] != i:
            raise ValueError('non-contiguous observation IDs')
        grouped[obs['keyframe_id']].append(obs)
    for frame, group in zip(ids, grouped):
        with np.load(cache / (frame + '.npz')) as data:
            if len(group) != len(data['mean']):
                raise ValueError('cache observation mismatch')
            for i, obs in enumerate(group):
                a, b = data['offsets'][i:i+2]
                obs.update(descriptor=normalize(data['mean'][i]), local_xyz=data['local_xyz'][a:b],
                           local_features=normalize(data['local_features'][a:b]))
    return meta, grouped, hashes


def write_frame(output, frame, group, labels, source_run):
    with np.load(source_run / 'frames' / (frame + '.npz')) as data:
        valid = data['valid']
    predicted = np.zeros(valid.shape, np.int32)
    for obs, label in zip(group, labels):
        with np.load(source_run / obs['support_file']) as support:
            pixels = support['pixels']
        u, v = pixels[:, 0] // 4, pixels[:, 1] // 4
        assert np.all(valid[v, u])
        predicted[v, u] = label
    np.savez_compressed(output / 'frames' / (frame + '.npz'), instance=predicted, valid=valid)


def run(sequence, cache, method, output, config=Config()):
    ids, _ = frame_index(sequence)
    meta, groups, hashes = load_cache(cache, sequence)
    source_run = Path(meta['source_run'])
    poses = np.array([read_pose(sequence / (f + '_pose.txt')) for f in ids])
    output.mkdir(parents=True, exist_ok=False); (output / 'frames').mkdir()
    protocol = dict(complete=False, sequence=str(sequence.resolve()), method=method, mode='predicted',
        depth_stride=4, frames_expected=len(ids), gt_segmentation_used_by_mapper=False,
        pose_source='provided poses', config=asdict(config), feature_cache=str(cache.resolve()),
        feature_cache_sha256=sha256(cache / 'cache.json'),
        source_sha256={p.name: sha256(p) for p in [Path(__file__), Path(__file__).with_name('evidence_association.py'), Path(__file__).with_name('dino_features.py')]},
        scope='causal per-frame instance output; all detections retained, no confirmed filtering',
        hsv_control='additive HSV distance; no hard appearance gate or ambiguity rejection',
        graph_scope='cost-ordered observation graph; bounded spatial candidates, co-visible cannot-links; no learned transformer',
        not_held_out=True)
    (output / 'run.json').write_text(json.dumps(protocol, indent=2))
    tracker = RevisableGraph(config) if method == 'dino_graph' else MemoryAssociator(method, config)
    emitted = set(); started = time.monotonic(); counts = []
    for k, (frame, group) in enumerate(zip(ids, groups)):
        labels = tracker.add_frame(group, poses)
        write_frame(output, frame, group, labels, source_run)
        emitted.update(labels)
        counts.append(dict(frame=frame, emitted_ids=len(emitted), map_objects=len(np.unique(tracker.labels()))))
        if k % 50 == 0:
            print(sequence.name, method, k, counts[-1], flush=True)
    labels = tracker.labels()
    doc = dict(observation_source=str((cache / 'observations.json').resolve()),
        assignments=[dict(observation_id=i, object_id=int(label)) for i, label in enumerate(labels)],
        objects=[dict(id=int(label), observation_ids=np.flatnonzero(labels == label).tolist()) for label in np.unique(labels)])
    (output / 'objects.json').write_text(json.dumps(doc, indent=2))
    (output / 'counts.json').write_text(json.dumps(counts, indent=2))
    if method == 'dino_graph':
        (output / 'revisions.json').write_text(json.dumps(tracker.events, indent=2))
    else:
        protocol.update(local_comparisons=tracker.local_comparisons, local_fallbacks=tracker.local_fallbacks)
    (output / 'input_sha256.json').write_text(json.dumps(hashes, indent=2))
    protocol.update(complete=True, frames_processed=len(ids), object_records=len(emitted), final_map_objects=len(doc['objects']),
                    wall_s=time.monotonic()-started)
    (output / 'run.json').write_text(json.dumps(protocol, indent=2))
    evaluate(sequence, output, output / 'metrics.json')
    if method == 'dino_graph':
        # Explicit retrospective view, evaluated separately, NEVER overwrite the
        # causal predictions. Other four methods have no retrospective revisions.
        final = output / 'final'; (final / 'frames').mkdir(parents=True, exist_ok=False)
        for frame, group in zip(ids, groups):
            write_frame(final, frame, group, [labels[o['observation_id']] for o in group], source_run)
        final_protocol = dict(protocol, method='dino_graph_final', object_records=len(doc['objects']),
                              scope='retrospective final assignments; uses later evidence, not causal tracking')
        (final / 'run.json').write_text(json.dumps(final_protocol, indent=2))
        (final / 'input_sha256.json').write_text(json.dumps(hashes, indent=2))
        evaluate(sequence, final, final / 'metrics.json')
    return protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['sequence', 'cache', 'output']:
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--method', choices=['geometry','hsv','dino_mean','dino_local','dino_graph'], required=True)
    a = p.parse_args()
    print(json.dumps(run(a.sequence, a.cache, a.method, a.output), indent=2))

if __name__ == '__main__':
    main()
