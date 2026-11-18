"""Frozen-input feature diagnosis and replay-fusion v0 Replica experiments.

Uses an existing verified cache; never runs a detector or reads semantic GT.
Evaluation is a separate step. All eight scenes are development data now.
"""
import argparse
import gzip
import json
import pickle
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from PIL import Image
from .flat_benchmark import sha256
from .competitive_graph import CompetitiveGraph, CompetitiveConfig
from .replay_fusion import FusionObservation, ReplayableObjectMap, ConceptGraphsBackend


def load(source, scene):
    feature_path = source / 'conceptgraphs_features' / scene
    feature_meta = json.loads((feature_path / 'manifest.json').read_text())
    cache = Path(feature_meta['input_cache'])
    for path, expected in ((cache/'manifest.json', feature_meta['input_cache_sha256']),
                           (cache/'observations.json', feature_meta['input_observations_sha256']),
                           (feature_path/'clip.npz', feature_meta['feature_sha256'])):
        if sha256(path) != expected:
            raise ValueError(f'modified input {path}')
    manifest = json.loads((cache/'manifest.json').read_text())
    if not manifest['complete']:
        raise ValueError('incomplete cache')
    for name, expected in manifest['files_sha256'].items():
        if sha256(cache/name) != expected:
            raise ValueError(f'modified cache {name}')
    for name, expected in json.loads((cache/'input_sha256.json').read_text()).items():
        if sha256(name) != expected:
            raise ValueError(f'modified raw input {name}')
    with np.load(feature_path/'clip.npz') as features:
        features = {name: features[name].copy() for name in features.files}
    observations = json.loads((cache/'observations.json').read_text())
    for obs in observations:
        with np.load(cache/obs['feature_file']) as f:
            for key in ['descriptor', 'local_xyz', 'local_features', 'support_xyz', 'pixels']:
                obs[key] = f[key].copy()
    return (cache, observations, json.loads((cache/'frames.json').read_text()),
            np.load(cache/'poses.npy'), features)


def pooled_graph(source, output, scene):
    cache, observations, frames, poses, features = load(source, scene)
    data = cache.parents[1]/'data'/scene
    out = output/'graph_runs'/scene
    out.mkdir(parents=True, exist_ok=False)
    views = []
    for frame in frames:
        fid = frame['source_frame']
        with np.load(cache/'frames'/f'{fid:04d}.npz') as f:
            views.append(dict(owner=f['owner'].copy(), valid=f['valid'].copy(),
                depth=np.array(Image.open(data/'depth'/f'depth_{fid}.png'))[::8, ::8]/1000.,
                stride=8, intrinsic=np.array([[600.,0,599.5],[0,600.,339.5],[0,0,1.]])))
    stats = {}
    for name in ['dino', 'clip']:
        graph = CompetitiveGraph(views)
        current = []
        for obs in observations:
            o = dict(obs)
            # Keep local support coordinates/counts and every graph condition.
            # Replace only appearance by a repeated pooled descriptor.
            descriptor = obs['descriptor'] if name == 'dino' else features['image'][obs['observation_id']]
            o['local_features'] = np.repeat(descriptor[None], len(obs['local_xyz']), axis=0).astype(np.float32)
            current.append(o)
        start = time.monotonic()
        causal = np.zeros(len(observations), np.int32)
        for frame in frames:
            ids = frame['observations']
            causal[ids] = graph.add_frame([current[i] for i in ids], poses)
        np.savez_compressed(out/(name+'.npz'), causal=causal, final=graph.labels())
        stats[name] = dict(wall_s=time.monotonic()-start, config=asdict(CompetitiveConfig()))
    (out/'run.json').write_text(json.dumps(dict(complete=True, frames=len(frames),
        source_cache=str(cache), adapter_sha256=sha256(Path(__file__)), methods=stats), indent=2))


def _labels(values, n):
    assert set(values) == set(range(n))
    return np.array([values[i] for i in range(n)], np.int32)


def replay_stress(source, output, scene):
    from omegaconf import OmegaConf
    import torch
    torch.set_num_threads(1)
    cache, observations, frames, poses, features = load(source, scene)
    data = cache.parents[1]/'data'/scene
    out = output/'replay_runs'/scene
    out.mkdir(parents=True, exist_ok=False)
    cfg = OmegaConf.create(json.loads((source/'conceptgraphs_runs'/scene/'mapping_config.json').read_text()))
    records = []
    for k, frame in enumerate(frames):
        fid = frame['source_frame']
        rgb = np.array(Image.open(data/'rgb'/f'rgb_{fid}.png').convert('RGB'))
        depth = np.array(Image.open(data/'depth'/f'depth_{fid}.png'))/1000.
        group = []
        for oid in frame['observations']:
            obs = observations[oid]
            # Cached pixels are uint16: promote before forming linear indices.
            # Otherwise row*width overflows and changes native jitter order.
            uv = obs['pixels'].astype(np.int64)
            uv = uv[np.argsort(uv[:,1]*depth.shape[1]+uv[:,0])]
            u,v = uv[:,0],uv[:,1]
            z = depth[v,u]
            xyz = np.column_stack([(u-599.5)*z/600., (v-339.5)*z/600., z])
            group.append(FusionObservation(oid, k, int(obs['label'].split('_')[-1]), xyz,
                rgb[v,u]/255., features['image'][oid], features['text'][oid]))
        records.append(group)
    factory = lambda: ConceptGraphsBackend(cfg, seed=0)
    n = len(observations)
    start = time.monotonic()
    clean = ReplayableObjectMap(factory)
    for k, group in enumerate(records):
        clean.add_frame(k, group, poses[k])
    clean_labels = _labels(clean.labels(), n)
    clean_causal = _labels(clean.causal_labels(), n)
    with np.load(source/'conceptgraphs_runs'/scene/'assignments.npz') as reference:
        # This adapter cannot silently change the clean baseline.
        np.testing.assert_array_equal(clean_labels, reference['final'])
        np.testing.assert_array_equal(clean_causal, reference['causal'])
    np.savez_compressed(out/'clean.npz', causal=clean_causal, final=clean_labels)
    clean_s = time.monotonic()-start
    corrupted = poses.copy()
    for k in range(len(poses)):
        fraction = k/max(len(poses)-1, 1)
        angle = np.deg2rad(3)*fraction
        delta = np.eye(4)
        delta[:2,:2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle),np.cos(angle)]]
        delta[0,3] = .2*fraction
        corrupted[k] = delta@poses[k]
    np.savez_compressed(out/'pose_protocol.npz', provided=poses, synthetic_drift=corrupted)
    state = ReplayableObjectMap(factory)
    for k, group in enumerate(records):
        state.add_frame(k, group, corrupted[k])
    drift_labels = _labels(state.labels(), n)
    drift_causal = _labels(state.causal_labels(), n)
    np.savez_compressed(out/'frozen_identity.npz', causal=drift_causal, final=drift_labels)
    start = time.monotonic()
    corrected = _labels(state.revise_poses(dict(enumerate(poses))), n)
    correction_s = time.monotonic()-start
    np.testing.assert_array_equal(corrected, clean_labels)
    np.testing.assert_array_equal(_labels(state.causal_labels(), n), drift_causal)
    np.savez_compressed(out/'replayed.npz', causal=drift_causal, final=corrected)
    # Compare actual reconstructed geometry, not labels alone.
    clean_objects, corrected_objects = clean.export_objects(), state.export_objects()
    assert len(clean_objects) == len(corrected_objects)
    for expected, actual in zip(clean_objects, corrected_objects):
        assert expected['observation_ids'] == actual['observation_ids']
        for key in ['points', 'colors', 'box', 'image_feature', 'text_feature']:
            np.testing.assert_allclose(expected[key], actual[key], rtol=0, atol=1e-10)
    with gzip.open(out/'objects.pkl.gz', 'wb') as f:
        pickle.dump(dict(revision=state.revision, objects=corrected_objects), f)
    events = list(state.events)
    restored = _labels(state.revise_poses(dict(enumerate(corrupted))), n)
    np.testing.assert_array_equal(restored, drift_labels)
    np.testing.assert_array_equal(_labels(state.causal_labels(), n), drift_causal)
    (out/'events.json').write_text(json.dumps(events, indent=2))
    (out/'run.json').write_text(json.dumps(dict(complete=True, frames=len(frames), observations=n,
        clean_native_equivalent=True, corrected_matches_clean=True, corrected_geometry_matches_clean=True,
        roundtrip_restores_drift=True, causal_preserved=True, clean_s=clean_s, correction_s=correction_s,
        observations_changed=int(np.count_nonzero(corrected!=drift_labels)),
        source_cache=str(cache), adapter_sha256=sha256(Path(__file__)),
        core_sha256=sha256(Path(__file__).with_name('replay_fusion.py')),
        stress='synthetic 0.2m x / 3 degree z ramp; no actual SLAM error claim'), indent=2))
    print(scene, 'replay validation complete', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['graph', 'replay'])
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scene', required=True)
    args = parser.parse_args()
    (pooled_graph if args.action == 'graph' else replay_stress)(args.source, args.output, args.scene)


if __name__ == '__main__':
    main()
