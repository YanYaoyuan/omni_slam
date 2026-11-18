"""Same frozen revision mechanism on all archived garage frames.

CPU replay of existing raw predictions. Public efficacy remains failed;
this separately authorized robot comparison is a diagnostic, not promotion.
"""
import argparse
import copy
from pathlib import Path
import time
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from omni_semantic_map.discovery_measured_input import robot_measured_visibility

B = ROOT / 'build/benchmarks'
OUT = B / 'revision_robot_v1'
PUBLIC = B / 'revision_authorization_v1'
OLD = B / 'batch_transport_robot_v1'
RAW = B / 'robot_geometry_reference_v1/raw/predictions_complete.json'
SEM = OLD / 'semantics/predictions_complete.json'
KEY = 'rosbag2245'


def check_seal(path):
    data = read(path)
    assert data['complete'], path
    for f, h in data.get('files', {}).items():
        assert sha(f) == h, f
    return data


def freeze():
    from run_revision_authorization import verify as verify_public
    verify_public()
    assert not OUT.exists()
    assert read(PUBLIC / 'completion_record.json')['complete']
    assert read(PUBLIC / 'summary.json')['development_gate'] is False
    raw = check_seal(RAW)
    sem = check_seal(SEM)
    assert raw['old_replay_exact']
    scene = read(OLD / 'protocol.json')['scenes'][0]
    job = read(scene['job'])
    assert len(job['frames']) == 330
    ids = [f['frame_id'] for f in job['frames']]
    assert ids == [r['frame_id'] for r in raw['results'][KEY]]
    assert ids == [r['frame_id'] for r in sem['results'][KEY]]
    full = OLD / 'full_measured' / KEY
    base = OLD / 'final_maps' / KEY / 'ours_frame'
    field = OLD / 'fields' / KEY / 'field.npz'
    context = dict(key=KEY, kind='robot', job=scene['job'], full=str(full),
                   ours=str(base), field=str(field),
                   visibility=str(OUT / 'visibility/manifest.json'))
    paths = [RAW, SEM, PUBLIC / 'protocol.json', PUBLIC / 'summary.json',
             PUBLIC / 'completion_record.json', OLD / 'protocol.json',
             OLD / 'summary.json', OLD / 'prediction_audit.json',
             full / 'geometry.npz', base / 'geometry.npz',
             base / 'classification.json', field, Path(scene['job'])]
    # The previous robot run calls its verification artifact summary.json.
    paths = [f for f in paths if f.name != 'prediction_audit.json' or f.exists()]
    paths += [Path(f['cloud']) for f in job['frames']]
    paths += [Path(f['rgb']) for f in job['frames']]
    paths += [Path(f) for f in raw['files']] + [Path(f) for f in sem['files']]
    source_names = [
        'scripts/benchmarks/run_revision_robot.py',
        'scripts/benchmarks/prepare_censored_object_partition.py',
        'scripts/benchmarks/run_object_mask_cover.py',
        'scripts/benchmarks/run_revision_authorization.py',
        'omni_semantic_map/omni_semantic_map/discovery_measured_input.py',
        'omni_semantic_map/omni_semantic_map/mask_cover_revision.py',
        'omni_semantic_map/omni_semantic_map/revision_authorization.py',
        'omni_semantic_map/omni_semantic_map/batch_stable_transport.py',
        'omni_semantic_map/omni_semantic_map/batch_stable_transport.cpp',
    ]
    p = dict(created_epoch=time.time(), contexts=[context],
             inputs={str(f): sha(f) for f in paths},
             sources={str(ROOT / f): sha(ROOT / f) for f in source_names},
             public_gate=False,
             scope='Same-version robot diagnostic, not efficacy or candidate promotion. Public failed gate and old results remain immutable.',
             scope_amendment='Execute the explicitly requested robot comparison despite self-imposed development gate; no unused public assessment is unlocked by this amendment.',
             mechanism='Call unchanged public atom preparation, whole-mask cover, identity competition and write authorization functions. Only paths and frozen contexts differ. Inherited 20cm cells, strict0.4m graph, mask utility, minimum3birthviews, single-view deletion margin and competing-label certificate. No new fit or class exception.',
             visibility='Offline full archived measured map projected into all330frames, actual packet intrinsics/camera_pose and floor-pixel current-cloud depth gate0.05m. Replay all earlier cumulative-prefix visibility exactly. Full-future source availability is NOT online causality.',
             evidence='Reuse all400nativeSAM3queries and full-resolution SegFormerclassmaps from every330frame. No new model calls; pixel evidence recomputed on full final-source visibility. Robot projection adapter differs from public calibrated RGB-D adapter, algorithm remains same.',
             budget='One full330frame visibility pass, one atom/mask-cover/identity pass. Four parent maps and four authorization maps, one reused strong map. No AP, training, GPU inference, threshold search or new scenes.',
             metrics='No independent robotGT. Report exact source provenance, old/new identity changes, actual differences among controls, all object extents, RGB and map views. Do not equate added points with accuracy or claim S100 runtime.',
             methods=['baseline', 'geometry_history', 'single_history', 'cover_current',
                      'cover_history', 'identity_only', 'write_only', 'current_only', 'ours'])
    dump(OUT / 'protocol.json', p)
    print('ROBOT REVISION FROZEN / 330 frames / 8 new maps / no efficacy claim', flush=True)


def verify():
    p = read(OUT / 'protocol.json')
    for f, h in {**p['inputs'], **p['sources']}.items():
        assert sha(f) == h, f
    return p


def visibility():
    p = verify()
    assert not (OUT / 'visibility_complete.json').exists()
    c = p['contexts'][0]
    job = read(c['job'])
    source = np.load(Path(c['full']) / 'geometry.npz')['xyz']
    old = read(B / 'discovery_measured_inputs_v1' / KEY / 'manifest.json')
    byframe = {f['frame_id']: f for f in old['records']}
    rows, files = [], {}
    start = time.monotonic()
    for rank, f in enumerate(job['frames']):
        ids, xy, stats = robot_measured_visibility(source, np.load(f['cloud'])['points'], f)
        prev = byframe[f['frame_id']]
        cached = np.load(prev['file'])
        prefix = ids < prev['map_points']
        np.testing.assert_array_equal(ids[prefix], cached['visible_source_ids'])
        np.testing.assert_array_equal(xy[prefix], cached['pixel_xy'])
        dest = OUT / 'visibility' / 'frames' / f'{f["frame_id"]:06d}.npz'
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, visible_source_ids=ids, pixel_xy=xy)
        files[str(dest)] = sha(dest)
        rows.append(dict(frame_id=f['frame_id'], file=str(dest), rgb=f['rgb'],
                         visible_points=len(ids), prior_prefix_visible=int(prefix.sum()),
                         additional_final_map_visible=int((~prefix).sum()), **stats))
        if (rank + 1) % 30 == 0:
            print('ROBOT FULL VISIBILITY', rank + 1, '/330', flush=True)
    dest = OUT / 'visibility/manifest.json'
    dump(dest, dict(geometry=str(Path(c['full']) / 'geometry.npz'), records=rows))
    files[str(dest)] = sha(dest)
    dump(OUT / 'visibility_complete.json', dict(complete=True, files=files,
         prefix_replays=len(rows), visible_source_observations=sum(r['visible_points'] for r in rows),
         additional_final_map_observations=sum(r['additional_final_map_visible'] for r in rows),
         seconds=time.monotonic()-start, protocol_sha256=sha(OUT / 'protocol.json')))


def stage_protocol(path, dependencies):
    p = copy.deepcopy(verify())
    for dependency in dependencies:
        d = check_seal(dependency)
        p['inputs'][str(dependency)] = sha(dependency)
        p['inputs'].update(d['files'])
    if path.exists():
        assert read(path) == p, path
    else:
        dump(path, p)


def count_correcting_dump(path, data):
    # Public writers hard-code two-window counts. Correct only metadata, never
    # coordinates, labels, evidence, decisions or the public source files.
    data = copy.deepcopy(data)
    if Path(path).name == 'maps_complete.json':
        data['new_maps'] = sum(len(v) for v in data['results'].values())
    dump(path, data)


def run_stage(action):
    obs, cover, auth = OUT / 'observations', OUT / 'cover', OUT / 'authorization'
    if action == 'atoms':
        import prepare_censored_object_partition as mod
        stage_protocol(obs / 'input_protocol.json', [OUT / 'visibility_complete.json'])
        mod.OUT, mod.RAW, mod.SEM = obs, RAW, SEM
        mod.prepare()
    elif action in ('cover_prepare', 'cover_maps'):
        import run_object_mask_cover as mod
        stage_protocol(cover / 'protocol.json', [OUT / 'visibility_complete.json', obs / 'prepared.json'])
        mod.OUT, mod.PARENT, mod.RAW = cover, obs, RAW
        mod.dump = count_correcting_dump
        (mod.prepare if action == 'cover_prepare' else mod.maps)()
    else:
        import run_revision_authorization as mod
        stage_protocol(auth / 'protocol.json', [OUT / 'visibility_complete.json', obs / 'prepared.json',
                       cover / 'prepared.json', cover / 'maps_complete.json'])
        mod.OUT, mod.PARENT, mod.OBS, mod.RAW = auth, cover, obs, RAW
        mod.dump = count_correcting_dump
        (mod.prepare if action == 'identity_prepare' else mod.maps)()
    print('ROBOT REVISION STAGE COMPLETE', action, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'visibility', 'atoms', 'cover_prepare',
                                         'cover_maps', 'identity_prepare', 'identity_maps'])
    action = parser.parse_args().action
    if action == 'freeze':
        freeze()
    elif action == 'visibility':
        visibility()
    else:
        run_stage(action)
