"""Frozen four-method execution on the reserved 0009/0010 windows.

All source kernels are isolated from old output roots. Subsetting parent export
loops avoids unregistered extra map variants; exposed golden exports must match
before any new image-model inference. Scoring is intentionally separate.
"""
import argparse
import copy
import inspect
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from prepare_revision_assessment import OUT, INPUT, FULL, verify as verify_inputs
from run_intrinsic_public_holdout import isolated_adapter
from run_batch_transport_assessment_empty import adapted as backbone_adapter
from audit_transport_unknown_seeds import corrected_graph_and_seeds

B = ROOT / 'build/benchmarks'
OLD = B / 'batch_transport_assessment_v1'
PUBLIC = B / 'revision_authorization_v1'
OBS = OUT / 'observations'
COVER = OUT / 'cover'
AUTH = OUT / 'authorization'


def sealed(path):
    data = read(path)
    assert data['complete'], path
    for f, h in data.get('files', {}).items():
        assert sha(f) == h, f
    return data


def subset_cover(module):
    """Execute only the registered parent cover_history arm."""
    source = inspect.getsource(module.maps)
    substitutions = [
        ("for mode in ['geometry','single','cover']:", "for mode in ['cover']:"),
        ("names=['cover_current','cover_history'] if mode=='cover' else [mode+'_history']",
         "names=['cover_history']"),
        ('int(oldids.max())+1', 'int(oldids.max(initial=0))+1')]
    for old, new in substitutions:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    exec(compile(source, str(__file__) + ':cover_subset', 'exec'), module.__dict__)
    return module


def counted_dump(path, data):
    data = copy.deepcopy(data)
    if Path(path).name == 'maps_complete.json':
        data['new_maps'] = sum(len(v) for v in data['results'].values())
    dump(path, data)


def verify(data=False):
    e = read(OUT / 'execution_protocol.json')
    for f, h in {**e['sources'], **e['inputs']}.items():
        assert sha(f) == h, f
    p = copy.deepcopy(read(OUT / 'protocol.json'))
    if data:
        verify_inputs()
        sealed(OUT / 'prepared_complete.json')
    p.update(prompts=['car', 'building'], semantic_ids=[26, 11], sam3_threshold=.5,
             sam3_resolution=1008, checkpoint=p['sam3_checkpoint'],
             checkpoint_sha256=p['sam3_sha256'],
             rotations=read(B / 'intrinsic_surface_ownership_v1/protocol.json')['rotations'])
    return p


def stage_start(name):
    verify(True)
    path = OUT / 'stages' / (name + '.json')
    assert not path.exists(), 'Existing stage: verify its live handle or terminal result before recovery'
    dump(path, dict(stage=name, started_epoch=time.time(),
         execution_protocol_sha256=sha(OUT / 'execution_protocol.json')))


def backbone():
    n = isolated_adapter('run_batch_transport_assessment.py')
    n.OUT, n.INPUT, n.FULL = OUT, INPUT, FULL
    n.BASE, n.SEM, n.BACK = OUT / 'baseline', OUT / 'semantics', OUT / 'backbone'
    n.verify, n.stage_start = verify, stage_start
    n.isolated_adapter = backbone_adapter
    n.graph_and_seeds = corrected_graph_and_seeds
    n.METHODS = ['ours_frame']  # Sole current strong backbone, no extra scored variants.
    assert n.maps.__globals__ is n.__dict__
    return n


def validate():
    p = verify_inputs()
    dest = OUT / 'execution_validation'
    assert not dest.exists()
    dest.mkdir()
    from run_revision_authorization import verify as verify_public
    public = verify_public()
    # The only changed cover arithmetic is empty old-ID compatibility. Nonempty
    # old scenes must exactly replay the complete original exported maps.
    parent = B / 'object_mask_cover_v1'
    oldobs = B / 'censored_object_partition_v1'
    for root, oldroot in [(dest / 'cover', parent), (dest / 'authorization', PUBLIC)]:
        shutil.copytree(oldroot / 'prepared', root / 'prepared')
        shutil.copyfile(oldroot / 'prepared.json', root / 'prepared.json')
        shutil.copyfile(oldroot / 'protocol.json', root / 'protocol.json')
    n = subset_cover(isolated_adapter('run_object_mask_cover.py'))
    n.OUT, n.PARENT, n.RAW = dest / 'cover', oldobs, B / 'prethreshold_candidates_v1/predictions_complete.json'
    n.verify = lambda: public
    n.dump = counted_dump
    n.maps()
    a = isolated_adapter('run_revision_authorization.py')
    a.OUT, a.PARENT, a.OBS, a.RAW = dest / 'authorization', dest / 'cover', oldobs, n.RAW
    a.verify = lambda: public
    a.NEW, a.dump = ['ours'], counted_dump
    a.maps()
    rows = {}
    for c in public['contexts']:
        key = c['key']
        rows[key] = {}
        for method, actual, expected in [
            ('unconditional', dest / 'cover/maps' / key / 'cover_history', parent / 'maps' / key / 'cover_history'),
            ('full_authorization', dest / 'authorization/maps' / key / 'ours', PUBLIC / 'maps' / key / 'ours')]:
            got, want = dict(np.load(actual / 'geometry.npz')), dict(np.load(expected / 'geometry.npz'))
            assert set(got) == set(want)
            for name in got:
                np.testing.assert_array_equal(got[name], want[name])
            assert read(actual / 'classification.json') == read(expected / 'classification.json')
            rows[key][method] = dict(all_geometry_arrays_exact=True, class_scores_exact=True,
                expected_sha256=sha(expected / 'geometry.npz'), actual_sha256=sha(actual / 'geometry.npz'))
        base = np.load(Path(c['ours']) / 'geometry.npz')
        parent_map = dict(np.load(dest / 'cover/maps' / key / 'cover_history/geometry.npz'))
        authorized = np.load(dest / 'authorization/maps' / key / 'ours/geometry.npz')['instance']
        labels = np.r_[authorized[:len(base['xyz'])], parent_map['instance'][len(base['xyz']):]]
        expected = np.load(B / 'revision_factorial_v1/maps' / key / 'authorized_history_only/geometry.npz')
        np.testing.assert_array_equal(labels, expected['instance'])
        rows[key]['history_only'] = dict(all_labels_exact=True)
    # Check the same validated empty-input adapters are still used, and replay
    # source ownership on both old scenes. No references or new RGB are read.
    empty = read(OLD / 'empty_input_amendment.json')
    assert sha(ROOT / 'scripts/benchmarks/run_batch_transport_assessment_empty.py') == empty['source_sha256']
    prior = read(OLD / 'unknown_semantic_amendment.json')
    for f, h in {**prior['source_hashes'], **prior['runtime_dependency_hashes']}.items():
        assert sha(f) == h, f
    n = backbone()
    for key in rows:
        expected = np.load(B / 'closed_vocabulary_semantics_v1/aggregate' / key / 'source.npz')['owners']
        got = n.source_owners(B / 'surface_ownership_v1/features' / key / 'core_sources.json',
            B / 'intrinsic_surface_ownership_v1/maps' / key / 'intrinsic_geometry',
            B / 'signed_query_association_v1/maps' / key / 'geometry_graph', len(expected))
        np.testing.assert_array_equal(got, expected)
        rows[key]['backbone_source_owners_exact'] = True
    # Prior full backbone validation is reused under pinned sources; not rerun
    # or miscounted as new-scene inference.
    old_execution = read(OLD / 'execution_protocol.json')
    for f, h in old_execution['source_hashes'].items():
        assert sha(f) == h, f
    dump(dest / 'complete.json', dict(complete=True, results=rows,
        source_sha256=sha(__file__), registration_sha256=sha(OUT / 'protocol.json'),
        reused_backbone_validation_sha256=sha(OLD / 'backbone_validation/validation.json'),
        reused_empty_validation_sha256=sha(OLD / 'empty_input_amendment.json'),
        new_rgb_inference=False, new_reference_decoded=False))
    print('EXECUTION GOLDEN PASSED', rows, flush=True)


def freeze():
    p = verify_inputs()
    assert not (OUT / 'execution_protocol.json').exists()
    v = read(OUT / 'execution_validation/complete.json')
    assert v['complete'] and v['source_sha256'] == sha(__file__)
    prior = read(OLD / 'execution_protocol.json')
    front = read(B / 'discovery_query_benchmark_v1/protocol.json')
    sources = dict(front['source_hashes'])
    sources.update(prior['source_hashes'])
    sources.update(p['source_hashes'])
    for f, h in sources.items():
        assert sha(f) == h, f
    for name in ['run_revision_assessment.py', 'run_batch_transport_assessment_empty.py',
                 'run_batch_transport_assessment_corrected.py', 'audit_transport_unknown_seeds.py',
                 'run_revision_robot.py', 'audit_full_source_visual_visibility.py']:
        path = ROOT / 'scripts/benchmarks' / name
        sources[str(path)] = sha(path)
    inputs = {str(OUT / f): sha(OUT / f) for f in [
        'protocol.json', 'prepared_complete.json', 'input_delivery_audit.json',
        'execution_validation/complete.json', 'robot_correspondence_audit.json']}
    for f in ['backbone_validation/validation.json', 'backbone_validation/golden.json',
              'unknown_semantic_amendment.json', 'empty_input_amendment.json']:
        inputs[str(OLD / f)] = sha(OLD / f)
    for path, row in read(B / 'ovo_full_20261110/model_artifacts.json')['files'].items():
        assert sha(path) == row['sha256']
        inputs[path] = row['sha256']
    model = ROOT / 'build/models/segformer_b5_cityscapes'
    for name, row in read(model / 'source_manifest.json')['files'].items():
        assert sha(model / name) == row['sha256']
        inputs[str(model / name)] = row['sha256']
    runtime = read(OLD / 'unknown_semantic_amendment.json')['runtime_dependency_hashes']
    inputs.update(runtime)
    dump(OUT / 'execution_protocol.json', dict(complete=True, sources=sources, inputs=inputs,
        created_epoch=time.time(), methods=p['methods'],
        unchanged_science=True, original_development_gate=False,
        adaptation='Isolated root bindings; inherited unknown-semantic and empty-input fixes; '
            'export only strong ours_frame, cover_history, history-only and authorization ours. '
            'Cover old-ID maximum uses default0 for empty input. Both exposed complete cover '
            'and authorization maps replay exactly, including ranking metadata.',
        public_visibility='Same causal full-source ledger, 5cm absolute/0relative gate, '
            'official camera/pose/depth. Full-source prefix selected by acquisition frame. '
            'Exact camera-source visibility replay for every new frame. Robot retains its '
            'already documented full-archive offline visibility adapter.',
        no_score_until='All8finalmaps sealed; this runner has no scoring action.',
        stage_policy='Separate started/terminal records; an existing stage cannot silently restart. '
            'Run GPU stages on GPU0 only; shared cached predictions across all methods.'))
    for sub in ['raw', 'ownership_axis', 'ownership_intrinsic', 'semantics']:
        folder = OUT / sub
        folder.mkdir(exist_ok=True)
        shutil.copyfile(OUT / 'protocol.json', folder / 'protocol.json')
    print('FOUR-METHOD EXECUTION FROZEN', flush=True)


def visibility():
    from omni_semantic_map.sparse_lidar_support import project_measured_visibility
    stage_start('visibility')
    p = verify()
    files, results = {}, {}
    for scene in p['scenes']:
        key = scene['key']
        job = read(scene['job'])
        m = read(INPUT / key / 'manifest.json')
        full = dict(np.load(FULL / key / 'geometry.npz'))
        mapping = np.load(FULL / key / 'old_to_full.npy')
        rows = []
        for rec, frame in zip(m['records'], job['frames']):
            assert rec['frame_id'] == frame['frame_id']
            n = int(np.searchsorted(full['source_frame'], frame['frame_id'], side='right'))
            depth = np.load(frame['measured'])['depth']
            x, y, _, visible = project_measured_visibility(full['xyz'][:n], np.asarray(frame['pose']),
                job['config']['cam'], depth, SimpleNamespace(depth_absolute_m=.05, depth_relative=0.))
            ids = np.flatnonzero(visible)
            pixels = np.column_stack([x[ids], y[ids]])
            cached = dict(np.load(rec['file']))
            selected = full['old_source_id'][ids] >= 0
            order = np.argsort(full['old_source_id'][ids[selected]])
            np.testing.assert_array_equal(ids[selected][order], mapping[cached['visible_source_ids']])
            np.testing.assert_array_equal(pixels[selected][order], cached['pixel_xy'])
            path = OUT / 'visibility' / key / f'{frame["frame_id"]:06d}.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, visible_source_ids=ids, pixel_xy=pixels)
            files[str(path)] = sha(path)
            rows.append(dict(frame_id=frame['frame_id'], file=str(path),
                             visible_points=len(ids), camera_prefix_exact=True))
        path = OUT / 'visibility' / key / 'manifest.json'
        dump(path, dict(complete=True, records=rows, geometry=str(FULL / key / 'geometry.npz')))
        files[str(path)] = sha(path)
        results[key] = dict(frames=len(rows), visible_observations=sum(r['visible_points'] for r in rows))
        print('ASSESSMENT FULL VISIBILITY', key, results[key], flush=True)
    dump(OUT / 'visibility_complete.json', dict(complete=True, files=files, results=results))


def contexts():
    p = verify()
    return [dict(key=s['key'], kind='outdoor', job=s['job'],
        full=str(FULL / s['key']), ours=str(OUT / 'final_maps' / s['key'] / 'ours_frame'),
        field=str(OUT / 'fields' / s['key'] / 'field.npz'),
        visibility=str(OUT / 'visibility' / s['key'] / 'manifest.json')) for s in p['scenes']]


def stage_protocol(path, dependencies):
    p = verify()
    e = read(OUT / 'execution_protocol.json')
    inputs = dict(e['inputs'])
    for dep in dependencies:
        d = sealed(dep)
        inputs[str(dep)] = sha(dep)
        inputs.update(d['files'])
    data = dict(contexts=contexts(), inputs=inputs, sources=e['sources'],
                parent_registration_sha256=sha(OUT / 'protocol.json'))
    if path.exists():
        assert read(path) == data, path
    else:
        dump(path, data)


def revision(action):
    stage_start(action)
    common = [OUT / 'visibility_complete.json', OUT / 'predictions_complete.json',
              OUT / 'fields_complete.json', OUT / 'prepared_complete.json',
              OUT / 'raw/predictions_complete.json', OUT / 'semantics/predictions_complete.json']
    if action == 'atoms':
        stage_protocol(OBS / 'input_protocol.json', common)
        n = isolated_adapter('prepare_censored_object_partition.py')
        n.OUT, n.RAW, n.SEM = OBS, OUT / 'raw/predictions_complete.json', OUT / 'semantics/predictions_complete.json'
        n.prepare()
    elif action in ['cover_prepare', 'cover_maps']:
        stage_protocol(COVER / 'protocol.json', common + [OBS / 'prepared.json'])
        n = subset_cover(isolated_adapter('run_object_mask_cover.py'))
        n.OUT, n.PARENT, n.RAW = COVER, OBS, OUT / 'raw/predictions_complete.json'
        n.dump = counted_dump
        (n.prepare if action == 'cover_prepare' else n.maps)()
    else:
        stage_protocol(AUTH / 'protocol.json', common + [OBS / 'prepared.json',
                       COVER / 'prepared.json', COVER / 'maps_complete.json'])
        n = isolated_adapter('run_revision_authorization.py')
        n.OUT, n.PARENT, n.OBS, n.RAW = AUTH, COVER, OBS, OUT / 'raw/predictions_complete.json'
        n.NEW, n.dump = ['ours'], counted_dump
        (n.prepare if action == 'identity_prepare' else n.maps)()


def finalize():
    stage_start('finalize')
    sealed(OUT / 'predictions_complete.json')
    sealed(COVER / 'maps_complete.json')
    sealed(AUTH / 'maps_complete.json')
    files, results = {}, {}
    for c in contexts():
        key = c['key']
        baseline = dict(np.load(Path(c['ours']) / 'geometry.npz'))
        parent = dict(np.load(COVER / 'maps' / key / 'cover_history/geometry.npz'))
        full = dict(np.load(AUTH / 'maps' / key / 'ours/geometry.npz'))
        history = dict(parent)
        history['instance'] = np.r_[full['instance'][:len(baseline['xyz'])], parent['instance'][len(baseline['xyz']):]]
        results[key] = {}
        for method, geometry, cls in [
            ('strong_baseline', baseline, Path(c['ours']) / 'classification.json'),
            ('unconditional', parent, COVER / 'maps' / key / 'cover_history/classification.json'),
            ('history_only', history, COVER / 'maps' / key / 'cover_history/classification.json'),
            ('full_authorization', full, AUTH / 'maps' / key / 'ours/classification.json')]:
            dest = OUT / 'comparison_maps' / key / method
            dest.mkdir(parents=True, exist_ok=False)
            np.savez_compressed(dest / 'geometry.npz', **geometry)
            shutil.copyfile(cls, dest / 'classification.json')
            stats = dict(points=len(geometry['xyz']),
                old_point_id_changes=int((geometry['instance'][:len(baseline['xyz'])] != baseline['instance']).sum()),
                nonempty_ids=len(np.unique(geometry['instance'][geometry['instance'] > 0])))
            dump(dest / 'run.json', stats)
            results[key][method] = stats
            for f in dest.iterdir():
                files[str(f)] = sha(f)
    dump(OUT / 'comparison_maps_complete.json', dict(complete=True, files=files,
        results=results, final_maps=8, registration_sha256=sha(OUT / 'protocol.json'),
        execution_sha256=sha(OUT / 'execution_protocol.json'), reference_labels_decoded=False))
    print('ALL EIGHT FINAL MAPS SEALED', results, flush=True)


def main():
    actions = ['validate', 'freeze', 'masks', 'features', 'supports', 'base_maps',
        'candidates', 'intrinsic', 'raw', 'geometry', 'anchors', 'semantics', 'aggregate',
        'fields', 'strong_maps', 'visibility', 'atoms', 'cover_prepare', 'cover_maps',
        'identity_prepare', 'identity_maps', 'finalize']
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=actions)
    action = parser.parse_args().action
    if action in ['validate', 'freeze', 'visibility', 'finalize']:
        return globals()[action]()
    if action in ['atoms', 'cover_prepare', 'cover_maps', 'identity_prepare', 'identity_maps']:
        return revision(action)
    n = backbone()
    if action in ['masks', 'features', 'supports']:
        n.frontend(action)
    else:
        getattr(n, 'maps' if action == 'strong_maps' else action)()


if __name__ == '__main__':
    main()
