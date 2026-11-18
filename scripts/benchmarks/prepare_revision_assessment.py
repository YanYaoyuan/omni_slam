"""Prepare the frozen 0009/0010 assessment with the unchanged measured kernel.

No semantic model inference or annotation decoding. Registration, data checks,
and exact old input-kernel validation are prerequisites. No old output is edited.
"""
import argparse
import time
from pathlib import Path
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from freeze_revision_assessment import OUT, DATA
from run_intrinsic_public_holdout import isolated_adapter

INPUT = OUT / 'measured'
FULL = OUT / 'full_measured'
META = ROOT / 'build/datasets/kitti360_prepare'
OLD = ROOT / 'build/benchmarks/batch_transport_assessment_v1'


def verify():
    p = read(OUT / 'protocol.json')
    seal = read(OUT / 'input_adapter_protocol.json')
    for f, h in {**p['input_hashes'], **p['source_hashes'],
                 **seal['inputs'], **seal['sources']}.items():
        assert sha(f) == h, f
    return p


def adapter():
    n = isolated_adapter('prepare_batch_transport_assessment.py')
    n.OUT, n.DATA, n.INPUT, n.FULL = OUT, DATA, INPUT, FULL
    n.verify = verify
    # Only an output description changes. Mathematical/source kernels stay exact.
    native_dump = n.dump
    def scoped_dump(path, data):
        if isinstance(data, dict) and data.get('kind') == 'kitti360_reserved_validation':
            data = dict(data, scope='revision_assessment_v1 frozen four-method replication; '
                        'opaque references until all eight final maps are sealed.')
        native_dump(path, data)
    n.dump = scoped_dump
    assert n.prepare.__globals__ is n.__dict__
    for name, value in [('OUT', OUT), ('DATA', DATA), ('INPUT', INPUT), ('FULL', FULL)]:
        assert n.prepare.__globals__[name] == value
    return n


def freeze():
    assert not (OUT / 'input_adapter_protocol.json').exists()
    p = read(OUT / 'protocol.json')
    for f, h in {**p['input_hashes'], **p['source_hashes']}.items():
        assert sha(f) == h, f
    v = read(DATA / 'verification.json')
    separation = read(DATA / 'geometry_separation_audit.json')
    assert v['complete'] and not v['reference_labels_decoded']
    assert separation['complete'] and separation['separation_proven']
    assert read(DATA / 'exposure_audit.json')['complete']
    kernel = ROOT / 'scripts/benchmarks/prepare_batch_transport_assessment.py'
    gold = read(OLD / 'input_golden.json')
    assert gold['complete'] and sha(kernel) == gold['source_sha256']
    # Check the actual code dependencies under which the original golden ran.
    prior = read(OLD / 'protocol.json')
    used = ['omni_semantic_map/omni_semantic_map/kitti360_input.py',
            'omni_semantic_map/omni_semantic_map/sparse_lidar_support.py',
            'omni_semantic_map/omni_semantic_map/measured_query_cells.py']
    old_execution = read(OLD / 'execution_protocol.json')
    old_front = read(ROOT / 'build/benchmarks/discovery_query_benchmark_v1/protocol.json')
    evidence_hashes = {**prior['source_hashes'], **old_execution['source_hashes'],
                       **old_front['source_hashes']}
    sources = {str(kernel): sha(kernel), str(Path(__file__)): sha(__file__)}
    for relative in used:
        path = ROOT / relative
        assert str(path) in evidence_hashes, path
        assert sha(path) == evidence_hashes[str(path)], path
        sources[str(path)] = sha(path)
    inputs = {str(OUT / 'protocol.json'): sha(OUT / 'protocol.json')}
    for f in ['verification.json', 'geometry_separation_audit.json', 'exposure_audit.json',
              'download_provenance.json', 'window_selection.json']:
        inputs[str(DATA / f)] = sha(DATA / f)
    inputs.update(v['files_sha256'])
    for path in [OLD / 'input_golden.json', OLD / 'protocol.json', OLD / 'execution_protocol.json',
                 ROOT / 'build/benchmarks/discovery_query_benchmark_v1/protocol.json',
                 ROOT / 'build/benchmarks/hypersim_shared_v1/jobs/ai_001_001.json']:
        inputs[str(path)] = sha(path)
    for scene in p['scenes']:
        for path in (META / 'extracted' / scene['window']['scene']).rglob('*.txt'):
            inputs[str(path)] = sha(path)
    for path in (META / 'extracted/calibration').glob('*.txt'):
        inputs[str(path)] = sha(path)
    adapter()  # Checks isolated namespace/path bindings before execution.
    dump(OUT / 'input_adapter_protocol.json', dict(
        complete=True, created_epoch=time.time(), inputs=inputs, sources=sources,
        no_algorithm_changes=True, no_reference_decode=True,
        change='Only isolated OUT/DATA/INPUT/FULL roots, verification, and job scope text.',
        golden_validation='Reuse, not rerun, the unchanged kernel validation: all108oldframes '
            'and7468489fullsourcefields plus12synthetic boundary frames. Source hashes '
            'and geometry dependencies verified against the original validation context.',
        actual_new_checks='After preparation: full source provenance, exact old-to-full '
            'mapping, per-frame raw-return coordinate reconstruction, finite geometry and '
            'causal visibility domain. No models or reference labels.'))
    dump(OUT / 'input_golden.json', dict(gold,
        reused_from=str(OLD / 'input_golden.json'),
        reused_sha256=sha(OLD / 'input_golden.json'), new_golden_run=False))
    print('MEASURED INPUT ADAPTER FROZEN', flush=True)


def prepare():
    verify()
    n = adapter()
    n.prepare()


def audit():
    p = verify()
    assert not (OUT / 'input_delivery_audit.json').exists()
    prepared = read(OUT / 'prepared_complete.json')
    assert prepared['complete'] and not prepared['reference_labels_decoded']
    for f, h in prepared['files'].items():
        assert sha(f) == h, f
    from omni_semantic_map.kitti360_input import load_left_calibration
    calib = load_left_calibration(META / 'extracted/calibration')
    transform = calib['rectified_from_velodyne']
    results = {}
    for scene in p['scenes']:
        key = scene['key']
        old = dict(np.load(INPUT / key / 'geometry.npz'))
        full = dict(np.load(FULL / key / 'geometry.npz'))
        mapping = np.load(FULL / key / 'old_to_full.npy')
        np.testing.assert_array_equal(full['xyz'][mapping], old['xyz'])
        for name in ['source_frame', 'source_raw_index']:
            np.testing.assert_array_equal(full[name][mapping], old[name])
        tokens = (full['source_frame'].astype(np.uint64) << np.uint64(32)) | full['source_raw_index'].astype(np.uint64)
        assert len(np.unique(tokens)) == len(tokens)
        assert np.all(tokens[1:] > tokens[:-1])
        assert np.isfinite(full['xyz']).all()
        job = read(scene['job'])
        frame_ids = {f['frame_id'] for f in job['frames']}
        assert set(full['source_frame']).issubset(frame_ids)
        largest_error = 0.
        count = 0
        for frame in job['frames']:
            ids = np.flatnonzero(full['source_frame'] == frame['frame_id'])
            raw = np.fromfile(frame['raw_scan'], dtype='<f4').reshape(-1, 4)
            ri = full['source_raw_index'][ids]
            assert np.all((ri >= 0) & (ri < len(raw)))
            xyz = raw[ri, :3].astype(float)
            camera = xyz @ transform[:3, :3].T + transform[:3, 3]
            pose = np.asarray(frame['pose'])
            world = camera @ pose[:3, :3].T + pose[:3, 3]
            # Array batch size can affect BLAS last-bit rounding; preserve an
            # explicit numerical tolerance, never label an allclose as exact.
            error = float(np.max(np.abs(world - full['xyz'][ids]), initial=0.))
            assert error <= 1e-9, (key, frame['frame_id'], error)
            largest_error = max(largest_error, error)
            count += len(ids)
        assert count == len(full['xyz'])
        manifest = read(INPUT / key / 'manifest.json')
        for row in manifest['records']:
            visible = np.load(row['file'])['visible_source_ids']
            assert np.all((visible >= 0) & (visible < row['map_points']))
            assert np.all(old['source_frame'][visible] <= row['frame_id'])
        results[key] = dict(frames=len(job['frames']), old_sources=len(old['xyz']),
            full_sources=len(full['xyz']), old_mapping_exact=True,
            all_full_sources_reconstructed=True, maximum_coordinate_error_m=largest_error,
            reconstruction_tolerance_m=1e-9, causal_visibility_domain=True)
        print('NEW MEASURED SOURCE AUDIT', key, results[key], flush=True)
    dump(OUT / 'input_delivery_audit.json', dict(complete=True, checks=results,
        files_verified=len(prepared['files']), source_sha256=sha(__file__),
        prepared_sha256=sha(OUT / 'prepared_complete.json'),
        input_adapter_sha256=sha(OUT / 'input_adapter_protocol.json'),
        reused_kernel_golden=True, semantic_inference=False,
        reference_labels_decoded=False, goal_complete=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'prepare', 'audit'])
    globals()[parser.parse_args().action]()
