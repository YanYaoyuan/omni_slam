"""Seal development predictions, then score one public static reference window.

Full-window AP50 uses the pinned official kernel with documented compatibility
adapters. The measured-support restriction is a separate custom diagnostic.
Neither is an official leaderboard submission or an independent test result.
"""
import argparse
from pathlib import Path
import runpy
import time
import numpy as np
from scipy.spatial import cKDTree
from paper_final_common import ROOT, read, dump, sha
from prepare_kitti360_frontend import OUT as NATIVE, DATA, META
from kitti360_sparse_mapping import OUT as SPARSE
from omni_semantic_map.kitti360_evaluation import evaluate, EVALUATOR_SHA256

OUT = SPARSE/'evaluation'
EVALUATOR = META/'evalInstanceLevelSemanticLabeling.py'
METHODS = dict(native_indoor_history2=NATIVE/'maps/cg/history2', cg_sparse=SPARSE/'maps/cg/cg',
    history2=SPARSE/'maps/cg/history2', sor=SPARSE/'maps/cleanup/sor',
    ours_visibility1=SPARSE/'maps/cleanup/visibility1', simple_ratio=SPARSE/'maps/cleanup/simple_ratio',
    mc_point_ratio=SPARSE/'maps/cleanup/mc_point_ratio')


def read_static_ply(path):
    """Strict reader for the released fixed binary schema, no faces/lists."""
    expected = [('float','x'), ('float','y'), ('float','z'), ('uchar','red'), ('uchar','green'),
                ('uchar','blue'), ('int','semantic'), ('int','instance'), ('uchar','visible'), ('float','confidence')]
    properties = []
    count = None
    with Path(path).open('rb') as f:
        assert f.readline() == b'ply\n'
        assert f.readline() == b'format binary_little_endian 1.0\n'
        for _ in range(50):
            line = f.readline().decode('ascii').strip()
            if line.startswith('element vertex '): count = int(line.split()[-1])
            elif line.startswith('property '): properties.append(tuple(line.split()[1:]))
            elif line.startswith('element '): raise ValueError('Unexpected PLY element')
            elif line == 'end_header': break
        else: raise ValueError('Incomplete PLY header')
        offset = f.tell()
    assert properties == expected and count is not None
    types = dict(float='<f4', uchar='u1', int='<i4')
    dtype = np.dtype([(name, types[kind]) for kind, name in properties])
    assert Path(path).stat().st_size == offset+count*dtype.itemsize
    return np.memmap(path, dtype=dtype, mode='r', offset=offset, shape=(count,))


def freeze():
    assert not (OUT/'protocol.json').exists()
    job = NATIVE/'jobs/2013_05_28_drive_0000_sync.json'
    m = read(job)
    cache = Path(m['output'])/'cache'
    rows = read(META/'window_selection.json')['windows']
    assert rows[0]['scene'] == m['scene']
    reference = DATA/'opaque_references'/rows[0]['member']
    hashes = {str(reference):sha(reference), str(job):sha(job), str(cache/'manifest.json'):sha(cache/'manifest.json'),
        str(cache/'geometry.npz'):sha(cache/'geometry.npz'), str(EVALUATOR):sha(EVALUATOR),
        str(META/'labels.py'):sha(META/'labels.py'), str(META/'annotation.py'):sha(META/'annotation.py')}
    assert hashes[str(EVALUATOR)] == EVALUATOR_SHA256
    for method, folder in METHODS.items():
        for name in ['geometry.npz', 'classification.json']:
            hashes[str(folder/name)] = sha(folder/name)
    for root in [NATIVE, SPARSE]:
        assert read(root/'maps/cg/run.json')['complete'] and read(root/'maps/cleanup/run.json')['complete']
        hashes[str(root/'mapping_protocol.json')] = sha(root/'mapping_protocol.json')
    dump(OUT/'protocol.json', dict(created_epoch=time.time(), no_reference_labels_decoded_yet=True,
        reference=str(reference), job=str(job), cache=str(cache), methods={k:str(v) for k,v in METHODS.items()},
        files_sha256=hashes, source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),
            ROOT/'omni_semantic_map/omni_semantic_map/kitti360_evaluation.py']},
        export_radius_m=.2, observed_support_radius_m=.2,
        export='Nearest point of the entire predicted map within 20cm, including non-target objects as label zero; deterministic source-index tie break. No GT class/instance access in projection.',
        primary='Full released static window, confidence-weighted AP50 for building/car using the pinned upstream matching/AP functions. Explicit confidence-count adapter repairs the upstream constructor mismatch; NumPy legacy dtype aliases supplied locally. Local adapted evaluation, not official server output.',
        secondary='Same AP kernel restricted to reference points within 20cm of the complete original measured-source cache. Shared domain independent of each predicted map. Observed-support diagnostic, not official AP.',
        ap_thresholds='Upstream 0.50..0.90 plus 0.25; primary AP50 unaffected by documented website/code AP95 discrepancy.',
        confidence='Prediction ranking uses fixed mean-descriptor cosine score; reference confidence weights areas/intersections. Minimum region is sum of confidence >=100.',
        scope='Exposed development window 0000 frames2..385; input60 sample times. No test-generalization or moving-object tracking claim.',
        stop='Evaluate these sealed outputs once, retain empty native transfer and all alternative filters. No threshold sweep or test-window access.'))


def verify():
    p = read(OUT/'protocol.json')
    for file, expected in p['files_sha256'].items(): assert sha(file) == expected, file
    for file, expected in p['source_sha256'].items(): assert sha(ROOT/file) == expected, file
    return p


def transfer(xyz, instance, reference, radius):
    result = np.zeros(len(reference), np.int64)
    if not len(xyz): return result
    tree = cKDTree(xyz)
    for start in range(0, len(reference), 200000):
        query = reference[start:start+200000]
        distance, nearest = tree.query(query, k=min(2, len(xyz)), workers=1, distance_upper_bound=radius)
        if len(xyz) == 1: distance, nearest = distance[:, None], nearest[:, None]
        valid = np.isfinite(distance[:, 0])
        selected = nearest[:, 0].copy()
        # Tie decisions cannot depend on cKDTree traversal order.
        if len(xyz) > 1:
            tied = np.flatnonzero(valid & (distance[:, 0] == distance[:, 1]))
            for i in tied:
                candidates = np.array(tree.query_ball_point(query[i], np.nextafter(distance[i, 0], np.inf)))
                d2 = ((xyz[candidates]-query[i])**2).sum(1)
                selected[i] = np.min(candidates[d2 == d2.min()])
        part = result[start:start+len(query)]
        part[valid] = instance[selected[valid]]
    return result


def clean_json(value):
    if isinstance(value, dict): return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray): return clean_json(value.tolist())
    if isinstance(value, np.generic): return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def run():
    p = verify()
    assert not (OUT/'summary.json').exists()
    job = read(p['job'])
    raw = read_static_ply(p['reference'])
    reference = np.column_stack([raw[k] for k in ['x','y','z']]).astype(np.float64)-np.array(job['origin_world'])
    assert np.isfinite(reference).all()
    labels = runpy.run_path(str(META/'labels.py'))['id2label']
    semantic = raw['semantic'].copy()
    train = np.array([labels[int(i)].trainId for i in semantic], dtype=np.int64)
    truth = train*1000+raw['instance'].astype(np.int64)%1000
    truth[train == 255] = 0
    weights = raw['confidence'].astype(np.float64)
    measured = np.load(Path(p['cache'])/'geometry.npz')['xyz']
    tree = cKDTree(measured)
    observed = np.zeros(len(reference), bool)
    for start in range(0, len(reference), 200000):
        distance, _ = tree.query(reference[start:start+200000], distance_upper_bound=p['observed_support_radius_m'], workers=1)
        observed[start:start+len(distance)] = np.isfinite(distance)
    np.savez_compressed(OUT/'reference.npz', xyz=reference, train_instance=truth, confidence=weights, observed_support=observed)
    coverage = []
    for oid in np.unique(truth[np.isin(train, [2, 13])]):
        target = truth == oid
        total, seen = weights[target].sum(), weights[target & observed].sum()
        coverage.append(dict(gt_id=int(oid), train_id=int(oid//1000), full_confidence_mass=float(total),
            observed_confidence_mass=float(seen), observed_fraction=float(seen/total) if total else None))
    results = {}
    for method, folder in p['methods'].items():
        folder = Path(folder)
        geometry = np.load(folder/'geometry.npz')
        objects = read(folder/'classification.json')['objects']
        predictions = {r['object_id']:r['predictions']['mean'] for r in objects if r['predictions']['mean']['retained']}
        source_labels = np.where(np.isin(geometry['instance'], list(predictions)), geometry['instance'], 0)
        projected = transfer(geometry['xyz'], source_labels, reference, p['export_radius_m'])
        np.savez_compressed(OUT/(method+'_prediction.npz'), instance=projected)
        domains = {}
        for name, domain in [('full_window', np.ones(len(truth), bool)), ('observed_support', observed)]:
            result = evaluate(truth[domain], weights[domain], projected[domain], predictions, EVALUATOR)
            domains[name] = dict(averages=result['averages'], thresholds=result['thresholds'],
                gt_eligible={label:sum(r['vert_count'] >= 100 for r in rows) for label, rows in result['matches']['window']['groundTruth'].items()},
                predictions_eligible={label:len(rows) for label, rows in result['matches']['window']['prediction'].items()})
            dump(OUT/(method+'_'+name+'_matches.json'), clean_json(result['matches']))
        results[method] = domains
        print(method, {name:round(row['averages']['allAp50%']*100, 4) for name, row in domains.items()}, flush=True)
    dump(OUT/'summary.json', clean_json(dict(complete=True, protocol_sha256=sha(OUT/'protocol.json'),
        scope=p['scope'], official_server_score=False, independent_test=False, reference_points=len(reference),
        observed_reference_points=int(observed.sum()), gt_coverage=coverage, methods=results,
        files_sha256={path.name:sha(path) for path in OUT.glob('*.npz')})))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze','run'])
    args = parser.parse_args()
    globals()[args.action]()
