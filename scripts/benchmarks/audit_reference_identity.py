"""Frozen, prediction-preserving reference-identity sensitivity audit.

Three encodings, the same five sealed predictions and full spatial domain.
This must never overwrite the published-code-compatible primary results.
"""
import argparse
import ast
from pathlib import Path
import runpy
import time
import numpy as np
from scipy.spatial import cKDTree

from paper_final_common import ROOT, read, dump, sha
from evaluate_kitti360_development import read_static_ply, clean_json
from run_map_credit_training import OUT as PARENT, METHODS, verify as verify_parent
from audit_map_credit_capacity import bounds
from run_discovery_query_benchmark import matched_ids
from omni_semantic_map.kitti360_evaluation import evaluate
from omni_semantic_map.kitti360_reference_identity import encode_reference, MODES

OUT = ROOT / 'build/benchmarks/reference_identity_audit_v1'
TAX = ROOT / 'build/datasets/kitti360_prepare/labels.py'
EVAL = ROOT / 'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py'
ARMS = ['baseline'] + METHODS


def freeze():
    assert not OUT.exists()
    p = verify_parent()
    inputs = {}
    for f in [PARENT/'protocol.json', PARENT/'summary.json', PARENT/'predictions_complete.json',
              PARENT/'capacity_audit.json', PARENT/'delivery_audit.json', TAX, EVAL]:
        inputs[str(f)] = sha(f)
    for c in p['contexts']:
        key = c['key']
        old_manifest = ROOT/'build/benchmarks/discovery_measured_inputs_v1'/key/'manifest.json'
        c['camera_geometry'] = read(old_manifest)['geometry']
        files = [c['raw_reference'], c['reference'], c['job'], old_manifest, c['camera_geometry'],
                 Path(c['full'])/'geometry.npz', Path(c['ours'])/'classification.json',
                 PARENT/'inputs'/key/'adjacency.npz', PARENT/'inputs'/key/'candidates.npz']
        for method in ARMS:
            folder = Path(c['ours']) if method == 'baseline' else PARENT/'maps'/key/method
            files += [folder/'geometry.npz', folder/'classification.json',
                      PARENT/'evaluation'/key/method/'reference_prediction.npz']
        for f in files:
            inputs[str(f)] = sha(f)
    sources = [Path(__file__), ROOT/'omni_semantic_map/omni_semantic_map/kitti360_reference_identity.py',
               ROOT/'omni_semantic_map/test/test_kitti360_reference_identity.py',
               ROOT/'scripts/benchmarks/audit_map_credit_capacity.py',
               ROOT/'scripts/benchmarks/evaluate_kitti360_development.py',
               ROOT/'scripts/benchmarks/run_discovery_query_benchmark.py',
               ROOT/'omni_semantic_map/omni_semantic_map/kitti360_evaluation.py']
    dump(OUT/'protocol.json', dict(created_epoch=time.time(), contexts=p['contexts'], methods=ARMS,
        inputs=inputs, sources={str(f):sha(f) for f in sources},
        discovery='Observed raw building11013 and garage34013 both become2013 in0002. The pinned upstream reader applies trainId*1000+rawInstance%1000 and also has this alias behavior. Audit ALL target objects, not only this example.',
        modes={
            'upstream_train':'Exact pinned public reader and frozen prior protocol. Primary replay, not an official server result.',
            'raw_identity':'Retain all target-class raw annotations and all confidence mass. Remap colliding raw semantic/instance pairs to distinct target-class instance IDs. Diagnostic only; tests identity folding without removing garages.',
            'target_only':'Original raw building11/car26 retain IDs; target-class aliases (garage34) become void. Keep every reference position/confidence and prediction; follows target taxonomy semantics but is NOT the pinned public reader or an official score.'},
        experiment='2 already-exposed windows x 5 EXACT frozen predictions x 3 reference encodings =30 complete scores. Zero new fits/maps/model calls, zero parameters changed. All source hashes fixed before scoring. No selection of best reference mode to promote a method.',
        coverage='Report both legacy camera-source geometry used in the earlier formation audit and current full360-degree measured LiDAR. Strict20cm nearest source, raw GT diagnosis only. Neither is new prediction or an achievable object-recovery score.',
        capacity='Recompute the same independent-query optimistic bound for the EXACT previous candidate/old-ID pool under each encoding. This audit cannot make the closed branch pass a changed protocol.',
        consequence='Object-formation diagnoses must preserve raw instance identity and distinguish unknown/ignored taxonomy. Keep upstream-compatible scores; do not teach or validate physical-object formation using unqualified folded aliases. Effective mechanism, same-version robot controls and independent assessment remain open.',
        upstream_urls=[
            'https://raw.githubusercontent.com/autonomousvision/kitti360Scripts/master/kitti360scripts/evaluation/semantic_3d/evalInstanceLevelSemanticLabeling.py',
            'https://raw.githubusercontent.com/autonomousvision/kitti360Scripts/master/kitti360scripts/helpers/labels.py']))
    print('REFERENCE IDENTITY AUDIT FROZEN', flush=True)


def verify():
    p = read(OUT/'protocol.json')
    for f, h in {**p['inputs'], **p['sources']}.items():
        assert sha(f) == h, f
    return p


def public_reader(raw, taxonomy):
    """Execute the pinned reader's actual AST, with only I/O substituted."""
    tree = ast.parse(EVAL.read_text())
    names = {'id2trainIdImg', 'readGTImage'}
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(selected) == 2
    from types import SimpleNamespace
    captured = {}
    class NP:
        def __getattr__(self, k):
            return getattr(np, k)
        def savez(self, name, **values):
            captured.update(values)  # never create/modify an upstream cache
    ns = dict(np=NP(), id2label=taxonomy, read_ply=lambda _:raw,
              os=SimpleNamespace(path=SimpleNamespace(splitext=lambda _:('memory',''), isfile=lambda _:False)))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(EVAL), 'exec'), ns)
    labels, conf = ns['readGTImage']('memory.ply')
    np.testing.assert_array_equal(labels, captured['groundTruthInstance'])
    return labels, conf


def coverage(xyz, query):
    tree = cKDTree(xyz)
    seen = np.zeros(len(query), bool)
    for lo in range(0, len(query), 200000):
        seen[lo:lo+200000] = tree.query(query[lo:lo+200000], distance_upper_bound=.2, workers=1)[0] < .2
    return seen


def run():
    p = verify()
    assert not (OUT/'summary.json').exists()
    taxonomy = runpy.run_path(str(TAX))['id2label']
    old = read(PARENT/'summary.json')
    result, files = {}, {}
    for c in p['contexts']:
        tick = time.monotonic()
        key = c['key']
        folder = OUT/key
        folder.mkdir(parents=True, exist_ok=False)
        raw = read_static_ply(c['raw_reference'])
        ref = np.load(c['reference'])
        weight = ref['confidence'].astype(float)
        original, conf = public_reader(raw, taxonomy)
        np.testing.assert_array_equal(original, ref['train_instance'])
        np.testing.assert_array_equal(conf, weight)
        xyz = np.column_stack([raw[k] for k in ['x','y','z']]).astype(float)
        xyz -= np.asarray(read(c['job'])['origin_world'])
        np.testing.assert_array_equal(xyz, ref['xyz'])
        labels, mappings = {}, {}
        for mode in MODES:
            labels[mode], mappings[mode] = encode_reference(raw['semantic'], raw['instance'], taxonomy, mode)
        np.testing.assert_array_equal(labels['upstream_train'], original)
        np.savez_compressed(folder/'diagnostic_labels.npz', **labels)
        dump(folder/'raw_identity_mapping.json', mappings)
        target = np.isin(original//1000, [2,13])
        support = {}
        for name, source in [('legacy_camera_sources', c['camera_geometry']),
                             ('full_measured_lidar', Path(c['full'])/'geometry.npz')]:
            measured = np.load(source)['xyz']
            seen = np.zeros(len(original), bool)
            seen[target] = coverage(measured, ref['xyz'][target])
            support[name] = seen
            print('REFERENCE COVERAGE', key, name, int(seen.sum()), flush=True)
        np.savez_compressed(folder/'measured_coverage.npz', **support)
        constituents = []
        for row in mappings['upstream_train']:
            take = (raw['semantic'] == row['raw_semantic']) & (raw['instance'] == row['raw_instance'])
            q = ref['xyz'][take]
            constituents.append(dict(**row, points=int(take.sum()), confidence_mass=float(weight[take].sum()),
                centroid=q.mean(0).tolist(), minimum=q.min(0).tolist(), maximum=q.max(0).tolist()))
        groups = []
        for gid in sorted({r['upstream_id'] for r in constituents}):
            parts = [r for r in constituents if r['upstream_id'] == gid]
            if len(parts) < 2:
                continue
            distances = []
            for i, a in enumerate(parts):
                for b in parts[i+1:]:
                    sa = (raw['semantic'] == a['raw_semantic']) & (raw['instance'] == a['raw_instance'])
                    sb = (raw['semantic'] == b['raw_semantic']) & (raw['instance'] == b['raw_instance'])
                    distance = cKDTree(ref['xyz'][sa]).query(ref['xyz'][sb], workers=1)[0].min()
                    distances.append(dict(raw_a=a['raw_instance'], raw_b=b['raw_instance'], minimum_distance_m=float(distance)))
            mass = float(sum(r['confidence_mass'] for r in parts))
            groups.append(dict(upstream_id=gid, confidence_mass=mass, eligible=mass>=100,
                               constituents=parts, separations=distances))
        dump(folder/'alias_groups.json', dict(collisions=groups, all_raw_target_objects=constituents))
        records = read(Path(c['ours'])/'classification.json')['objects']
        classes = {r['object_id']:r['predictions']['mean']['train_id'] for r in records}
        predictions = {r['object_id']:r['predictions']['mean'] for r in records}
        a = np.load(PARENT/'inputs'/key/'adjacency.npz')
        d = np.load(PARENT/'inputs'/key/'candidates.npz')
        query = np.repeat(np.arange(len(a['fallback'])), np.diff(a['offsets']))
        src = a['sources']; valid = d['valid'][src]
        rows = np.broadcast_to(query[:,None], valid.shape)[valid]
        owners = d['owners'][src][valid]
        modes = {}
        for mode in MODES:
            truth = labels[mode]
            mass = np.bincount(truth, weights=weight)
            eligible = np.flatnonzero(np.isin(np.arange(len(mass))//1000, [2,13]) & (mass>=100))
            upper = bounds(truth, weight, a['fallback'], rows, owners, sorted(classes))
            scores, detail = {}, {}
            for method in ARMS:
                predfile = PARENT/'evaluation'/key/method/'reference_prediction.npz'
                pred = np.load(predfile)['instance']
                scored = evaluate(truth, weight, pred, predictions, EVAL)
                scores[method] = dict(metrics=clean_json(scored['averages']),
                                      matched_gt_ids=sorted(matched_ids(scored['matches'])))
                dump(folder/mode/(method+'_matches.json'), clean_json(scored['matches']))
                if mode == 'upstream_train':
                    np.testing.assert_allclose(scores[method]['metrics']['allAp'],
                        old['results'][key]['methods'][method]['metrics']['allAp'], atol=1e-12, rtol=0)
                    assert scores[method]['matched_gt_ids'] == old['results'][key]['methods'][method]['matched_gt_ids']
                n = max(classes)+1
                table = np.bincount(truth*n+pred, weights=weight, minlength=len(mass)*n).reshape(-1,n)
                pm = table.sum(0)
                for gid in eligible:
                    ids = np.array([o for o, cls in classes.items() if cls == gid//1000])
                    inter = table[gid,ids]
                    iou = inter / np.maximum(1e-12, mass[gid]+pm[ids]-inter)
                    best = int(np.argmax(iou))
                    row = detail.setdefault(int(gid), dict(gt_id=int(gid), confidence_mass=float(mass[gid]), methods={}))
                    row['methods'][method] = dict(best_iou=float(iou[best]), object_id=int(ids[best]))
                    assert float(iou[best]) <= max(upper[o][gid] for o in ids)+1e-10
            matched = set(scores['baseline']['matched_gt_ids'])
            for scored in scores.values():
                observed = set(scored['matched_gt_ids'])
                scored['versus_baseline'] = dict(gained=sorted(observed-matched), lost=sorted(matched-observed))
            for gid, row in detail.items():
                ids = [o for o, cls in classes.items() if cls == gid//1000]
                best = max(ids, key=lambda o:upper[o][gid])
                row.update(baseline_matched=gid in matched,
                           fixed_pool_upper_iou=float(upper[best][gid]), bound_owner=int(best))
                row['raw_constituents'] = [v for v in mappings[mode] if v['encoded_id'] == gid]
                row['coverage'] = {name:float(weight[(truth==gid)&seen].sum()/mass[gid])
                                   for name, seen in support.items()}
            allrows = list(detail.values())
            modes[mode] = dict(methods=scores, objects=allrows,
                eligible_gt={name:int(sum(g//1000 == cls for g in eligible)) for cls, name in [(2,'building'),(13,'car')]},
                potential_new_ap50_in_old_pool=[r['gt_id'] for r in allrows if not r['baseline_matched'] and r['fixed_pool_upper_iou']>.5],
                missed_buildings_with_measured_half_coverage={name:[r['gt_id'] for r in allrows
                    if r['gt_id']//1000==2 and not r['baseline_matched'] and r['coverage'][name]>.5] for name in support})
            print('REFERENCE MODE', key, mode, {m:round(s['metrics']['allAp']*100,6) for m,s in scores.items()},
                  'pool possible new', modes[mode]['potential_new_ap50_in_old_pool'], flush=True)
        result[key] = dict(modes=modes, colliding_eligible_gt=[g['upstream_id'] for g in groups if g['eligible']],
            raw_target_objects=len(constituents), public_reader_exact=True, coordinates_confidence_exact=True,
            reference_points=len(original), seconds=time.monotonic()-tick)
        dump(folder/'result.json', result[key])
        for f in folder.rglob('*'):
            if f.is_file(): files[str(f)] = sha(f)
    verify()
    means = {mode:{m:float(np.mean([r['modes'][mode]['methods'][m]['metrics']['allAp']
                                  for r in result.values()])) for m in ARMS} for mode in MODES}
    dump(OUT/'summary.json', dict(complete=True, results=result, means=means, files=files,
        protocol_sha256=sha(OUT/'protocol.json'), original_inputs_unchanged=True,
        complete_score_replays=30, new_maps=0, new_fits=0, new_robot_runs=0, new_independent_scenes=0,
        reference_sensitivity_not_algorithm_improvement=True, goal_complete=False))
    print('REFERENCE IDENTITY AUDIT COMPLETE', means, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze','run'])
    globals()[parser.parse_args().action]()
