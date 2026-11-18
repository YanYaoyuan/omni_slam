"""Fixed six-scan evidence probe, before any proposed map revision.

Predicted anchors select candidate pairs. GT is read only after native feature
scores are sealed, and labels pairs of surfaces, not predicted object accuracy.
"""
import argparse
from pathlib import Path
import time
import numpy as np
from paper_final_common import ROOT, read, dump, sha

B = ROOT / 'build/benchmarks'
OUT = B / 'utonia_ownership_probe_v1'
ASSETS = B / 'utonia_evidence_v1'
SEED = 6985480


def verify():
    p = read(OUT / 'protocol.json')
    for f, h in {**p['inputs'], **p['sources']}.items():
        assert sha(f) == h, f
    return p


def freeze():
    from audit_observation_anchor_provenance import contexts
    assert not OUT.exists()
    smoke = read(ASSETS / 'execution_check.json')
    assert smoke['execution_passed'] and smoke['repeat_allclose']
    scenes = read(B / 'provenance_object_memory_v1/protocol.json')['scenes']
    contexts_by_key = {c['key']: c for c in contexts()}
    files = {str(ASSETS / name): sha(ASSETS / name) for name in ['assets.json', 'execution_check.json']}
    rows = []
    for scene in scenes:
        key = scene['key']
        assert key in ['kitti360_0000', 'kitti360_0002']
        c = contexts_by_key[key]
        job = read(scene['job'])
        n = len(job['frames'])
        frames = [job['frames'][i] for i in [n // 4, n // 2, 3 * n // 4]]
        row = dict(key=key, frames=frames, geometry=str(c['full'] / 'geometry.npz'),
                   mapping=str(c['full'] / 'old_to_full.npy'), semantic=str(c['sem']),
                   reference=scene['reference'], job=scene['job'])
        rows.append(row)
        for path in [row[k] for k in ['geometry', 'mapping', 'semantic', 'reference', 'job']] + [f['raw_scan'] for f in frames]:
            files[str(path)] = sha(path)
    dump(OUT / 'protocol.json', dict(
        created_epoch=time.time(), scenes=rows, inputs=files, sources={str(Path(__file__)): sha(__file__)},
        question='Does native pretrained 3D cosine distinguish same-object surfaces from nearby environment beyond distance?',
        frames='Exactly floor(N/4),floor(N/2),floor(3N/4) of each existing job; no outcome-based replacement.',
        candidates='Every retained measured source in these scans with original predicted owner=0; up to 3 nearest same-scan sources with original predicted owner>0, Euclidean distance<=2m. Exact distance then source-ID ties. No GT in candidate selection or features.',
        features='Official Utonia base; full raw scan; zero color/normal; native outdoor scale0.2 without center shift; fixed seed6985480; SDPA FP32 patch1024; four-level upcast1386D; no feature fitting. Score raw feature cosine; negative Euclidean distance control. Same pairs, no feature-conditioned filtering.',
        readout='After scores sealed: nearest existing reference within0.2m; equal-distance GT-index ties. Saved truth0 conservatively unknown (ignored/road0 conflation). Both references known train_id1..18 and confidence>0; anchor GT building/car with window mass>=100. Positive iff same train_instance, negative iff other known train_instance. This labels surfaces, not correctness of predicted object IDs.',
        metrics='Confidence-weighted AUROC (min endpoint confidence), all pairs and target absent from original camera-source cache. Also per-frame, different-class-only, same-class-different-instance, and four fixed distance bins [0,.2,.5,1,2]. No independence/significance claim for correlated pairs.',
        gate='In BOTH windows, all-pair and no-old-camera-cache subgroups each have>=200 positives and>=200 negatives, cosine AUROC>=.55 and >=distance AUROC+.02; equal-weight mean over eligible distance-bin AUCs also>=distance+.02 with at least2 eligible bins (>=100 positives and negatives per bin). This is only an information-screen gate, never map quality acceptance.',
        budget='6 new native forwards,0fits,0maps,0APscores. No patch/checkpoint/radius/threshold variants after readout. Failed information screen closes this raw-cosine candidate; passed screen only permits defining one map mechanism with same-feature simple control.',
        scope='Already exposed development data; not independent test, leaderboard score, robot validation, or Ours contribution.'
    ))
    print('UTONIA EVIDENCE PROTOCOL FROZEN', flush=True)


def stable_neighbors(tree, query, points, ids, k=3, radius=2.):
    """Sorted nearest k, retaining exactly <=radius; deterministic kth ties."""
    k = min(k, len(points))
    if not k:
        return np.empty((len(query), 0)), np.empty((len(query), 0), np.int64)
    dist, idx = tree.query(query, k=min(k + 1, len(points)), workers=1)
    if dist.ndim == 1:
        dist, idx = dist[:, None], idx[:, None]
    for i in range(len(query)):
        # Only boundary ties require expanding the neighbor list.
        if dist.shape[1] > k and dist[i, k - 1] == dist[i, k] and dist[i, k - 1] <= radius:
            candidates = np.asarray(tree.query_ball_point(query[i], np.nextafter(dist[i, k], np.inf)))
            d = np.linalg.norm(points[candidates] - query[i], axis=1)
            order = np.lexsort((ids[candidates], d))[:k]
            idx[i, :k], dist[i, :k] = candidates[order], d[order]
        else:
            order = np.lexsort((ids[idx[i]], dist[i]))
            idx[i], dist[i] = idx[i, order], dist[i, order]
    return dist[:, :k], idx[:, :k]


def prepare():
    from scipy.spatial import cKDTree
    p = verify()
    assert not (OUT / 'prepared.json').exists()
    files, stats = {}, {}
    for c in p['scenes']:
        with np.load(c['geometry']) as f:
            xyz, acquisition, raw_index, old_id = [f[k] for k in ['xyz', 'source_frame', 'source_raw_index', 'old_source_id']]
        owner = np.zeros(len(xyz), np.int64)
        owner[np.load(c['mapping'])] = np.load(c['semantic'])['owners']
        stats[c['key']] = []
        for frame in c['frames']:
            fid = frame['frame_id']
            source = np.flatnonzero(acquisition == fid)
            raw = np.fromfile(frame['raw_scan'], np.float32).reshape(-1, 4)
            ri = raw_index[source]
            assert (ri >= 0).all() and (ri < len(raw)).all() and len(np.unique(ri)) == len(ri)
            a = np.flatnonzero(owner[source] > 0)
            q = np.flatnonzero(owner[source] == 0)
            if len(a):
                d, nn = stable_neighbors(cKDTree(xyz[source[a]]), xyz[source[q]], xyz[source[a]], source[a])
                pair_q, rank = np.nonzero(d <= 2.)
                left, right, distance = q[pair_q], a[nn[pair_q, rank]], d[pair_q, rank]
            else:
                left = right = np.empty(0, np.int64)
                distance = np.empty(0)
            path = OUT / 'inputs' / c['key'] / f'{fid:06d}.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, source_id=source, raw_index=ri, xyz=xyz[source], owner=owner[source],
                                no_old_camera_cache=old_id[source] < 0, left=left, right=right, distance=distance)
            files[str(path)] = sha(path)
            stat = dict(frame_id=fid, sources=len(source), raw_points=len(raw), anchors=len(a), queries=len(q), pairs=len(left))
            stats[c['key']].append(stat)
            print('PREPARED', c['key'], stat, flush=True)
    dump(OUT / 'prepared.json', dict(complete=True, files=files, stats=stats, protocol_sha256=sha(OUT / 'protocol.json')))


def infer():
    import torch
    import utonia
    p = verify()
    seal = read(OUT / 'prepared.json')
    assert seal['complete'] and not (OUT / 'scores_complete.json').exists()
    for f, h in seal['files'].items(): assert sha(f) == h
    assets = read(ASSETS / 'assets.json')
    for f, h in assets['files'].items(): assert sha(f) == h
    torch.cuda.set_per_process_memory_fraction(.85)
    utonia.utils.set_seed(SEED)
    model = utonia.load(assets['checkpoint'], custom_config=dict(enable_flash=False, enc_patch_size=[1024] * 5)).cuda().eval()
    transform = utonia.transform.default(.2, apply_z_positive=False)
    files, stats = {}, []
    for c in p['scenes']:
        for frame in c['frames']:
            fid = frame['frame_id']
            output = OUT / 'scores' / c['key'] / f'{fid:06d}.npz'
            assert not output.exists()
            raw = np.fromfile(frame['raw_scan'], np.float32).reshape(-1, 4)
            inp = np.load(OUT / 'inputs' / c['key'] / f'{fid:06d}.npz')
            utonia.utils.set_seed(SEED)
            point = transform(dict(coord=raw[:, :3].copy(), color=np.zeros_like(raw[:, :3]), normal=np.zeros_like(raw[:, :3])))
            inverse = point['inverse'].numpy().copy()
            assert np.linalg.norm(point['coord'].numpy()[inverse] - raw[:, :3] * .2, axis=1).max() < np.sqrt(3) * .01 + 1e-5
            started = time.monotonic()
            with torch.inference_mode():
                point = model({k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in point.items()})
                for _ in range(4):
                    parent = point.pop('pooling_parent')
                    parent.feat = torch.cat([parent.feat, point.feat[point.pop('pooling_inverse')]], -1)
                    point = parent
                assert 'pooling_parent' not in point and point.feat.shape[1] == 1386
                feat = point.feat[inverse[inp['raw_index']]].cpu().numpy().copy()
            assert np.isfinite(feat).all()
            norm = np.linalg.norm(feat, axis=1)
            assert (norm > 0).all()
            feat /= norm[:, None]
            left, right = inp['left'], inp['right']
            cosine = np.empty(len(left), np.float32)
            for start in range(0, len(left), 2048):
                sl = slice(start, start + 2048)
                cosine[sl] = np.einsum('ij,ij->i', feat[left[sl]], feat[right[sl]])
            assert np.isfinite(cosine).all() and (np.abs(cosine) <= 1.00001).all()
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output, cosine=cosine, negative_distance=-inp['distance'])
            files[str(output)] = sha(output)
            stats.append(dict(key=c['key'], frame_id=fid, pairs=len(left), seconds=time.monotonic() - started,
                              peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30))
            print('UTONIA SCORES', stats[-1], flush=True)
            del point, feat
    dump(OUT / 'scores_complete.json', dict(complete=True, files=files, stats=stats, gt_read=False, new_maps=0,
                                           prepared_sha256=sha(OUT / 'prepared.json'), protocol_sha256=sha(OUT / 'protocol.json')))


def summarize(y, score, weights, distance):
    from sklearn.metrics import roc_auc_score
    def auc(s, take):
        return float(roc_auc_score(y[take], s[take], sample_weight=weights[take])) if len(np.unique(y[take])) == 2 else None
    all_ = np.ones(len(y), bool)
    rows = []
    bounds = [0, .2, .5, 1, 2.000000001]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        t = (distance >= lo) & (distance < hi)
        rows.append(dict(lower=lo, upper=min(hi, 2), positive=int(y[t].sum()), negative=int((~y[t]).sum()),
                         cosine_auc=auc(score, t), distance_auc=auc(-distance, t)))
    eligible = [r for r in rows if r['positive'] >= 100 and r['negative'] >= 100]
    return dict(positive=int(y.sum()), negative=int((~y).sum()), cosine_auc=auc(score, all_),
                distance_auc=auc(-distance, all_), distance_bins=rows, eligible_bins=len(eligible),
                bin_macro_cosine_auc=float(np.mean([r['cosine_auc'] for r in eligible])) if eligible else None,
                bin_macro_distance_auc=float(np.mean([r['distance_auc'] for r in eligible])) if eligible else None)


def score():
    from scipy.spatial import cKDTree
    p = verify()
    assert not (OUT / 'summary.json').exists()
    for name in ['prepared', 'scores_complete']:
        s = read(OUT / f'{name}.json'); assert s['complete']
        for f, h in s['files'].items(): assert sha(f) == h
    results = {}
    for c in p['scenes']:
        ref = np.load(c['reference'])
        rxyz, truth, confidence = [ref[k] for k in ['xyz', 'train_instance', 'confidence']]
        tree = cKDTree(rxyz)
        mass = np.bincount(truth, weights=confidence)
        combined = []
        frames = []
        for frame in c['frames']:
            fid = frame['frame_id']
            data = np.load(OUT / 'inputs' / c['key'] / f'{fid:06d}.npz')
            predicted = np.load(OUT / 'scores' / c['key'] / f'{fid:06d}.npz')
            distances, indices = stable_neighbors(tree, data['xyz'], rxyz, np.arange(len(rxyz)), k=1, radius=.2)
            rd, ri = distances[:, 0], indices[:, 0]
            labels, w = truth[ri], confidence[ri]
            known = (rd <= .2) & (labels >= 1000) & (labels < 19000) & (w > 0)
            left, right = data['left'], data['right']
            take = known[left] & known[right] & np.isin(labels[right] // 1000, [2, 13]) & (mass[labels[right]] >= 100)
            left, right = left[take], right[take]
            same = labels[left] == labels[right]
            same_class = labels[left] // 1000 == labels[right] // 1000
            d = data['distance'][take]
            cosine = predicted['cosine'][take]
            weights = np.minimum(w[left], w[right])
            no_camera = data['no_old_camera_cache'][left]
            combined.append((same, same_class, d, cosine, weights, no_camera))
            frames.append(dict(frame_id=fid, candidate_pairs=len(take), scored_pairs=int(take.sum()),
                               excluded_pairs=int((~take).sum()), unique_queries=len(np.unique(left)),
                               all_pairs=summarize(same, cosine, weights, d)))
            path = OUT / 'readout' / c['key'] / f'{fid:06d}.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, eligible=take, same_instance=same, same_class=same_class,
                                weights=weights, no_old_camera_cache=no_camera, reference_index=ri,
                                reference_distance=rd)
        same, same_class, d, cosine, weights, no_camera = [np.concatenate([x[i] for x in combined]) for i in range(6)]
        groups = {}
        for name, t in dict(all_pairs=np.ones(len(same), bool), no_old_camera_cache=no_camera,
                            cross_class_negatives=same | ~same_class, within_class=same_class).items():
            groups[name] = summarize(same[t], cosine[t], weights[t], d[t])
        def passes(r):
            return bool(r['positive'] >= 200 and r['negative'] >= 200 and r['cosine_auc'] >= .55 and
                        r['cosine_auc'] >= r['distance_auc'] + .02 and r['eligible_bins'] >= 2 and
                        r['bin_macro_cosine_auc'] >= r['bin_macro_distance_auc'] + .02)
        passed = all(passes(groups[k]) for k in ['all_pairs', 'no_old_camera_cache'])
        results[c['key']] = dict(frames=frames, groups=groups, information_gate=passed)
        print('READOUT', c['key'], 'GATE', passed, {k: {a: v for a, v in r.items() if a != 'distance_bins'} for k, r in groups.items()}, flush=True)
    dump(OUT / 'summary.json', dict(complete=True, results=results,
         information_gate=all(r['information_gate'] for r in results.values()),
         protocol_sha256=sha(OUT / 'protocol.json'), scores_sha256=sha(OUT / 'scores_complete.json'),
         readout_hashes={str(f): sha(f) for f in (OUT / 'readout').rglob('*.npz')},
         scope=p['scope'], new_maps=0, new_ap_scores=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'prepare', 'infer', 'score'])
    globals()[parser.parse_args().action]()
