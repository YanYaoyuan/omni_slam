"""Measure available local visual evidence on the frozen full LiDAR graph.

No training, reference labels, map generation, new RGB inference or thresholds.
Frozen predicted owners serve only as conditional diagnostic pseudo labels.
"""
import argparse
import time
from pathlib import Path
import numpy as np
from PIL import Image
from paper_final_common import ROOT, read, dump, sha
from omni_semantic_map.patch_edge_evidence import bilinear_edge_cosine

B = ROOT/'build/benchmarks'
OUT = B/'local_visual_boundary_audit_v1'
KEYS = ['kitti360_0000', 'kitti360_0002']
KINDS = ['same_background', 'same_foreground', 'different_foreground', 'foreground_background']


def freeze():
    assert not OUT.exists()
    capture = read(B/'persistent_source_memory_v1/capture_complete.json')
    edge_seal = read(B/'surface_continuation_probe_v1/features_complete.json')
    assert capture['complete'] and edge_seal['complete']
    files = {}
    for key in KEYS:
        mpath = B/'discovery_measured_inputs_v1'/key/'manifest.json'
        m = read(mpath); assert m['complete']
        paths = [mpath, Path(m['geometry']), B/'full_lidar_surface_inputs_v1'/key/'old_to_full.npy',
                 B/'batch_stable_transport_v1/fields'/key/'full/field.npz',
                 B/'surface_continuation_probe_v1/features'/key/'edges.npz']
        for r in m['records']:
            feature = B/'persistent_source_memory_v1/features'/key/f'{r["frame_id"]:06d}.npz'
            assert sha(feature) == capture['files'][str(feature)]
            assert sha(r['file']) == r['sha256']
            paths.extend([Path(r['file']), Path(r['rgb']), feature])
        assert sha(m['geometry']) == m['geometry_sha256']
        edge = paths[4]; assert sha(edge) == edge_seal['files'][str(edge)]
        files.update({str(p):sha(p) for p in paths})
    sources = [Path(__file__), ROOT/'omni_semantic_map/omni_semantic_map/patch_edge_evidence.py',
               ROOT/'omni_semantic_map/test/test_patch_edge_evidence.py']
    dump(OUT/'protocol.json', dict(created_epoch=time.time(), inputs=files,
        sources={str(p):sha(p) for p in sources}, scenes=KEYS,
        question='Do cached local image tokens distinguish graph neighbors beyond projected pixel distance, and cover cells denied by the source-frame certificate?',
        scope='Two already exposed development windows; conditional on existing predicted anchor ownership. No GT correctness, independent test, official leaderboard or map improvement claim.',
        features='Reuse all108 causal VGGT layer23 grids,10x37x2048. Exact bilinear pixel-center border sampling, cosine via Gram matrix; arithmetic mean over available same-frame edge observations. No missing-feature imputation.',
        projection='Each observed20cm graph cell uses its visible measured source closest to the actual representative, with sourceID tie-break. The pixel is a measured cell proxy, not a projection/visibility claim for the representative point itself. No label-dependent source selection.',
        controls='Rank same-vs-different frozen owners by raw cosine, negative mean pixel distance, negative geometric edge length on identical observed edges. Foreground comparisons separate from background-dominated overall AUROC. Same-patch rate describes resolution, not proof that bilinear features are identical.',
        output='All edge observation counts,mean cosine,mean pixel distance,same-nearest-token fractions; per-type coverage; denied cell observation/incident-edge coverage. Readout is descriptive, no tuneable acceptance gate.',
        budget='0training fits,0new model forwards,0newmaps,0reference reads. No automatic promotion or fullmap experiment from this audit.'))
    print('LOCAL VISUAL AUDIT FROZEN', flush=True)


def run():
    from sklearn.metrics import roc_auc_score
    p = read(OUT/'protocol.json')
    assert not (OUT/'summary.json').exists()
    for f, h in {**p['inputs'], **p['sources']}.items(): assert sha(f) == h, f
    results, files = {}, {}
    for key in KEYS:
        started = time.monotonic()
        m = read(B/'discovery_measured_inputs_v1'/key/'manifest.json')
        xyz = np.load(m['geometry'])['xyz']
        mapping = np.load(B/'full_lidar_surface_inputs_v1'/key/'old_to_full.npy')
        with np.load(B/'batch_stable_transport_v1/fields'/key/'full/field.npz') as f:
            cell = f['source_to_cell'][mapping]; centers = f['centers']
            denied = (f['cell_owner'] > 0) & (f['frame_owner'] != f['cell_owner'])
        with np.load(B/'surface_continuation_probe_v1/features'/key/'edges.npz') as f:
            left, right, kinds, length = f['left'], f['right'], f['pair_kind'], f['length']
        count = np.zeros(len(left), np.int32)
        cosine_sum = np.zeros(len(left)); pixel_sum = np.zeros(len(left)); alias = np.zeros(len(left), np.int32)
        node_seen = np.zeros(len(centers), np.int32)
        proxy_dist = np.linalg.norm(xyz-centers[cell], axis=1)
        frame_stats = []
        for i, rec in enumerate(m['records']):
            with np.load(rec['file']) as f: ids, pixels = f['visible_source_ids'], f['pixel_xy']
            order = np.lexsort((ids, proxy_dist[ids], cell[ids]))
            _, first = np.unique(cell[ids][order], return_index=True)
            select = order[first]; nodes = cell[ids[select]]; xy = pixels[select]
            assert len(nodes) == len(np.unique(nodes))
            node_seen[nodes] += 1
            lookup = np.full(len(centers), -1, np.int32); lookup[nodes] = np.arange(len(nodes))
            a, b = lookup[left], lookup[right]
            edges = np.flatnonzero((a >= 0) & (b >= 0)); aa, bb = a[edges], b[edges]
            gridpath = B/'persistent_source_memory_v1/features'/key/f'{rec["frame_id"]:06d}.npz'
            with np.load(gridpath) as f:
                grid = f['grid']; causal = f['causal_frame_ids']
            np.testing.assert_array_equal(causal, [r['frame_id'] for r in m['records'][max(0, i-3):i+1]])
            assert grid.shape == (10, 37, 2048)
            with Image.open(rec['rgb']) as image: w, h = image.size
            cosine, valid = bilinear_edge_cosine(grid, xy, (h, w), aa, bb)
            edges, aa, bb, cosine = edges[valid], aa[valid], bb[valid], cosine[valid]
            pixel_distance = np.linalg.norm(xy[aa].astype(float)-xy[bb], axis=1)
            token = np.floor((xy+.5)*[37/w, 10/h]).astype(int)
            count[edges] += 1; cosine_sum[edges] += cosine; pixel_sum[edges] += pixel_distance
            alias[edges] += (token[aa] == token[bb]).all(1)
            frame_stats.append(dict(frame_id=rec['frame_id'], visible_cells=len(nodes), observed_edges=len(edges),
                proxy_offset_quantiles=np.quantile(proxy_dist[ids[select]], [0, .5, .95, 1]).tolist()))
            if (i+1) % 10 == 0: print('VISUAL EDGES', key, i+1, '/', len(m['records']), flush=True)
        observed = count > 0
        mean_cosine = np.full(len(left), np.nan); mean_pixel = mean_cosine.copy(); same_patch = mean_cosine.copy()
        mean_cosine[observed] = cosine_sum[observed]/count[observed]
        mean_pixel[observed] = pixel_sum[observed]/count[observed]
        same_patch[observed] = alias[observed]/count[observed]
        ranking = {}
        for name, pos, neg in [('all_known', [0, 1], [2, 3]), ('foreground_vs_foreground', [1], [2]),
                               ('foreground_vs_background_boundary', [1], [3])]:
            take = observed & np.isin(kinds, pos+neg); y = np.isin(kinds[take], pos)
            scores = {'visual_cosine':mean_cosine[take], 'negative_pixel_distance':-mean_pixel[take], 'negative_edge_length':-length[take]}
            ranking[name] = dict(positive_edges=int(y.sum()), negative_edges=int((~y).sum()),
                auroc={k:float(roc_auc_score(y, s)) if len(np.unique(y))==2 else None for k,s in scores.items()})
        per_type = {}
        for j, label in enumerate(KINDS+['unknown']):
            kind = j if j < 4 else -1; all_edges = kinds == kind; take = observed & all_edges
            per_type[label] = dict(total=int(all_edges.sum()), observed=int(take.sum()),
                all_views_same_nearest_patch=int((take & (alias == count)).sum()),
                cosine_quantiles=np.quantile(mean_cosine[take], [0, .1, .5, .9, 1]).tolist() if take.any() else [],
                pixel_distance_quantiles=np.quantile(mean_pixel[take], [0, .1, .5, .9, 1]).tolist() if take.any() else [])
        incident = np.zeros(len(centers), bool); incident[left[observed]] = True; incident[right[observed]] = True
        stats = dict(nodes=len(centers), edges=len(left), observed_edges=int(observed.sum()),
            observed_cells=int((node_seen > 0).sum()), denied_cells=int(denied.sum()),
            denied_cells_observed=int((denied & (node_seen > 0)).sum()),
            denied_cells_with_observed_incident_edge=int((denied & incident).sum()),
            rankings=ranking, pair_types=per_type, frames=frame_stats, seconds=time.monotonic()-started)
        folder = OUT/key; folder.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(folder/'readout.npz', count=count, mean_cosine=mean_cosine, mean_pixel_distance=mean_pixel,
                            same_patch_fraction=same_patch, node_observation_count=node_seen)
        dump(folder/'summary.json', stats)
        for path in folder.iterdir(): files[str(path)] = sha(path)
        results[key] = stats
        print('VISUAL AUDIT', key, {k:v for k,v in stats.items() if k not in ['frames', 'pair_types']}, flush=True)
    dump(OUT/'summary.json', dict(complete=True, results=results, files=files,
        protocol_sha256=sha(OUT/'protocol.json'), new_maps=0, training_fits=0,
        new_foundation_model_calls=0, reference_labels_read=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['freeze', 'run'])
    globals()[parser.parse_args().action]()
