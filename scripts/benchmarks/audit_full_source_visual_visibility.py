"""Check whether old camera-only visibility omitted later-visible full sources.

Reuses the same cameras, frames, measured depth and 5cm gate. No masks,
semantic predictions, reference labels or new model calls.
"""
import argparse
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from omni_semantic_map.sparse_lidar_support import project_measured_visibility

B = ROOT/'build/benchmarks'
OUT = B/'full_source_visual_visibility_audit_v1'


def freeze():
    assert not OUT.exists()
    specs = [s for s in read(B/'discovery_measured_inputs_v1/protocol.json')['scenes'] if s['kind']=='outdoor']
    inputs = {}
    for s in specs:
        key = s['key']; m = read(B/'discovery_measured_inputs_v1'/key/'manifest.json')
        paths = [Path(s['job']), B/'discovery_measured_inputs_v1'/key/'manifest.json',
                 B/'full_lidar_surface_inputs_v1'/key/'geometry.npz',
                 B/'full_lidar_surface_inputs_v1'/key/'old_to_full.npy',
                 B/'batch_stable_transport_v1/fields'/key/'full/field.npz',
                 B/'local_visual_boundary_audit_v1'/key/'readout.npz',
                 B/'surface_continuation_probe_v1/features'/key/'edges.npz']
        paths += [Path(f['measured']) for f in read(s['job'])['frames']]
        paths += [Path(r['file']) for r in m['records']]
        inputs.update({str(f):sha(f) for f in paths})
    sources = [Path(__file__), ROOT/'omni_semantic_map/omni_semantic_map/sparse_lidar_support.py']
    dump(OUT/'protocol.json', dict(created_epoch=time.time(), scenes=specs, inputs=inputs,
        sources={str(f):sha(f) for f in sources},
        question='Previous visual audit uses only old camera-source visibility. Are additional full LiDAR sources actually visible later in the same existing images?',
        procedure='Same full chronological measured-source ledger, exact old camera/pose/measured depth, absolute5cm gate and zero relative tolerance; no future sources, depth filling, dilation or tolerance search. Independently require exact replay of every old visible source and pixel.',
        scope='Input availability diagnosis, not a method or map-quality claim. Old visual audit remains correct for its stated cached input domain, not all potentially visible full-source geometry.',
        budget='Same108frames,0newRGB,0foundationmodelcalls,0fits,0maps,0reference reads. Save all visible full-source IDs/pixels and coverage; do not promote a mechanism from coverage alone.'))
    print('FULL SOURCE VISIBILITY FROZEN', flush=True)


def run():
    p = read(OUT/'protocol.json'); assert not (OUT/'summary.json').exists()
    for f, h in {**p['inputs'], **p['sources']}.items(): assert sha(f)==h, f
    results, files = {}, {}
    for s in p['scenes']:
        start = time.monotonic(); key = s['key']; job = read(s['job'])
        m = read(B/'discovery_measured_inputs_v1'/key/'manifest.json')
        with np.load(B/'full_lidar_surface_inputs_v1'/key/'geometry.npz') as f:
            xyz, source_frames, old_ids = f['xyz'], f['source_frame'], f['old_source_id']
        mapping = np.load(B/'full_lidar_surface_inputs_v1'/key/'old_to_full.npy')
        with np.load(B/'batch_stable_transport_v1/fields'/key/'full/field.npz') as f:
            cells, centers = f['source_to_cell'], f['centers']
            denied = (f['cell_owner'] > 0) & (f['frame_owner'] != f['cell_owner'])
        with np.load(B/'surface_continuation_probe_v1/features'/key/'edges.npz') as f:
            left, right, kinds = f['left'], f['right'], f['pair_kind']
        count = np.zeros(len(left), np.int32); node_seen = np.zeros(len(centers), np.int32)
        source_seen = np.zeros(len(xyz), np.int32); newly_seen = np.zeros(len(centers), bool)
        records = []; folder = OUT/key; folder.mkdir(parents=True, exist_ok=False)
        for rec, frame in zip(m['records'], job['frames']):
            assert rec['frame_id'] == frame['frame_id']
            n = int(np.searchsorted(source_frames, frame['frame_id'], side='right'))
            depth = np.load(frame['measured'])['depth']
            x, y, _, visible = project_measured_visibility(xyz[:n], np.asarray(frame['pose']), job['config']['cam'], depth,
                SimpleNamespace(depth_absolute_m=.05, depth_relative=0.))
            ids = np.flatnonzero(visible); pixels = np.column_stack([x[ids], y[ids]])
            with np.load(rec['file']) as f: old, oldpx = f['visible_source_ids'], f['pixel_xy']
            keep_old = old_ids[ids]>=0
            order = np.argsort(old_ids[ids[keep_old]])
            np.testing.assert_array_equal(old_ids[ids[keep_old]][order], old)
            np.testing.assert_array_equal(pixels[keep_old][order], oldpx)
            np.testing.assert_array_equal(ids[keep_old][order], mapping[old])
            nodes = np.unique(cells[ids]); node_seen[nodes] += 1; source_seen[ids] += 1
            obs = np.zeros(len(centers), bool); obs[nodes] = True
            count += obs[left] & obs[right]
            extra = ids[~keep_old]; newly_seen[cells[extra]] = True
            dest = folder/f'{frame["frame_id"]:06d}.npz'
            np.savez_compressed(dest, visible_source_ids=ids, pixel_xy=pixels)
            files[str(dest)] = sha(dest)
            records.append(dict(frame_id=frame['frame_id'], file=str(dest), sha256=files[str(dest)],
                visible_sources=len(ids), old_visible_sources=len(old), additional_sources=len(extra),
                old_visibility_exact=True, additional_observed_later=int((source_frames[extra]<frame['frame_id']).sum())))
        previous = np.load(B/'local_visual_boundary_audit_v1'/key/'readout.npz')
        old_nodes = previous['node_observation_count']>0; old_edges = previous['count']>0
        assert np.all(node_seen[old_nodes]>0) and np.all(count[old_edges]>0)
        inc = np.zeros(len(centers), bool); inc[left[count>0]]=True; inc[right[count>0]]=True
        stats = dict(full_sources=len(xyz), observed_sources=int((source_seen>0).sum()),
            newly_observed_sources=int(((source_seen>0)&(old_ids<0)).sum()),
            old_observed_cells=int(old_nodes.sum()), full_observed_cells=int((node_seen>0).sum()),
            additional_observed_cells=int(((node_seen>0)&~old_nodes).sum()),
            old_observed_edges=int(old_edges.sum()), full_observed_edges=int((count>0).sum()),
            additional_observed_edges=int(((count>0)&~old_edges).sum()),
            denied_cells=int(denied.sum()), denied_cells_observed=int((denied&(node_seen>0)).sum()),
            denied_cells_newly_observed=int((denied&(node_seen>0)&~old_nodes).sum()),
            denied_cells_with_observed_incident_edge=int((denied&inc).sum()),
            foreground_boundary_edges_observed=int(((count>0)&(kinds==2)).sum()),
            frames=len(records), all_old_visibility_exact=True, seconds=time.monotonic()-start)
        np.savez_compressed(folder/'coverage.npz', source_observation_count=source_seen, node_observation_count=node_seen,
                            edge_observation_count=count, additional_source_cells=newly_seen)
        dump(folder/'manifest.json', dict(complete=True, records=records, stats=stats))
        for f in [folder/'coverage.npz', folder/'manifest.json']: files[str(f)] = sha(f)
        results[key]=stats; print('FULL VISUAL COVERAGE',key,stats,flush=True)
    dump(OUT/'summary.json', dict(complete=True, results=results, files=files,
        protocol_sha256=sha(OUT/'protocol.json'), new_maps=0, training_fits=0, new_foundation_model_calls=0, reference_labels_read=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['freeze','run']); globals()[parser.parse_args().action]()
