"""Pre-assessment bug audit: an ambiguous semantic winner is not background.

Never rewrites frozen experiments or reads reference labels. Recomputes all
development graph assignments, so a harmless-looking seed change is not assumed
to preserve the reported maps.
"""
import time
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from paper_final_common import ROOT, read, dump, sha
from run_batch_stable_transport import graph_and_seeds
from omni_semantic_map.batch_stable_transport import native_library, nearest_groups, nearest_owner, certify

OUT = ROOT / 'build/benchmarks/batch_transport_assessment_v1/unknown_seed_audit'


def known_semantics(sem):
    result = {k: sem[k] for k in ['owners', 'seen', 'mean_response']}
    # The frozen graph helper tests seen > 0 for background seeds. Keep the
    # cached observations intact; only suppress invalid winners at this boundary.
    result['seen'] = result['seen'].copy()
    result['seen'][(result['mean_response'] < 0) | (result['mean_response'] >= 19)] = 0
    return result


def corrected_graph_and_seeds(xyz, old_to_domain, sem, source_frames):
    return graph_and_seeds(xyz, old_to_domain, known_semantics(sem), source_frames)


def main():
    assert not (OUT / 'complete.json').exists()
    OUT.mkdir(parents=True, exist_ok=True)
    lib = native_library(OUT / 'native')
    results = {}; all_equal = True
    for key in ['kitti360_0000', 'kitti360_0002']:
        sem_path = ROOT / 'build/benchmarks/closed_vocabulary_semantics_v1/aggregate' / key / 'source.npz'
        sem = np.load(sem_path); fixed = known_semantics(sem)
        affected = np.flatnonzero((sem['owners'] == 0) & (sem['seen'] > 0) & (fixed['seen'] == 0))
        for domain in ['full', 'camera']:
            start = time.monotonic()
            old_path = ROOT / 'build/benchmarks/batch_stable_transport_v1/fields' / key / domain / 'field.npz'
            old = np.load(old_path)
            if domain == 'full':
                folder = ROOT / 'build/benchmarks/full_lidar_surface_inputs_v1' / key
                geometry = np.load(folder / 'geometry.npz'); mapping = np.load(folder / 'old_to_full.npy')
            else:
                manifest = read(ROOT / 'build/benchmarks/discovery_measured_inputs_v1' / key / 'manifest.json')
                geometry = np.load(manifest['geometry']); mapping = np.arange(len(sem['owners']))
            owner = fixed['owners']; active = (owner > 0) | ((owner == 0) & (fixed['seen'] > 0) & ~np.isin(fixed['mean_response'], [2, 13]))
            ids = np.flatnonzero(active)
            seeds = np.unique(np.column_stack([old['source_to_cell'][mapping[ids]], owner[ids], geometry['source_frame'][mapping[ids]]]), axis=0)
            same_seeds = np.array_equal(seeds, old['seeds'])
            changes = {}; fields = {}
            if not same_seeds:
                n = len(old['centers'])
                graph = csr_matrix((old['graph_weights'], old['graph_indices'], old['graph_indptr']), shape=(n, n))
                labels, dist = nearest_groups(graph, seeds[:, 0], seeds[:, 1], lib)
                fields['nearest_owner'] = nearest_owner(labels, dist)
                fields['same_cell_owner'] = np.where(dist[:, 0] == 0, fields['nearest_owner'], -1)
                for grouping, column in [('cell', 0), ('frame', 2)]:
                    if grouping == 'cell' and domain == 'camera': continue
                    pairs, tokens = np.unique(seeds[:, [1, column]], axis=0, return_inverse=True)
                    tg, td = nearest_groups(graph, seeds[:, 0], tokens, lib)
                    fields[grouping + '_owner'] = certify(labels, dist, tg, td, pairs[:, 0])
                for name, value in fields.items():
                    original = old[name] if name != 'same_cell_owner' else np.where(old['label_distance'][:, 0] == 0, old['nearest_owner'], -1)
                    changes[name] = int(np.count_nonzero(value != original))
                    all_equal &= changes[name] == 0
                np.savez_compressed(OUT / (key + '_' + domain + '.npz'), seeds=seeds, **fields)
            row = dict(affected_source_ids=affected.tolist(), original_seed_records=len(old['seeds']), corrected_seed_records=len(seeds), identical_seeds=same_seeds, changed_cells=changes, original_field_sha256=sha(old_path), semantic_sha256=sha(sem_path), seconds=time.monotonic()-start)
            results[key + '/' + domain] = row
            print(key, domain, row, flush=True)
    dump(OUT / 'complete.json', dict(complete=True, all_development_assignments_identical=bool(all_equal), results=results, source_sha256=sha(__file__), reference_labels_read=False, source_domain_and_export_unchanged=True))


if __name__ == '__main__': main()
