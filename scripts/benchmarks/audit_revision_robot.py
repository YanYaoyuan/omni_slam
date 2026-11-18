"""Independent robot export, observation and local-identity checks (no GT)."""
from pathlib import Path
import itertools
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from paper_final_common import read, dump, sha
from run_revision_robot import OUT, KEY, RAW, SEM, verify, check_seal
from omni_semantic_map.batch_stable_transport import native_library, nearest_groups, nearest_owner


def map_paths(p):
    rows = {'baseline': Path(p['contexts'][0]['ours'])}
    rows.update({m: OUT / 'cover/maps' / KEY / m for m in
                 ['geometry_history', 'single_history', 'cover_current', 'cover_history']})
    rows.update({m: OUT / 'authorization/maps' / KEY / m for m in
                 ['identity_only', 'write_only', 'current_only', 'ours']})
    return rows


def run():
    tick = time.monotonic()
    p = verify()
    assert not (OUT / 'delivery_audit.json').exists()
    check_seal(OUT / 'visibility_complete.json')
    check_seal(OUT / 'observations/prepared.json')
    check_seal(OUT / 'cover/prepared.json')
    covers = check_seal(OUT / 'cover/maps_complete.json')
    auths = check_seal(OUT / 'authorization/maps_complete.json')
    check_seal(OUT / 'authorization/prepared.json')
    assert covers['new_maps'] == auths['new_maps'] == 4
    c = p['contexts'][0]
    full = np.load(Path(c['full']) / 'geometry.npz')['xyz']
    base = dict(np.load(Path(c['ours']) / 'geometry.npz'))
    old = base['instance']; nold = len(old)
    field = dict(np.load(c['field'])); sourcecell = field['source_to_cell']; nc = len(field['cells'])
    atoms = dict(np.load(OUT / 'observations/prepared' / KEY / 'atoms.npz'))
    part = dict(np.load(OUT / 'cover/prepared' / KEY / 'components.npz'))
    d = dict(np.load(OUT / 'authorization/prepared' / KEY / 'field.npz'))
    group = read(OUT / 'authorization/prepared' / KEY / 'groups.json')
    parentfield = dict(np.load(OUT / 'cover/transport' / KEY / 'cover/field.npz'))
    n = len(atoms['cell_ids'])
    # Independent packed-byte union readout; native implementation unpacked bits.
    rawby = {r['frame_id']: r for r in read(RAW)['results'][KEY]}
    semby = {r['frame_id']: r for r in read(SEM)['results'][KEY]}
    visby = {r['frame_id']: r for r in read(c['visibility'])['records']}
    explanations = read(OUT / 'cover/prepared' / KEY / 'explanations.json')
    witnesses = []
    views = read(OUT / 'observations/prepared' / KEY / 'views.json')
    for rank, rec in enumerate(views):
        v = np.load(rec['file']); px = v['pixel_xy']; node = d['cell_atom'][v['cells']]
        visible = np.load(visby[rec['frame_id']]['file'])
        np.testing.assert_array_equal(v['source_ids'], visible['visible_source_ids'])
        np.testing.assert_array_equal(px, visible['pixel_xy'])
        np.testing.assert_array_equal(v['cells'], sourcecell[v['source_ids']])
        semantic = np.load(semby[rec['frame_id']]['file'])['class_map']
        np.testing.assert_array_equal(v['semantic'], semantic[px[:,1], px[:,0]])
        raw = np.load(rawby[rec['frame_id']]['file']); masks = raw['masks_packed']
        for r in explanations:
            co = r['component']
            if r['frame_id'] != rec['frame_id'] or not part['cover_accepted'][co]:
                continue
            if parentfield['root_owner'][part['roots'][co]] <= 0:
                continue
            inside = np.zeros(len(px), bool)
            for q in r['cover']['native_rows']:
                inside |= ((masks[q, px[:, 1], px[:, 0] // 8] >> (7 - px[:, 0] % 8)) & 1).astype(bool)
            take = np.flatnonzero(inside & (node >= 0))
            nn = np.unique(node[take][part['component'][node[take]] == co])
            witnesses.extend((int(i), co, rank) for i in nn)
        if (rank + 1) % 60 == 0:
            print('ROBOT WITNESS AUDIT', rank + 1, '/330', flush=True)
    expected = np.unique(np.asarray(witnesses, np.int64).reshape(-1, 3), axis=0)
    np.testing.assert_array_equal(expected, d['witnesses'])
    dtype = np.dtype([('x',np.int64),('y',np.int64),('z',np.int64)])
    keys = np.ascontiguousarray(field['cells'],np.int64).view(dtype).ravel()
    oldkeys = np.ascontiguousarray(np.floor(base['xyz']/.2),np.int64).view(dtype).ravel()
    pos = np.searchsorted(keys,oldkeys); valid = pos<len(keys)
    valid[valid] &= keys[pos[valid]]==oldkeys[valid]
    oldcell=np.full(nold,-1,np.int64);oldcell[valid]=pos[valid]
    np.testing.assert_array_equal(oldcell,part['old_cell'])
    low=np.full(nc,np.iinfo(np.int64).max,np.int64);high=np.full(nc,-1,np.int64)
    np.minimum.at(low,pos[valid],old[valid]);np.maximum.at(high,pos[valid],old[valid])
    np.testing.assert_array_equal(np.where((low==high)&(low>0),low,0),d['pure_old_cell_id'])
    witness_set = set(map(tuple, expected[:, [0, 2]].tolist()))
    for node, label, frame in d['seeds']:
        assert (int(node), int(frame)) in witness_set
        component = str(int(part['component'][node]))
        oldid = str(int(d['pure_old_cell_id'][atoms['cell_ids'][node]]))
        if oldid in group['aliases'][component]:
            want = group['aliases'][component][oldid]
        else:
            assert not group['members'][component]
            want = group['groups'][component]['recipient']
        assert label == want
    # Analytic approval must equal explicit deletion of every actual view.
    evidence = read(OUT / 'authorization/prepared' / KEY / 'evidence.json')
    deletions = 0
    for pair, r in evidence.items():
        delta = [v['shared'] - v['separate'] for v in r['observations']]
        margins = [sum(delta)] + [sum(delta[:i] + delta[i+1:]) for i in range(len(delta))]
        assert abs(min(margins) - r['worst_after_one_deletion']) < 1e-8, pair
        assert r['approved'] == (min(margins) > 1e-10)
        deletions += len(delta)
    for g in group['groups'].values():
        for a, b in itertools.combinations(g['members'], 2):
            assert evidence['_'.join(map(str, sorted((a, b))))]['approved']
    # Recompute all local nearest-identity distances with a separate solver.
    edges = atoms['local_edges']; a, b = edges // n, edges % n
    centers = field['centers'][atoms['cell_ids']]
    length = np.linalg.norm(centers[a] - centers[b], axis=1)
    graph = csr_matrix((np.r_[length, length], (np.r_[a, b], np.r_[b, a])), shape=(n, n))
    graph.sort_indices()
    np.testing.assert_array_equal(graph.indptr, d['graph_indptr'])
    np.testing.assert_array_equal(graph.indices, d['graph_indices'])
    np.testing.assert_array_equal(graph.data, d['graph_weights'])
    seeds = d['seeds']; labels = np.unique(seeds[:, 1]); distances = []
    for label in labels:
        points = np.unique(seeds[seeds[:, 1] == label, 0])
        distances.append(dijkstra(graph, directed=True, indices=points, min_only=True))
    ordered = np.sort(np.asarray(distances), axis=0)
    nearest = np.full((n, 2), np.inf)
    nearest[:, :min(2, len(labels))] = ordered[:2].T
    np.testing.assert_allclose(nearest, d['owner_distance'], atol=1e-7, rtol=0)
    unambiguous = np.isfinite(nearest[:, 0]) & (nearest[:, 0] < nearest[:, 1] - 1e-10)
    if len(labels):
        actual = labels[np.argmin(np.asarray(distances)[:, unambiguous], axis=0)]
        np.testing.assert_array_equal(actual, d['nearest_owner'][unambiguous, 0])
    counts = np.bincount(seeds[:, 2])
    frames = sorted(np.flatnonzero(counts), key=lambda f: (-counts[f], f))[:2]
    lib = native_library(OUT / 'authorization/native'); removals = []
    for frame in frames:
        remain = seeds[seeds[:, 2] != frame]
        owners, dist = nearest_groups(graph, remain[:, 0], remain[:, 1], lib)
        decision = nearest_owner(owners, dist); known = d['certified_owner'] >= 0
        np.testing.assert_array_equal(decision[known], d['certified_owner'][known])
        removals.append(dict(frame_rank=int(frame), frame_id=views[frame]['frame_id'],
                             removed_seeds=int(counts[frame]), certified_cells=int(known.sum()), changed=0))
    paths = map_paths(p); maps = {m: dict(np.load(path / 'geometry.npz')) for m, path in paths.items()}
    parent = maps['cover_history']; source = parent['added_full_source_ids']
    nodes = d['cell_atom'][sourcecell[source]]; direct = nodes >= 0
    direct[direct] &= part['cover_accepted'][part['component'][nodes[direct]]]
    direct[direct] &= parentfield['root_owner'][part['root'][nodes[direct]]] > 0
    witnessed = np.zeros(len(source), bool); witnessed[direct] = d['witness_node'][nodes[direct]]
    local = np.full(len(source), -1, np.int64); local[direct] = d['certified_owner'][nodes[direct]]
    history = old.copy()
    for co, g in group['groups'].items():
        selected = (d['old_component'] == int(co)) & np.isin(old, g['accepted']) & (parent['instance'][:nold] == g['recipient'])
        ii = np.flatnonzero(selected)
        selected[ii] &= d['certified_owner'][d['cell_atom'][part['old_cell'][ii]]] == g['recipient']
        history[selected] = g['recipient']
    original_class = {r['object_id']: r for r in read(Path(c['ours']) / 'classification.json')['objects']}
    result = {}; assignments = {}; nearest_base = cKDTree(base['xyz']).query(full, workers=1)[0]
    for method, g in maps.items():
        np.testing.assert_array_equal(g['xyz'][:nold], base['xyz'])
        records = read(paths[method] / 'classification.json')
        metadata = {r['object_id']: r for r in records['objects']}
        for oid, record in original_class.items():
            assert metadata[oid] == record
        assert set(np.unique(g['instance'])) <= set(metadata)
        added = np.empty(0, np.int64) if method == 'baseline' else g['added_full_source_ids']
        assert len(added) == len(np.unique(added))
        np.testing.assert_array_equal(g['xyz'][nold:], full[added])
        if method in ['geometry_history', 'single_history', 'cover_current', 'cover_history']:
            mode = method.split('_')[0]
            pf = np.load(OUT / 'cover/transport' / KEY / mode / 'field.npz')
            np.testing.assert_array_equal(pf['duplicate_free'], nearest_base > .05)
            expected_ids = np.flatnonzero((nearest_base > .05) & (pf['certified_owner'][sourcecell] > 0))
            np.testing.assert_array_equal(added, expected_ids)
            np.testing.assert_array_equal(g['instance'][nold:], pf['certified_owner'][sourcecell[added]])
            prefix = old.copy()
            if method != 'cover_current':
                eligible = part['old_cell'] >= 0
                eligible[eligible] &= (pf['cell_direct_owner'][part['old_cell'][eligible]] > 0) & (pf['certified_owner'][part['old_cell'][eligible]] == pf['cell_direct_owner'][part['old_cell'][eligible]])
                prefix[eligible] = pf['cell_direct_owner'][part['old_cell'][eligible]]
        elif method != 'baseline':
            assert records == read(paths['cover_history'] / 'classification.json')
            if method == 'identity_only':
                keep = ~direct | (local >= 0); new = parent['instance'][nold:].copy()
                new[direct] = local[direct]; prefix = history
            elif method == 'write_only':
                keep = direct & witnessed; new = parent['instance'][nold:]; prefix = parent['instance'][:nold]
            else:
                keep = direct & witnessed & (local >= 0); new = local
                prefix = old if method == 'current_only' else history
            np.testing.assert_array_equal(added, source[keep])
            np.testing.assert_array_equal(g['instance'][nold:], new[keep])
        else:
            prefix = old
        np.testing.assert_array_equal(g['instance'][:nold], prefix)
        assigned = np.zeros(len(full), np.int64); assigned[added] = g['instance'][nold:]
        assignments[method] = assigned
        objects = []
        for oid in sorted(metadata):
            before = base['xyz'][old == oid]; after = g['xyz'][g['instance'] == oid]
            objects.append(dict(object_id=int(oid), train_class=metadata[oid]['predictions']['mean']['train_id'], old_points=len(before), points=len(after),
                                net_points=len(after)-len(before),
                                extent_m=np.ptp(after, axis=0).tolist() if len(after) else None))
        changed = old != prefix
        pairs, cnt = np.unique(np.column_stack((old[changed], prefix[changed])), axis=0, return_counts=True)
        result[method] = dict(points=len(g['xyz']), added_measured_sources=len(added),
                              nonempty_ids=len(np.unique(g['instance'])), metadata_ids=len(metadata),
                              class_points={str(t):sum(o['points'] for o in objects if o['train_class']==t) for t in [2,13]},
                              old_labels_changed=int(changed.sum()),
                              transitions=[dict(old_id=int(a), new_id=int(b), points=int(v)) for (a, b), v in zip(pairs, cnt)],
                              objects=objects)
    differences = []
    for a, b in itertools.combinations(paths, 2):
        differences.append(dict(a=a, b=b, added_source_label_differences=int((assignments[a] != assignments[b]).sum()),
                                old_label_differences=int((maps[a]['instance'][:nold] != maps[b]['instance'][:nold]).sum())))
    np.testing.assert_array_equal(maps['ours']['xyz'], maps['current_only']['xyz'])
    np.testing.assert_array_equal(maps['ours']['instance'][nold:], maps['current_only']['instance'][nold:])
    dump(OUT / 'delivery_audit.json', dict(complete=True, results=result, differences=differences,
         raw_cell_frame_witnesses_replayed=len(expected), explicit_pair_view_deletions=deletions,
         scipy_graph_nodes=n, actual_frame_removals=removals, map_export_replays=8,
         source_sha256=sha(__file__), public_gate=False, semantic_accuracy_available=False,
         new_robot_maps=8, new_independent_scenes=0, goal_complete=False,
         seconds=time.monotonic()-tick, protocol_sha256=sha(OUT / 'protocol.json'),
         files={str(path / f):sha(path / f) for path in paths.values() for f in ['geometry.npz', 'classification.json']}))
    print('ROBOT REVISION AUDITED', {m:{k:v[k] for k in ['points', 'old_labels_changed', 'nonempty_ids']} for m,v in result.items()}, flush=True)


if __name__ == '__main__':
    run()
