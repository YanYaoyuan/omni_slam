"""Frozen offline history recovery controls on the exact same causal map.

No new candidate search: all proposals come from the 41 native ambiguous
decisions already recorded before this experiment. Finalization is offline.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

from audit_evidence_retention import ROOT, OUT, dump
from omni_semantic_map.flat_benchmark import sha256
from omni_semantic_map.history_graph import build_history_graph, contrastive_bridge
from omni_semantic_map.shared_observations import observation_table, select_views
from omni_semantic_map.evidence_readout import aggregate
from omni_semantic_map.full_ovo_evaluation import BACKGROUND, cg_geometry

DEST = OUT / 'history_graph_v1'
CACHE = OUT / 'livo_full_strong_v2/cache_supportcc'
ASSOCIATION = OUT / 'dino_association_v1/baseline/association.json'
BASE = OUT / 'scale_ablation_v1/cache_supportcc/both'
VARIANTS = [('native_bridge_1', 1, False), ('dino_bridge_1', 1, True),
            ('native_bridge_2', 2, False), ('dino_bridge_2', 2, True)]


def read(path):
    return json.loads(path.read_text())


def freeze():
    assert not DEST.exists(), 'do not overwrite a prior experiment'
    sources = [Path(__file__), ROOT / 'omni_semantic_map/omni_semantic_map/history_graph.py',
               ROOT / 'omni_semantic_map/test/test_history_graph.py']
    dependencies = [ASSOCIATION, BASE / 'run.json', BASE / 'classification.json',
                    OUT / 'scale_ablation_v1/protocol.json', CACHE / 'manifest.json',
                    OUT / 'dino_livo_tokens_v2/manifest.json']
    dump(DEST / 'protocol.json', dict(scope=__doc__, variants=VARIANTS,
        sources={str(p): sha256(p) for p in sources},
        dependencies={str(p): sha256(p) for p in dependencies},
        candidates='All native ambiguous query-to-historical-candidate links; map immutable observations to the pre-final baseline fragments. Skip links already internal to a fragment. No class filtering.',
        native_bridge='A proposal is already accepted by the native geometry+SigLIP threshold. This control is NOT pure geometry.',
        dino_bridge='At query time both native and alternate historical branches must have a valid pooled-DINO maximum strictly above the query-to-all-other-valid-same-frame-observations maximum. Missing competitors or evidence gives no DINO vote. Operational competitors are not identity GT.',
        repeated_support='Two predeclared controls: 1 or 2 distinct query frames per edge. Same query frame cannot vote twice. No threshold sweep after results.',
        conflict='Veto any new graph union whose full components share a frame; includes transitive conflicts. This is conservative and can reject same-object split masks.',
        order='Common ordering by native distinct-frame vote count descending, then node IDs, for all variants.',
        finalization='Merge selected pre-final fragments with native CG merge kernel; then unchanged denoise/filter/native guarded final merge and SigLIP unique-frame readout. Final native merges are logged separately and need not obey the graph coframe veto.',
        offline=True, no_gt=True, no_training=True,
        scope_limit='Already viewed rosbag2245, mechanism development only. Graph components use final baseline fragments, not a causal or online claim. One query frame can be insufficient identity evidence even when DINO passes.'))
    for p in sources:
        target = DEST / 'sources' / p.name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(p.read_bytes())


def finalize(engine, groups, output, table):
    from conceptgraph.slam.slam_classes import MapObjectList
    from omni_semantic_map.fusion_diagnostics import guarded_merge, membership_audit
    cfg, utils = engine.config, engine.utils
    originals = {min(o['observation_ids'])+1: o for o in engine._objects}
    objects = MapObjectList()
    for group in groups:
        obj = deepcopy(originals[group[0]])
        for key in group[1:]:
            obj = utils.merge_obj2_into_obj1(cfg, obj, deepcopy(originals[key]), run_dbscan=False)
        objects.append(obj)
    before = membership_audit(objects)
    assert before['repeated_memberships'] == 0
    assert before['unique_observations'] == sum(len(o['observation_ids']) for o in engine._objects)
    filtered = utils.filter_objects(cfg, utils.denoise_objects(cfg, deepcopy(objects)))
    objects = guarded_merge(cfg, deepcopy(filtered))
    after = membership_audit(objects)
    assert after['repeated_memberships'] == 0
    points, labels, identity = cg_geometry([dict(object_id=min(o['observation_ids'])+1,
                                              points=np.asarray(o['pcd'].points)) for o in objects])
    features, frames, areas, text, vocab, pose_by_frame = table
    rows = []
    class_ids = np.array(vocab['replica_class_ids'])
    for record in identity:
        oid = record['evaluation_id']
        members = objects[record['object_index']]['observation_ids']
        selected = np.array(select_views(members, frames, areas, top_k=0))
        center = points[labels == oid].mean(0)
        cameras = np.array([pose_by_frame[int(frames[i])][:3, 3] for i in selected])
        row = dict(object_id=oid, observation_ids=sorted(set(map(int, members))),
                   selected_observation_ids=selected.tolist(), center=center.tolist(),
                   unique_frames=len(selected), predictions={})
        for method in ['mean', 'area_density']:
            result = aggregate(features[selected], areas[selected], cameras, center,
                               method=method, observation_ids=selected)
            scores = text @ result['descriptor']
            order = np.argsort(-scores, kind='stable')
            best = order[0]
            cls = int(class_ids[best])
            row['predictions'][method] = dict(name=vocab['names'][best], replica_class_id=cls,
                score=float(scores[best]), margin=float(scores[best]-scores[order[1]]),
                retained=cls not in BACKGROUND, effective_views=result['effective_views'])
        rows.append(row)
    output.mkdir(exist_ok=False)
    np.savez_compressed(output / 'geometry.npz', xyz=points, instance=labels)
    dump(output / 'classification.json', dict(objects=rows))
    return dict(before_filter=before, after_final=after, objects=rows)


def run():
    spec = read(DEST / 'protocol.json')
    for p, h in {**spec['sources'], **spec['dependencies']}.items():
        assert sha256(Path(p)) == h, p
    assert not (DEST / 'snapshot').exists()
    import run_livo_scale_ablation as scale
    sys.path.insert(0, str(ROOT / 'build/benchmarks/expanded_semantic_20261109/concept-graphs'))
    from omni_semantic_map.replay_fusion import ConceptGraphsBackend
    from conceptgraph.slam import utils
    holder = {}
    original = ConceptGraphsBackend.__init__
    def capture(self, *args, **kwargs):
        original(self, *args, **kwargs)
        holder['engine'] = self
    ConceptGraphsBackend.__init__ = capture
    replay = DEST / 'replay'
    replay.mkdir(exist_ok=False)
    (replay / 'protocol.json').write_bytes((OUT / 'scale_ablation_v1/protocol.json').read_bytes())
    scale.ROOT_OUT = replay
    try:
        scale.execute('cache_supportcc', 'both')
    finally:
        ConceptGraphsBackend.__init__ = original
    exact = replay / 'cache_supportcc/both'
    assert read(exact / 'classification.json') == read(BASE / 'classification.json')
    for k, values in dict(np.load(exact / 'geometry.npz')).items():
        np.testing.assert_array_equal(values, np.load(BASE / 'geometry.npz')[k])
    engine = holder['engine']
    features, frames, areas = observation_table(CACHE)
    manifest = read(CACHE / 'manifest.json')
    poses = np.load(CACHE / 'poses.npy')
    pose_by_frame = {r['frame_id']: poses[i] for i, r in enumerate(manifest['records'])}
    text = np.load(CACHE / 'text_features.npy')
    text = text / np.linalg.norm(text, axis=1, keepdims=True)
    vocab = read(CACHE / 'vocabulary.json')
    table = features, frames, areas, text, vocab, pose_by_frame
    snapshot = DEST / 'snapshot'
    snapshot.mkdir()
    nodes, node_frames, ownership, all_points, point_labels = [], {}, {}, [], []
    for obj in engine._objects:
        ids = list(map(int, obj['observation_ids']))
        alias = min(ids)+1
        node_frames[alias] = sorted(set(map(int, frames[ids])))
        nodes.append(dict(alias=alias, members=ids, frames=node_frames[alias], points=len(obj['pcd'].points)))
        for oid in ids:
            assert oid not in ownership
            ownership[oid] = alias
        all_points.append(np.asarray(obj['pcd'].points))
        point_labels.append(np.full(len(obj['pcd'].points), alias, np.int32))
    dump(snapshot / 'nodes.json', nodes)
    np.savez_compressed(snapshot / 'geometry.npz', xyz=np.concatenate(all_points), instance=np.concatenate(point_labels))
    events = read(ASSOCIATION)['events']
    # Freeze the full candidate universe before reading any DINO scores.
    raw = []
    for event in events:
        query = event['observation_id']
        a = ownership[query]
        native = next(c for c in event['candidates'] if c['alias'] == event['native_alias'])
        assert all(ownership[i] == a for i in native['members'])
        for candidate in event['candidates']:
            targets = {ownership[i] for i in candidate['members']}
            assert len(targets) == 1
            b = targets.pop()
            if a == b:
                continue
            raw.append(dict(a=a, b=b, frame=event['frame'], query=query,
                native_members=native['members'], alternate_members=candidate['members'],
                native_score=next(c['native_score'] for c in event['candidates'] if c['alias'] == event['native_alias']),
                alternate_score=candidate['native_score']))
    dump(DEST / 'candidate_universe.json', dict(events=len(events), nodes=len(nodes), proposals=raw,
        independent_of_dino_selection=True, source_sha256=sha256(ASSOCIATION)))
    droot = OUT / 'dino_livo_tokens_v2'
    dm = read(droot / 'manifest.json')
    descriptors = np.zeros((len(frames), 384), np.float32)
    for rec in dm['records']:
        path = droot / rec['file']
        assert sha256(path) == rec['sha256']
        values = np.load(path)['mean']
        start = rec['observation_offset']
        descriptors[start:start+len(values)] = values
    proposals = []
    for p in raw:
        q = p['query']
        competitors = np.flatnonzero((frames == p['frame']) & (np.arange(len(frames)) != q))
        assert all(frames[i] < frames[q] for i in p['native_members']+p['alternate_members'])
        contrast = contrastive_bridge(descriptors[q], descriptors[p['native_members']],
                                      descriptors[p['alternate_members']], descriptors[competitors])
        proposals.append(dict(**p, dino_pass=contrast['pass_vote'], contrast=contrast))
    dump(DEST / 'proposals.json', proposals)
    # The exact same sensor-radius override used by the frozen engineering base.
    overlap = utils.compute_overlap_matrix
    def sensor_overlap(cfg, *args):
        cfg = deepcopy(cfg)
        cfg.downsample_voxel_size = .2
        return overlap(cfg, *args)
    utils.compute_overlap_matrix = sensor_overlap
    try:
        control = finalize(engine, [[n['alias']] for n in nodes], DEST / 'baseline', table)
        assert read(DEST / 'baseline/classification.json') == read(BASE / 'classification.json')
        for key, value in dict(np.load(DEST / 'baseline/geometry.npz')).items():
            np.testing.assert_array_equal(value, np.load(BASE / 'geometry.npz')[key])
        summaries = []
        original_members = {i for o in control['objects'] for i in o['observation_ids']}
        original_fg = {i for o in control['objects'] if o['predictions']['mean']['retained'] for i in o['observation_ids']}
        for variant, votes, use_dino in VARIANTS:
            graph = build_history_graph(node_frames, proposals, votes, use_dino)
            output = DEST / variant
            result = finalize(engine, graph['groups'], output, table)
            dump(output / 'graph.json', graph)
            kept = {i for o in result['objects'] for i in o['observation_ids']}
            fg = [o for o in result['objects'] if o['predictions']['mean']['retained']]
            fg_members = {i for o in fg for i in o['observation_ids']}
            same_frame = lambda rows: sum(len(o['observation_ids'])-len(set(frames[o['observation_ids']])) for o in rows)
            summary = dict(variant=variant, joined_edges=sum(e['reason']=='joined' for e in graph['decisions']),
                reasons=dict(Counter(e['reason'] for e in graph['decisions'])), regions=len(result['objects']), foreground=len(fg),
                retained_observations=len(kept), recovered_observations=sorted(kept-original_members),
                lost_observations=sorted(original_members-kept), foreground_observations=len(fg_members),
                foreground_added_observations=sorted(fg_members-original_fg), foreground_lost_observations=sorted(original_fg-fg_members),
                final_same_frame_extras=same_frame(result['objects']), foreground_same_frame_extras=same_frame(fg),
                graph_components=[g for g in graph['groups'] if len(g)>1],
                hashes={p.name:sha256(p) for p in output.iterdir()})
            dump(output / 'summary.json', summary)
            summaries.append(summary)
            print('COMPLETE', variant, {k:v for k,v in summary.items() if k in ['joined_edges','regions','foreground','retained_observations','foreground_observations','foreground_same_frame_extras']}, flush=True)
    finally:
        utils.compute_overlap_matrix = overlap
    dump(DEST / 'summary.json', dict(complete=True, baseline_exact=True, protocol_sha256=sha256(DEST / 'protocol.json'),
        nodes=len(nodes), proposals=len(proposals), dino_passing_proposals=sum(p['dino_pass'] for p in proposals),
        unique_edges=len({tuple(sorted([p['a'],p['b']])) for p in proposals}),
        baseline=dict(regions=len(control['objects']), foreground=sum(o['predictions']['mean']['retained'] for o in control['objects']),
                      observations=len(original_members), foreground_observations=len(original_fg)), variants=summaries))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        freeze()
    if args.run:
        run()
