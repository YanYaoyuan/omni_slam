"""Replay frozen outputs and diagnose every object without changing predictions.

GT is used only in this post-scoring audit. No new fitting or map variants.
"""
import numpy as np
from scipy.special import logit
from paper_final_common import read, dump, sha
from run_signed_query_association import OUT, BASE, INPUT, METHODS, verify, sealed, pieces
from omni_semantic_map.hypothesis_admission import predict_quality
from evaluate_kitti360_development import transfer


def replay_trace(count, left, right, weight, trace):
    matrix = np.zeros((count, count))
    matrix[left, right] = weight
    matrix[right, left] = weight
    groups = {j: {j} for j in range(count)}
    owner = np.arange(count)
    objective = 0.
    for event in trace:
        a, b = event['left'], event['right']
        ia, ib = int(owner[a[0]]), int(owner[b[0]])
        assert ia != ib and groups[ia] == set(a) and groups[ib] == set(b)
        gain = matrix[np.ix_(a, b)].sum()
        np.testing.assert_allclose(gain, event['gain'], rtol=1e-10, atol=1e-8)
        assert gain > 0
        objective += gain
        merged = groups.pop(ia) | groups.pop(ib)
        groups[min(ia, ib)] = merged
        owner[list(merged)] = min(ia, ib)
    expected = sum(matrix[np.ix_(list(g), list(g))].sum()/2 for g in groups.values())
    np.testing.assert_allclose(objective, expected, rtol=1e-10, atol=1e-8)
    values = list(groups.values())
    largest_remaining = max((matrix[np.ix_(list(a), list(b))].sum()
                             for i, a in enumerate(values) for b in values[i+1:]), default=0.)
    assert largest_remaining <= 1e-8
    return set(map(frozenset, values)), dict(merges=len(trace), objective=float(objective),
                                           maximum_remaining_gain=float(largest_remaining))


def main():
    protocol = verify()
    for stage in ['nodes_complete', 'pairs_complete', 'labels_complete', 'trained', 'predictions_complete']:
        sealed(stage)
    summary = read(OUT/'summary.json')
    assert summary['protocol_sha256'] == sha(OUT/'protocol.json')
    assert summary['prediction_seal_sha256'] == sha(OUT/'predictions_complete.json')
    details, checks = {}, {}
    for scene in protocol['scenes']:
        key = scene['key']
        nodes = dict(np.load(OUT/'nodes'/key/'nodes.npz'))
        metadata = read(OUT/'nodes'/key/'nodes.json')
        sets = pieces(nodes['source_ids'], nodes['source_offsets'])
        pairs = dict(np.load(OUT/'pairs'/key/'features.npz'))
        probabilities = dict(np.load(OUT/'models'/key/'probabilities.npz'))
        labels = dict(np.load(OUT/'training_labels'/key/'TRAINING_ONLY.npz'))
        basefolder = BASE/'maps'/key/'intrinsic_geometry'
        base = dict(np.load(basefolder/'geometry.npz'))
        source = np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz']
        locked = np.unique(np.r_[np.concatenate([sets[j] for j in np.flatnonzero(nodes['old'])]),
                                  base['added_source_ids']])
        reference = dict(np.load(scene['reference']))
        checks[key] = {'models': {}, 'maps': {}}
        for name in ['geometry', 'evidence']:
            model = read(OUT/'models'/key/(name+'.json'))
            assert model['target'] == key and model['training'] != key
            x = np.load(OUT/'pairs'/model['training']/'features.npz')['features']
            eligible = np.load(OUT/'training_labels'/model['training']/'TRAINING_ONLY.npz')['eligible']
            train = x[eligible, :model['dimensions']]
            mean, scale = train.mean(0), train.std(0)
            scale[scale < 1e-6] = 1.
            np.testing.assert_array_equal(mean, model['mean'])
            np.testing.assert_array_equal(scale, model['scale'])
            actual = predict_quality(model, pairs['features'][:, :model['dimensions']])
            np.testing.assert_array_equal(actual, probabilities[name])
            checks[key]['models'][name] = dict(training=model['training'], rows=len(train),
                                               probabilities_exact=True, statistics_exact=True)
        scene_details = {}
        for method in METHODS:
            folder = OUT/'maps'/key/method
            geometry = dict(np.load(folder/'geometry.npz'))
            objects = read(folder/'classification.json')['objects']
            by_id = {o['object_id']: o for o in objects}
            if method == 'baseline':
                for filename in ['geometry.npz', 'classification.json']:
                    assert sha(folder/filename) == sha(basefolder/filename)
                groups = {}
                checks[key]['maps'][method] = dict(byte_identical=True)
            else:
                grouping = read(folder/'groups.json')
                groups = {g['object_id']: g for g in grouping['groups']}
                assign = {int(a): b for a, b in grouping['old_id_assignment'].items()}
                assert set(assign) == set(map(int, np.unique(base['instance'])))
                n = len(base['xyz'])
                np.testing.assert_array_equal(geometry['xyz'][:n], base['xyz'])
                np.testing.assert_array_equal(geometry['instance'][:n], [assign[int(i)] for i in base['instance']])
                added = geometry['added_source_ids']
                assert len(np.unique(added)) == len(added) and not np.isin(added, locked).any()
                np.testing.assert_array_equal(geometry['xyz'][n:], source[added])
                assert set(groups) == set(by_id) == set(map(int, np.unique(geometry['instance'])))
                used = np.zeros(len(source), bool)
                used[locked] = True
                expected_ids, expected_instances = [], []
                for g in sorted(groups.values(), key=lambda q: (-q['score'], q['object_id'])):
                    member_ids = np.unique(np.concatenate([sets[j] for j in g['node_ids']]))
                    free = member_ids[~used[member_ids]]
                    used[free] = True
                    assert len(free) == g['added_points']
                    assert np.all(nodes['semantic_ids'][g['node_ids']] == g['semantic_id'])
                    assert sorted(metadata[j]['old_object_id'] for j in g['node_ids']
                                  if metadata[j]['old_object_id'] >= 0) == g['old_ids']
                    expected_ids.extend(free.tolist())
                    expected_instances.extend([g['object_id']]*len(free))
                np.testing.assert_array_equal(added, expected_ids)
                np.testing.assert_array_equal(geometry['instance'][n:], expected_instances)
                check = dict(old_coordinates_exact=True, old_identity_assignment_exact=True,
                             measured_sources_unique=True, source_allocation_replayed=True)
                if method in ['ours_signed', 'geometry_signed']:
                    name = 'evidence' if method == 'ours_signed' else 'geometry'
                    weight = np.where(pairs['opportunity'], logit(np.clip(probabilities[name], 1e-6, 1-1e-6)), 0.)
                    partition, trace_check = replay_trace(len(metadata), pairs['left'], pairs['right'], weight, grouping['merge_trace'])
                    assert all(frozenset(g['node_ids']) in partition for g in groups.values())
                    check['trace'] = trace_check
                checks[key]['maps'][method] = check
            # Recover evaluator's sequential pred_id from actual projected masses.
            projected = transfer(geometry['xyz'], geometry['instance'], reference['xyz'], .2)
            masses = np.bincount(projected, weights=reference['confidence'], minlength=max(by_id)+1)
            accepted = [o for o in sorted(objects, key=lambda o: o['object_id'])
                        if o['predictions']['mean']['semantic_id'] in [11, 26] and masses[o['object_id']] >= 100]
            pred_ids = {o['object_id']: i for i, o in enumerate(accepted)}
            matches = read(folder/'matches.json')['window']['prediction']
            predictions = {r['pred_id']: r for rows in matches.values() for r in rows}
            assert set(predictions) == set(pred_ids.values())
            rows = []
            for oid, obj in sorted(by_id.items()):
                row = dict(object_id=oid, semantic_id=obj['predictions']['mean']['semantic_id'],
                           points=int((geometry['instance'] == oid).sum()),
                           score=obj['predictions']['mean']['score'], reference_mass=float(masses[oid]), overlaps=[])
                if oid in pred_ids:
                    pred = predictions[pred_ids[oid]]
                    np.testing.assert_allclose(masses[oid], pred['vert_count'], atol=1e-7)
                    assert pred['confidence'] == row['score']
                    row['void_fraction'] = pred['voidIntersection']/pred['vert_count']
                    for gt in pred['matchedGt']:
                        row['overlaps'].append(dict(gt_id=gt['instance_id'],
                            iou=gt['intersection']/(gt['vert_count']+pred['vert_count']-gt['intersection']),
                            intersection=gt['intersection'], gt_mass=gt['vert_count']))
                    row['overlaps'].sort(key=lambda q: -q['iou'])
                if oid in groups:
                    group = groups[oid]
                    row.update(old_ids=group['old_ids'], node_ids=group['node_ids'], frames=group['frames'],
                               added_points=group['added_points'])
                    anchors = labels['node_anchor'][group['node_ids']]
                    known = labels['node_valid'][group['node_ids']]
                    ids, counts = np.unique(anchors[known], return_counts=True)
                    row['posthoc_node_anchor_counts'] = {str(int(i)): int(c) for i, c in zip(ids, counts)}
                rows.append(row)
            scene_details[method] = rows
            print('AUDIT', key, method, len(rows), flush=True)
        details[key] = scene_details
    dump(OUT/'object_diagnostics.json', dict(posthoc_gt=True, not_inference=True, results=details))
    dump(OUT/'artifact_audit.json', dict(complete=True, checks=checks,
        summary_sha256=sha(OUT/'summary.json'), diagnostics_sha256=sha(OUT/'object_diagnostics.json'),
        auditor_sha256=sha(__file__), limitations='Post-score replay of existing artifacts. No new fitting, maps or independent test. Does not certify task adaptation against official benchmark submission.'))
    print('AUDIT COMPLETE', flush=True)


if __name__ == '__main__':
    main()
