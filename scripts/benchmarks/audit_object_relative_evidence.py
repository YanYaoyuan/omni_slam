"""Independent brute rank checks and source-exact map replay."""
import argparse
import numpy as np
from paper_final_common import read, dump, sha
from run_object_relative_evidence import OUT, METHODS, verify, seal, parent


def inputs():
    parent.verify(); p = verify(); seal('prepared'); results = {}
    for scene in p['scenes']:
        key = scene['key']; d = dict(np.load(parent.OUT / 'features' / key / 'candidates.npz'))
        s = np.load(OUT / 'features' / key / 'static.npz'); refs = read(OUT / 'features' / key / 'references.json')
        drop = d['static'].copy(); drop[:, :, 7:10] = 0
        np.testing.assert_array_equal(s['shape_drop'], drop)
        rank = s['ours_relative']; np.testing.assert_array_equal(rank[:, :, 10:], d['static'][:, :, 10:])
        np.testing.assert_array_equal(rank[~d['valid']], d['static'][~d['valid']])
        ii, jj = np.where(d['valid']); sample = np.linspace(0, len(ii)-1, min(2048, len(ii)), dtype=int)
        exclusions = 0
        for q in sample:
            i, j = ii[q], jj[q]; r = refs[str(d['owners'][i, j])]
            cells = np.array(r['cells']); values = np.array(r['values'])
            other = (cells != np.floor(d['query_xyz'][i]/.2).astype(np.int64)).any(1)
            exclusions += int(not other.all()); query = d['static'][i, j, :10]
            # Direct comparisons, not the module's sorted-array/search algorithm.
            expected = ((values[other] < query).mean(0) + (values[other] <= query).mean(0)) - 1
            np.testing.assert_allclose(rank[i, j, :10], expected, rtol=1e-6, atol=1e-7)
        results[key] = dict(candidate_pairs=len(ii), brute_rank_rows=len(sample),
            ten_channels_per_row=True, sampled_cell_exclusions=exclusions,
            all_other_channels_and_padding_unchanged=True, shape_drop_exact=True)
        print('RELATIVE INPUT AUDIT', key, results[key], flush=True)
    dump(OUT / 'input_audit.json', dict(complete=True, results=results, source_sha256=sha(__file__),
         prepared_sha256=sha(OUT / 'prepared.json'), parent_protocol_sha256=sha(parent.OUT / 'protocol.json')))


def maps():
    parent.verify(); p = verify(); seal('prepared'); seal('trained'); seal('predictions_complete'); results = {}
    assert read(OUT / 'input_audit.json')['prepared_sha256'] == sha(OUT / 'prepared.json')
    for scene in p['scenes']:
        key = scene['key']; base = np.load(parent.OUT / 'maps' / key / 'baseline/geometry.npz')
        d = dict(np.load(parent.OUT / 'features' / key / 'candidates.npz')); oi = d['old_indices']; n = len(oi)
        source = np.load(read(parent.INPUT / key / 'manifest.json')['geometry'])['xyz']
        original = {o['object_id']: o for o in read(parent.OUT / 'maps' / key / 'baseline/classification.json')['objects']}
        checked = np.zeros(len(base['xyz']), bool); checked[oi[d['valid'][:n].any(1)]] = True
        decisions = {}; rows = {}
        for method in METHODS:
            folder = OUT / 'maps' / key / method
            if method in ['baseline', 'frozen_raw']:
                old = parent.OUT / 'maps' / key / ('baseline' if method=='baseline' else 'ours_revision')
                for name in ['geometry.npz', 'classification.json']: assert sha(folder/name) == sha(old/name)
                rows[method] = dict(exact_copy=True); continue
            g = np.load(folder / 'geometry.npz'); choice = np.load(folder / 'decisions.npz')['owners']; revised = base['instance'].copy()
            if method != 'ours_append_only': revised[oi] = choice[:n]
            np.testing.assert_array_equal(g['revised_base_owners'], revised)
            np.testing.assert_array_equal(revised[~checked], base['instance'][~checked])
            keep = np.flatnonzero(revised>0); add = choice[n:]>0; sid = d['new_source_ids'][add]
            np.testing.assert_array_equal(g['retained_base_indices'], keep); np.testing.assert_array_equal(g['added_source_ids'], sid)
            np.testing.assert_array_equal(g['xyz'], np.r_[base['xyz'][keep], source[sid]])
            np.testing.assert_array_equal(g['instance'], np.r_[revised[keep], choice[n:][add]])
            assert len(np.unique(sid)) == len(sid)
            cls = read(folder / 'classification.json')['objects']; assert all(o == original[o['object_id']] for o in cls)
            assert {o['object_id'] for o in cls} == set(g['instance'])
            decisions[method] = choice
            rows[method] = dict(points=len(g['xyz']), exact_source_coordinates=True, unchanged_unexamined_points=True, unchanged_identity_class_score=True)
        np.testing.assert_array_equal(decisions['ours_append_only'], decisions['ours_relative'])
        rows['decision_differences_shape_vs_relative'] = int((decisions['shape_drop'] != decisions['ours_relative']).sum())
        results[key] = rows; print('RELATIVE MAP AUDIT', key, rows, flush=True)
    dump(OUT / 'map_audit.json', dict(complete=True, results=results, source_sha256=sha(__file__),
         prediction_sha256=sha(OUT / 'predictions_complete.json')))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['inputs','maps']); globals()[ap.parse_args().action]()
