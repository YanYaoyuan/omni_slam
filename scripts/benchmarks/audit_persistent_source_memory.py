"""Replay causal input bookkeeping independently of the training/inference loop."""
import argparse
import time
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from run_persistent_source_memory import OUT, INPUT, SEM, PREPARED, PREPARED_SEAL, verify, seal, METHODS


def inputs():
    import torch
    torch.set_num_threads(1)
    p = verify(); seal('capture_complete'); prepared = seal(PREPARED_SEAL)
    results = {}
    for scene in p['scenes']:
        key = scene['key']; manifest = read(INPUT/key/'manifest.json'); records = manifest['records']
        xyz = np.load(manifest['geometry'])['xyz']; n = len(xyz)
        latest = np.full(n, -1, np.int32); pixels = np.zeros((n,2), int)
        sums = np.zeros((n,19), np.float64); counts = np.zeros(n, int)
        grids = []
        for i, r in enumerate(records):
            f = np.load(OUT/'features'/key/f'{r["frame_id"]:06d}.npz')
            np.testing.assert_array_equal(f['causal_frame_ids'], [t['frame_id'] for t in records[max(0,i-3):i+1]])
            assert f['grid'].shape == (10,37,2048) and np.isfinite(f['grid']).all()
            grids.append(f['grid'])
        repeated = conflicts = independent_feature_rows = 0
        for i,r in enumerate(records):
            d = np.load(PREPARED/key/f'{i:03d}.npz'); obs = np.load(r['file'])
            sem = np.load(SEM/'frames'/key/f'{r["frame_id"]:06d}.npz')
            sid = obs['visible_source_ids']; np.testing.assert_array_equal(sid,sem['source_ids'])
            context_ids = d['context_source_ids']; cf = d['frame_ids']; historical = cf<i
            assert (cf<=i).all() and (cf>=0).all() and len(cf)<=512
            np.testing.assert_array_equal(latest[context_ids[historical]], cf[historical])
            np.testing.assert_array_equal(pixels[context_ids[historical]], d['context_pixels'][historical])
            expected_sem = np.zeros((len(cf),19))
            expected_sem[historical] = sums[context_ids[historical]]/counts[context_ids[historical],None]
            # Update an independent sum/count accumulator, not the online-mean implementation.
            latest[sid] = i; pixels[sid] = obs['pixel_xy']; counts[sid] += 1
            sums[sid] += sem['source_probabilities'].astype(float)
            np.testing.assert_array_equal(latest[context_ids[~historical]], cf[~historical])
            np.testing.assert_array_equal(pixels[context_ids[~historical]], d['context_pixels'][~historical])
            expected_sem[~historical] = sums[context_ids[~historical]]/counts[context_ids[~historical],None]
            np.testing.assert_allclose(d['context'][:,-19:], expected_sem, rtol=8e-4, atol=5e-4)
            qi = d['query_ids']; assert len(qi)==len(np.unique(qi)) and np.all(latest[qi]>=0)
            np.testing.assert_array_equal(d['query_frame_ids'],latest[qi])
            np.testing.assert_allclose(d['query'][:,-19:], sums[qi]/counts[qi,None], rtol=8e-4, atol=5e-4)
            np.testing.assert_allclose(d['query'][:,2048:2051],xyz[qi]/20,rtol=8e-4,atol=5e-4)
            rel=d['relations']; np.testing.assert_array_equal(rel,rel.T)
            repeated += int((rel>0).sum()//2); conflicts += int((rel<0).sum()//2)
            eq=context_ids[:,None]==context_ids[None];np.fill_diagonal(eq,False)
            np.testing.assert_array_equal(rel>0,eq)
            # All query appearances are independently interpolated by torch.
            expected=np.zeros((len(qi),2048),np.float32)
            for frame in np.unique(latest[qi]):
                take=latest[qi]==frame
                xy=pixels[qi[take]]; normalized=((xy+.5)/[1408,376]*2-1).astype(np.float32)
                grid=torch.tensor(grids[frame].astype(np.float32).transpose(2,0,1)[None])
                sampled=torch.nn.functional.grid_sample(grid,torch.tensor(normalized)[None,None],
                    padding_mode='border',align_corners=False)[0,:,0].T
                expected[take]=sampled.numpy()
            expected/=np.maximum(np.linalg.norm(expected,axis=1,keepdims=True),1e-12)
            expected*=2048**.5
            np.testing.assert_allclose(d['query'][:,:2048],expected,rtol=1e-3,atol=.004)
            independent_feature_rows+=len(qi)
        final=np.load(PREPARED/key/'ledger.npz')
        np.testing.assert_array_equal(final['latest'],latest);np.testing.assert_array_equal(final['pixel'],pixels)
        np.testing.assert_array_equal(final['count'],counts);assert (counts>0).all()
        np.testing.assert_allclose(final['probability'],sums/counts[:,None],rtol=2e-6,atol=2e-7)
        assert repeated>=128*(len(records)-1), 'Retrieval must actually exercise repeated sources'
        results[key]=dict(frames=len(records),all_measured_sources=n,exact_repeat_pairs=repeated,
            ray_conflict_pairs=conflicts,independent_query_feature_rows=independent_feature_rows,
            every_source_registered=True,online_semantics_replayed=True,causal_appearance_verified=True)
        print('INPUT AUDIT',key,results[key],flush=True)
    dump(OUT/'input_audit.json',dict(complete=True,results=results,source_sha256=sha(__file__),
        prepared_sha256=sha(OUT/f'{PREPARED_SEAL}.json'),created_epoch=time.time()))


def maps():
    p=verify();seal('trained');prediction=seal('predictions_complete');summary=read(OUT/'summary.json')
    results={}
    for s in p['scenes']:
        key=s['key'];xyz=np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz'];rows={}
        for method in METHODS[1:]:
            folder=OUT/'maps'/key/method;g=np.load(folder/'geometry.npz');own=np.load(folder/'source_ownership.npz')
            ids=g['source_ids'];assert len(np.unique(ids))==len(ids)
            np.testing.assert_array_equal(g['xyz'],xyz[ids]);np.testing.assert_array_equal(g['instance'],own['retained'][ids])
            np.testing.assert_array_equal(np.flatnonzero(own['retained']),ids)
            assert np.all((own['retained']==0)|(own['retained']==own['raw']))
            objects=read(folder/'classification.json')['objects']
            assert set(g['instance'])=={o['object_id'] for o in objects}
            assert all(o['predictions']['mean']['train_id'] in [2,13] for o in objects)
            rows[method]=dict(sources=len(ids),objects=len(objects),exact_measured_xyz=True,unique_source_ownership=True)
        a=np.load(OUT/'maps'/key/'current_commit/source_ownership.npz');b=np.load(OUT/'maps'/key/'ours_history/source_ownership.npz')
        changes=int((a['raw']!=b['raw']).sum());assert changes==prediction['stats'][key]['ours_history']['raw_historical_changes']
        final=np.load(PREPARED/key/'ledger.npz');last=int(final['latest'].max())
        last_changed=int((a['raw'][final['latest']==last]!=b['raw'][final['latest']==last]).sum())
        rows['revision']=dict(raw_changed_sources=changes,retained_changed_sources=int((a['retained']!=b['retained']).sum()),
            final_frame_unchanged=last_changed==0,final_frame_changed=last_changed,
            interpretation='The final-frame inputs and state are the same. Any differing discrete IDs expose numerical/batch sensitivity, not historical correction.')
        results[key]=rows
    dump(OUT/'map_audit.json',dict(complete=True,results=results,source_sha256=sha(__file__),
        prediction_sha256=sha(OUT/'predictions_complete.json'),advance_gate=summary['advance_gate'],
        all_final_frame_invariance_checks_passed=all(r['revision']['final_frame_unchanged'] for r in results.values())))
    print('MAP AUDIT VERIFIED',results,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['inputs','maps']);globals()[parser.parse_args().action]()
