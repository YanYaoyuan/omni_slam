"""One bounded test of object-relative geometry versus dropping shape features.

Four fits, ten maps. Reuses frozen perception, labels, proposals and attention
architecture. No selection of epochs, thresholds, seeds or downstream cleanup.
"""
import argparse, copy, shutil, time
from pathlib import Path
import numpy as np
from paper_final_common import ROOT, read, dump, sha
import run_provenance_object_memory as parent
from omni_semantic_map.object_relative_evidence import ObjectReferenceRanks
from omni_semantic_map.provenance_object_memory import ViewEvidenceAllocator, revision_map
from omni_semantic_map.surface_ownership import grouped_loss, allocate
from omni_semantic_map.intrinsic_surface_geometry import geometry_features
from audit_mask_observation_code import source_anchors

OUT = ROOT / 'build/benchmarks/object_relative_evidence_v1'
TRAINED = ['shape_drop', 'ours_relative']
METHODS = ['baseline', 'frozen_raw', 'shape_drop', 'ours_append_only', 'ours_relative']


def freeze():
    assert not OUT.exists()
    p = parent.verify(); inputs = {}
    for name in ['features_complete', 'labels_complete', 'trained', 'predictions_complete']:
        d = parent.seal(name); inputs[str(parent.OUT / (name + '.json'))] = sha(parent.OUT / (name + '.json'))
        inputs.update(d['files'])
    audit = ROOT / 'build/benchmarks/object_geometry_shortcut_audit_v1/summary.json'
    assert read(audit)['complete']; inputs[str(audit)] = sha(audit)
    sources = [Path(__file__), ROOT / 'omni_semantic_map/omni_semantic_map/object_relative_evidence.py',
               ROOT / 'omni_semantic_map/test/test_object_relative_evidence.py']
    inputs.update(p['source_hashes'])
    dump(OUT / 'protocol.json', dict(created_epoch=time.time(), scenes=p['scenes'], references=p['references'],
        inputs=inputs, source_hashes={str(f): sha(f) for f in sources}, methods=METHODS,
        evidence='Frozen-input audit: false rejection despite positive point targets; shape-channel intervention restores642/655false-rejectedcar13009points and2661/2661building2011points, but also restores260negativebuildingpoints. Intervention is sensitivity evidence, not a deployable improvement or unique physical cause.',
        hypothesis='Normalize local point-to-object geometry against the same predicted object observed surface, so partial-object extent acts as context instead of an absolute class/identity shortcut. Must outperform simply dropping3globalshapechannels, not merely the failed raw model.',
        reference='For every existing object, all original fusedmap points under its predicted ID, without GT/learned rejection selection. Compute identical intrinsic geometry using the unchanged original source anchors; reference needs no image support, while query/memory eligibility is unchanged. One actual representative per20cm world-axis cell, nearestcellcentre, lexXYZties. Use10geometrychannels. At least2cells required; assert all common candidates meet this rather than silently dropping them.',
        transform='Per-object empirical midrank of query features, scaled[-1,1], excluding the whole querycell if present. Constant object-wide shape maps0. Each candidate reads its own object reference. Per-channel strictly increasing transforms preserve ranks when ties are preserved. Grid is fixed world-axis, not rotation-invariant. Predicted reference may contain errors; ranks are not confidence probabilities or a correctness certificate.',
        controls='baseline exact strong reference; frozen_raw exact failed provenanceOurs; shape_drop same architecture/attention/supervision but static7:10zero; ours_relative static0:10 replaced by object ranks; ours_append_only same Ours decisions, no old geometry edits. Remaining features, views, eligibility, proposals, coordinates, classes and ranking identical.',
        training='4fits=2methods x2directions; other exposed window only; identical seed17,Adam1e-3,40epochs,batch1024,clip1,train-only feature standardization,finalcheckpoint. Same inherited labels/weights and groupedNLL. No new backbone calls, warmstarts, epoch/threshold/seed sweeps.',
        inference='Parent categorical allocation: unique probability>.5 else null; same precise source provenance; unseen oldpoints unchanged; no newIDs,merge,split,coordinate synthesis,SOR,reranking or classchange. Reference normalization uses predicted geometry only, never targetGT.',
        budget='4fixedfits,2windows x5maps including exactcopies. Freeze predictions before scoring. Failed parent artifacts unchanged.',
        gate='Ours>=baseline eachwindow,mean>=baseline+.01,strictly>shape_drop,frozen_raw,appendonly means,>=1gainedAP50GT and0lost; actual oldrevision; source/selfexclusion invariants pass. Otherwise close version without rescue sweeps or robot/independent expansion.',
        scope='Fixed-object offline geometry revision, not complete object birth/discovery. Both0000/0002exposed; inherited parent supervision, not nestedCV, independenttest or officialleaderboard. Empirical normalization is an established operation; no novelty claim from normalization alone. Required effective mechanism/robot/new independent evaluation remain pending.'))
    print('FROZEN OBJECT RELATIVE', flush=True)


def verify():
    p = read(OUT / 'protocol.json')
    for f, h in {**p['inputs'], **p['source_hashes']}.items(): assert sha(f) == h, f
    return p


def seal(name):
    d = read(OUT / (name + '.json')); assert d['complete']
    for f, h in d['files'].items(): assert sha(f) == h, f
    return d


def prepare():
    verify(); assert not (OUT / 'prepared.json').exists(); files = {}; stats = {}
    for key in parent.KEYS:
        d = dict(np.load(parent.OUT / 'features' / key / 'candidates.npz'))
        base = np.load(parent.BASE / key / 'geometry_graph/geometry.npz')
        xyz = np.load(read(parent.INPUT / key / 'manifest.json')['geometry'])['xyz']
        anchors = source_anchors(key, len(xyz)); allcore = np.flatnonzero(anchors > 0)
        classes = {o['object_id']: o['predictions']['mean']['train_id'] for o in read(parent.BASE / key / 'geometry_graph/classification.json')['objects']}
        reference_features = np.empty((len(base['xyz']), 10), np.float32)
        for oid in np.unique(base['instance']):
            use = base['instance'] == oid
            reference_features[use] = geometry_features(base['xyz'][use], xyz[anchors==oid], xyz[allcore[anchors[allcore]!=oid]], 26 if classes[int(oid)]==13 else 11)[:, :10]
        # Recomputed old-query geometry must match the previously sealed cache.
        good = d['valid'][:len(d['old_indices']), 0]
        np.testing.assert_array_equal(reference_features[d['old_indices'][good]], d['static'][:len(d['old_indices']), 0, :10][good])
        bank = ObjectReferenceRanks().fit(base['xyz'], base['instance'], reference_features)
        assert set(d['owners'][d['valid']]).issubset(bank.references)
        rank = d['static'].copy(); ii, jj = np.where(d['valid'])
        rank[ii, jj, :10] = bank.transform(d['query_xyz'][ii], d['owners'][ii, jj], d['static'][ii, jj, :10])
        # Every object-global covariance eigenvalue must cancel, including new queries.
        assert np.max(np.abs(rank[ii, jj, 7:10])) == 0
        drop = d['static'].copy(); drop[:, :, 7:10] = 0
        folder = OUT / 'features' / key; folder.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(folder / 'static.npz', shape_drop=drop, ours_relative=rank)
        ref = {str(o): dict(cells=r['cells'].tolist(), values=r['values'].tolist()) for o, r in bank.references.items()}
        dump(folder / 'references.json', ref)
        row = dict(objects=len(ref), reference_cells=sum(len(r['cells']) for r in bank.references.values()),
                   min_reference_cells=min(len(r['cells']) for r in bank.references.values()),
                   candidate_pairs=len(ii), unchanged_eligibility=True, no_gt_read=True)
        for f in folder.iterdir(): files[str(f)] = sha(f)
        stats[key] = row; print('RELATIVE PREPARED', key, row, flush=True)
    dump(OUT / 'prepared.json', dict(complete=True, files=files, stats=stats, protocol_sha256=sha(OUT / 'protocol.json')))


def train():
    verify(); seal('prepared'); assert not (OUT / 'training_execution.json').exists(); torch = parent.gpu()
    dump(OUT / 'training_execution.json', dict(started_epoch=time.time())); files = {}; stats = {}
    for target in parent.KEYS:
        key = next(k for k in parent.KEYS if k != target)
        d = dict(np.load(parent.OUT / 'features' / key / 'candidates.npz'))
        y = dict(np.load(parent.OUT / 'training_only' / key / 'labels.npz')); use = y['eligible']
        static = np.load(OUT / 'features' / key / 'static.npz')
        valid = torch.tensor(d['valid'][use], device='cuda'); positive = torch.tensor(y['positive'][use], device='cuda')
        views = torch.tensor(d['views'][use].astype('float32'), device='cuda'); seen = torch.tensor(d['separated'][use], device='cuda')
        w = y['weight'][use]; weight = torch.tensor(w / w.mean(), device='cuda', dtype=torch.float32)
        for method in TRAINED:
            xx = static[method][use]; v = d['valid'][use]; mean = xx[v].mean(0); std = np.maximum(xx[v].std(0), 1e-6)
            x = torch.tensor((xx - mean) / std, device='cuda')
            torch.manual_seed(17); torch.cuda.manual_seed_all(17); rng = np.random.default_rng(17)
            model = ViewEvidenceAllocator().cuda(); optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            curve = []; start = time.monotonic(); torch.cuda.reset_peak_memory_stats()
            for epoch in range(40):
                order = rng.permutation(len(x)); total = 0.
                for lo in range(0, len(order), 1024):
                    ix = torch.tensor(order[lo:lo+1024], device='cuda')
                    logits = model(x[ix], views[ix], seen[ix], valid[ix], 'attention')
                    loss = (grouped_loss(logits, valid[ix], positive[ix]) * weight[ix]).mean(); assert torch.isfinite(loss)
                    optimizer.zero_grad(); loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    assert torch.isfinite(norm); optimizer.step(); total += float(loss.detach()) * len(ix)
                curve.append(total / len(x))
                if epoch == 0 or (epoch + 1) % 10 == 0: print('RELATIVE FIT', target, method, epoch+1, curve[-1], flush=True)
            folder = OUT / 'models' / target / method; folder.mkdir(parents=True, exist_ok=False)
            torch.save(dict(state={k: v.detach().cpu() for k, v in model.state_dict().items()}, mean=mean.tolist(), std=std.tolist(), training=[key], method=method), folder / 'model.pt')
            row = dict(training=[key], loss=curve, epochs=40, queries=len(x), seconds=time.monotonic()-start,
                       peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30)
            dump(folder / 'training.json', row); stats[target+'/'+method] = row
            for f in folder.iterdir(): files[str(f)] = sha(f)
            del model, optimizer, x
        del views, seen, valid, positive, weight; torch.cuda.empty_cache()
    dump(OUT / 'trained.json', dict(complete=True, files=files, stats=stats, protocol_sha256=sha(OUT / 'protocol.json')))


def predict():
    p = verify(); seal('prepared'); seal('trained'); assert not (OUT / 'prediction_execution.json').exists(); torch = parent.gpu()
    dump(OUT / 'prediction_execution.json', dict(started_epoch=time.time())); files = {}; stats = {}
    for scene in p['scenes']:
        key = scene['key']; d = dict(np.load(parent.OUT / 'features' / key / 'candidates.npz'))
        static = np.load(OUT / 'features' / key / 'static.npz'); base = np.load(parent.BASE / key / 'geometry_graph/geometry.npz')
        cls = read(parent.BASE / key / 'geometry_graph/classification.json'); known = [o['object_id'] for o in cls['objects']]
        source = np.load(read(parent.INPUT / key / 'manifest.json')['geometry'])['xyz']; oi = d['old_indices']; n = len(oi)
        indices = np.flatnonzero(d['valid'].any(1)); decisions = {}; stats[key] = {}
        for method in TRAINED:
            c = torch.load(OUT / 'models' / key / method / 'model.pt', map_location='cpu', weights_only=True)
            assert c['training'] == [k for k in parent.KEYS if k != key]
            model = ViewEvidenceAllocator().cuda().eval(); model.load_state_dict(c['state']); model.requires_grad_(False)
            choice = np.r_[base['instance'][oi], np.zeros(len(d['new_source_ids']), np.int64)]
            confidence = np.full(len(choice), np.nan); xx = static[method]
            with torch.inference_mode():
                for lo in range(0, len(indices), 2048):
                    ix = indices[lo:lo+2048]
                    x = torch.tensor(((xx[ix]-np.array(c['mean']))/np.array(c['std'])).astype('float32'), device='cuda')
                    z = model(x, torch.tensor(d['views'][ix].astype('float32'), device='cuda'),
                              torch.tensor(d['separated'][ix], device='cuda'), torch.tensor(d['valid'][ix], device='cuda'), 'attention').cpu().numpy()
                    choice[ix], confidence[ix] = allocate(z, d['valid'][ix], d['owners'][ix])
            decisions[method] = (choice, confidence); del model; torch.cuda.empty_cache()
        for method in METHODS:
            folder = OUT / 'maps' / key / method; folder.mkdir(parents=True, exist_ok=False)
            if method in ['baseline', 'frozen_raw']:
                source_folder = parent.OUT / 'maps' / key / ('baseline' if method == 'baseline' else 'ours_revision')
                for name in ['geometry.npz', 'classification.json']: shutil.copyfile(source_folder / name, folder / name)
                row = dict(exact_copy=True)
            else:
                choice, confidence = decisions['ours_relative' if method == 'ours_append_only' else method]
                g = revision_map(base['xyz'], base['instance'], source, oi, choice[:n], d['new_source_ids'], choice[n:], known, revise=method != 'ours_append_only')
                np.savez_compressed(folder / 'geometry.npz', **g); np.savez_compressed(folder / 'decisions.npz', owners=choice, confidence=confidence)
                classification = copy.deepcopy(cls); present = set(g['instance'])
                classification['objects'] = [o for o in cls['objects'] if o['object_id'] in present]; dump(folder / 'classification.json', classification)
                row = dict(points=len(g['xyz']), objects=len(present), added_sources=len(g['added_source_ids']),
                    removed_old=int((g['revised_base_owners']==0).sum()), reassigned_old=int(((g['revised_base_owners']!=base['instance']) & (g['revised_base_owners']>0)).sum()))
            dump(folder / 'run.json', row); stats[key][method] = row
            for f in folder.iterdir(): files[str(f)] = sha(f)
            print('RELATIVE MAP', key, method, row, flush=True)
    dump(OUT / 'predictions_complete.json', dict(complete=True, files=files, stats=stats, sealed_before_scoring=True, trained_seal_sha256=sha(OUT / 'trained.json')))


def score():
    from evaluate_kitti360_development import transfer, clean_json
    from omni_semantic_map.kitti360_evaluation import evaluate
    from run_discovery_query_benchmark import matched_ids
    p = verify(); pred = seal('predictions_complete'); assert not (OUT / 'summary.json').exists()
    results = {}; means = {m: [] for m in METHODS}; gained = lost = 0; gate = True
    prior = read(parent.OUT / 'summary.json')
    for scene in p['scenes']:
        key = scene['key']; ref = np.load(scene['reference']); rows = {}
        for method in METHODS:
            folder = OUT / 'maps' / key / method; g = np.load(folder / 'geometry.npz'); objects = read(folder / 'classification.json')['objects']
            prediction = transfer(g['xyz'], g['instance'], ref['xyz'], .2)
            r = evaluate(ref['train_instance'], ref['confidence'], prediction, {o['object_id']: o['predictions']['mean'] for o in objects}, parent.EVALUATOR)
            dump(folder / 'matches.json', clean_json(r['matches'])); np.savez_compressed(folder / 'reference_prediction.npz', instance=prediction)
            rows[method] = dict(metrics=clean_json(r['averages']), matched_gt_ids=sorted(matched_ids(r['matches'])))
            means[method].append(float(r['averages']['allAp'])); print('RELATIVE SCORE', key, method, r['averages']['allAp'], flush=True)
        for method, old in [('baseline', 'baseline'), ('frozen_raw', 'ours_revision')]:
            assert abs(rows[method]['metrics']['allAp'] - prior['results'][key]['methods'][old]['metrics']['allAp']) < 1e-12
        original = set(rows['baseline']['matched_gt_ids'])
        for row in rows.values():
            ids = set(row['matched_gt_ids']); row['changes_vs_baseline'] = dict(gained=sorted(ids-original), lost=sorted(original-ids))
        change = rows['ours_relative']['changes_vs_baseline']; gained += len(change['gained']); lost += len(change['lost'])
        gate &= rows['ours_relative']['metrics']['allAp'] >= rows['baseline']['metrics']['allAp']; results[key] = dict(methods=rows)
    mean = {m: float(np.mean(v)) for m, v in means.items()}
    gate &= mean['ours_relative'] >= mean['baseline']+.01 and mean['ours_relative'] > max(v for m,v in mean.items() if m!='ours_relative') and gained>=1 and lost==0
    changed = any(s['ours_relative']['removed_old'] + s['ours_relative']['reassigned_old'] > 0 for s in pred['stats'].values()); gate &= changed
    dump(OUT / 'summary.json', dict(complete=True, results=results, mean_primary_ap=mean, gained=gained, lost=lost, advance_gate=bool(gate),
         historical_changes=changed, official_leaderboard=False, independent_test=False, prediction_seal_sha256=sha(OUT / 'predictions_complete.json')))
    print('RELATIVE GATE', gate, mean, 'gained', gained, 'lost', lost, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['freeze', 'prepare', 'train', 'predict', 'score'])
    globals()[ap.parse_args().action]()
