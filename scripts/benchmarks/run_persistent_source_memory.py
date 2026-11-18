"""Finite training/full-map test of persistent source ownership revision.

freeze -> capture -> prepare -> labels -> train -> predict -> score.
Input preparation and prediction never load instance ground truth. Two exposed
windows exchange training direction: development only, never an official score.
"""
import argparse
from pathlib import Path
import runpy
import shutil
import sys
import time
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from omni_semantic_map.persistent_source_memory import (
    PersistentSourceMemory, SourceFeatureLedger, sample_sources, measured_relations, retrieve_context)

OUT = ROOT/'build/benchmarks/persistent_source_memory_v1'
INPUT = ROOT/'build/benchmarks/discovery_measured_inputs_v1'
SEM = ROOT/'build/benchmarks/closed_vocabulary_semantics_v1'
ASSETS = ROOT/'build/benchmarks/vggt_object_probe_v1'
BASE = ROOT/'build/benchmarks/signed_query_association_v1/maps'
KEYS = ['kitti360_0000', 'kitti360_0002']
METHODS = ['baseline', 'current_commit', 'no_geometry', 'ours_history']
CONFIG = dict(input_dim=2070, width=128, heads=4, layers=2, max_objects=128)
PREPARED = OUT/'prepared_retrieved'
PREPARED_SEAL = 'prepared_retrieved'
EVALUATOR = ROOT/'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py'


def freeze():
    assert not OUT.exists()
    parent = read(SEM/'protocol.json'); assets = read(ASSETS/'assets.json')
    assert sha(assets['checkpoint']) == assets['sha256']
    inputs = {str(ASSETS/'assets.json'): sha(ASSETS/'assets.json')}
    expected = ROOT/'build/benchmarks/signed_query_association_v1/summary.json'
    inputs[str(expected)] = sha(expected)
    for s in parent['scenes']:
        key = s['key']; manifest = read(INPUT/key/'manifest.json')
        files = [s['job'], INPUT/key/'manifest.json', manifest['geometry'],
                 s['reference'], parent['references'][key]['raw']]
        files += [BASE/key/'geometry_graph'/n for n in ['geometry.npz', 'classification.json']]
        for r in manifest['records']:
            files += [r['file'], r['rgb'], SEM/'frames'/key/f'{r["frame_id"]:06d}.npz']
        for f in files:inputs[str(f)] = sha(f)
    files = [Path(__file__), EVALUATOR, ROOT/'scripts/benchmarks/evaluate_kitti360_development.py',
             ROOT/'build/datasets/kitti360_prepare/labels.py']
    files += [ROOT/'omni_semantic_map/omni_semantic_map'/n for n in
              ['joint_object_memory.py', 'persistent_source_memory.py', 'kitti360_evaluation.py']]
    files += [ROOT/'omni_semantic_map/test'/n for n in ['test_joint_object_memory.py', 'test_persistent_source_memory.py']]
    files += list((ASSETS/'upstream/vggt').rglob('*.py'))
    dump(OUT/'protocol.json', dict(created_epoch=time.time(), scenes=parent['scenes'],
        references=parent['references'], assets=assets, configuration=CONFIG, methods=METHODS,
        input_hashes=inputs, source_hashes={str(f):sha(f) for f in files},
        hypothesis='A persistent object state can re-decode already stored measured sources when subsequent views arrive, correcting historical ownership beyond committing only the currently visible sources. Not a novelty claim for VGGT, attention or set matching.',
        features='Frozen pinned VGGT aggregator layer23, each incoming image jointly encoded with up to3 immediately preceding available images; only latest image tokens retained. 108 causal forwards, no future RGB. Official crop preprocessing518wide, no actual crop/pad on KITTI. Bilinear pixel-center patch sampling; L2 appearance times sqrt2048, original world XYZ/20, online mean19class SegFormer posteriors. No new depth/XYZ, GT or old map anchors.',
        context='At each frame,256 GT-free class-stratified sources from current visible measurements and256 historical sources sampled BEFORE current observation update. Sampling groups predicted car/building/other equally with unused budget redistribution, deterministic seed17+frame. Historical and current occurrences can share exact source ID. Stored context<=512tokens. Query training512current+512previous sources, unique IDs, latest appearance and online class evidence after update. All source provenance retained; ledger is O(N), not bounded total memory.',
        geometry='Symmetric soft relations+1 for exact repeated physical source; -1 if one source projects into another token measured pixel but differs in axial depth>5cm; unknown0, diagonal0. Conflicting rays do not supervise different instances. No interpolated depth, hard geometric mask, semantic same-instance assumption or radius tuning.',
        lifecycle='128 persistent slots initialized once per sequence, fixed slot-ID=1..128, never recycled. Competition includes separate background0. Unused slots may receive new objects; no inference GT binding or nearest-GT ID fix. This finite-window capacity cannot establish arbitrary long-term tracking. Training must fail on capacity exhaustion rather than drop instances.',
        training='Four fits: train0000/predict0002 and train0002/predict0000, each fullgeometry/no_geometry. Same initialization17,40epochs, AdamW lr3e-4 weight_decay1e-4, clipnorm1,4-frame truncated BPTT with averaged losses; detach memory only at each optimizer update, reset slots/binding each epoch. No early stopping, validation checkpoint selection or parameter sweep. Per-epoch shared deterministic random yaw and Gaussian5m translation on XYZ features. Frozen visual/semantic models. Final weights only.',
        supervision='Training-only nearest raw nonignored reference within strict20cm, confidence>0. Known car/building instance IDs positive, other known categories background0 (including road), ignored/unmatched=-1. Original source_instance_loss equal weight per present instance/background plus objectness BCE, persistent training-only Hungarian binding; absent established slots not false negatives. Same loss on unique current and history queries. No target-window labels loaded by its trainer.',
        prediction='Full chronological stream, same prepared contexts. Decode ALL visible measured sources at every frame, chunks4096, unmodified XYZ. Learned slot logits plus log-sigmoid(objectness), competing with background. current_commit and ours_history share exact fullgeometry weights/states and visible-source commits; ours_history additionally re-decodes every registered source once using final memory. Equal-compute control discards this final historical readout. no_geometry independently trained without relation bias and also revises history. Window-end historical revision, not online full-history rewrite at every frame.',
        classification='For each nonzero slot with>=16sources, choose argmax of mean online19class posteriors of its members; retain only car13/building2. Confidence=final slot objectness sigmoid times that mean category probability. Same readout/all postprocessing for every learned arm; no GT labels, SOR, NMS, connected components or baseline fallback. Baseline copied exactly.',
        budget='108 frozen VGGT forwards,4fixed fits,8complete maps (2windows x4conditions). No robot or independent expansion before development gate. A trained slot model is a candidate, not automatically an improvement or proof that learning is necessary.',
        gate='Ours primary AP50:90>=baseline eachwindow, mean>=baseline+0.01 and strictly>current_commit and no_geometry;>=1gained and0lostAP50GT. At least one historical source ownership actually changes relative to current_commit. Fail closes version without threshold/epoch/seed sweep. Do not call exposed0000/0002/0003/0004/0005 independent.',
        scope='Two exposed outdoor development windows. Candidate inputs contain no inherited baseline predictions; shared Cityscapes supervision is external. Existing baseline includes earlier cross-window supervision; comparison is not nested CV, zero-shot generalization or official leaderboard. External strong-method reproduction and same-version robot/independent evaluation remain required after positive development evidence.'))
    print('FROZEN PERSISTENT SOURCE MEMORY', flush=True)


def verify():
    p = read(OUT/'protocol.json')
    for f, h in {**p['input_hashes'], **p['source_hashes']}.items():assert sha(f) == h, f
    return p


def seal(name):
    p = read(OUT/f'{name}.json'); assert p['complete']
    for f, h in p['files'].items():assert sha(f) == h, f
    return p


def gpu():
    import torch
    assert torch.cuda.is_available()
    torch.set_num_threads(4); torch.manual_seed(17); torch.cuda.manual_seed_all(17)
    free, total = torch.cuda.mem_get_info(); assert free >= 14*1024**3
    torch.cuda.set_per_process_memory_fraction(12*1024**3/total)
    return torch


def capture():
    p = verify(); assert not (OUT/'capture_execution.json').exists()
    assert sha(p['assets']['checkpoint']) == p['assets']['sha256']
    torch = gpu(); sys.path.insert(0, str(ASSETS/'upstream'))
    from vggt.models.aggregator import Aggregator
    from vggt.utils.load_fn import load_and_preprocess_images
    model = Aggregator().eval()
    state = torch.load(p['assets']['checkpoint'], map_location='cpu', weights_only=True, mmap=True)
    result = model.load_state_dict({k.removeprefix('aggregator.'):v for k,v in state.items() if k.startswith('aggregator.')}, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    del state
    model.cached_layer_indices = {23}; model.requires_grad_(False).cuda()
    dump(OUT/'capture_execution.json', dict(started_epoch=time.time(), torch=torch.__version__, gpu=torch.cuda.get_device_name(0)))
    files = {}; stats = {}; start = time.monotonic()
    for scene in p['scenes']:
        key = scene['key']; records = read(INPUT/key/'manifest.json')['records']; stats[key] = []
        for i, record in enumerate(records):
            paths = [r['rgb'] for r in records[max(0, i-3):i+1]]
            images = load_and_preprocess_images(paths).cuda()
            assert tuple(images.shape[-2:]) == (140, 518)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                layers, offset = model(images[None])
                grid = layers[23][0, -1, offset:].float().cpu().numpy().reshape(10, 37, 2048).astype(np.float16)
            assert np.isfinite(grid).all()
            path = OUT/'features'/key/f'{record["frame_id"]:06d}.npz'; path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, grid=grid, causal_frame_ids=np.array([r['frame_id'] for r in records[max(0, i-3):i+1]]))
            files[str(path)] = sha(path); stats[key].append(dict(frame_id=record['frame_id'], context_frames=len(paths)))
            del images, layers
            if (i+1)%10 == 0:print('CAUSAL FEATURES', key, i+1, '/', len(records), flush=True)
    dump(OUT/'capture_complete.json', dict(complete=True, files=files, stats=stats, seconds=time.monotonic()-start,
        peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30, no_gt_read=True, no_future_images=True))


def scene_inputs(scene):
    key = scene['key']; manifest = read(INPUT/key/'manifest.json'); records = manifest['records']
    xyz = np.load(manifest['geometry'])['xyz']; job = read(scene['job'])
    grids = [np.load(OUT/'features'/key/f'{r["frame_id"]:06d}.npz')['grid'] for r in records]
    return records, xyz, grids, job['config']['cam']


def observe(ledger, key, i, record):
    obs = np.load(record['file']); sem = np.load(SEM/'frames'/key/f'{record["frame_id"]:06d}.npz')
    np.testing.assert_array_equal(obs['visible_source_ids'], sem['source_ids'])
    np.testing.assert_array_equal(obs['pixel_xy'], sem['source_pixel_xy'])
    ids = obs['visible_source_ids']
    ledger.observe(i, ids, obs['pixel_xy'], sem['source_probabilities'])
    return ids


def prepare():
    p = verify(); seal('capture_complete'); assert not (OUT/f'{PREPARED_SEAL}.json').exists()
    files = {}; stats = {}
    for scene in p['scenes']:
        key = scene['key']; records, xyz, grids, camera = scene_inputs(scene)
        ledger = SourceFeatureLedger(xyz); poses = [r['camera_pose'] for r in records]; rows = []
        for i, record in enumerate(records):
            previous = np.flatnonzero(ledger.latest >= 0)
            history = sample_sources(previous, ledger.probability[previous], 512, 17+i)
            observed = np.load(record['file'])['visible_source_ids']
            current_sem = np.load(SEM/'frames'/key/f'{record["frame_id"]:06d}.npz')
            np.testing.assert_array_equal(observed, current_sem['source_ids'])
            history_context, current_context = retrieve_context(previous, observed,
                ledger.probability[previous], current_sem['source_probabilities'], 256, 17+i)
            old = ledger.packet(history_context, grids, (camera['H'], camera['W']))
            current = observe(ledger, key, i, record)
            selected = sample_sources(current, ledger.probability[current], 512, 17+i)
            now = ledger.packet(current_context, grids, (camera['H'], camera['W']))
            packet = {k:np.concatenate([now[k], old[k]]) for k in now}
            relation = measured_relations(packet, xyz, poses, camera)
            query_ids = np.union1d(selected, history)
            query = ledger.packet(query_ids, grids, (camera['H'], camera['W']))
            path = PREPARED/key/f'{i:03d}.npz'; path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, context=packet['features'].astype(np.float16), frame_ids=packet['frame_ids'],
                context_source_ids=packet['source_ids'], context_pixels=packet['pixel_xy'], relations=relation,
                query=query['features'].astype(np.float16), query_ids=query_ids,
                query_frame_ids=query['frame_ids'], current_ids=current)
            files[str(path)] = sha(path)
            rows.append(dict(frame_id=record['frame_id'], current_sources=len(current), history_queries=len(history),
                context=len(packet['source_ids']), unique_queries=len(query_ids),
                repeat_relations=int((relation>0).sum()//2), conflict_relations=int((relation<0).sum()//2),
                measured_sources_seen=int((ledger.latest>=0).sum())))
        assert np.all(ledger.latest >= 0), 'All measured geometry must be represented'
        final = PREPARED/key/'ledger.npz'
        np.savez_compressed(final, latest=ledger.latest, pixel=ledger.pixel, probability=ledger.probability, count=ledger.count)
        files[str(final)] = sha(final); stats[key] = rows
        print('PREPARED STREAM', key, len(records), 'sources', len(xyz), 'relations', sum(r['repeat_relations']+r['conflict_relations'] for r in rows), flush=True)
    dump(OUT/f'{PREPARED_SEAL}.json', dict(complete=True, files=files, stats=stats, no_gt_read=True,
        capture_seal_sha256=sha(OUT/'capture_complete.json')))


def labels():
    from evaluate_kitti360_development import read_static_ply, transfer
    p = verify(); seal(PREPARED_SEAL); assert not (OUT/'training_labels.json').exists()
    taxonomy = runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['id2label']
    files = {}; stats = {}
    for scene in p['scenes']:
        key = scene['key']; raw = read_static_ply(p['references'][key]['raw']); ref = np.load(scene['reference'])
        train = np.array([taxonomy[int(s)].trainId for s in raw['semantic']]); truth = train*1000+raw['instance'].astype(np.int64)%1000
        truth[train==255] = 0; np.testing.assert_array_equal(truth, ref['train_instance'])
        xyz = np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz']
        nearest = transfer(ref['xyz'], np.arange(1,len(truth)+1), xyz, .2)
        index = np.maximum(nearest-1, 0); known = (nearest>0)&(train[index]!=255)&(ref['confidence'][index]>0)
        target = np.full(len(xyz), -1, np.int64); target[known] = 0
        obj = known & np.isin(train[index], [2,13]); target[obj] = truth[index[obj]]
        assert len(np.unique(target[target>0])) <= CONFIG['max_objects'], 'Insufficient persistent slot capacity'
        path = OUT/'training_only'/key/'labels.npz'; path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, target=target); files[str(path)] = sha(path)
        stats[key] = dict(known=int(known.sum()), objects=len(np.unique(target[target>0])), foreground=int(obj.sum()), unknown=int((target<0).sum()))
        print('TRAIN LABELS', key, stats[key], flush=True)
    dump(OUT/'training_labels.json', dict(complete=True, files=files, stats=stats, training_only=True))


def tensor_packet(torch, data, rng=None):
    context = data['context'].astype(np.float32).copy(); query = data['query'].astype(np.float32).copy()
    if rng is not None:
        rotation, shift = rng
        context[:, 2048:2051] = context[:, 2048:2051] @ rotation.T + shift/20
        query[:, 2048:2051] = query[:, 2048:2051] @ rotation.T + shift/20
    return (torch.tensor(context, device='cuda'), torch.tensor(query, device='cuda'),
            torch.tensor(data['frame_ids'], device='cuda'), torch.tensor(data['relations'], device='cuda'))


def train():
    from omni_semantic_map.joint_object_memory import source_instance_loss
    p = verify(); seal(PREPARED_SEAL); seal('training_labels')
    assert not (OUT/'training_execution.json').exists(); torch = gpu()
    dump(OUT/'training_execution.json', dict(started_epoch=time.time(), device=torch.cuda.get_device_name(0)))
    files = {}; stats = {}
    for target in KEYS:
        training = next(k for k in KEYS if k != target)
        paths = sorted((PREPARED/training).glob('[0-9][0-9][0-9].npz'))
        packets = [dict(np.load(f)) for f in paths]
        labels = np.load(OUT/'training_only'/training/'labels.npz')['target']
        for mode in ['geometry', 'no_geometry']:
            torch.manual_seed(17); torch.cuda.manual_seed_all(17)
            model = PersistentSourceMemory(**CONFIG).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
            rng = np.random.default_rng(17); curve = []; start = time.monotonic(); torch.cuda.reset_peak_memory_stats()
            for epoch in range(40):
                memory = None; binding = {}; losses = []; values = []; optimizer.zero_grad()
                yaw = rng.uniform(-np.pi, np.pi); c, s = np.cos(yaw), np.sin(yaw)
                rotation = np.array([[c,-s,0],[s,c,0],[0,0,1]], np.float32); shift = rng.normal(0, 5, 3).astype(np.float32)
                for i, data in enumerate(packets):
                    context, query, frame, relations = tensor_packet(torch, data, (rotation, shift))
                    out = model(context, frame, relations, memory=memory, use_geometry=mode=='geometry')
                    logits = model.decode_sources(query, out['memory'])
                    target_labels = torch.tensor(labels[data['query_ids']], device='cuda')
                    result = source_instance_loss(logits, target_labels, out['objectness'], binding)
                    binding = result['binding']; losses.append(result['loss']); values.append(float(result['loss'].detach()))
                    memory = out['memory']
                    if len(losses)==4 or i==len(packets)-1:
                        loss = torch.stack(losses).mean(); assert torch.isfinite(loss)
                        loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        assert torch.isfinite(norm)
                        optimizer.step(); optimizer.zero_grad(); memory = memory.detach(); losses = []
                curve.append(float(np.mean(values)))
                if epoch==0 or (epoch+1)%5==0:print('TRAIN MEMORY', target, mode, epoch+1, '/', 40, 'loss', curve[-1], 'slots', len(binding), flush=True)
            folder = OUT/'models'/target/mode; folder.mkdir(parents=True, exist_ok=False)
            torch.save(dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()}, configuration=CONFIG,
                training=[training], target=target, geometry=mode=='geometry'), folder/'model.pt')
            row = dict(epochs=40, steps=40*((len(packets)+3)//4), loss=curve, seconds=time.monotonic()-start,
                peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30, training=[training], target=target)
            dump(folder/'training.json', row); stats[target+'/'+mode] = row
            for f in folder.iterdir():files[str(f)] = sha(f)
            del model, optimizer, memory, out, logits, result, loss; torch.cuda.empty_cache()
    dump(OUT/'trained.json', dict(complete=True, files=files, stats=stats, protocol_sha256=sha(OUT/'protocol.json')))


def decode(torch, model, memory, features, objectness):
    logits = model.decode_sources(torch.tensor(features, device='cuda'), memory)
    logits[:, 1:] += torch.nn.functional.logsigmoid(objectness)[None]
    return logits.argmax(1).cpu().numpy()


def export(folder, xyz, owners, probability, existence, extra):
    objects = []; retained = np.zeros_like(owners)
    for slot in np.unique(owners[owners>0]):
        take = owners==slot
        if take.sum()<16:continue
        p = probability[take].mean(0); cls = int(p.argmax())
        if cls not in [2,13]:continue
        retained[take] = slot
        objects.append(dict(object_id=int(slot), observation_ids=[], selected_observation_ids=[],
            predictions=dict(mean=dict(name='building' if cls==2 else 'car', semantic_id=11 if cls==2 else 26,
                train_id=cls, score=float(existence[slot-1]*p[cls]), retained=True))))
    folder.mkdir(parents=True, exist_ok=False); source_ids = np.flatnonzero(retained>0)
    np.savez_compressed(folder/'geometry.npz', xyz=xyz[source_ids], instance=retained[source_ids], source_ids=source_ids)
    np.savez_compressed(folder/'source_ownership.npz', raw=owners, retained=retained)
    dump(folder/'classification.json', dict(objects=objects))
    row = dict(objects=len(objects), points=len(source_ids), **extra); dump(folder/'run.json', row)
    print('MAP MEMORY', folder.parent.name, folder.name, row, flush=True)
    return row


def predict():
    p = verify(); seal(PREPARED_SEAL); seal('trained'); assert not (OUT/'prediction_execution.json').exists()
    torch = gpu(); dump(OUT/'prediction_execution.json', dict(started_epoch=time.time()))
    files = {}; stats = {}
    for scene in p['scenes']:
        key = scene['key']; records, xyz, grids, camera = scene_inputs(scene); hw = (camera['H'], camera['W'])
        folder = OUT/'maps'/key/'baseline'; folder.mkdir(parents=True, exist_ok=False)
        for name in ['geometry.npz', 'classification.json']:shutil.copyfile(BASE/key/'geometry_graph'/name, folder/name)
        dump(folder/'run.json', dict(exact_copy=True))
        stats[key] = {}
        for mode in ['geometry', 'no_geometry']:
            checkpoint = torch.load(OUT/'models'/key/mode/'model.pt', map_location='cpu', weights_only=True)
            assert checkpoint['training']==[k for k in KEYS if k!=key]
            model = PersistentSourceMemory(**checkpoint['configuration']).eval().cuda()
            model.load_state_dict(checkpoint['state'], strict=True); model.requires_grad_(False)
            ledger = SourceFeatureLedger(xyz); owners = np.zeros(len(xyz), np.int64); memory = None; started = time.monotonic()
            with torch.inference_mode():
                for i, rec in enumerate(records):
                    data = dict(np.load(PREPARED/key/f'{i:03d}.npz'))
                    current = observe(ledger, key, i, rec)
                    out = model(torch.tensor(data['context'].astype('float32'), device='cuda'),
                        torch.tensor(data['frame_ids'], device='cuda'), torch.tensor(data['relations'], device='cuda'),
                        memory=memory, use_geometry=mode=='geometry')
                    memory = out['memory']
                    for lo in range(0,len(current),4096):
                        ids = current[lo:lo+4096]; q = ledger.packet(ids, grids, hw)
                        owners[ids] = decode(torch, model, memory, q['features'], out['objectness'])
                    if (i+1)%15==0:print('STREAM MAP', key, mode, i+1, '/', len(records), flush=True)
                assert np.all(ledger.latest>=0)
                existence = out['objectness'].sigmoid().cpu().numpy()
                revised = np.empty_like(owners)
                for lo in range(0,len(xyz),4096):
                    ids = np.arange(lo,min(lo+4096,len(xyz))); q = ledger.packet(ids, grids, hw)
                    revised[ids] = decode(torch, model, memory, q['features'], out['objectness'])
            extra = dict(seconds=time.monotonic()-started, measured_sources=len(xyz),
                raw_historical_changes=int((revised!=owners).sum()), slot_capacity=CONFIG['max_objects'], inference_gt_read=False)
            if mode=='geometry':
                stats[key]['current_commit'] = export(OUT/'maps'/key/'current_commit', xyz, owners, ledger.probability, existence,
                    dict(**extra, final_history_readout_discarded=True))
                stats[key]['ours_history'] = export(OUT/'maps'/key/'ours_history', xyz, revised, ledger.probability, existence,
                    dict(**extra, final_history_readout_discarded=False))
            else:stats[key]['no_geometry'] = export(OUT/'maps'/key/'no_geometry', xyz, revised, ledger.probability, existence, extra)
            del model, memory, out; torch.cuda.empty_cache()
        for f in (OUT/'maps'/key).rglob('*'):
            if f.is_file():files[str(f)] = sha(f)
    dump(OUT/'predictions_complete.json', dict(complete=True, files=files, stats=stats, sealed_before_scoring=True,
        trained_seal_sha256=sha(OUT/'trained.json')))


def score():
    from evaluate_kitti360_development import transfer, clean_json
    from omni_semantic_map.kitti360_evaluation import evaluate
    from run_discovery_query_benchmark import matched_ids
    p = verify(); predictions = seal('predictions_complete'); assert not (OUT/'summary.json').exists()
    results = {}; means = {m:[] for m in METHODS}; gained = lost = 0; gate = True
    for scene in p['scenes']:
        key = scene['key']; ref = np.load(scene['reference']); rows = {}
        for method in METHODS:
            folder = OUT/'maps'/key/method; g = np.load(folder/'geometry.npz'); objects = read(folder/'classification.json')['objects']
            projection = transfer(g['xyz'], g['instance'], ref['xyz'], .2)
            scored = evaluate(ref['train_instance'], ref['confidence'], projection,
                {o['object_id']:o['predictions']['mean'] for o in objects}, EVALUATOR)
            dump(folder/'matches.json', clean_json(scored['matches'])); np.savez_compressed(folder/'reference_prediction.npz', instance=projection)
            rows[method] = dict(metrics=clean_json(scored['averages']), matched_gt_ids=sorted(matched_ids(scored['matches'])))
            means[method].append(float(scored['averages']['allAp']))
            print('MEMORY FULL MAP SCORE', key, method, rows[method]['metrics'], flush=True)
        expected = read(ROOT/'build/benchmarks/signed_query_association_v1/summary.json')['results'][key]['methods']['geometry_graph']['metrics']
        assert abs(rows['baseline']['metrics']['allAp']-expected['allAp'])<1e-12
        base = set(rows['baseline']['matched_gt_ids'])
        for row in rows.values():
            ids = set(row['matched_gt_ids']); row['changes_vs_baseline'] = dict(gained=sorted(ids-base), lost=sorted(base-ids))
        change = rows['ours_history']['changes_vs_baseline']; gained += len(change['gained']); lost += len(change['lost'])
        gate &= rows['ours_history']['metrics']['allAp']>=rows['baseline']['metrics']['allAp']
        results[key] = dict(methods=rows)
    mean = {k:float(np.mean(v)) for k,v in means.items()}
    gate &= mean['ours_history']>=mean['baseline']+.01 and mean['ours_history']>max(mean['current_commit'],mean['no_geometry']) and gained>=1 and lost==0
    gate &= any(row['ours_history']['raw_historical_changes']>0 for row in predictions['stats'].values())
    dump(OUT/'summary.json', dict(complete=True, results=results, mean_primary_ap=mean, gained=gained, lost=lost,
        advance_gate=bool(gate), independent_test=False, official_leaderboard=False,
        prediction_seal_sha256=sha(OUT/'predictions_complete.json')))
    print('MEMORY GATE', bool(gate), mean, 'gained', gained, 'lost', lost, flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['freeze','capture','prepare','labels','train','predict','score'])
    globals()[parser.parse_args().action]()
