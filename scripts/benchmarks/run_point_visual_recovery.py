"""Finite, development-first test of measured-point visual ownership evidence."""
import argparse
import time
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from paper_final_common import ROOT, read, dump, sha
from run_intrinsic_surface_ownership import OUT as GEOM, verify as verify_geom, sealed as geom_sealed, inference
from run_surface_ownership import OUT as PARENT, PRIOR, INPUT, OUTDOOR, sealed as parent_sealed
from omni_semantic_map.point_visual_evidence import VISUAL_FEATURES, patch_indices, similarity_matrix, view_evidence, summarize_evidence
from omni_semantic_map.surface_ownership import OwnershipScorer, append_owned

OUT = ROOT / 'build/benchmarks/point_visual_recovery_v1'
MODEL = ROOT / 'build/benchmarks/flat_dino_20261109/model'
METHODS = ['baseline', 'nearest_core', 'intrinsic_geometry', 'rgb', 'dino_first', 'ours_dino', 'shuffled_dino']
TRAINED = METHODS[3:]


def freeze():
    assert not OUT.exists()
    parent = verify_geom(True)
    for name in ['features_complete', 'trained', 'predictions_complete']: geom_sealed(name)
    parent_sealed('training_labels_complete')
    sources = [Path(__file__), ROOT/'omni_semantic_map/omni_semantic_map/point_visual_evidence.py', ROOT/'omni_semantic_map/test/test_point_visual_evidence.py']
    models = [MODEL/'config.json', MODEL/'model.safetensors']
    dump(OUT/'protocol.json', dict(created_epoch=time.time(), scenes=parent['scenes'], methods=METHODS,
        sources={str(f): sha(f) for f in sources}, model_files={str(f): sha(f) for f in models},
        geometry_protocol_sha256=sha(GEOM/'protocol.json'),
        hypothesis='Visibility-linked local appearance can distinguish correct discarded measurements from background/other-object support better than geometry alone. This is a point-allocation hypothesis, not a new object discovery or pretrained-backbone novelty claim.',
        extraction='Frozen DINOv2-small, original local weights. Uncropped bicubic RGB resize to width896; height nearest positive multiple of14 preserving aspect. ImageNet normalization; FP32 inference, FP16 stored patch descriptors. No model fine-tuning, prompt change, new masks or unseen-window data. GPU0 allocator budget8GiB.',
        evidence='Project only already depth-verified measured visibility into patch cells using pixel centers. Predicted original core patches are anchors. Remove anchors shared by own and other cores; exclude query patch on both sides. Average top min(3,n) distinct anchor-patch cosines. One vote per source/owner/frame. Compare own and other cores per frame, then frame-balanced means/extrema/std and support counts. No physical coordinate-axis dependence.',
        rgb='Same patch grid, image, visibility and aggregation as DINO. Patch mean RGB, descriptor [r,g,b,1-r,1-g,1-b], cosine normalized. Tests simple color affinity with same learned head and dimensions.',
        ablations='First valid frame only; full multi-view; same-class fixed-seed shuffled full visual features. Shuffle within each scene and predicted category, never using GT. Original intrinsic geometry and nearest-core/strong baseline copied exactly.',
        features=list(VISUAL_FEATURES), training='Independent BCE MLP24->32->32->1; source-confidence weights, eligible old training labels, train-only standardization. Adam0.001,40epochs,batch2048,seed0,CPU. Same model/head/loss/budget for all four new methods. Threshold sigmoid>0.5; exact ties abstain.',
        splits='Crossfit only exposed0000/0002. Each target is trained only on the other scene. No0004/0005 access. Robot2245 permitted only if the predeclared development gate passes; train on both public development scenes, no robot labels.',
        budget='108 development RGB encodings,8fits,8new maps plus6copied controls. No resolution/threshold/epoch sweep. If gate passes:330robot encodings,4fits and4new robot maps plus3copied controls. Independent public evaluation needs a separate prospectively sealed split; current0004/0005 are exposed.',
        gate='Ours mean AP50:90 >= baseline+0.02 and nearest_core+0.02 and intrinsic_geometry+0.01; strictly greater than RGB, first-frame and shuffled controls. No per-window AP50:90/AP50 regression vs baseline; at least one new AP50 GT match and no lost matches. Otherwise seal branch; no robot expansion or parameter search.',
        scope='Window-end measured support restoration. Old core XYZ/IDs and classification scores immutable; source may only join an existing eligible candidate owner once. No object birth, existing core deletion, synthetic geometry, online causality or building-recall claim.',
        headroom_audit_sha256=sha(ROOT/'build/benchmarks/surface_recovery_headroom_v1/summary.json')))
    print('FROZEN', flush=True)


def verify(data=False):
    p = read(OUT/'protocol.json'); verify_geom(data)
    assert sha(GEOM/'protocol.json') == p['geometry_protocol_sha256']
    for path, h in {**p['sources'], **p['model_files']}.items(): assert sha(path) == h, path
    return p


def selected(p, robot):
    if robot:
        assert read(OUT/'summary.json')['advance_gate'], 'Robot expansion requires development gate'
    return [s for s in p['scenes'] if (s['key'] not in OUTDOOR) == robot]


def seal(name, robot=False):
    d = read(OUT/(name+('_robot' if robot else '')+'.json'))
    for path, h in d['files'].items(): assert sha(path) == h, path
    return d


def write_seal(name, files, stats, robot=False, **extra):
    dump(OUT/(name+('_robot' if robot else '')+'.json'), dict(complete=True, files=files, stats=stats,
        protocol_sha256=sha(OUT/'protocol.json'), **extra))


def tokens(robot=False):
    import torch
    from transformers import Dinov2Config, Dinov2Model
    from safetensors.torch import load_file
    p = verify(True); scenes = selected(p, robot); files = {}; stats = {}
    assert not (OUT/('tokens_complete'+('_robot' if robot else '')+'.json')).exists()
    torch.set_num_threads(2); torch.set_float32_matmul_precision('highest')
    torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory, 0)
    model = Dinov2Model(Dinov2Config.from_pretrained(str(MODEL), local_files_only=True))
    model.load_state_dict(load_file(str(MODEL/'model.safetensors')), strict=True)
    model = model.eval().to('cuda').requires_grad_(False)
    assert model.config.patch_size == 14
    mean = torch.tensor([.485, .456, .406], device='cuda')[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device='cuda')[None, :, None, None]
    for scene in scenes:
        key = scene['key']; manifest = read(INPUT/key/'manifest.json'); start = time.monotonic(); records = []
        folder = OUT/'tokens'/key; folder.mkdir(parents=True, exist_ok=True)
        progress = folder/'progress.json'; previous = read(progress)['records'] if progress.exists() else []
        for k, r in enumerate(manifest['records']):
            assert sha(r['file']) == r['sha256']
            expected = manifest['input_hashes'][r['rgb']]
            assert sha(r['rgb']) == expected
            if k < len(previous):
                record = previous[k]; assert record['frame_id'] == r['frame_id'] and record['rgb_sha256'] == expected
                assert sha(record['file']) == record['sha256']; records.append(record); files[record['file']] = record['sha256']; continue
            image = Image.open(r['rgb']).convert('RGB'); ow, oh = image.size
            width = 896; height = max(14, 14*round(oh/ow*width/14)); gw, gh = width//14, height//14
            rgb = np.array(image.resize((width, height), Image.Resampling.BICUBIC))
            x = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].to('cuda', torch.float32)/255
            with torch.inference_mode(): feature = model(pixel_values=(x-mean)/std).last_hidden_state[0, 1:].cpu().numpy()
            assert feature.shape == (gh*gw, 384) and np.isfinite(feature).all()
            color = rgb.reshape(gh, 14, gw, 14, 3).mean((1, 3)).reshape(-1, 3).astype(np.float32)/255
            path = folder/f'{r["frame_id"]:06d}.npz'
            np.savez_compressed(path, dino=feature.astype(np.float16), rgb=np.c_[color, 1-color], grid=np.array([gw, gh]), original_size=np.array([ow, oh]))
            record = dict(frame_id=r['frame_id'], file=str(path), sha256=sha(path), rgb_sha256=expected, grid=[gw, gh])
            records.append(record); files[str(path)] = record['sha256']
            dump(progress, dict(records=records, protocol_sha256=sha(OUT/'protocol.json')))
            if (k+1)%10 == 0: print('TOKENS', key, k+1, len(manifest['records']), flush=True)
        stats[key] = dict(frames=len(records), seconds=time.monotonic()-start, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    write_seal('tokens_complete', files, stats, robot)
    print('TOKENS COMPLETE', stats, flush=True)


def features(robot=False):
    p = verify(True); seal('tokens_complete', robot); files = {}; stats = {}
    for scene in selected(p, robot):
        key = scene['key']; start = time.monotonic(); d = np.load(GEOM/'features'/key/'candidates.npz')
        groups, slots = np.where(d['valid']); owners = d['owners'][groups, slots]; sources = d['source_ids'][groups]
        manifest = read(INPUT/key/'manifest.json'); total = manifest['source_points']
        core = read(PARENT/'features'/key/'core_sources.json'); visible = np.zeros(len(groups), np.int32)
        evidence = {mode: [] for mode in ['dino', 'rgb']}
        for index, r in enumerate(manifest['records']):
            measured = np.load(r['file']); sid = measured['visible_source_ids']; pixels = measured['pixel_xy']
            token = np.load(OUT/'tokens'/key/f'{r["frame_id"]:06d}.npz'); width, height = token['original_size']; gw, gh = token['grid']
            patches = patch_indices(pixels, width, height, gw, gh)
            lookup = np.full(total, -1, np.int32); lookup[sid] = patches
            query = lookup[sources]; present = query >= 0; visible += present
            anchors = {int(oid): np.unique(lookup[np.asarray(ids, np.int64)][lookup[np.asarray(ids, np.int64)] >= 0]) for oid, ids in core.items()}
            similarity = {mode: similarity_matrix(token[mode]) for mode in evidence}
            for oid in np.unique(owners[present]):
                rows = np.flatnonzero(present & (owners == oid)); other = np.unique(np.concatenate([v for o, v in anchors.items() if o != oid]))
                for mode in evidence:
                    own, negative, valid, competing = view_evidence(similarity[mode], query[rows], anchors[oid], other)
                    if valid.any(): evidence[mode].append(np.column_stack((rows[valid], np.full(valid.sum(), r['frame_id']), own[valid], negative[valid], competing[valid])))
            if (index+1)%15 == 0: print('EVIDENCE', key, index+1, len(manifest['records']), flush=True)
        visual = {}; trace = {}
        for mode, chunks in evidence.items():
            trace[mode] = np.concatenate(chunks) if chunks else np.empty((0, 5))
            a = trace[mode]; args = (a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2], a[:, 3], a[:, 4].astype(bool), visible)
            visual[mode] = summarize_evidence(*args)
            if mode == 'dino': visual['first'] = summarize_evidence(*args, first_only=True)
        shuffled = visual['dino'].copy(); rng = np.random.RandomState(0)
        semantic = d['features'][groups, slots, 10:12].argmax(1)
        for category in np.unique(semantic):
            ix = np.flatnonzero(semantic == category); shuffled[ix] = visual['dino'][rng.permutation(ix)]
        visual['shuffled'] = shuffled
        folder = OUT/'features'/key; folder.mkdir(parents=True, exist_ok=False)
        arrays = dict(valid=d['valid'], owners=d['owners'], source_ids=d['source_ids'])
        for method, mode in dict(rgb='rgb', dino_first='first', ours_dino='dino', shuffled_dino='shuffled').items():
            x = np.zeros(d['valid'].shape+(24,), np.float32); x[:, :, :12] = d['features'][:, :, :12]; x[groups, slots, 12:] = visual[mode]
            assert np.isfinite(x).all(); arrays[method] = x
        np.savez_compressed(folder/'candidates.npz', **arrays)
        np.savez_compressed(folder/'view_evidence.npz', pair_groups=groups, pair_slots=slots, visible_views=visible, **trace)
        for path in folder.iterdir(): files[str(path)] = sha(path)
        stats[key] = dict(candidate_pairs=len(groups), pairs_with_visual_evidence=int((visual['dino'][:, 0]>0).sum()), pairs_with_multiple_views=int((visual['dino'][:, 0]>np.log(2)+1e-6).sum()), rows=len(trace['dino']), seconds=time.monotonic()-start)
        print('FEATURES COMPLETE', key, stats[key], flush=True)
    write_seal('features_complete', files, stats, robot)


def train(robot=False):
    import torch
    p = verify(True); seal('features_complete'); parent_sealed('training_labels_complete')
    if robot: seal('features_complete', True)
    files = {}; stats = {}; torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    for scene in selected(p, robot):
        key = scene['key']; training = [k for k in OUTDOOR if k != key]
        for method in TRAINED:
            torch.manual_seed(0); rng = np.random.RandomState(0); parts = []; width = 0
            for k in training:
                d = np.load(OUT/'features'/k/'candidates.npz'); y = np.load(PARENT/'training_labels'/k/'TRAINING_ONLY.npz'); keep = y['eligible']
                width = max(width, d['valid'].shape[1]); parts.append((d[method][keep], d['valid'][keep], y['positive'][keep], y['weight'][keep]))
            arrays = []
            for i in range(4):
                arrays.append(np.concatenate([np.pad(part[i], [(0, 0), (0, width-part[i].shape[1])]+([(0, 0)] if i == 0 else [])) if i < 3 else part[i] for part in parts]))
            xx, v, positive, w = arrays; mean = xx[v].mean(0); std = np.maximum(xx[v].std(0), 1e-6)
            x = torch.tensor((xx-mean)/std); valid = torch.tensor(v); pos = torch.tensor(positive); weight = torch.tensor(w/w.mean(), dtype=torch.float32)
            model = OwnershipScorer(24); optimizer = torch.optim.Adam(model.parameters(), lr=.001); curve = []; start = time.monotonic()
            for epoch in range(40):
                order = rng.permutation(len(x)); total = 0.
                for lo in range(0, len(order), 2048):
                    ix = order[lo:lo+2048]; logits = model(x[ix]); loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, pos[ix].float(), reduction='none')*valid[ix]).sum(1)/valid[ix].sum(1)
                    objective = (loss*weight[ix]).mean(); optimizer.zero_grad(); objective.backward(); optimizer.step(); total += float(objective.detach())*len(ix)
                curve.append(total/len(x))
            folder = OUT/'models'/key/method; folder.mkdir(parents=True, exist_ok=False)
            torch.save(dict(state=model.state_dict(), mean=mean, std=std, dimensions=24, training=training, method=method), folder/'model.pt')
            stat = dict(training=training, groups=len(x), loss=curve, seconds=time.monotonic()-start); dump(folder/'training.json', stat)
            for path in folder.iterdir(): files[str(path)] = sha(path)
            stats[key+'/'+method] = stat; print('TRAINED', key, method, curve[0], curve[-1], flush=True)
    write_seal('trained', files, stats, robot)


def predict(robot=False):
    import torch
    p = verify(True); seal('features_complete', robot); seal('trained', robot); files = {}; stats = {}; torch.set_num_threads(1)
    for scene in selected(p, robot):
        key = scene['key']; d = np.load(OUT/'features'/key/'candidates.npz'); core = np.load(PRIOR/'maps'/key/'largest/geometry.npz'); classification = read(PRIOR/'maps'/key/'largest/classification.json')
        xyz = np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz']
        for method in METHODS:
            folder = OUT/'maps'/key/method; folder.mkdir(parents=True, exist_ok=False)
            if method not in TRAINED:
                source = GEOM/'maps'/key/method
                for name in ['geometry.npz', 'decisions.npz', 'classification.json']: shutil.copyfile(source/name, folder/name); assert sha(source/name) == sha(folder/name)
                stat = dict(read(source/'run.json'), copied_from=str(source), copied_exact=True)
            else:
                path = OUT/'models'/key/method/'model.pt'; m = torch.load(path, map_location='cpu', weights_only=False); assert key not in m['training']
                owners, confidence = inference(m, d[method], d['valid'], d['owners']); g = append_owned(core['xyz'], core['instance'], xyz, d['source_ids'], owners)
                np.testing.assert_array_equal(g['xyz'][:len(core['xyz'])], core['xyz']); np.testing.assert_array_equal(g['instance'][:len(core['xyz'])], core['instance'])
                assert np.all((owners == 0) | ((owners[:, None] == d['owners']) & d['valid']).any(1))
                np.savez_compressed(folder/'geometry.npz', **g); np.savez_compressed(folder/'decisions.npz', source_ids=d['source_ids'], owners=owners, confidence=confidence); dump(folder/'classification.json', classification)
                stat = dict(objects=len(classification['objects']), old_points=len(core['xyz']), added_points=int((owners>0).sum()), model_sha256=sha(path), core_bitwise_equal=True, exact_measured_additions=True)
            dump(folder/'run.json', stat)
            for path in folder.iterdir(): files[str(path)] = sha(path)
            stats[key+'/'+method] = stat; print('PREDICTED', key, method, stat['added_points'], flush=True)
    write_seal('predictions_complete', files, stats, robot)


def score(robot=False):
    assert not robot, 'Robot accuracy unavailable; report execution only'
    from evaluate_kitti360_development import transfer, clean_json
    from omni_semantic_map.kitti360_evaluation import evaluate
    from run_discovery_query_benchmark import matched_ids
    p = verify(); seal('predictions_complete'); assert not (OUT/'summary.json').exists()
    parent = read(GEOM/'summary.json'); results = {}; values = {m: [] for m in METHODS}; no_harm = True; gains = 0
    for scene in selected(p, False):
        key = scene['key']; ref = np.load(scene['reference']); rows = {}; base = set(parent['results'][key]['methods']['baseline']['matched_gt_ids'])
        for method in METHODS:
            folder = OUT/'maps'/key/method; g = np.load(folder/'geometry.npz'); cls = read(folder/'classification.json')['objects']
            prediction = transfer(g['xyz'], g['instance'], ref['xyz'], .2)
            result = evaluate(ref['train_instance'], ref['confidence'], prediction, {r['object_id']: r['predictions']['mean'] for r in cls}, ROOT/'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py')
            hits = matched_ids(result['matches']); metrics = clean_json(result['averages']); changes = dict(gained=sorted(hits-base), lost=sorted(base-hits))
            rows[method] = dict(metrics=metrics, matched_gt_ids=sorted(hits), changes_vs_baseline=changes, run=read(folder/'run.json'))
            dump(folder/'matches.json', clean_json(result['matches'])); values[method].append(metrics['allAp'])
            if method not in TRAINED: assert metrics == parent['results'][key]['methods'][method]['metrics']
            print('SCORED', key, method, metrics['allAp'], metrics['allAp50%'], changes, flush=True)
        ours, base_row = rows['ours_dino'], rows['baseline']; no_harm &= not ours['changes_vs_baseline']['lost'] and all(ours['metrics'][m] >= base_row['metrics'][m]-1e-12 for m in ['allAp', 'allAp50%']); gains += len(ours['changes_vs_baseline']['gained'])
        results[key] = dict(methods=rows)
    means = {m: float(np.mean(v)) for m, v in values.items()}
    checks = dict(no_harm=bool(no_harm), gained_gt=gains>0, baseline_margin=means['ours_dino'] >= means['baseline']+.02, nearest_margin=means['ours_dino'] >= means['nearest_core']+.02, geometry_margin=means['ours_dino'] >= means['intrinsic_geometry']+.01, ablations=all(means['ours_dino'] > means[m] for m in ['rgb', 'dino_first', 'shuffled_dino']))
    dump(OUT/'summary.json', dict(complete=True, results=results, mean_primary_ap=means, advance_gate=all(checks.values()), gate_checks=checks, gained_gt=gains, development_crossfit=True, robot_accuracy_available=False, prediction_seal_sha256=sha(OUT/'predictions_complete.json')))
    print('ADVANCE', all(checks.values()), checks, means, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['freeze', 'tokens', 'features', 'train', 'predict', 'score']); parser.add_argument('--robot', action='store_true'); args = parser.parse_args()
    freeze() if args.action == 'freeze' else globals()[args.action](args.robot)
