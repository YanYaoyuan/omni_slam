"""Prospectively bounded crop-query feasibility check on exposed outdoor data.

No model fitting, new test windows, map replacement, or accuracy claim from point
counts. Current full-frame masks are shared inputs; additional crops cost extra.
"""
import argparse
import time
import runpy
from pathlib import Path
import numpy as np
from PIL import Image
from paper_final_common import ROOT, read, dump, sha
from run_discovery_query_benchmark import OUT as FRONT, INPUT, verify as verify_front
from run_deferred_surface_support import OUT as SUPPORT
from run_surface_ownership import OUT as OWNERSHIP
from omni_semantic_map.surface_query_regions import SurfaceQueryMemory, crop_grid, vertical_surface_weight

OUT = ROOT/'build/benchmarks/surface_query_regions_probe_v1'
POLICIES = ['uniform', 'geometry', 'current', 'memory']
SEMANTICS = {26: 13, 11: 2}


def freeze():
    assert not OUT.exists(); parent = verify_front(True)
    scenes = [s for s in parent['scenes'] if s['key'] in ['kitti360_0000', 'kitti360_0002']]
    for s in scenes:
        n = len(read(INPUT/s['key']/'manifest.json')['records']); s['probe_ranks'] = [n*i//5 for i in [1, 2, 3, 4]]
    paths = [Path(__file__), ROOT/'omni_semantic_map/omni_semantic_map/surface_query_regions.py', ROOT/'omni_semantic_map/test/test_surface_query_regions.py']
    dump(OUT/'protocol.json', dict(created_epoch=time.time(), scenes=scenes, policies=POLICIES, source_hashes={str(f): sha(f) for f in paths},
        parent_protocol_sha256=sha(FRONT/'protocol.json'), parent_masks_sha256=sha(FRONT/'masks_complete.json'),
        checkpoint=parent['sam3_checkpoint'], checkpoint_sha256=parent['sam3_sha256'], prompts=parent['prompts'], semantic_ids=parent['semantic_ids'],
        input='Existing depth-verified measured visibility, fixed20cm causal cell IDs, full-frame SAM3 masks. Read only current/past masks when selecting crops. No GT, reference semantic features, learned class filters or future masks.',
        grid='Six fixed overlapping boxes: width ceil(W/2), height ceil(2H/3); x0 in [0,floor((W-w)/2),W-w], y0 in [0,H-h]. Uniform visits indices[0,2,3,5] on the four probe frames. Other policies choose highest score, first-index tie.',
        geometry='Current visible-source nearest16 covariance, include all numerical boundary ties within1e-9*max(1,d). Planarity=(lambda1-lambda0)/lambda2 times 1-(normal dot gravity)^2. Public world frame gravity=[0,0,1]. Average weights/pixel coordinates within each20cm cell; one cell contributes once per crop. Shared preference for vertical surfaces, not a general objectness estimate.',
        memory='On every received frame, count unique visible cells and unique mask-covered cells. Geometry sums current weights; current multiplies by uncovered-current-frame indicator; memory multiplies by smoothed miss frequency(seen-covered+1)/(seen+2). Existing image-wide masks are available before additional crop selection. No claim of calibrated uncertainty or independent Bernoulli observations.',
        extraction='Same native SAM3 weights, confidence0.5,resolution1008,BF16, car/building text. Real RGB crops, returned masks lifted to original coordinates. No box-exemplar prompt, prompt expansion, fine-tuning or mask search. Identical boxes within frame reuse exactly one prediction; logical budgets and actual calls reported separately.',
        budget='8 index-selected images x4 policies x1crop x2text prompts = maximum32extra image encodes/64decodes; shared full-image inference was previously paid. No extra frame/crop/prompt/threshold selection after results. No new full maps at this stage.',
        pre_inference_gate='Memory must choose a different crop from current-only in at least one of the four fixed probe frames in each scene. Otherwise stop before new model inference.',
        post_inference_gate='In EACH scene: memory adds at least100 confidence-weighted correct semantic source mass outside the full-sequence original same-class raw-mask support; known-label new-support semantic precision>=0.5; correct new mass strictly exceeds all three equal-budget controls. Both stages must pass before proposing complete-map evaluation. This is a feasibility gate, not AP or object-instance accuracy.',
        diagnostic='Only after all crop predictions are sealed, use old development reference labels to audit new measured support; ignore raw semantic255/unmatched/zero confidence. Unique source/class decisions counted across the four probe frames. New semantics without proper identities do not establish map quality.',
        limits='Exposed development windows,8frames,vertical-surface prior andfixed tiling. Cannot establish novelty, official benchmark advantage, robot accuracy, online SLAM performance or independent generalization. Original maps remain immutable.',
        related_work={'SAM-V': 'https://arxiv.org/html/2609.25490v1', 'OVI-MAP': 'https://arxiv.org/html/2603.26541v1', 'ActiveSAM': 'https://arxiv.org/abs/2606.16996', 'Urbex': 'https://arxiv.org/abs/2610.18356'}))
    print('FROZEN', [(s['key'], s['probe_ranks']) for s in scenes], flush=True)


def verify():
    p = read(OUT/'protocol.json'); verify_front(True)
    assert sha(FRONT/'protocol.json') == p['parent_protocol_sha256'] and sha(FRONT/'masks_complete.json') == p['parent_masks_sha256']
    for f, h in p['source_hashes'].items(): assert sha(f) == h, f
    return p


def select():
    p = verify(); assert not (OUT/'selection.json').exists(); original = read(FRONT/'masks_complete.json'); results = {}; files = {}; gates = {}
    for s in p['scenes']:
        key = s['key']; m = read(INPUT/key/'manifest.json'); xyz = np.load(m['geometry'])['xyz']; cells = np.load(INPUT/'query_cells'/key/'source_to_cell.npy')
        mask_rows = {r['frame_id']: r for r in original['scenes'][key]['records']}; memory = SurfaceQueryMemory(); probes = []
        for rank, r in enumerate(m['records']):
            f = np.load(r['file']); ids = f['visible_source_ids']; pixels = f['pixel_xy']; segmentation = np.load(mask_rows[r['frame_id']]['file'])['segmentation']
            positive = segmentation[pixels[:, 1], pixels[:, 0]] >= 0; memory.observe(r['frame_id'], cells[ids], positive)
            if rank not in s['probe_ranks']: continue
            image = Image.open(r['rgb']); width, height = image.size; boxes = crop_grid(width, height); weights = vertical_surface_weight(xyz[ids], [0, 0, 1])
            record = memory.rank(r['frame_id'], cells[ids], pixels, weights, positive, boxes, [0, 2, 3, 5][len(probes)])
            record.update(frame_id=r['frame_id'], frame_rank=rank, boxes=boxes.tolist(), rgb=r['rgb'], width=width, height=height, measured=r['file'])
            folder = OUT/'selection'/key; folder.mkdir(parents=True, exist_ok=True); path = folder/f'{r["frame_id"]:06d}.npz'
            np.savez_compressed(path, source_ids=ids, cells=cells[ids], pixel_xy=pixels, weights=weights, covered=positive, seen_count=memory.seen[cells[ids]], covered_count=memory.covered[cells[ids]])
            files[str(path)] = sha(path); probes.append(record)
            print('SELECT', key, r['frame_id'], record['chosen'], flush=True)
        gates[key] = any(r['chosen']['memory'] != r['chosen']['current'] for r in probes)
        results[key] = probes
    dump(OUT/'selection.json', dict(complete=True, results=results, files=files, gate_by_scene=gates, pre_inference_gate=all(gates.values()), protocol_sha256=sha(OUT/'protocol.json')))
    print('SELECTION GATE', gates, flush=True)


def predict():
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    p = verify(); chosen = read(OUT/'selection.json'); assert chosen['pre_inference_gate'], 'No distinct memory action; stop before inference'
    assert not (OUT/'predictions_complete.json').exists()
    for f, h in chosen['files'].items(): assert sha(f) == h
    assert sha(p['checkpoint']) == p['checkpoint_sha256']
    torch.set_num_threads(4); free, _ = torch.cuda.mem_get_info(); assert free >= 18*1024**3
    torch.cuda.set_per_process_memory_fraction(.5); torch.manual_seed(0)
    model = build_sam3_image_model(checkpoint_path=p['checkpoint'], load_from_HF=False, device='cuda', compile=False)
    proc = Sam3Processor(model, resolution=1008, confidence_threshold=.5); files = {}; results = {}; calls = 0; start = time.monotonic()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        warm = proc.set_image(Image.new('RGB', (640, 360))); proc.set_text_prompt(prompt='car', state=warm); del warm
        for key, records in chosen['results'].items():
            results[key] = []
            for r in records:
                image = Image.open(r['rgb']).convert('RGB'); generated = {}
                for box_id in sorted(set(r['chosen'].values())):
                    box = r['boxes'][box_id]; crop = image.crop(box); state = proc.set_image(crop); masks = []; scores = []; semantics = []
                    for prompt, semantic in zip(p['prompts'], p['semantic_ids']):
                        proc.reset_all_prompts(state); result = proc.set_text_prompt(prompt=prompt, state=state)
                        small = result['masks'].cpu().numpy().reshape(-1, crop.height, crop.width).astype(bool); probability = result['scores'].float().cpu().numpy()
                        lifted = np.zeros((len(small), image.height, image.width), bool); lifted[:, box[1]:box[3], box[0]:box[2]] = small
                        masks.extend(lifted); scores.extend(probability.tolist()); semantics.extend([semantic]*len(small)); calls += 1
                    dense = np.asarray(masks, bool).reshape(-1, image.height, image.width); path = OUT/'predictions'/key/f'{r["frame_id"]:06d}_{box_id}.npz'; path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, masks_packed=np.packbits(dense, axis=2), scores=np.asarray(scores, np.float32), semantic_ids=np.asarray(semantics, np.int32), box=np.asarray(box))
                    files[str(path)] = sha(path); generated[str(box_id)] = str(path); del state
                    print('CROP', key, r['frame_id'], box_id, len(scores), 'masks', flush=True)
                results[key].append(dict(frame_id=r['frame_id'], predictions=generated, chosen=r['chosen']))
    dump(OUT/'predictions_complete.json', dict(complete=True, files=files, results=results, actual_decodes=calls, logical_decodes=64, actual_encodes=calls//2, wall_seconds=time.monotonic()-start, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), selection_sha256=sha(OUT/'selection.json'), protocol_sha256=sha(OUT/'protocol.json')))


def score():
    from evaluate_kitti360_development import read_static_ply, transfer
    p = verify(); prediction = read(OUT/'predictions_complete.json'); selection = read(OUT/'selection.json')
    assert prediction['selection_sha256'] == sha(OUT/'selection.json')
    for f, h in prediction['files'].items(): assert sha(f) == h
    assert not (OUT/'summary.json').exists(); original = read(FRONT/'masks_complete.json'); owner_protocol = read(OWNERSHIP/'protocol.json'); taxonomy = runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['id2label']; results = {}; gates = {}
    for s in p['scenes']:
        key = s['key']; manifest = read(INPUT/key/'manifest.json'); xyz = np.load(manifest['geometry'])['xyz']; ref = np.load(s['reference']); info = owner_protocol['references'][key]
        assert sha(info['raw']) == info['sha256']; raw = read_static_ply(info['raw']); train = np.asarray([taxonomy[int(v)].trainId for v in raw['semantic']], np.int64)
        nearest = transfer(ref['xyz'], np.arange(1, len(ref['xyz'])+1), xyz, .2); ix = np.maximum(nearest-1, 0); known = (nearest>0)&(train[ix]!=255)&(ref['confidence'][ix]>0); label = train[ix]; weight = ref['confidence'][ix]
        baseline = {sem: set() for sem in SEMANTICS}
        for r in original['scenes'][key]['records']:
            masks = np.load(r['file']); support = np.load(SUPPORT/'supports'/key/'all'/f'{r["frame_id"]:06d}.npz')
            for j, sem in enumerate(masks['semantic_ids']): baseline[int(sem)].update(map(int, support['point_ids'][support['offsets'][j]:support['offsets'][j+1]]))
        selected = {r['frame_id']: r for r in selection['results'][key]}; new = {method: {sem: set() for sem in SEMANTICS} for method in POLICIES}
        for r in prediction['results'][key]:
            request = selected[r['frame_id']]; f = np.load(request['measured']); ids = f['visible_source_ids']; px = f['pixel_xy']
            for method in POLICIES:
                data = np.load(r['predictions'][str(r['chosen'][method])]); masks = np.unpackbits(data['masks_packed'], axis=2)[:, :, :request['width']].astype(bool)
                for sem in SEMANTICS:
                    active = masks[data['semantic_ids']==sem].any(0); new[method][sem].update(set(map(int, ids[active[px[:, 1], px[:, 0]]]))-baseline[sem])
        rows = {}
        for method in POLICIES:
            by_class = {}; correct = 0.; incorrect = 0.; unknown = 0
            for sem, train_id in SEMANTICS.items():
                ids = np.asarray(sorted(new[method][sem]), np.int64); good = known[ids]&(label[ids]==train_id); bad = known[ids]&~good
                pos = float(weight[ids[good]].sum()); neg = float(weight[ids[bad]].sum()); unk = int((~known[ids]).sum())
                by_class[str(sem)] = dict(new_sources=len(ids), correct_mass=pos, incorrect_mass=neg, unknown_sources=unk); correct += pos; incorrect += neg; unknown += unk
                path = OUT/'diagnostic_support'/key/f'{method}_{sem}.npy'; path.parent.mkdir(parents=True, exist_ok=True); np.save(path, ids)
            rows[method] = dict(classes=by_class, correct_mass=correct, incorrect_mass=incorrect, known_precision=correct/(correct+incorrect) if correct+incorrect else 0., unknown_sources=unknown)
        ours = rows['memory']; gates[key] = ours['correct_mass'] >= 100 and ours['known_precision'] >= .5 and all(ours['correct_mass'] > rows[m]['correct_mass'] for m in ['uniform', 'geometry', 'current'])
        results[key] = rows; print('SUPPORT', key, rows, 'GATE', gates[key], flush=True)
    dump(OUT/'summary.json', dict(complete=True, results=results, gate_by_scene=gates, feasibility_gate=all(gates.values()), probe_only=True, not_map_accuracy=True, predictions_sha256=sha(OUT/'predictions_complete.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['freeze', 'select', 'predict', 'score']); globals()[parser.parse_args().action]()
