"""Read-only native decoder capture before SAM3's confidence acceptance gate.

All original high-confidence outputs must reproduce the frozen frontend. This
is a development information audit, not a new predictor or threshold search.
"""
import argparse
import time
from pathlib import Path
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from run_discovery_query_benchmark import OUT as FRONT, INPUT, verify as verify_front

OUT = ROOT/'build/benchmarks/prethreshold_candidates_v1'


def freeze():
    assert not OUT.exists(); parent = verify_front(True)
    sources = [Path(__file__)]
    dump(OUT/'protocol.json', dict(created_epoch=time.time(), scenes=[s for s in parent['scenes'] if s['key'] in ['kitti360_0000', 'kitti360_0002']],
        parent_protocol_sha256=sha(FRONT/'protocol.json'), parent_masks_sha256=sha(FRONT/'masks_complete.json'), sources={str(f): sha(f) for f in sources},
        checkpoint=parent['sam3_checkpoint'], checkpoint_sha256=parent['sam3_sha256'], prompts=parent['prompts'], semantic_ids=parent['semantic_ids'],
        capture='Intercept return of the native forward_grounding on this process-local model instance, immediately before the processor confidence gate. Original native processor still runs unchanged. No second decoder call, text/prompt/model/source changes. Internal native suppression is not undone.',
        stored='Every native query: class logit, image-level presence logit, their native product score, normalized box, query index and binary full-resolution mask. Use native bilinear align_corners=False and sigmoid>0.5, in bounded16query chunks. Store packed masks, not decoder latent embeddings.',
        replay='Same RGB1008 native encoder, car/building texts, BF16, seed0, native warmup. Select stored scores>0.5 and require exact mask/score/class agreement with both current native processor and all frozen108-frame outputs. Stop on any disagreement. Original masks/maps remain unchanged.',
        budget='108 image encodes and216 text decodes on old development0000/0002, no training or new maps. No0004/0005 or robot inputs. GPU0,>=18GiB free,50percent allocator cap. No threshold/resolution/prompt search.',
        prospective_analysis='Partition queries into high(score>0.5), deferred(0.1<score<=0.5), and other(score<=0.1), fixed before inference. 0.1 defines one future permissive baseline, not a chosen winning threshold. All queries are stored so discarded information is inspectable; only high/deferred participate in the proposed measured-support audit.',
        gate='First exact native/old replay. Then diagnostic reference labels may test whether deferred proposals contain object-pure measured supports across>=3 distinct frames for targets not already AP50-matched. Such GT-assisted evidence only justifies designing a delayed-admission mechanism; it is neither a method score nor proof of novelty. If absent, do not train or build another candidate.',
        limitation='Threshold-free capture refers only to processor acceptance; native decoder suppression and mask0.5 binarization remain. Exposed development data, not independent evaluation. Low-confidence boxes/masks are predictions, never assumed true.'))
    print('FROZEN', flush=True)


def verify():
    p = read(OUT/'protocol.json'); verify_front(True)
    assert sha(FRONT/'protocol.json') == p['parent_protocol_sha256'] and sha(FRONT/'masks_complete.json') == p['parent_masks_sha256']
    for path, h in p['sources'].items(): assert sha(path) == h, path
    return p


def predict():
    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model.data_misc import interpolate
    p = verify(); assert not (OUT/'predictions_complete.json').exists(); assert sha(p['checkpoint']) == p['checkpoint_sha256']
    torch.set_num_threads(4); free, _ = torch.cuda.mem_get_info(); assert free >= 18*1024**3
    torch.cuda.set_per_process_memory_fraction(.5); torch.manual_seed(0)
    model = build_sam3_image_model(checkpoint_path=p['checkpoint'], load_from_HF=False, device='cuda', compile=False)
    proc = Sam3Processor(model, resolution=1008, confidence_threshold=.5)
    native = model.forward_grounding; captured = {}
    def capture(*args, **kwargs):
        result = native(*args, **kwargs); captured['output'] = result; return result
    model.forward_grounding = capture
    originals = read(FRONT/'masks_complete.json'); files = {}; results = {}; started = time.monotonic()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        warm = proc.set_image(Image.new('RGB', (640, 360))); proc.set_text_prompt(prompt='car', state=warm); del warm; captured.clear()
        for scene in p['scenes']:
            key = scene['key']; manifest = read(INPUT/key/'manifest.json'); old = {r['frame_id']: r for r in originals['scenes'][key]['records']}; records = []
            folder = OUT/'frames'/key; folder.mkdir(parents=True, exist_ok=False)
            for index, r in enumerate(manifest['records']):
                assert sha(r['rgb']) == manifest['input_hashes'][r['rgb']]
                image = Image.open(r['rgb']).convert('RGB'); state = proc.set_image(image); combined = {name: [] for name in ['masks_packed', 'scores', 'class_logits', 'presence_logits', 'boxes', 'query_ids', 'semantic_ids']}; frame_start = time.monotonic()
                for prompt, semantic in zip(p['prompts'], p['semantic_ids']):
                    proc.reset_all_prompts(state); native_result = proc.set_text_prompt(prompt=prompt, state=state); raw = captured['output']
                    prob = (raw['pred_logits'].sigmoid()*raw['presence_logit_dec'].sigmoid().unsqueeze(1)).squeeze(-1).reshape(-1)
                    logits = raw['pred_logits'].reshape(-1); masks = raw['pred_masks'][0]; boxes = raw['pred_boxes'][0]
                    assert len(masks) == len(prob) == len(boxes) and raw['pred_masks'].shape[0] == 1
                    packed = []
                    for lo in range(0, len(masks), 16):
                        full = interpolate(masks[lo:lo+16, None], (image.height, image.width), mode='bilinear', align_corners=False).sigmoid() > .5
                        packed.append(np.packbits(full[:, 0].cpu().numpy(), axis=2))
                    packed = np.concatenate(packed); scores = prob.float().cpu().numpy(); high = scores > .5
                    expected = native_result['masks'].cpu().numpy().reshape(-1, image.height, image.width)
                    np.testing.assert_array_equal(packed[high], np.packbits(expected, axis=2)); np.testing.assert_array_equal(scores[high], native_result['scores'].float().cpu().numpy())
                    values = dict(masks_packed=packed, scores=scores, class_logits=logits.float().cpu().numpy(), presence_logits=np.full(len(scores), float(raw['presence_logit_dec'].float().item()), np.float32), boxes=boxes.float().cpu().numpy(), query_ids=np.arange(len(scores), dtype=np.int32), semantic_ids=np.full(len(scores), semantic, np.int32))
                    for name, value in values.items(): combined[name].append(value)
                    captured.clear()
                arrays = {name: np.concatenate(value) for name, value in combined.items()}; keep = arrays['scores'] > .5; previous = np.load(old[r['frame_id']]['file'])
                for name in ['masks_packed', 'scores', 'semantic_ids']: np.testing.assert_array_equal(arrays[name][keep], previous[name], err_msg=f'{key}/{r["frame_id"]}/{name} failed original replay')
                path = folder/f'{r["frame_id"]:06d}.npz'; np.savez_compressed(path, **arrays, original_size=np.array([image.width, image.height]))
                files[str(path)] = sha(path); record = dict(frame_id=r['frame_id'], file=str(path), queries=len(keep), high=int(keep.sum()), deferred=int(((arrays['scores']>.1)&~keep).sum()), other=int((arrays['scores']<=.1).sum()), old_replay_exact=True, seconds=time.monotonic()-frame_start)
                records.append(record); del state; dump(OUT/'progress.json', dict(scene=key, frames=index+1, total=len(manifest['records']), record=record))
                if (index+1)%10 == 0: print('CAPTURE', key, index+1, len(manifest['records']), 'queries', record['queries'], 'high', record['high'], 'deferred', record['deferred'], flush=True)
            results[key] = records
    dump(OUT/'predictions_complete.json', dict(complete=True, files=files, results=results, old_replay_exact=True, wall_seconds=time.monotonic()-started, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), protocol_sha256=sha(OUT/'protocol.json')))
    print('CAPTURE COMPLETE', {k: dict(frames=len(v), high=sum(r['high'] for r in v), deferred=sum(r['deferred'] for r in v), other=sum(r['other'] for r in v)) for k, v in results.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['freeze', 'predict']); globals()[parser.parse_args().action]()
