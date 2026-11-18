"""Frozen DINOv2 features for Flat. No GT inputs, no center crop, no training."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from .flat_benchmark import frame_index, sha256


def normalize(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def pool_tokens(tokens, pixels, points, max_tokens=24):
    """Mask/depth support weighted pooling; patch centres keep local 3D anchors.

    The 640x480 image is resized to 448x336, giving a 32x24 patch grid.
    Tiny masks are retained, although their patches can contain background.
    """
    h, w, _ = tokens.shape
    px = np.minimum(pixels[:, 0] * w // 640, w - 1)
    py = np.minimum(pixels[:, 1] * h // 480, h - 1)
    bins = py * w + px
    unique, counts = np.unique(bins, return_counts=True)
    features = normalize(tokens.reshape(-1, tokens.shape[-1])[unique])
    descriptor = normalize(np.average(features, axis=0, weights=counts))
    selected = np.unique(np.linspace(0, len(unique) - 1, min(max_tokens, len(unique))).astype(int))
    anchors = np.array([np.median(points[bins == unique[i]], axis=0) for i in selected])
    return descriptor, anchors.astype(np.float32), features[selected]


def extract(sequence, source_run, model_path, output, threads=8, device="cpu"):
    import torch
    from transformers import Dinov2Model
    import transformers
    ids, _ = frame_index(sequence)
    source = json.loads((source_run / 'run.json').read_text())
    if not source['complete'] or source['mode'] != 'predicted' or Path(source['sequence']).resolve() != sequence.resolve():
        raise ValueError('requires complete predicted run of the same sequence')
    hashes = json.loads((source_run / 'input_sha256.json').read_text())
    for p, digest in hashes.items():
        if sha256(p) != digest:
            raise ValueError('source input changed: ' + p)
    objects_path = source_run / 'objects.json'
    by_support = {o['support_file']: o for obj in json.loads(objects_path.read_text())['objects'] for o in obj['observations']}
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    model = Dinov2Model.from_pretrained(str(model_path), local_files_only=True).to(device).eval()
    model.requires_grad_(False)
    assert model.config.patch_size == 14
    provenance = dict(complete=False, frames_expected=len(ids), model='facebook/dinov2-small',
        model_revision=json.loads((model_path / 'upstream.json').read_text())['sha'],
        model_sha256={p.name: sha256(p) for p in model_path.iterdir() if p.is_file()},
        preprocessing='RGB, bicubic resize 448x336, no crop, ImageNet mean/std; 24x32 patch tokens',
        pooling='valid depth mask weighted normalized patch mean; <=24 spatially sampled local tokens',
        torch=torch.__version__, transformers=transformers.__version__, device=device, threads=threads,
        device_name=torch.cuda.get_device_name(device) if device.startswith('cuda') else 'CPU',
        source_objects_sha256=sha256(objects_path), source_run=str(source_run.resolve()),
        sequence=str(sequence.resolve()), source_sha256=sha256(Path(__file__)), gt_used=False)
    (output / 'cache.json').write_text(json.dumps(provenance, indent=2))
    all_obs, cache_hashes = [], {}
    started = time.monotonic()
    mean = torch.tensor([.485, .456, .406])[None, :, None, None]
    std = torch.tensor([.229, .224, .225])[None, :, None, None]
    for k, frame in enumerate(ids):
        image_path = sequence / (frame + '_color.png')
        arr = np.asarray(Image.open(image_path).convert('RGB').resize((448, 336), Image.Resampling.BICUBIC)).copy()
        tensor = torch.from_numpy(arr).permute(2, 0, 1)[None].float() / 255
        with torch.inference_mode():
            tokens = model(pixel_values=((tensor - mean) / std).to(device)).last_hidden_state[0, 1:].reshape(24, 32, -1).cpu().numpy()
        obs, descriptors, positions, features, offsets = [], [], [], [], [0]
        for seg in json.loads((sequence / (frame + '_labels.json')).read_text()):
            support = f'supports/{frame}_{int(seg["id"])}.npz'
            if not seg['isthing'] or support not in by_support:
                continue
            o = copy.deepcopy(by_support[support])
            for key in ['object_id', 'numeric_object_id']:
                o.pop(key, None)
            support_path = source_run / support
            hashes[str(support_path.resolve())] = sha256(support_path)
            with np.load(support_path) as data:
                descriptor, anchor, feature = pool_tokens(tokens, data['pixels'], data['local_points'])
            o.update(observation_id=len(all_obs) + len(obs), segment_id=int(seg['id']))
            obs.append(o); descriptors.append(descriptor); positions.append(anchor); features.append(feature); offsets.append(offsets[-1] + len(anchor))
        path = output / (frame + '.npz')
        np.savez_compressed(path, mean=np.asarray(descriptors, np.float32).reshape(-1, model.config.hidden_size),
            local_xyz=np.concatenate(positions) if positions else np.zeros((0, 3), np.float32),
            local_features=np.concatenate(features).astype(np.float16) if features else np.zeros((0, model.config.hidden_size), np.float16),
            offsets=np.asarray(offsets, np.int32))
        cache_hashes[path.name] = sha256(path)
        all_obs.extend(obs)
        if k % 50 == 0:
            print(sequence.name, k, 'observations', len(all_obs), 'elapsed', round(time.monotonic()-started, 1), flush=True)
    (output / 'observations.json').write_text(json.dumps(all_obs))
    cache_hashes['observations.json'] = sha256(output / 'observations.json')
    (output / 'input_sha256.json').write_text(json.dumps(hashes, indent=2))
    provenance.update(complete=True, frames_processed=len(ids), observations=len(all_obs), wall_s=time.monotonic()-started,
                      files_sha256=cache_hashes)
    (output / 'cache.json').write_text(json.dumps(provenance, indent=2))
    return provenance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['sequence', 'source-run', 'model', 'output']:
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    print(json.dumps(extract(a.sequence, a.source_run, a.model, a.output, a.threads, a.device), indent=2))

if __name__ == '__main__':
    main()
