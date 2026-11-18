"""Frozen DINOv2-small on the same full-recording LIVO observations, no training."""
import json
from pathlib import Path
import subprocess
import time
import resource
import numpy as np
from PIL import Image
from audit_evidence_retention import ROOT,OUT,dump
from omni_semantic_map.flat_benchmark import sha256
from omni_semantic_map.livo_dino_support import pool


def main():
    import torch
    from transformers import Dinov2Model, Dinov2Config
    from safetensors.torch import load_file
    output=OUT/'dino_livo_tokens_v2';output.mkdir(exist_ok=False)
    job_path=OUT/'jobs/livo_full_v2.json';job=json.loads(job_path.read_text())
    cache=OUT/'livo_full_strong_v2/cache_supportcc';manifest=json.loads((cache/'manifest.json').read_text())
    model_path=ROOT/'build/benchmarks/flat_dino_20261109/model'
    protocol=dict(model='frozen facebook/dinov2-small',model_sha256={p.name:sha256(p) for p in model_path.iterdir() if p.is_file()},
        job_sha256=sha256(job_path),cache_manifest_sha256=sha256(cache/'manifest.json'),
        source_sha256={str(p):sha256(p) for p in [Path(__file__),ROOT/'omni_semantic_map/omni_semantic_map/livo_dino_support.py']},
        preprocessing='640x360 RGB -> bicubic 448x252, no crop, ImageNet normalization; 18x32 patch tokens, float32 inference, float16 local feature storage.',
        pooling='Measured connected supports projected at image epoch; patch-count weighted normalized descriptor and <=24 spatially sampled local world anchors.',
        scope='Feature cache only; original SAM2 masks, sparse supports and SigLIP semantics unchanged. No new inference predictions or GT access.',
        loading='Instantiate saved Dinov2Config and strict-load original safetensors; avoids installed Transformers optional accelerate initialization failure. No weights changed.',
        gpu_policy='GPU0 below 10% utilization and >=7GiB free twice 10 seconds apart; allocator cap 4GiB.')
    dump(output/'protocol.json',protocol)
    idle=0
    while idle<2:
        free,util=map(int,subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
        idle=idle+1 if free>=7168 and util<10 else 0
        dump(output/'status.json',dict(complete=False,phase='waiting_gpu',free_mib=free,utilization=util,idle_polls=idle))
        if idle<2:time.sleep(10)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    torch.cuda.set_per_process_memory_fraction(4*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
    model=Dinov2Model(Dinov2Config.from_pretrained(str(model_path),local_files_only=True))
    model.load_state_dict(load_file(str(model_path/'model.safetensors')),strict=True)
    model=model.eval().to('cuda:0');model.requires_grad_(False)
    assert model.config.patch_size==14
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    geometry=np.load(cache/'geometry.npz')['xyz'];byframe={f['frame_id']:f for f in job['frames']};records=[];started=time.monotonic()
    for k,r in enumerate(manifest['records']):
        frame=byframe[r['frame_id']];assert frame['width']==640 and frame['height']==360
        assert sha256(Path(frame['rgb']))==job['input_sha256'][frame['rgb']]
        image=np.array(Image.open(frame['rgb']).convert('RGB').resize((448,252),Image.Resampling.BICUBIC))
        tensor=torch.from_numpy(image.copy()).permute(2,0,1)[None].to('cuda',torch.float32)/255
        start=time.monotonic()
        with torch.inference_mode():tokens=model(pixel_values=(tensor-mean)/std).last_hidden_state[0,1:].reshape(18,32,-1).cpu().numpy()
        path=cache/'frames'/r['file'];assert sha256(path)==r['sha256'];f=dict(np.load(path))
        T=np.array(frame['camera_from_tracking'])@np.linalg.inv(np.array(frame['pose']))
        descriptors=[];positions=[];features=[];offsets=[0]
        for i in range(r['masks']):
            ids=f['point_ids'][f['offsets'][i]:f['offsets'][i+1]]
            d,a,t=pool(tokens,geometry[ids],T,frame['intrinsic'],640,360)
            descriptors.append(d);positions.append(a);features.append(t);offsets.append(offsets[-1]+len(a))
        save=output/f'{r["frame_id"]:06d}.npz'
        np.savez_compressed(save,mean=np.asarray(descriptors,np.float32).reshape(-1,model.config.hidden_size),
            local_xyz=np.concatenate(positions) if positions else np.empty((0,3),np.float32),
            local_features=np.concatenate(features).astype(np.float16) if features else np.empty((0,model.config.hidden_size),np.float16),
            offsets=np.array(offsets,np.int32))
        records.append(dict(frame_id=r['frame_id'],file=save.name,sha256=sha256(save),observation_offset=r['observation_offset'],masks=r['masks'],wall_s=time.monotonic()-start))
        dump(output/'status.json',dict(complete=False,phase='extracting',frames=len(records),total_frames=len(manifest['records'])))
        if (k+1)%25==0:print('DINO',k+1,len(manifest['records']),flush=True)
    result=dict(complete=True,protocol_sha256=sha256(output/'protocol.json'),records=records,frames=len(records),observations=manifest['observations'],
        wall_s=time.monotonic()-started,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    dump(output/'manifest.json',result);dump(output/'status.json',dict(complete=True,phase='complete',frames=len(records)));print('COMPLETE DINO',len(records),flush=True)


if __name__=='__main__':main()
