"""Extract frozen measured-support DINO descriptors on complete public scenes."""
import argparse
import subprocess
import time
from pathlib import Path
import numpy as np
from PIL import Image
from public_dino_protocol import DEST,MODEL,read,verify
from audit_evidence_retention import dump
from omni_semantic_map.flat_benchmark import sha256
from omni_semantic_map.livo_dino_support import pool


def run(keys,gpu):
    spec=read(DEST/'protocol.json');verify(spec)
    for p,h in spec['model_sha256'].items():assert sha256(p)==h
    import torch
    from transformers import Dinov2Model,Dinov2Config
    from safetensors.torch import load_file
    idle=0
    while idle<2:
        free,util=map(int,subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
        idle=idle+1 if free>=7168 and util<10 else 0
        dump(DEST/f'gpu{gpu}_status.json',dict(phase='waiting',free_mib=free,util=util,idle_polls=idle))
        if idle<2:time.sleep(10)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.cuda.set_device(gpu)
    torch.cuda.set_per_process_memory_fraction(4*1024**3/torch.cuda.get_device_properties(gpu).total_memory,gpu)
    device=f'cuda:{gpu}'
    model=Dinov2Model(Dinov2Config.from_pretrained(str(MODEL),local_files_only=True))
    model.load_state_dict(load_file(str(MODEL/'model.safetensors')),strict=True)
    model=model.eval().to(device).requires_grad_(False)
    mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    for key in keys:
        entry=next(s for s in spec['scenes'] if s['key']==key);cache=Path(entry['cache'])
        m=read(cache/'manifest.json');out=DEST/'features'/key;out.mkdir(parents=True,exist_ok=False)
        assert sha256(cache/'manifest.json')==entry['hashes'][str(cache/'manifest.json')]
        for name,h in m['files_sha256'].items():assert sha256(cache/name)==h
        xyz=np.load(cache/'geometry.npz')['xyz'];poses=np.load(cache/'poses.npy')
        cam=m['config']['cam'];width,height=cam['W'],cam['H'];K=np.array([[cam['fx'],0,cam['cx']],[0,cam['fy'],cam['cy']],[0,0,1.]])
        iw=448;ih=int(round((height/width*iw)/14))*14
        image_by_id={r['frame_id']:r for r in entry['images']};records=[];began=time.monotonic()
        for index,rec in enumerate(m['records']):
            rgb=image_by_id[rec['frame_id']];assert sha256(rgb['rgb'])==rgb['sha256']
            image=Image.open(rgb['rgb']).convert('RGB');assert image.size==(width,height)
            tensor=torch.from_numpy(np.array(image.resize((iw,ih),Image.Resampling.BICUBIC))).permute(2,0,1)[None].to(device,torch.float32)/255
            with torch.inference_mode():tokens=model(pixel_values=(tensor-mean)/std).last_hidden_state[0,1:].reshape(ih//14,iw//14,-1).cpu().numpy()
            fpath=cache/'frames'/rec['file'];assert sha256(fpath)==rec['sha256'];f=dict(np.load(fpath))
            T=np.linalg.inv(poses[index]);descriptors=[];features=[];offsets=[0]
            for i in range(rec['masks']):
                ids=f['point_ids'][f['offsets'][i]:f['offsets'][i+1]]
                d,_,t=pool(tokens,xyz[ids],T,K,width,height)
                descriptors.append(d);features.append(t);offsets.append(offsets[-1]+len(t))
            path=out/f'{rec["frame_id"]:06d}.npz'
            np.savez_compressed(path,mean=np.asarray(descriptors,np.float32).reshape(-1,384),
                local_features=np.concatenate(features).astype(np.float16) if features else np.empty((0,384),np.float16),offsets=np.array(offsets,np.int32))
            records.append(dict(frame_id=rec['frame_id'],observation_offset=rec['observation_offset'],masks=rec['masks'],file=path.name,sha256=sha256(path)))
            if index%25==0:
                dump(out/'status.json',dict(complete=False,frames=index+1,total=len(m['records']),gpu=gpu))
                print('DINO',key,index+1,flush=True)
        dump(out/'manifest.json',dict(complete=True,key=key,records=records,frames=len(records),observations=m['observations'],
            elapsed_s=time.monotonic()-began,protocol_sha256=sha256(DEST/'protocol.json'),gpu=gpu,peak_cuda_bytes=torch.cuda.max_memory_allocated(gpu),resize=[iw,ih]))
        dump(out/'status.json',dict(complete=True,frames=len(records),gpu=gpu));print('COMPLETE DINO',key,time.monotonic()-began,flush=True)
    dump(DEST/f'gpu{gpu}_status.json',dict(complete=True,phase='complete',scenes=keys))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scenes',nargs='+',required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args();run(a.scenes,a.gpu)
