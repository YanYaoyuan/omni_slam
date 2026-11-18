"""Nine bounded local-evidence fits; no development inputs or labels opened."""
import argparse
import time
import numpy as np
import torch
from torch.nn import functional as F
from memory_reseg_common import OUT,SEEDS,MODES,spec,dump,sha,ROOT
from omni_semantic_map.memory_resegmentation import LocalEvidenceReader

SAVE_EPOCHS=[0,1,5,10,25,50]

def train():
    p=spec();torch.set_num_threads(2);torch.manual_seed(0)
    assert torch.cuda.is_available();device='cuda'
    folder=OUT/'training_data/train'
    data={k:torch.from_numpy(np.load(folder/(k+'.npy'))).to(device) for k in
          ['features','xyz','memory','pairs','targets','weights','foreground','scenes']}
    scene_ids=[torch.nonzero(data['scenes']==i).flatten() for i in range(len(p['splits']['train']))]
    assert all(len(x)>0 for x in scene_ids)
    sources=[ROOT/'scripts/benchmarks/train_memory_resegmentation.py',ROOT/'omni_semantic_map/omni_semantic_map/memory_resegmentation.py']
    dump(OUT/'training_execution.json',dict(protocol_sha256=sha(OUT/'protocol.json'),
        source_sha256={str(x):sha(x) for x in sources},save_epochs=SAVE_EPOCHS,
        data_sha256={x.name:sha(x) for x in folder.glob('*.npy')},device=torch.cuda.get_device_name(0),
        label_scope='train only; validation performed by separate SAM decoding process',
        loss='Class-balanced fractional same-instance BCE plus .25 foreground BCE; valid-support weighted.'))
    for mode in MODES:
        for seed in SEEDS:
            target=OUT/'fits'/f'{mode}_{seed}';target.mkdir(parents=True,exist_ok=False)
            torch.manual_seed(seed);rng=np.random.default_rng(seed)
            model=LocalEvidenceReader(mode).to(device)
            # Unused parameters are explicitly frozen, not called equal-capacity controls.
            if mode!='attention':
                model.attention.requires_grad_(False)
            optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=p['training']['lr'],weight_decay=p['training']['weight_decay'])
            initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            log=[];steps=0;start=time.monotonic()
            for epoch in range(51):
                if epoch:
                    model.train();losses=[]
                    chosen=torch.cat([ids[torch.tensor(rng.integers(0,len(ids),p['training']['pairs_per_scene_epoch']),device=device)] for ids in scene_ids])
                    chosen=chosen[torch.tensor(rng.permutation(len(chosen)),device=device)]
                    for pick in chosen.split(p['training']['batch']):
                        pair=data['pairs'][pick].long();flat=pair.flatten()
                        mem=data['memory'][flat].long();valid=mem>=0
                        x=data['features'][flat].float();h=data['features'][mem.clamp_min(0)].float()
                        z=model.encode(x,h,valid).reshape(-1,2,64)
                        raw=x.reshape(-1,2,384);cos=(raw[:,0]*raw[:,1]).sum(-1)
                        distance=(data['xyz'][pair[:,0]]-data['xyz'][pair[:,1]]).norm(dim=-1)
                        score=model.affinity(z[:,0],z[:,1],cos,distance)
                        y=data['targets'][pick];weight=data['weights'][pick]
                        pos=y.mean().clamp(.05,.95);balance=y/(2*pos)+(1-y)/(2*(1-pos))
                        loss=(F.binary_cross_entropy_with_logits(score,y,reduction='none')*weight*balance).mean()
                        fg=data['foreground'][flat].float()
                        loss=loss+.25*F.binary_cross_entropy_with_logits(model.foreground(z.reshape(-1,64)).flatten(),fg)
                        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                        optimizer.step();steps+=1;losses.append(float(loss.detach()))
                    log.append(dict(epoch=epoch,loss=float(np.mean(losses))))
                if epoch in SAVE_EPOCHS:
                    torch.save(dict(mode=mode,seed=seed,epoch=epoch,state={k:v.detach().cpu() for k,v in model.state_dict().items()}),target/f'epoch_{epoch:02d}.pt')
                    dump(target/'progress.json',dict(complete=False,mode=mode,seed=seed,epoch=epoch,steps=steps,curve=log))
                    print('FIT',mode,seed,epoch,log[-1] if log else {},flush=True)
            delta={k:float((model.state_dict()[k].detach().cpu()-v).norm()) for k,v in initial.items()}
            dump(target/'fit.json',dict(complete=True,mode=mode,seed=seed,steps=steps,epochs=50,wall_s=time.monotonic()-start,
                curve=log,weight_change_norm=delta,trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
                checkpoints={x.name:sha(x) for x in target.glob('*.pt')}))
    dump(OUT/'training_complete.json',dict(complete=True,fits=9,save_epochs=SAVE_EPOCHS))

if __name__=='__main__':train()
