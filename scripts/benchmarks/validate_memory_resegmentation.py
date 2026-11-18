"""Actual SAM decoding on validation only, with identical four-point budgets."""
import hashlib
import time
import cv2
import numpy as np
import torch
from memory_reseg_common import OUT,PRIOR,SEEDS,MODES,read,dump,spec,archive,sam_predictor,sha
from omni_semantic_map.memory_resegmentation import CausalTokenMemory,LocalEvidenceReader,prompt_sets

def project(xyz,pose,cam):
    t=np.linalg.inv(pose);x=xyz@t[:3,:3].T+t[:3,3]
    u=np.rint(cam['fx']*x[:,0]/np.maximum(x[:,2],1e-8)+cam['cx']).astype(int)
    v=np.rint(cam['fy']*x[:,1]/np.maximum(x[:,2],1e-8)+cam['cy']).astype(int)
    return np.clip(u,0,cam['W']-1),np.clip(v,0,cam['H']-1)

def local_score(y,pred):
    valid=y>=0;y=y[valid];pred=pred[valid]
    gids,gi=np.unique(np.r_[0,y],return_inverse=True);gi=gi[1:]
    _,pi=np.unique(np.r_[0,pred],return_inverse=True);pi=pi[1:];npred=int(pi.max(initial=0))+1
    table=np.bincount(gi*npred+pi,minlength=len(gids)*npred).reshape(len(gids),npred)
    union=table.sum(1)[:,None]+table.sum(0)[None]-table
    iou=np.divide(table,union,out=np.zeros_like(table,float),where=union>0)
    hit=iou>.5;hit[0]=False;hit[:,0]=False
    tp=int(hit.sum());fp=int((table.sum(0)[1:]>0).sum())-tp;fn=len(gids)-1-tp
    denom=tp+.5*(fp+fn)
    return dict(pq=float(iou[hit].sum()/denom) if denom else 0.,tp=tp,fp=fp,fn=fn,
        gt=len(gids)-1,matched=gids[np.flatnonzero(hit.any(1))].tolist())

def scores_for(model,z,h,valid,event):
    lo,hi=z['offsets'][event['mask']:event['mask']+2]
    if model is None:
        # Simple spatial affinity; all methods have two seeds and four prompts/call.
        return np.stack([-np.linalg.norm(z['xyz'][lo:hi]-z['xyz'][anchor],axis=1) for anchor in [event['a'],event['b']]])
    with torch.inference_mode():
        x=torch.from_numpy(z['features'].astype(np.float32)).cuda()
        encoded=model.encode(x,torch.from_numpy(h).cuda(),torch.from_numpy(valid).cuda())
        out=[]
        for anchor in [event['a'],event['b']]:
            distance=torch.from_numpy(np.linalg.norm(z['xyz'][lo:hi]-z['xyz'][anchor],axis=1)).cuda()
            scores=model.affinity(encoded[anchor].expand(hi-lo,-1),encoded[lo:hi],
                (x[lo:hi]*x[anchor]).sum(-1),distance)
            out.append(scores.cpu().numpy())
        return np.array(out)

def decode_prompts(predictor,requests):
    """Deduplicate identical requests, then batch actual frozen SAM calls."""
    unique={};keys=[]
    for pixels,labels in requests:
        key=hashlib.sha256(pixels.tobytes()+labels.tobytes()).hexdigest()
        unique.setdefault(key,(pixels,labels));keys.append(key)
    results={};items=list(unique.items())
    for first in range(0,len(items),8):
        batch=items[first:first+8]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            masks,quality,_=predictor.predict(point_coords=np.stack([x[1][0] for x in batch]),
                point_labels=np.stack([x[1][1] for x in batch]),multimask_output=False)
        masks=np.asarray(masks).reshape(len(batch),*predictor._orig_hw[0]);quality=np.asarray(quality).reshape(len(batch))
        for (key,_),m,q in zip(batch,masks,quality):results[key]=(m.astype(bool),float(q))
    return [results[k] for k in keys],len(unique)

def validate():
    p=spec();assert read(OUT/'training_complete.json')['complete']
    key=p['splits']['validation'][0];s=next(x for x in p['scenes'] if x['key']==key)
    from pathlib import Path
    cache=Path(s['cache']);m=read(cache/'manifest.json');im=read(OUT/'inputs'/key/'manifest.json')
    geo=archive(cache/'geometry.npz');poses=np.load(cache/'poses.npy');cam=m['config']['cam']
    labels=archive(PRIOR/'supervision/validation'/key/'point_targets.npz')['instance']
    models={'simple':None}
    for mode in MODES:
        for seed in SEEDS:
            for ckpt in sorted((OUT/'fits'/f'{mode}_{seed}').glob('epoch_*.pt')):
                saved=torch.load(ckpt,map_location='cpu',weights_only=False)
                model=LocalEvidenceReader(mode);model.load_state_dict(saved['state']);model.cuda().eval()
                models[f'{mode}_{seed}_{saved["epoch"]}']=model
    metrics={k:dict(pq_sum=0.,fg_observations=0,fp=0,tp=0,fn=0,gained=0,lost=0,mixed_gained=0,mixed_lost=0,changed_points=0) for k in models}
    metrics['original']=dict(metrics['simple'])
    memory=CausalTokenMemory(**p['history']);predictor=sam_predictor();images={x['frame_id']:x['rgb'] for x in s['images']}
    start=time.monotonic();events_out=[];sam_calls=0
    for index,(rec,ir) in enumerate(zip(m['records'],im['records'])):
        z=archive(OUT/'inputs'/key/ir['file']);h,valid,_=memory.query(z['xyz'],rec['frame_id'])
        memory.add(z['xyz'],z['features'].astype(np.float32),rec['frame_id'],np.arange(len(z['features'])))
        raw=archive(cache/'frames'/rec['file']);requests=[];jobs=[];events={e['mask']:e for e in ir['events']}
        for e in ir['events']:
            for name,model in models.items():
                prompts=prompt_sets(e,z['offsets'],z['pixels'],scores_for(model,z,h,valid,e))
                jobs.append((e['mask'],name,len(requests)));requests.extend(prompts)
        decoded=[]
        if requests:
            image=cv2.cvtColor(cv2.imread(images[rec['frame_id']]),cv2.COLOR_BGR2RGB)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):predictor.set_image(image)
            decoded,calls=decode_prompts(predictor,requests);sam_calls+=calls
        perjob={(j,name):offset for j,name,offset in jobs}
        for j,(lo,hi) in enumerate(zip(raw['offsets'][:-1],raw['offsets'][1:])):
            ids=raw['point_ids'][lo:hi];y=labels[ids];u,v=project(geo['xyz'][ids],poses[index],cam)
            original=local_score(y,np.ones(len(y),np.int32));mix=len(np.unique(y[y>0]))>1
            for name in metrics:
                pred=np.ones(len(y),np.int32)
                if name!='original' and j in events:
                    off=perjob[j,name];(a,qa),(b,qb)=decoded[off:off+2]
                    footprint=raw['segmentation'][v,u]==j
                    aa,bb=a[v,u]&footprint,b[v,u]&footprint;pred[:]=3;pred[aa]=1;pred[bb]=2
                    pred[aa&bb]=1 if qa>=qb else 2
                result=original if name=='original' or j not in events else local_score(y,pred)
                mm=metrics[name];mm['pq_sum']+=result['pq'] if result['gt'] else 0.;mm['fg_observations']+=int(result['gt']>0)
                for k in ['tp','fp','fn']:mm[k]+=result[k]
                gained=len(set(result['matched'])-set(original['matched']));lost=len(set(original['matched'])-set(result['matched']))
                mm['gained']+=gained;mm['lost']+=lost
                mm['mixed_gained']+=gained*mix;mm['mixed_lost']+=lost*mix
                # Different partition, not a mere integer-ID permutation.
                mm['changed_points']+=int(len(pred)-max(np.bincount(pred),default=0))
                if j in events and name!='original':events_out.append(dict(frame=rec['frame_id'],mask=j,model=name,**result,mixed=bool(mix),gained=gained,lost=lost))
        if index%10==0:
            dump(OUT/'validation_progress.json',dict(frame=index+1,total=len(im['records']),wall_s=time.monotonic()-start,actual_deduplicated_sam_calls=sam_calls))
            print('VALIDATION',index+1,'/',len(im['records']),'SAM calls',sam_calls,flush=True)
    selected={}
    for mode in MODES:
        for seed in SEEDS:
            names=[k for k in models if k.startswith(f'{mode}_{seed}_')]
            best=max(names,key=lambda k:(metrics[k]['pq_sum']/max(metrics[k]['fg_observations'],1),-metrics[k]['fp'],-int(k.rsplit('_',1)[1])))
            epoch=int(best.rsplit('_',1)[1]);path=OUT/'fits'/f'{mode}_{seed}'/f'epoch_{epoch:02d}.pt'
            selected[f'{mode}_{seed}']=dict(epoch=epoch,checkpoint=str(path),sha256=sha(path),metrics=metrics[best])
    dump(OUT/'validation_events.json',events_out)
    dump(OUT/'validation.json',dict(complete=True,metrics=metrics,selected=selected,wall_s=time.monotonic()-start,
        actual_sam_calls=sam_calls,evaluation_scenes_accessed=False,logical_calls_per_model=sum(len(r['events'])*2 for r in im['records'])))
    print('VALIDATION COMPLETE', {k:v['epoch'] for k,v in selected.items()},flush=True)

if __name__=='__main__':validate()
