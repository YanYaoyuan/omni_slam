"""Actual new masks -> measured-support partitions -> independent native CG maps."""
import argparse
from pathlib import Path
import resource
import time
import sys
import numpy as np
from memory_reseg_common import OUT,ROOT,HISTORY,read,dump,spec,archive,sha

KEYS=['replica_office_0','replica_office_1','scenenn_014','scenenn_015']

def regenerate(key,variant):
    import cv2
    import torch
    from memory_reseg_common import sam_predictor
    from omni_semantic_map.memory_resegmentation import CausalTokenMemory,LocalEvidenceReader,prompt_sets,partition_candidates
    from validate_memory_resegmentation import scores_for,decode_prompts,project
    p=spec();s=next(x for x in p['scenes'] if x['key']==key)
    c=Path(s['cache']);cm=read(c/'manifest.json');clean=OUT/'inputs'/key;im=read(clean/'manifest.json')
    geo=archive(c/'geometry.npz');poses=np.load(c/'poses.npy');cam=cm['config']['cam']
    images={x['frame_id']:x['rgb'] for x in s['images']}
    output=OUT/'regenerated'/key/variant;output.mkdir(parents=True,exist_ok=False)
    model=None
    if variant=='B2':
        selected=read(OUT/'selection.json')['selected'];path=Path(selected['checkpoint']);assert sha(path)==selected['sha256']
        fit=torch.load(path,map_location='cpu',weights_only=False)
        model=LocalEvidenceReader(fit['mode']);model.load_state_dict(fit['state']);model.cuda().eval()
    predictor=sam_predictor();memory=CausalTokenMemory(**p['history']);rows=[];events=[];offset=0;start=time.monotonic()
    for i,(rec,ir) in enumerate(zip(cm['records'],im['records'])):
        began=time.monotonic();z=archive(clean/ir['file']);raw=archive(c/'frames'/rec['file'])
        h,valid,_=memory.query(z['xyz'],rec['frame_id'])
        memory.add(z['xyz'],z['features'].astype(np.float32),rec['frame_id'],np.arange(len(z['features'])))
        requests=[];event_index={}
        for e in ir['events']:
            event_index[e['mask']]=len(requests)
            requests.extend(prompt_sets(e,z['offsets'],z['pixels'],scores_for(model,z,h,valid,e)))
        decoded=[];encoding=0.;decoding=0.;calls=0
        if requests:
            image=cv2.cvtColor(cv2.imread(images[rec['frame_id']]),cv2.COLOR_BGR2RGB)
            tic=time.monotonic()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):predictor.set_image(image)
            torch.cuda.synchronize();encoding=time.monotonic()-tic;tic=time.monotonic()
            decoded,calls=decode_prompts(predictor,requests);decoding=time.monotonic()-tic
        supports=[];features=[];areas=[];parents=[];newmasks=[];changed_supports=0
        for j,(lo,hi) in enumerate(zip(raw['offsets'][:-1],raw['offsets'][1:])):
            ids=raw['point_ids'][lo:hi]
            if j not in event_index:
                parts=[(ids,int(raw['areas'][j]))]
            else:
                off=event_index[j];(a,qa),(b,qb)=decoded[off:off+2]
                orig=raw['segmentation']==j
                parts_dense=partition_candidates(orig,[a,b],[qa,qb])
                u,v=project(geo['xyz'][ids],poses[i],cam)
                aa=a&orig;bb=b&orig;overlap=aa&bb
                if qa>=qb:bb[overlap]=False
                else:aa[overlap]=False
                remainder=orig&~(aa|bb)
                owner=np.full(len(ids),2,int);owner[aa[v,u]]=0;owner[bb[v,u]]=1
                # Rounded projection outside cached footprint joins the SAME remainder.
                # This exactly matches validation; no measured support is dropped.
                parts=[(ids[owner==k],int(mask.sum())) for k,mask in enumerate([aa,bb,remainder]) if (owner==k).any()]
                changed_supports+=len(ids)-max((len(x[0]) for x in parts),default=0)
                events.append(dict(frame=rec['frame_id'],mask=j,sam_quality=[qa,qb],input_points=len(ids),
                    output_points=[len(x[0]) for x in parts],logical_calls=2,prompt_points_per_call=4))
                newmasks.extend(parts_dense)
                if i in [10,50,100] and j==ir['events'][0]['mask']:
                    canvas=image.copy();canvas[orig]=(canvas[orig]*.4+np.array([255,100,60])*.6).astype(np.uint8)
                    after=image.copy();palette=np.array([[30,200,255],[245,80,180],[220,200,40]])
                    for k,mask in enumerate(parts_dense):after[mask]=(after[mask]*.4+palette[k%3]*.6).astype(np.uint8)
                    cv2.imwrite(str(output/f'example_{i:03d}.jpg'),cv2.cvtColor(np.concatenate([image,canvas,after],axis=1),cv2.COLOR_RGB2BGR))
            for points,area in parts:
                supports.append(points);features.append(raw['features'][j]);areas.append(area);parents.append(rec['observation_offset']+j)
        joined=np.concatenate(supports) if supports else np.empty(0,np.int32)
        np.testing.assert_array_equal(np.sort(joined),np.sort(raw['point_ids']))
        file=f'{rec["frame_id"]:06d}.npz'
        np.savez_compressed(output/file,point_ids=joined,offsets=np.r_[0,np.cumsum([len(x) for x in supports])],
            features=np.array(features,np.float32).reshape(-1,1152),areas=np.array(areas),parents=np.array(parents))
        # Archive actual dense predicted masks, separate from geometry-only mapping input.
        if newmasks:np.savez_compressed(output/f'masks_{rec["frame_id"]:06d}.npz',masks_packed=np.packbits(np.array(newmasks),axis=2),width=cam['W'])
        rows.append(dict(frame_id=rec['frame_id'],file=file,masks=len(features),observation_offset=offset,
            sha256=sha(output/file),wall_s=time.monotonic()-began,encoding_s=encoding,decoding_s=decoding,
            logical_sam_calls=len(requests),actual_sam_calls=calls,changed_supports=changed_supports))
        offset+=len(features)
        if i%20==0:
            dump(output/'progress.json',dict(frame=i+1,total=len(cm['records']),wall_s=time.monotonic()-start))
            print('NEW MASKS',key,variant,i+1,flush=True)
    dump(output/'events.json',events)
    dump(output/'manifest.json',dict(complete=True,key=key,variant=variant,records=rows,observations=offset,
        wall_s=time.monotonic()-start,all_measured_supports_conserved=True,gt_accessed=False,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),source_sha256=sha(Path(__file__)),
        descriptor_policy='Original frozen SigLIP feature inherited by each partition; no semantic re-encoding.'))

def map_scene(key,variant,limit=None):
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0,str(ROOT/'build/benchmarks/expanded_semantic_20261109/concept-graphs'))
    from omegaconf import OmegaConf
    from omni_semantic_map.replay_fusion import ConceptGraphsBackend,FusionObservation
    from run_livo_history_graph import finalize
    p=spec();s=next(x for x in p['scenes'] if x['key']==key);cache=Path(s['cache'])
    original=read(cache/'manifest.json');source=cache/'frames' if variant=='B0' else OUT/'regenerated'/key/variant
    m=original if variant=='B0' else read(source/'manifest.json');assert m['complete']
    output=OUT/'maps'/key/variant;output.mkdir(parents=True,exist_ok=False)
    geo=archive(cache/'geometry.npz');poses=np.load(cache/'poses.npy');rawtext=np.load(cache/'text_features.npy')
    text=rawtext/np.linalg.norm(rawtext,axis=1,keepdims=True);vocab=read(cache/'vocabulary.json')
    cfg=OmegaConf.create(read(s['config']));engine=ConceptGraphsBackend(cfg,seed=0)
    feature_rows=[];frame_rows=[];area_rows=[];times=[];seen=set();start=time.monotonic();prefix=[]
    for index,rec in enumerate(m['records']):
        if limit is not None and index>=limit:break
        raw=archive(source/rec['file']);tic=time.monotonic();obs=[]
        for j,(lo,hi) in enumerate(zip(raw['offsets'][:-1],raw['offsets'][1:])):
            oid=rec['observation_offset']+j;ids=raw['point_ids'][lo:hi];feature=raw['features'][j]
            obs.append(FusionObservation(oid,rec['frame_id'],0,geo['xyz'][ids],geo['colors'][ids].astype(float)/255,
                feature,rawtext[int((rawtext@feature).argmax())]))
        engine.integrate(obs,np.eye(4));times.append(time.monotonic()-tic)
        feature_rows.extend(raw['features']);frame_rows.extend([rec['frame_id']]*len(raw['features']));area_rows.extend(raw['areas'])
        seen.update(map(int,raw['point_ids']))
        if index+1 in sorted(set([len(m['records'])*n//4 for n in [1,2,3,4]])):
            table=(np.array(feature_rows),np.array(frame_rows),np.array(area_rows),text,vocab,
                {r['frame_id']:poses[i] for i,r in enumerate(original['records'][:index+1])})
            groups=[[min(o['observation_ids'])+1] for o in engine._objects]
            path=output/('final' if index+1==len(m['records']) else f'prefix_{index+1:03d}')
            finalize(engine,groups,path,table)
            np.save(path/'seen_point_ids.npy',np.array(sorted(seen),np.int32));prefix.append(str(path))
        if index%20==0:
            dump(output/'progress.json',dict(frame=index+1,total=len(m['records']),wall_s=time.monotonic()-start))
            print('MAP',key,variant,index+1,'objects',len(engine._objects),flush=True)
    exact=None
    if variant=='B0' and limit is None:
        a=archive(output/'final/geometry.npz');b=archive(HISTORY/'runs'/key/'baseline/geometry.npz')
        exact=all(np.array_equal(a[k],b[k]) for k in a)
        assert exact,'No-op native baseline differs from archived geometry'
        a=read(output/'final/classification.json')['objects'];b=read(HISTORY/'runs'/key/'baseline/classification.json')['objects']
        assert [(x['observation_ids'],x['predictions']['mean']['replica_class_id']) for x in a]==[(x['observation_ids'],x['predictions']['mean']['replica_class_id']) for x in b]
    dump(output/'run.json',dict(complete=limit is None,key=key,variant=variant,gt_accessed=False,frames=len(times),
        wall_s=time.monotonic()-start,frame_times_s=times,frame_p95_s=float(np.quantile(times,.95)),
        frame_p50_s=float(np.median(times)),native_baseline_exact=exact,prefixes=prefix,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        artifacts={str(x.relative_to(output)):sha(x) for x in output.rglob('*') if x.is_file() and x.name not in ['run.json','progress.json']}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['regenerate','map']);ap.add_argument('--scene',required=True)
    ap.add_argument('--variant',required=True,choices=['B0','B1','B2']);ap.add_argument('--limit',type=int);a=ap.parse_args()
    regenerate(a.scene,a.variant) if a.action=='regenerate' else map_scene(a.scene,a.variant,a.limit)
