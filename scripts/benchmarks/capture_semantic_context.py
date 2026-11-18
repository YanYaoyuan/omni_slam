"""Fixed19-class SAM3 evidence on original measured pixels, target replay checked."""
import argparse,time,runpy
from pathlib import Path
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from omni_semantic_map.semantic_context_evidence import reduce_support

OUT=ROOT/'build/benchmarks/semantic_context_observations_v1'
INPUT=ROOT/'build/benchmarks/discovery_measured_inputs_v1'
OLD=ROOT/'build/benchmarks/prethreshold_candidates_v1'
SAM=Path('/home/user/vbot/sam3')
KEYS=['kitti360_0000','kitti360_0002']


def freeze():
    assert not OUT.exists();old=read(OLD/'protocol.json');seal=read(OLD/'predictions_complete.json')
    taxonomy=ROOT/'build/datasets/kitti360_prepare/labels.py';vocab={}
    for label in runpy.run_path(str(taxonomy))['labels']:
        if 0<=label.trainId<255 and label.trainId not in vocab:
            vocab[label.trainId]=dict(prompt=label.name,semantic_id=label.id,train_id=label.trainId)
    assert sorted(vocab)==list(range(19))
    order=[13,2]+[i for i in range(19) if i not in [13,2]]
    inputs={str(taxonomy):sha(taxonomy),str(OLD/'protocol.json'):sha(OLD/'protocol.json'),str(OLD/'predictions_complete.json'):sha(OLD/'predictions_complete.json')}
    for key in KEYS:
        m=read(INPUT/key/'manifest.json');inputs[str(INPUT/key/'manifest.json')]=sha(INPUT/key/'manifest.json')
        for row in m['records']:
            for f in [row['file'],row['rgb']]:inputs[f]=sha(f)
        for row in seal['results'][key]:assert sha(row['file'])==seal['files'][row['file']];inputs[row['file']]=sha(row['file'])
    sources=[Path(__file__),ROOT/'omni_semantic_map/omni_semantic_map/semantic_context_evidence.py',ROOT/'omni_semantic_map/test/test_semantic_context_evidence.py']+sorted((SAM/'sam3').rglob('*.py'))
    assert sha(old['checkpoint'])==old['checkpoint_sha256']
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),scenes=KEYS,vocabulary=[vocab[i] for i in order],
         input_hashes=inputs,source_hashes={str(f):sha(f) for f in sources},checkpoint=old['checkpoint'],checkpoint_sha256=old['checkpoint_sha256'],
         purpose='New semantic context evidence for object-conditioned measured-support acceptance, not an evaluated new map. All19canonical nonignored train classes from public taxonomy; no class subset/prompt/synonym selected by observed error frequencies.',
         inference='SAM3 original model/checkpoint,1008,BF16,seed0,native car warmup. Encode once per frame; native reset/text decoding in frozen vocabulary order. GPU0>=18GiB free,50percent allocator cap.',
         capture='Native forward_grounding before processor confidence acceptance. Native class sigmoid times image-presence sigmoid; internal suppression unchanged. Original bilinear full-resolution mask interpolation and sigmoid>0.5. Chunk16 queries. Record binary memberships ONLY at exact measured-visible source pixels for all200queries/class. Missing source pixels do not become zeros/negative observations.',
         fields='Source IDs/pixels;19x200 native scores/logits/boxes/presence; packed measured membership; Nx19max native scores inside masks and witness query, -1if no positive support. Max reduction is not a calibrated class probability or semantic label.',
         replay='car/building precede new classes. Require all native scores/class logits/boxes and all full-image binary masks to equal frozen old raw outputs, not just selected detections. Stop on any mismatch; no rerun with changed tolerance.',
         budget='Only old108images,2052text decodes,108image encodes plus one warmup. No GT, training, output maps,0004/0005,robot or independent data. All controls later share this sealed evidence. No further prompt sweep.',
         next='Production mechanism must compare frozen strong baseline, expanded geometry control, simple native-context rejection, category-conditioned learned acceptance and object-conditioned acceptance. Full-map protocol and train/eval separation must be frozen before training. New evidence alone is not innovation or benchmark gain.'))
    print('FROZEN SEMANTIC CONTEXT',flush=True)


def verify():
    p=read(OUT/'protocol.json')
    for f,h in {**p['input_hashes'],**p['source_hashes']}.items():assert sha(f)==h,f
    assert sha(p['checkpoint'])==p['checkpoint_sha256']
    return p


def predict():
    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model.data_misc import interpolate
    p=verify();assert not (OUT/'predictions_complete.json').exists()
    torch.set_num_threads(4);free,_=torch.cuda.mem_get_info();assert free>=18*1024**3
    torch.cuda.set_per_process_memory_fraction(.5);torch.manual_seed(0)
    model=build_sam3_image_model(checkpoint_path=p['checkpoint'],load_from_HF=False,device='cuda',compile=False)
    proc=Sam3Processor(model,resolution=1008,confidence_threshold=.5);native=model.forward_grounding;captured={}
    def hook(*args,**kwargs):
        result=native(*args,**kwargs);captured['raw']=result;return result
    model.forward_grounding=hook;old=read(OLD/'predictions_complete.json');files={};results={};start=time.monotonic()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        warm=proc.set_image(Image.new('RGB',(640,360)));proc.set_text_prompt(prompt='car',state=warm);del warm;captured.clear()
        for key in KEYS:
            manifest=read(INPUT/key/'manifest.json');previous={r['frame_id']:r for r in old['results'][key]};rows=[]
            folder=OUT/'frames'/key;folder.mkdir(parents=True,exist_ok=False)
            for rank,rec in enumerate(manifest['records']):
                tick=time.monotonic();v=dict(np.load(rec['file']));xy=v['pixel_xy'];ids=v['visible_source_ids']
                before=dict(np.load(previous[rec['frame_id']]['file']));image=Image.open(rec['rgb']).convert('RGB');state=proc.set_image(image)
                gx=torch.as_tensor(xy[:,0],device='cuda',dtype=torch.long);gy=torch.as_tensor(xy[:,1],device='cuda',dtype=torch.long)
                values={k:[] for k in ['membership_packed','query_scores','class_logits','presence_logits','boxes','point_scores','point_queries']}
                for item in p['vocabulary']:
                    proc.reset_all_prompts(state);proc.set_text_prompt(prompt=item['prompt'],state=state);raw=captured['raw']
                    masks=raw['pred_masks'][0];scores=(raw['pred_logits'].sigmoid()*raw['presence_logit_dec'].sigmoid().unsqueeze(1)).squeeze(-1).reshape(-1).float().cpu().numpy()
                    logits=raw['pred_logits'].reshape(-1).float().cpu().numpy();boxes=raw['pred_boxes'][0].float().cpu().numpy();parts=[]
                    replay=item['semantic_id'] in [26,11]
                    if replay:
                        old_ids=np.flatnonzero(before['semantic_ids']==item['semantic_id'])
                        for actual,name in [(scores,'scores'),(logits,'class_logits'),(boxes,'boxes')]:np.testing.assert_array_equal(actual,before[name][old_ids],err_msg=f'{key}/{rec["frame_id"]}/{item["prompt"]}/{name}')
                    for lo in range(0,len(masks),16):
                        full=interpolate(masks[lo:lo+16,None],(image.height,image.width),mode='bilinear',align_corners=False).sigmoid()>.5
                        parts.append(full[:,0,gy,gx].cpu().numpy())
                        if replay:np.testing.assert_array_equal(np.packbits(full[:,0].cpu().numpy(),axis=2),before['masks_packed'][old_ids[lo:lo+16]])
                    membership=np.concatenate(parts);point_scores,point_queries=reduce_support(membership,scores)
                    new=dict(membership_packed=np.packbits(membership,axis=1),query_scores=scores,class_logits=logits,
                             presence_logits=np.full(len(scores),float(raw['presence_logit_dec'].float().item()),np.float32),boxes=boxes,point_scores=point_scores,point_queries=point_queries)
                    for name,value in new.items():values[name].append(value)
                    captured.clear()
                arrays={name:np.stack(value,axis=1 if name.startswith('point_') else 0) for name,value in values.items()}
                dest=folder/f'{rec["frame_id"]:06d}.npz'
                np.savez_compressed(dest,source_ids=ids,pixel_xy=xy,semantic_ids=np.array([r['semantic_id'] for r in p['vocabulary']]),**arrays)
                files[str(dest)]=sha(dest);row=dict(frame_id=rec['frame_id'],file=str(dest),sources=len(ids),classes=len(p['vocabulary']),queries=int(arrays['query_scores'].size),target_full_replay_exact=True,seconds=time.monotonic()-tick)
                rows.append(row);del state
                dump(OUT/'progress.json',dict(scene=key,frame=rank+1,total=len(manifest['records']),last=row))
                if (rank+1)%10==0:print('CONTEXT',key,rank+1,len(manifest['records']),flush=True)
            results[key]=rows
    dump(OUT/'predictions_complete.json',dict(complete=True,files=files,results=results,protocol_sha256=sha(OUT/'protocol.json'),
         all_target_raw_outputs_exact=True,wall_seconds=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),gt_read=False,new_maps=False))
    print('CONTEXT COMPLETE',time.monotonic()-start,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','predict']);globals()[parser.parse_args().action]()
