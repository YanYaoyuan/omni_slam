"""Bounded symmetric partition prompts and unconditioned unused-view queries."""
import time
from pathlib import Path
import numpy as np
from PIL import Image
from paper_final_common import ROOT, read, dump, sha

OUT=ROOT/'build/benchmarks/partition_query_probe_v1'


def main():
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model.data_misc import interpolate
    p=read(OUT/'protocol.json'); v=read(OUT/'verification_inputs.json')
    assert v['complete'] and v['protocol_sha256']==sha(OUT/'protocol.json')
    assert not (OUT/'predictions_complete.json').exists() and not (OUT/'execution.json').exists()
    for f,h in {**p['source_hashes'],**v['files']}.items(): assert sha(f)==h,f
    assert sha(p['checkpoint'])==p['checkpoint_sha256']
    torch.set_num_threads(4); torch.manual_seed(0)
    free,total=torch.cuda.mem_get_info(); assert free>=18*1024**3
    torch.cuda.set_per_process_memory_fraction(.5)
    dump(OUT/'execution.json',dict(protocol_sha256=sha(OUT/'protocol.json'), inputs_sha256=sha(OUT/'verification_inputs.json'),
        source_sha256=sha(__file__), sam3_sources={str(f):sha(f) for f in [Path('/home/user/vbot/sam3/sam3/model_builder.py'),Path('/home/user/vbot/sam3/sam3/model/sam3_image.py'),Path('/home/user/vbot/sam3/sam3/model/sam3_image_processor.py')]},
        physical_gpu=1, free_bytes=free, total_bytes=total, started_epoch=time.time()))
    loading=[]; native_load=torch.nn.Module.load_state_dict
    def checked_load(self,*args,**kwargs):
        result=native_load(self,*args,**kwargs)
        if self.__class__.__name__=='Sam3Image': loading.append(dict(missing=list(result.missing_keys),unexpected=list(result.unexpected_keys)))
        return result
    torch.nn.Module.load_state_dict=checked_load
    start=time.monotonic()
    try:
        model=build_sam3_image_model(checkpoint_path=p['checkpoint'],load_from_HF=False,device='cuda',compile=False,enable_inst_interactivity=True)
    finally: torch.nn.Module.load_state_dict=native_load
    assert len(loading)==1 and not loading[0]['missing'] and not loading[0]['unexpected'],loading
    assert model.inst_interactive_predictor is not None
    processor=Sam3Processor(model,resolution=1008,confidence_threshold=.5)
    results=[]; files={}; encodes=instance_decodes=grounding_decodes=0
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for row in p['pairs']:
            if row['status']!='ready':continue
            image=Image.open(row['acquisition_rgb']).convert('RGB');state=processor.set_image(image);encodes+=1
            a,b=row['point_a'],row['point_b']
            prompts=[('split_a',[a[0],b[0]],[1,0]),('split_b',[b[0],a[0]],[1,0]),
                     ('joint_1',[a[0],b[0]],[1,1]),('joint_2',[a[1],b[1]],[1,1])]
            prefix=f'{row["key"]}_{row["object_ids"][0]}_{row["object_ids"][1]}'
            record=dict(key=row['key'],object_ids=row['object_ids'],acquisition_frame=row['acquisition_frame'],queries=[])
            for name,points,labels in prompts:
                before=time.monotonic()
                masks,scores,logits=model.predict_inst(state,point_coords=np.array(points,np.float32),point_labels=np.array(labels,np.int32),multimask_output=True)
                instance_decodes+=1;selected=int(np.argmax(scores))
                assert masks.ndim==3 and masks.shape[1:]==(image.height,image.width)
                f=OUT/'predictions'/prefix/(name+'.npz');f.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(f,masks_packed=np.packbits(masks.astype(bool),axis=2),scores=scores,selected=np.array(selected),
                                    point_coords=np.array(points),point_labels=np.array(labels),original_size=np.array(image.size))
                files[str(f)]=sha(f);record['queries'].append(dict(name=name,file=str(f),selected=selected,scores=scores.tolist(),seconds=time.monotonic()-before))
            results.append(record);del state
            print('INSTANCE',prefix,flush=True)
        requests={}
        for row in p['pairs']:
            if row['status']=='ready':
                for frame in row['verification_frames']:requests.setdefault((row['key'],frame),set()).add(row['semantic_id'])
        native=model.forward_grounding;capture={}
        def capture_raw(*args,**kwargs):
            result=native(*args,**kwargs);capture['output']=result;return result
        model.forward_grounding=capture_raw
        checks=[]
        for row in v['records']:
            image=Image.open(row['rgb']).convert('RGB');state=processor.set_image(image);encodes+=1
            for semantic in sorted(requests[(row['key'],row['frame_id'])]):
                processor.reset_all_prompts(state)
                processor.set_text_prompt(prompt='car' if semantic==26 else 'building',state=state);grounding_decodes+=1
                raw=capture['output'];prob=(raw['pred_logits'].sigmoid()*raw['presence_logit_dec'].sigmoid().unsqueeze(1)).squeeze(-1).reshape(-1)
                masks=raw['pred_masks'][0];packed=[]
                for lo in range(0,len(masks),16):
                    binary=interpolate(masks[lo:lo+16,None],(image.height,image.width),mode='bilinear',align_corners=False).sigmoid()>.5
                    packed.append(np.packbits(binary[:,0].cpu().numpy(),axis=2))
                f=OUT/'predictions'/row['key']/f'{row["frame_id"]:06d}_{semantic}.npz';f.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(f,masks_packed=np.concatenate(packed),scores=prob.float().cpu().numpy(),semantic_id=np.array(semantic),original_size=np.array(image.size))
                files[str(f)]=sha(f);checks.append(dict(key=row['key'],frame_id=row['frame_id'],semantic_id=semantic,file=str(f)))
                capture.clear()
            del state
            print('VERIFY',row['key'],row['frame_id'],flush=True)
    dump(OUT/'predictions_complete.json',dict(complete=True,protocol_sha256=sha(OUT/'protocol.json'),execution_sha256=sha(OUT/'execution.json'),
         files=files,acquisition=results,verification=checks,image_encodes=encodes,instance_decodes=instance_decodes,grounding_decodes=grounding_decodes,
         wall_seconds=time.monotonic()-start,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),checkpoint_loading=loading,
         no_maps_changed=True,no_gt_read=True))
    print('PREDICTIONS COMPLETE',encodes,instance_decodes,grounding_decodes,flush=True)


if __name__=='__main__':main()
