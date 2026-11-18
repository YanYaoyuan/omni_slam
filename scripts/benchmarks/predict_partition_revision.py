"""Frozen full-map experiment: ordinary/symmetric prompts and shared queries."""
import time
from pathlib import Path
import numpy as np
from PIL import Image
from paper_final_common import ROOT,read,dump,sha

OUT=ROOT/'build/benchmarks/partition_revision_v1'


def main():
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model.data_misc import interpolate
    p=read(OUT/'protocol.json');assert not (OUT/'model_outputs_complete.json').exists()
    for f,h in {**p['source_hashes'],**p['input_hashes']}.items():assert sha(f)==h,f
    assert sha(p['checkpoint'])==p['checkpoint_sha256']
    assert not (OUT/'model_execution.json').exists(),'Inspect previous attempt before resuming'
    torch.set_num_threads(4);torch.manual_seed(0);free,total=torch.cuda.mem_get_info();assert free>=18*1024**3
    torch.cuda.set_per_process_memory_fraction(.5)
    dump(OUT/'model_execution.json',dict(protocol_sha256=sha(OUT/'protocol.json'),started_epoch=time.time(),physical_gpu=1,free_bytes=free))
    loading=[];native_load=torch.nn.Module.load_state_dict
    def check_load(self,*args,**kwargs):
        result=native_load(self,*args,**kwargs)
        if self.__class__.__name__=='Sam3Image':loading.append(dict(missing=list(result.missing_keys),unexpected=list(result.unexpected_keys)))
        return result
    torch.nn.Module.load_state_dict=check_load;started=time.monotonic()
    try:model=build_sam3_image_model(checkpoint_path=p['checkpoint'],load_from_HF=False,device='cuda',compile=False,enable_inst_interactivity=True)
    finally:torch.nn.Module.load_state_dict=native_load
    assert loading==[dict(missing=[],unexpected=[])],loading
    processor=Sam3Processor(model,resolution=1008,confidence_threshold=.5)
    native=model.forward_grounding;capture={}
    def capture_raw(*args,**kwargs):
        result=native(*args,**kwargs);capture['output']=result;return result
    model.forward_grounding=capture_raw
    files={};instances=[];queries=[];encodes=decodes=grounding=0
    def raw_query(state,image,key,frame,semantic,role):
        nonlocal grounding
        processor.reset_all_prompts(state);processor.set_text_prompt(prompt='car' if semantic==26 else 'building',state=state);grounding+=1
        raw=capture['output'];prob=(raw['pred_logits'].sigmoid()*raw['presence_logit_dec'].sigmoid().unsqueeze(1)).squeeze(-1).reshape(-1)
        masks=raw['pred_masks'][0];packed=[]
        for lo in range(0,len(masks),16):
            full=interpolate(masks[lo:lo+16,None],(image.height,image.width),mode='bilinear',align_corners=False).sigmoid()>.5
            packed.append(np.packbits(full[:,0].cpu().numpy(),axis=2))
        path=OUT/'model_outputs'/role/f'{key}_{frame}_{semantic}.npz';path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,masks_packed=np.concatenate(packed),scores=prob.float().cpu().numpy(),original_size=np.array(image.size))
        files[str(path)]=sha(path);queries.append(dict(key=key,frame_id=frame,semantic_id=semantic,role=role,file=str(path)));capture.clear()
    selected={s['key']:read(OUT/s['key']/'selected.json') for s in p['scenes']}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for key,rows in selected.items():
            frames={}
            for row in rows:frames.setdefault(row['policies']['max_visible']['frame_id'],[]).append(row)
            for frame,cases in sorted(frames.items()):
                image=Image.open(cases[0]['policies']['max_visible']['rgb']).convert('RGB');state=processor.set_image(image);encodes+=1
                for row in cases:
                    action=row['policies']['max_visible'];a,b=action['point_a'],action['point_b'];ga,gb=row['groups']
                    policies=dict(ordinary=[([a[0]],[1]),([b[0]],[1]),([a[1]],[1]),([b[1]],[1])],
                        symmetric=[([a[0],b[0]],[1,0]),([b[0],a[0]],[1,0]),([a[0],b[0]],[1,1]),([a[1],b[1]],[1,1])])
                    for policy,prompts in policies.items():
                        paths=[]
                        for j,(points,labels) in enumerate(prompts):
                            masks,scores,_=model.predict_inst(state,point_coords=np.asarray(points,np.float32),point_labels=np.asarray(labels,np.int32),multimask_output=True);decodes+=1
                            pick=int(np.argmax(scores));assert masks.shape[1:]==(image.height,image.width)
                            path=OUT/'model_outputs'/'instances'/f'{key}_{ga}_{gb}_{policy}_{j}.npz';path.parent.mkdir(parents=True,exist_ok=True)
                            np.savez_compressed(path,masks_packed=np.packbits(masks.astype(bool),axis=2),scores=scores,selected=np.array(pick),point_coords=np.asarray(points),point_labels=np.asarray(labels),original_size=np.array(image.size))
                            paths.append(str(path));files[str(path)]=sha(path)
                        instances.append(dict(key=key,groups=[ga,gb],policy=policy,files=paths))
                for semantic in sorted({r['semantic_id'] for r in cases}):raw_query(state,image,key,frame,semantic,'acquisition')
                del state;print('ACQUISITION',key,frame,'decodes',decodes,flush=True)
        validation=read(OUT/'validation_cells_complete.json')
        for key,rows in selected.items():
            requests={}
            for row in rows:
                for frame in row['policies']['max_visible']['verification_frames']:requests.setdefault(frame,set()).add(row['semantic_id'])
            for r in validation['results'][key]['records']:
                image=Image.open(r['rgb']).convert('RGB');state=processor.set_image(image);encodes+=1
                for semantic in sorted(requests[r['frame_id']]):raw_query(state,image,key,r['frame_id'],semantic,'validation')
                del state;print('VALIDATION',key,r['frame_id'],flush=True)
    dump(OUT/'model_outputs_complete.json',dict(complete=True,files=files,instances=instances,queries=queries,
        protocol_sha256=sha(OUT/'protocol.json'),execution_sha256=sha(OUT/'model_execution.json'),
        image_encodes=encodes,instance_decodes=decodes,grounding_decodes=grounding,checkpoint_loading=loading,
        wall_seconds=time.monotonic()-started,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
        reference_labels_read=False,full_maps_scored=False))
    print('MODEL OUTPUTS COMPLETE',encodes,decodes,grounding,flush=True)


if __name__=='__main__':main()
