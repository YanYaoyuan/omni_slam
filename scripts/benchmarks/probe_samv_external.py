"""Fixed external SAM-V evidence comparison on the prior eight object clips.

No new Ours claim, no model fitting, no map output or independent benchmark.
Freeze -> capture all predictions -> evaluate known development observations.
"""
import argparse
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
from paper_final_common import ROOT,read,dump,sha

B=ROOT/'build/benchmarks'
ASSET=B/'samv_external_v1'
OUT=B/'samv_evidence_comparison_v1'
OLD=B/'world_anchor_tracking_probe_v1'
METHODS=['samv_joint_seed','samv_joint_current','samv_single_current']
CONTROL=['image_reprompt','video_seed','video_world_feedback']


def freeze():
    assert not OUT.exists();assets=read(ASSET/'assets.json');assert assets['complete']
    prepared=read(OLD/'prepared.json');assert prepared['complete'] and len(prepared['cases'])==8
    previous=read(OLD/'predictions_complete.json');assert previous['complete']
    eligible=sum(q['eligible'] for c in prepared['cases'] for q in c['prompts'])
    inputs={str(ASSET/'assets.json'):sha(ASSET/'assets.json')}
    for k,r in assets['models'].items():assert sha(r['path'])==r['sha256'];inputs[r['path']]=r['sha256']
    for f,h in {**prepared['files'],**previous['files']}.items():assert sha(f)==h;inputs[f]=h
    for f in [OLD/'protocol.json',OLD/'prepared.json',OLD/'predictions_complete.json',OLD/'summary.json']:
        inputs[str(f)]=sha(f)
    old_protocol=read(OLD/'protocol.json')
    for scene in old_protocol['scenes']:
        for f in [scene['reference'],old_protocol['references'][scene['key']]['raw'],
                  B/'world_anchor_tracking_probe_v1'/(scene['key']+'_anchors.npz'),
                  B/'discovery_measured_inputs_v1'/scene['key']/'manifest.json',
                  read(B/'discovery_measured_inputs_v1'/scene['key']/'manifest.json')['geometry']]:
            inputs[str(f)]=sha(f)
    inputs[str(ROOT/'build/datasets/kitti360_prepare/labels.py')]=sha(ROOT/'build/datasets/kitti360_prepare/labels.py')
    for c in prepared['cases']:
        for r in c['records']:
            for name in ['rgb','file']:inputs[r[name]]=sha(r[name])
    sources=[Path(__file__),ROOT/'omni_semantic_map/omni_semantic_map/samv_adapter.py',ROOT/'omni_semantic_map/test/test_samv_adapter.py']
    sources+=list((ASSET/'upstream').rglob('*.py'))
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),inputs=inputs,sources={str(f):sha(f) for f in sources},
        cases=prepared['cases'],models=assets['models'],methods=METHODS,existing_controls=CONTROL,
        code_commit=assets['code_commit'],submodules=dict(vggt='44b3afbd1869d8bde4894dd8ea1e293112dd5eba',sam_hq='8c0ef8de98c5b84f3c050339d91e953c7249f10e'),
        expected_calls=dict(sam_frames=88,vggt_calls=9+eligible,vggt_frames=99+eligible,decoder_calls=9+2*eligible),
        selection='Reuse exactly the eight prior fixed object clips,11frames each,all objects retained including missing-GT case. Same measured positive/negative calibration prompts. No case replacement or new GT prompt. Already exposed development data; not independent.',
        prompting_scope='Inherited prompts derive from final-window predicted map anchors,so no arm establishes end-to-end online causality. Retain3positive+up to3negative prompts exactly,including6-point prompts; do not adapt them to improve external-model scores. Releasedstage2wasfine-tunedonScanNet++;thisisconditionaloutdoortransfer,not its published evaluation protocol.',
        inference='Released stage2 weights, complete strict base SAM/VGGT loads, verified142trained tensors, eval,BF16,SAMchunk1,1024square RGB resize as author path,VGGT896square. Native logit>=0 threshold; split panoramic logits per frame, resize256to1024 then tooriginal before threshold.',
        conditions='joint_seed:all11frames encoded jointly,onlyoriginalseed prompts,outputs allframes. joint_current:samejointfeatures,each eligible currentframe prompts in separate decoder call,retain only that current output. single_current:sameSAMimagefeature,VGGTonlycurrentimage,samecurrentprompts,retain currentoutput. No across-object mutual exclusion or map updates.',
        interpretation='joint_current vs single_current isolates multiview feature context for the same prompted frame with shared trained model,not equal compute. joint_seed has no later prompts but may use future images:offline,not causal tracking. ExistingSAM3 controls use the same11images with different architecture/preprocessing and prompt schedules; contextual comparison only.',
        parity='First case joint seed cached decoder logits/IoU must exactly equal pinned native model.forward with the same precomputed SAM features. Two unit tests cover per-axis prompt scaling and frame-separated interpolation. Basecheckpoint loads strict; trainedprefix state verified exactly.',
        budget='8jointVGGTencodes +1additionalnativeparityencode;88SAMframes;singleVGGTonlyeligibleframes;8jointseeddecodes +2pereligibleframe +1paritydecode.0fits,0maps. No prompt,threshold,seed,resolution orcheckpointsearch.',
        evaluation='Seal all masks before loading reference labels. Exactly old common eligible nonseedframes and baseline fixedGT targets; keep missingGT asN/A. Per-case and macro/pooled confidence-weighted measured-observationF1,precision,recall,unique correctly selected unowned sources and heldout-anchor support/intrusion. Not complete-mapAP.',
        decision='Only justify considering a separately registered complete-map candidate if joint_current macroF1 is higher than single_current and no lower than existing image_reprompt/video_seed in bothwindows,with>=1newcorrectunownedsource each. This information gate is not mechanism validation or a newOurs contribution; missingGTlimitsremain. Fail stops this use of the released model without tuning.'))
    print('SAMV EXTERNAL COMPARISON FROZEN',flush=True)


def verify():
    p=read(OUT/'protocol.json')
    for f,h in {**p['inputs'],**p['sources']}.items():assert sha(f)==h,f
    return p


def capture():
    import torch
    import torch.nn.functional as F
    from omni_semantic_map.samv_adapter import encode_context,decode_context,scale_prompts,logits_to_original
    p=verify();assert not (OUT/'capture_execution.json').exists()
    for path in [ASSET/'upstream/submodules/vggt',ASSET/'upstream/submodules/sam-hq',ASSET/'upstream']:sys.path.insert(0,str(path))
    from model.sam_vggt_model import SamVGGT
    from utils.checkpoint import load_partial_checkpoint,is_partial_checkpoint
    torch.set_num_threads(2);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    free,total=torch.cuda.mem_get_info();assert free>=27*1024**3
    torch.cuda.set_per_process_memory_fraction(.9)
    torch.backends.cuda.matmul.allow_tf32=False
    model=SamVGGT(sam_checkpoint=p['models']['sam_vit_h']['path'],vggt_checkpoint=p['models']['vggt']['path'],sam_encode_chunk=1,device='cuda')
    # Refuse silently ignored/missing frozen weights from upstream nonstrict loads.
    for module,key in [(model.sam,'sam_vit_h'),(model.vggt,'vggt')]:
        state=torch.load(p['models'][key]['path'],map_location='cpu',weights_only=True,mmap=True)
        module.load_state_dict(state,strict=True);del state
    checkpoint=torch.load(p['models']['samv_stage2']['path'],map_location='cpu',weights_only=True)
    assert is_partial_checkpoint(checkpoint) and len(checkpoint['model_state_dict'])==142
    load_partial_checkpoint(model,checkpoint)
    for k,v in checkpoint['model_state_dict'].items():assert torch.equal(model.state_dict()[k].detach().cpu(),v),k
    del checkpoint;model.eval().requires_grad_(False)
    counts=dict(sam_frames=0,vggt_calls=0,vggt_frames=0,decoder_calls=0)
    def sam_hook(module,args,output):counts['sam_frames']+=int(args[0].shape[0])
    def vggt_hook(module,args,output):counts['vggt_calls']+=1;counts['vggt_frames']+=int(args[0].shape[1])
    def decoder_hook(module,args,output):counts['decoder_calls']+=1
    hooks=[model.sam.image_encoder.register_forward_hook(sam_hook),model.vggt.aggregator.register_forward_hook(vggt_hook),model.sam.mask_decoder.register_forward_hook(decoder_hook)]
    dump(OUT/'capture_execution.json',dict(started_epoch=time.time(),torch=torch.__version__,gpu=torch.cuda.get_device_name(0),strict_base_weights=True,trained_tensors_exact=142,protocol_sha256=sha(OUT/'protocol.json')))
    files={};stats={};start=time.monotonic()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for ci,case in enumerate(p['cases']):
            tick=time.monotonic();before=counts.copy();images=[];sizes=[]
            for r in case['records']:
                array=np.array(Image.open(r['rgb']).convert('RGB'));sizes.append(array.shape[:2])
                image=torch.from_numpy(array).permute(2,0,1)[None].float()
                images.append(F.interpolate(image,size=(1024,1024),mode='bilinear',align_corners=False)[0])
            assert len(set(sizes))==1
            h,w=sizes[0];images=torch.stack(images)[None].cuda();joint=encode_context(model,images)
            packed={name:np.zeros((11,h,(w+7)//8),np.uint8) for name in METHODS}
            lowres={name:np.full((11,256,256),np.nan,np.float16) for name in METHODS}
            valid={name:np.zeros(11,bool) for name in METHODS}
            def point(j,frame_index):
                xy,lab=scale_prompts(case['prompts'][j]['pixels'],case['prompts'][j]['labels'],(h,w))
                return torch.tensor(xy,device='cuda'),torch.tensor(lab,device='cuda'),torch.full((len(lab),),frame_index,device='cuda',dtype=torch.long)
            xy,lab,fr=point(0,0);logits,iou=decode_context(model,joint,xy,lab,fr)
            if ci==0:
                native=model.forward(sam_pre=images,sam_feats_precomputed=joint['sam'],point_coords_list=[xy],point_labels_list=[lab],point_frame_indices_list=[fr],multimask_output=False)
                torch.testing.assert_close(logits,native['low_res_logits'],rtol=0,atol=0)
                torch.testing.assert_close(iou,native['iou_predictions'],rtol=0,atol=0)
                dump(OUT/'native_parity.json',dict(complete=True,case=case['name'],logits_exact=True,iou_exact=True,extra_vggt_encodes=1,extra_decodes=1));del native
            masks=(logits_to_original(logits,11,(h,w))>=0).cpu().numpy()
            packed['samv_joint_seed']=np.packbits(masks,axis=2);valid['samv_joint_seed'][:]=True
            lowres['samv_joint_seed']=logits[0,0].reshape(256,11,256).permute(1,0,2).float().cpu().numpy().astype(np.float16)
            del logits,iou,masks
            for j,q in enumerate(case['prompts']):
                if not q['eligible']:continue
                xy,lab,fr=point(j,j);logits,_=decode_context(model,joint,xy,lab,fr)
                current=logits[:,:, :,j*256:(j+1)*256]
                packed['samv_joint_current'][j]=np.packbits((logits_to_original(current,1,(h,w))[0]>=0).cpu().numpy(),axis=1)
                lowres['samv_joint_current'][j]=current[0,0].float().cpu().numpy();valid['samv_joint_current'][j]=True
                single=encode_context(model,images[:,j:j+1],joint['sam'][:,j:j+1])
                fr=torch.zeros_like(fr);current,_=decode_context(model,single,xy,lab,fr)
                packed['samv_single_current'][j]=np.packbits((logits_to_original(current,1,(h,w))[0]>=0).cpu().numpy(),axis=1)
                lowres['samv_single_current'][j]=current[0,0].float().cpu().numpy();valid['samv_single_current'][j]=True
                del single,current,logits
            np.testing.assert_array_equal(packed['samv_joint_seed'][0],packed['samv_joint_current'][0])
            folder=OUT/'cases'/case['name'];folder.mkdir(parents=True,exist_ok=False)
            for method in METHODS:
                path=folder/(method+'.npz');np.savez_compressed(path,masks_packed=packed[method],valid=valid[method],height=h,width=w,lowres_logits=lowres[method]);files[str(path)]=sha(path)
            common=case['common_frames']
            differences={other:int(np.unpackbits(packed['samv_joint_current'][common]^packed[other][common],axis=2)[:,:,:w].sum()) for other in ['samv_single_current','samv_joint_seed']}
            foreground={method:[int(np.unpackbits(packed[method][j],axis=1)[:,:w].sum()) if valid[method][j] else None for j in range(11)] for method in METHODS}
            stats[case['name']]=dict(counts={k:counts[k]-before[k] for k in counts},seconds=time.monotonic()-tick,eligible_frames=sum(q['eligible'] for q in case['prompts']),seed_exact=True,
                different_original_pixels_on_common_frames=differences,foreground_pixels=foreground)
            dump(folder/'run.json',stats[case['name']]);files[str(folder/'run.json')]=sha(folder/'run.json')
            print('SAMV CAPTURED',case['name'],stats[case['name']],flush=True)
            del joint,images;torch.cuda.empty_cache()
    for h in hooks:h.remove()
    assert counts==p['expected_calls'],(counts,p['expected_calls'])
    files[str(OUT/'native_parity.json')]=sha(OUT/'native_parity.json')
    dump(OUT/'predictions_complete.json',dict(complete=True,files=files,results=stats,counts=counts,seconds=time.monotonic()-start,peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,protocol_sha256=sha(OUT/'protocol.json'),reference_read=False,new_maps=0,training_fits=0))
    print('SAMV PREDICTIONS COMPLETE',counts,flush=True)


def evaluate():
    import runpy
    from evaluate_kitti360_development import transfer,read_static_ply
    from evaluate_world_anchor_tracking import rates
    p=verify();seal=read(OUT/'predictions_complete.json');assert seal['complete'] and not (OUT/'summary.json').exists()
    for f,h in seal['files'].items():assert sha(f)==h,f
    oldp=read(OLD/'protocol.json');oldsummary=read(OLD/'summary.json');taxonomy=runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))
    taxonomy=taxonomy['id2label'];results={};gate=True
    for scene in oldp['scenes']:
        key=scene['key'];ref=dict(np.load(scene['reference']));raw=read_static_ply(oldp['references'][key]['raw'])
        train=np.array([taxonomy[int(x)].trainId for x in raw['semantic']]);expected=train*1000+raw['instance'].astype(np.int64)%1000;expected[train==255]=0
        np.testing.assert_array_equal(expected,ref['train_instance'])
        xyz=np.load(read(B/'discovery_measured_inputs_v1'/key/'manifest.json')['geometry'])['xyz']
        nn=transfer(ref['xyz'],np.arange(1,len(train)+1),xyz,.2);ix=np.maximum(nn-1,0)
        known=(nn>0)&(train[ix]!=255)&(ref['confidence'][ix]>0);truth=ref['train_instance'][ix];weight=ref['confidence'][ix]
        anchors=np.load(OLD/(key+'_anchors.npz'));owners,cal=anchors['owners'],anchors['calibration'];rows=[]
        unique={m:set() for m in METHODS+CONTROL}
        for case in [c for c in p['cases'] if c['key']==key]:
            old=next(r for r in oldsummary['results'][key]['cases'] if r['case']==case['name']);gid=old['gt_target'];methods={}
            for method in METHODS+CONTROL:
                root=OUT if method in METHODS else OLD;masks=np.load(root/'cases'/case['name']/(method+'.npz'))
                counts=dict(tp=0.,fp=0.,fn=0.,known_observations=0,unknown_selected=0,heldout_own_seen=0,heldout_own_selected=0,heldout_foreign_seen=0,heldout_foreign_selected=0)
                newly=set();frame_rows=[]
                for j in case['common_frames']:
                    assert masks['valid'][j];o=np.load(case['records'][j]['file']);sid,xy=o['visible_source_ids'],o['pixel_xy'];bits=masks['masks_packed'][j]
                    inside=((bits[xy[:,1],xy[:,0]//8]>>(7-xy[:,0]%8))&1).astype(bool)
                    k=known[sid];correct=truth[sid]==gid if gid is not None else np.zeros(len(sid),bool);w=weight[sid]
                    if gid is not None:
                        values=dict(tp=float(w[k&inside&correct].sum()),fp=float(w[k&inside&~correct].sum()),fn=float(w[k&~inside&correct].sum()))
                        for name,value in values.items():counts[name]+=value
                        counts['known_observations']+=int(k.sum());newly.update(sid[k&inside&correct&(owners[sid]==0)].tolist())
                        frame_rows.append(dict(frame_index=j,frame_id=case['records'][j]['frame_id'],**values,**rates(values)))
                    counts['unknown_selected']+=int((inside&~k).sum())
                    for name,take in [('own',(owners[sid]==case['object_id'])&~cal[sid]),('foreign',(owners[sid]>0)&(owners[sid]!=case['object_id'])&~cal[sid])]:
                        counts['heldout_'+name+'_seen']+=int(take.sum());counts['heldout_'+name+'_selected']+=int((take&inside).sum())
                metrics=rates(counts) if gid is not None else dict(precision=None,recall=None,f1=None)
                methods[method]=dict(**counts,**metrics,new_correct_unowned_sources=len(newly),gt_target_known=gid is not None,frames=frame_rows)
                unique[method].update(newly)
                if method in CONTROL and gid is not None:
                    for name in ['tp','fp','fn','f1','precision','recall']:
                        assert abs(methods[method][name]-old['methods'][method][name])<1e-10,(case['name'],method,name)
            rows.append(dict(case=case['name'],object_id=case['object_id'],semantic_id=case['semantic_id'],gt_target=gid,common_nonseed_frames=len(case['common_frames']),methods=methods))
            print('SAMV EVIDENCE',case['name'],gid,{m:v['f1'] for m,v in methods.items()},flush=True)
        eligible=[r for r in rows if r['gt_target'] is not None];macro={m:float(np.mean([r['methods'][m]['f1'] for r in eligible])) for m in METHODS+CONTROL}
        pooled={m:rates({x:sum(r['methods'][m][x] for r in eligible) for x in ['tp','fp','fn']}) for m in METHODS+CONTROL}
        passed=macro['samv_joint_current']>macro['samv_single_current'] and macro['samv_joint_current']>=max(macro['image_reprompt'],macro['video_seed']) and len(unique['samv_joint_current'])>0
        gate &= passed
        results[key]=dict(cases=rows,macro_case_f1=macro,pooled=pooled,new_correct_unowned_sources={m:len(s) for m,s in unique.items()},missing_gt_cases=len(rows)-len(eligible),information_gate=bool(passed))
    dump(OUT/'summary.json',dict(complete=True,results=results,information_gate=bool(gate),old_controls_replayed_exactly=True,prediction_seal_sha256=sha(OUT/'predictions_complete.json'),protocol_sha256=sha(OUT/'protocol.json'),new_maps=0,new_ours_contribution=False,independent_test=False,official_leaderboard=False))
    print('SAMV INFORMATION GATE',bool(gate),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','capture','evaluate']);globals()[parser.parse_args().action]()
