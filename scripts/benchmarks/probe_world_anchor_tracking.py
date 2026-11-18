"""Bounded evidence-acquisition probe, not a full-map method or paper result."""
import argparse,time,runpy
from pathlib import Path
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from audit_mask_observation_code import source_anchors
from omni_semantic_map.world_anchor_prompts import prompts,calibration_blocks

OUT=ROOT/'build/benchmarks/world_anchor_tracking_probe_v1'
INPUT=ROOT/'build/benchmarks/discovery_measured_inputs_v1'
PARENT=ROOT/'build/benchmarks/object_context_acceptance_v1'
SAM=Path('/home/user/vbot/sam3')
KEYS=['kitti360_0000','kitti360_0002']
METHODS=['image_reprompt','video_seed','video_world_feedback']


def freeze():
    assert not OUT.exists();p=read(PARENT/'protocol.json');inputs={}
    for f,h in p['input_hashes'].items():assert sha(f)==h;inputs[f]=h
    for f in [PARENT/'protocol.json',PARENT/'object_diagnostics.json',ROOT/'build/benchmarks/prethreshold_candidates_v1/protocol.json']:
        inputs[str(f)]=sha(f)
    for key in KEYS:
        m=read(INPUT/key/'manifest.json')
        for f in [INPUT/key/'manifest.json',m['geometry'],INPUT/'query_cells'/key/'source_to_cell.npy']+[r[t] for r in m['records'] for t in ['file','rgb']]:inputs[str(f)]=sha(f)
    paths=[Path(__file__),ROOT/'scripts/benchmarks/evaluate_world_anchor_tracking.py',ROOT/'scripts/benchmarks/audit_mask_observation_code.py',ROOT/'omni_semantic_map/omni_semantic_map/world_anchor_prompts.py',ROOT/'omni_semantic_map/test/test_world_anchor_prompts.py']+sorted((SAM/'sam3').rglob('*.py'))
    old=read(ROOT/'build/benchmarks/prethreshold_candidates_v1/protocol.json')
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),scenes=p['scenes'],references=p['references'],methods=METHODS,input_hashes=inputs,source_hashes={str(f):sha(f) for f in paths},checkpoint=old['checkpoint'],checkpoint_sha256=old['checkpoint_sha256'],
         purpose='Test new identity-conditioned temporal masks beyond the frozen weak-query candidate pool. This is an evidence capability probe; no old-map revision/newAP/robot/independent success claimed.',
         selection='Per exposed window choose first2car+first2building object IDs that admit an11-record clip. Start at earliest frame with>=16distinct20cm calibration anchor cells and>=3eligible later frames in next10records. No GT or model-result selection. No replacement of unlabelable/bad cases. Spatial2m parity split as previous code probe; withheld blocks never used in point prompts.',
         prompts='Exact measured-visible source pixels; original+intrinsic+geometry_graph source provenance; conflicts ignored. Positive3FPS starting closest to mean pixel from distinct own calibration cells. Negative up to3FPS from32nearest foreign calibration cells to positive centroid. No unobserved/unknown region is a negative point. Same native point prompts used by image and feedback conditions.',
         inference='SAM3 existing checkpoint strict video-model loading; native tracker and shared detector backbone per official video notebook;1008,BF16,seed0,eval,no compile or fine-tuning. Per clip11image encodes shared across all conditions. Native binary logit>0, no confidence search. Single-object sessions do not enforce mutual exclusion between objects.',
         controls='image_reprompt:fresh state, current geometry points only when eligible, otherwise unknown. video_seed:seed points only, ordinary forward memory. video_world_feedback:same seed/propagation, then correct eligible later frames using current world-anchor points with use_prev_mem_frame=True; corrected outputs encoded into native memory before next frame. Costs differ and are reported: NOT equal-decoder-budget ablation and not an original Any3DIS/SAM2Object reproduction.',
         budgets='At most8clips x11frames x3conditions; up to88image encodes and native tracking/correction decodes counted. No extra clips,prompt sweep,GT-drivencasechange,training,maps or full-map AP. All masks sealed before GT diagnostics.',
         diagnostics='On eligible nonseed frames common to all3conditions: predicted-anchor heldout recall/foreign intrusion; then exact measured-source nearest reference20cm labels after mask seal. Fixed GT target=baseline best same-class eligibleGT, not prompt/GTchosen. Ignore rawtrain255/confidence0/missing reference. Report every case, pooled and macro case F1, precision,recall,unknown selected points,newly supported unowned source points. These are measured observation diagnostics, not full-map AP or independent accuracy.',
         screening='Only consider a separately frozen full-map revision candidate if all8cases exist and have baseline GT diagnostic targets, feedback macro measured-source F1 exceeds video_seed AND is not below image_reprompt in BOTH windows, with>=1new correctly supported unowned source in each. No full-method validation inferred from this necessary screening; missing GT cases and unequal decoder costs remain limitations.',
         scope='All inputs exposed and strong parent inherits earlier cross-window supervised features. Whole system not training-free. Offline window-end map anchors may come from later frames; do not claim online causality. Temporal tracking,geometric prompting,memory correction have published precedents; probe does not establish novelty. New-method robot and independent evaluation remain pending.'))
    print('FROZEN WORLD ANCHOR TRACKING PROBE',flush=True)


def verify():
    p=read(OUT/'protocol.json')
    for f,h in {**p['input_hashes'],**p['source_hashes']}.items():assert sha(f)==h,f
    assert sha(p['checkpoint'])==p['checkpoint_sha256'];return p


def prepare():
    p=verify();assert not (OUT/'prepared.json').exists();cases=[];all_selection={};files={}
    for key in KEYS:
        m=read(INPUT/key/'manifest.json');xyz=np.load(m['geometry'])['xyz'];owners=source_anchors(key,len(xyz));cell=np.load(INPUT/'query_cells'/key/'source_to_cell.npy');cal=calibration_blocks(xyz);records=m['records'];observations=[dict(np.load(r['file'])) for r in records];objects=read(PARENT/'maps'/key/'baseline/classification.json')['objects'];selections=[]
        for semantic in [26,11]:
            taken=0
            for obj in sorted(objects,key=lambda x:x['object_id']):
                if obj['predictions']['mean']['semantic_id']!=semantic:continue
                oid=obj['object_id'];pr=[prompts(o['visible_source_ids'],o['pixel_xy'],owners,cell,cal,oid) for o in observations]
                starts=[i for i in range(len(pr)-10) if pr[i]['eligible'] and sum(r['eligible'] for r in pr[i+1:i+11])>=3]
                selections.append(dict(object_id=oid,semantic_id=semantic,eligible_starts=starts,chosen=bool(starts and taken<2)))
                if not starts or taken>=2:continue
                start=starts[0];name=key+f'_object{oid:03d}';folder=OUT/'cases'/name;folder.mkdir(parents=True,exist_ok=False);video=folder/'images';video.mkdir()
                for j,r in enumerate(records[start:start+11]):(video/f'{j:06d}.png').symlink_to(r['rgb'])
                case=dict(name=name,key=key,object_id=oid,semantic_id=semantic,records=records[start:start+11],prompts=pr[start:start+11],video=str(video),common_frames=[j for j,q in enumerate(pr[start:start+11]) if j>0 and q['eligible']])
                dump(folder/'case.json',case);files[str(folder/'case.json')]=sha(folder/'case.json');cases.append(case);taken+=1
        all_selection[key]=selections
        np.savez_compressed(OUT/f'{key}_anchors.npz',owners=owners,calibration=cal);files[str(OUT/f'{key}_anchors.npz')]=sha(OUT/f'{key}_anchors.npz')
    dump(OUT/'selection.json',all_selection);files[str(OUT/'selection.json')]=sha(OUT/'selection.json')
    dump(OUT/'prepared.json',dict(complete=True,cases=cases,files=files,protocol_sha256=sha(OUT/'protocol.json')))
    print('CASES',[(c['name'],len(c['common_frames'])) for c in cases],flush=True)


def prepared():
    d=read(OUT/'prepared.json');assert d['complete']
    for f,h in d['files'].items():assert sha(f)==h
    return d


def infer():
    import torch
    from sam3.model_builder import build_sam3_video_model
    p=verify();d=prepared();assert not (OUT/'predictions_complete.json').exists();torch.manual_seed(0);torch.set_num_threads(4)
    free,_=torch.cuda.mem_get_info();assert free>20*1024**3;torch.cuda.set_per_process_memory_fraction(.75)
    model=build_sam3_video_model(checkpoint_path=p['checkpoint'],load_from_HF=False,strict_state_dict_loading=True,compile=False);model.eval();tracker=model.tracker;tracker.backbone=model.detector.backbone
    counts=dict(encodes=0,decodes=0);native=tracker.forward_image
    def encode(*args,**kwargs):counts['encodes']+=1;return native(*args,**kwargs)
    tracker.forward_image=encode
    hook=tracker.sam_mask_decoder.register_forward_hook(lambda *args:counts.__setitem__('decodes',counts['decodes']+1))
    files={};results={};tick=time.monotonic()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for case in d['cases']:
            start=time.monotonic();before=counts.copy();folder=OUT/'cases'/case['name'];loader=tracker.init_state(video_path=case['video'],offload_video_to_cpu=True);h=loader['video_height'];w=loader['video_width'];cache={}
            for j in range(11):
                tracker._get_image_feature(loader,j,1);cache[j]=loader['cached_features'][j]
            def state():return tracker.init_state(video_height=h,video_width=w,num_frames=11,cached_features=cache.copy(),offload_state_to_cpu=True)
            def point(s,j):
                q=case['prompts'][j];xy=np.asarray(q['pixels'],np.float32)/np.array([w,h],np.float32)
                return tracker.add_new_points_or_box(s,j,case['object_id'],points=xy,labels=np.array(q['labels'],np.int32),use_prev_mem_frame=True)[3][0,0]
            seed=None;runs={}
            for method in METHODS:
                c0=counts.copy();masks=[];valid=[];s=state()
                if method!='image_reprompt':point(s,0)
                for j in range(11):
                    ok=case['prompts'][j]['eligible']
                    if method=='image_reprompt':
                        if ok:mask=point(state(),j)>0
                        else:mask=torch.zeros((h,w),dtype=torch.bool,device='cuda')
                    else:
                        frame,oids,low,full,score=next(tracker.propagate_in_video(s,j,0,False,tqdm_disable=True,propagate_preflight=True));assert frame==j and oids==[case['object_id']];mask=full[0,0]>0
                        if method=='video_world_feedback' and j>0 and ok:
                            mask=point(s,j)>0;tracker.propagate_in_video_preflight(s)
                    array=mask.cpu().numpy();masks.append(np.packbits(array,axis=1));valid.append(ok if method=='image_reprompt' else True)
                packed=np.stack(masks)
                if seed is None:seed=packed[0]
                else:np.testing.assert_array_equal(packed[0],seed,err_msg='Shared seed mask differs')
                f=folder/(method+'.npz');np.savez_compressed(f,masks_packed=packed,valid=np.array(valid),height=h,width=w);files[str(f)]=sha(f);runs[method]=dict(decodes=counts['decodes']-c0['decodes'],encodes=counts['encodes']-c0['encodes'],valid_frames=sum(valid));del s
            assert counts['encodes']-before['encodes']==11
            result=dict(methods=runs,seconds=time.monotonic()-start,seed_exact=True);results[case['name']]=result;print('TRACKED',case['name'],result,flush=True)
            del cache,loader;torch.cuda.empty_cache()
    hook.remove();dump(OUT/'predictions_complete.json',dict(complete=True,files=files,results=results,counts=counts,seconds=time.monotonic()-tick,peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,protocol_sha256=sha(OUT/'protocol.json'),prepared_sha256=sha(OUT/'prepared.json'),gt_read=False,new_map=False))
    print('TRACKING CAPTURE COMPLETE',counts,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','prepare','infer']);globals()[parser.parse_args().action]()
