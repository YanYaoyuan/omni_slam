"""One frozen equal-budget discovery experiment, shared SAM3/CG-history2.

All scene predictions precede target scoring. Cached unqueried masks are never
read by policy selection; access logs and deterministic policy replay are saved.
"""
import argparse,time
from pathlib import Path
import numpy as np
from paper_final_common import ROOT,FROZEN,read,dump,sha

OUT=ROOT/'build/benchmarks/discovery_query_benchmark_v1'
INPUT=ROOT/'build/benchmarks/discovery_measured_inputs_v1'
MODEL=Path('/home/user/vbot/sam3')
METHODS=['uniform','geometric','residual','all_frames']


def freeze():
    assert not OUT.exists()
    prepared=read(INPUT/'protocol.json');probe=read(ROOT/'build/benchmarks/sam3_discovery_probe_v1/protocol.json');scenes=[]
    refs=dict(kitti360_0000=ROOT/'build/benchmarks/kitti360_connected_development_v1/evaluation/reference.npz',
        kitti360_0002=ROOT/'build/benchmarks/kitti360_reserved_v1/2013_05_28_drive_0002_sync/evaluation/reference.npz')
    for s in prepared['scenes']:
        item=dict(s,reference=str(refs[s['key']]) if s['key'] in refs else None);scenes.append(item)
    modules=['discovery_query_memory','measured_query_cells','sparse_support_components','replay_fusion','history_graph','fusion_diagnostics','full_ovo_evaluation','shared_observations','kitti360_evaluation']
    sources=[Path(__file__),ROOT/'scripts/benchmarks/discovery_cg_backend.py',FROZEN/'config/cg_rgbd.json',ROOT/'scripts/benchmarks/evaluate_kitti360_development.py']
    sources += [ROOT/'omni_semantic_map/omni_semantic_map'/f'{m}.py' for m in modules]
    sources += [ROOT/'build/benchmarks/expanded_semantic_20261109/concept-graphs/conceptgraph/slam'/f for f in ['mapping.py','utils.py','slam_classes.py']]
    sources += list((ROOT/'build/benchmarks/ovo_full_20261110/upstream/ovo/entities').glob('*clip*.py'))
    sources += [ROOT/'build/benchmarks/ovo_full_20261110/upstream/ovo/utils/clip_utils.py',ROOT/'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py']
    inputs=[INPUT/'complete.json',INPUT/'verification.json',INPUT/'query_cells/complete.json',INPUT/'query_cells/protocol.json',
        ROOT/'build/benchmarks/sam3_discovery_probe_v1/protocol.json',ROOT/'build/benchmarks/ovo_full_20261110/model_artifacts.json']
    input_hashes={str(f):sha(f) for f in inputs}
    for name in ['text_features.npy','vocabulary.json']:
        f=ROOT/'build/benchmarks/kitti360_development_v1/taxonomy'/name;input_hashes[str(f)]=sha(f)
    for f,h in read(INPUT/'complete.json')['files'].items():assert sha(f)==h
    for f,h in read(INPUT/'query_cells/complete.json')['files'].items():assert sha(f)==h
    for s in scenes:
        m=read(INPUT/s['key']/'manifest.json');input_hashes[m['geometry']]=m['geometry_sha256']
        input_hashes[s['job']]=sha(s['job'])
    source_hashes={str(f):sha(f) for f in sources};source_hashes.update(probe['source_hashes'])
    # The probe driver is not used by this benchmark; pin native model sources.
    source_hashes.pop(str(ROOT/'scripts/benchmarks/probe_sam3_discovery.py'),None)
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),scenes=scenes,methods=METHODS,input_hashes=input_hashes,source_hashes=source_hashes,
        reference_hashes={str(f):sha(f) for f in refs.values()},sam3_checkpoint=probe['checkpoint'],sam3_sha256=probe['input_hashes'][probe['checkpoint']],
        prompts=['car','building'],semantic_ids=[26,11],sam3_threshold=.5,sam3_resolution=1008,buffer_size=4,
        selection='One query per completed buffer of4 received frames (including final partial buffer), delayed by at most3 sampled frames. Uniform:last; geometric:sum1/(1+queries); residual:exclude units explained by previously queried accepted mask supports. Latest-frame stable ties. No future masks/GT.',
        memory='20cm measured query cells; original measured geometry unchanged. Query result updates from connected mask supports with >=16 points, before final object persistence filtering. This is observation memory, not confirmed-object truth.',
        masks='Native SAM3 car/building with0.5 threshold. Cross-mask pixels assigned to highest native score, stable prompt/index ties. No added NMS or GT filter.',
        features='Original frozen SigLIP global/masked/bbox triplets and learned merger, CUDA BF16 batches8, native preprocessing. Same feature array for every policy; feature model is not the claimed mechanism.',
        support='Existing largest measured component radius0.4m, minimum16. Same current measured depth gates and causal source IDs as prepared inputs.',
        backend='Existing native CG with DBSCAN disabled,20cm association/final-overlap radius,2.5cm geometric processing; all similarity/persistence thresholds unchanged. History2 graph and guarded final merge. Native seeded4mm jitter retained.',
        semantics='Native SAM3 prompt class, one largest-area observation per distinct frame; confidence is winning class sum of native detector scores divided by number of selected frames. Shared by every policy. No SigLIP category relabeling.',
        scoring='Native CG final geometry, nearest20cm full static reference; zero if no source within radius. Official adapted KITTI kernel, building/car confidence weighted AP50. No extra source blockers. Scores are paired among these four arms; old SAM2 scores are not used as attribution for policy gains.',
        budget='12 maps:4 arms times2 exposed outdoor windows and1 robot. Equal-budget arms15/12/83 image queries,2 prompts per image. All-frames reference60/48/330; report its extra compute. Full cached encodings are counterfactual input reuse, never counted as online replay speed.',
        gate='Residual must exceed BOTH uniform and geometric by at least2 AP50 points on EACH outdoor window, with at least one additional unique matched GT versus each and no lost matched GT. Otherwise freeze without parameter search. All-frame arm is a larger-budget reference, not an equal-budget competitor.',
        limitation='Previously exposed development windows, not independent/official leaderboard. Robot unlabelled; qualitative only. Shared new frontend is not ours. Object existence/persistence policy unchanged. Query memory may suppress revisiting a false mask; unknown pixels never vote.'))
    print('FROZEN',[(s['key'],len(read(INPUT/s['key']/'manifest.json')['records'])) for s in scenes],flush=True)


def verify(data=False):
    p=read(OUT/'protocol.json')
    for f,h in {**p['source_hashes'],**p['input_hashes']}.items():assert sha(f)==h,f
    if data:
        for seal in [INPUT/'complete.json',INPUT/'query_cells/complete.json']:
            for f,h in read(seal)['files'].items():assert sha(f)==h,f
        for s in p['scenes']:
            for f,h in read(INPUT/s['key']/'manifest.json')['input_hashes'].items():assert sha(f)==h,f
    return p


def masks():
    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    p=verify(True);assert not (OUT/'masks_complete.json').exists();assert sha(p['sam3_checkpoint'])==p['sam3_sha256']
    torch.set_num_threads(4);assert torch.cuda.is_available();free,total=torch.cuda.mem_get_info();assert free>=18*1024**3
    torch.cuda.set_per_process_memory_fraction(.5);torch.manual_seed(0)
    model=build_sam3_image_model(checkpoint_path=p['sam3_checkpoint'],load_from_HF=False,device='cuda',compile=False)
    proc=Sam3Processor(model,resolution=p['sam3_resolution'],confidence_threshold=p['sam3_threshold']);files={};scenes={}
    # Fixed synthetic warm-up; not a selected dataset frame or method input.
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        warm=proc.set_image(Image.new('RGB',(640,360)));proc.set_text_prompt(prompt='car',state=warm);del warm
        for s in p['scenes']:
            m=read(INPUT/s['key']/'manifest.json');dest=OUT/'frontend'/s['key']/'masks';dest.mkdir(parents=True,exist_ok=False);records=[];offset=0
            for rank,r in enumerate(m['records']):
                assert sha(r['rgb'])==m['input_hashes'][r['rgb']];image=Image.open(r['rgb']).convert('RGB')
                torch.cuda.synchronize();start=time.monotonic();state=proc.set_image(image);torch.cuda.synchronize();encode=time.monotonic()-start
                binary=[];scores=[];classes=[];decode=0.
                for prompt,semantic in zip(p['prompts'],p['semantic_ids']):
                    proc.reset_all_prompts(state);torch.cuda.synchronize();start=time.monotonic();result=proc.set_text_prompt(prompt=prompt,state=state);torch.cuda.synchronize();decode+=time.monotonic()-start
                    masks=result['masks'].cpu().numpy().reshape(-1,image.height,image.width).astype(bool);confidence=result['scores'].float().cpu().numpy()
                    binary.extend(masks);scores.extend(confidence.tolist());classes.extend([semantic]*len(masks))
                binary=np.asarray(binary,bool).reshape(-1,image.height,image.width);scores=np.asarray(scores,np.float32);n=len(scores)
                seg=np.full((image.height,image.width),-1,np.int16)
                for j in np.argsort(-scores,kind='stable'):seg[binary[j]&(seg<0)]=j
                file=dest/f'{r["frame_id"]:06d}.npz';np.savez_compressed(file,masks_packed=np.packbits(binary,axis=2),segmentation=seg,
                    scores=scores,semantic_ids=np.asarray(classes,np.int32),areas=binary.sum((1,2)).astype(np.int32))
                files[str(file)]=sha(file);records.append(dict(frame_id=r['frame_id'],file=str(file),masks=n,observation_offset=offset,image_encode_seconds=encode,prompt_decode_seconds=decode));offset+=n;del state
                if rank%25==0:print('MASKS',s['key'],rank,len(m['records']),n,flush=True)
            scenes[s['key']]=dict(records=records,observations=offset)
    dump(OUT/'masks_complete.json',dict(complete=True,files=files,scenes=scenes,protocol_sha256=sha(OUT/'protocol.json'),peak_allocated_bytes=torch.cuda.max_memory_allocated()))


def features():
    import sys,torch,yaml
    from PIL import Image
    p=verify();seal=read(OUT/'masks_complete.json');assert not (OUT/'features_complete.json').exists()
    for f,h in seal['files'].items():assert sha(f)==h
    artifacts=read(ROOT/'build/benchmarks/ovo_full_20261110/model_artifacts.json')
    for f,item in artifacts['files'].items():assert sha(f)==item['sha256'],f
    sys.path.insert(0,str(ROOT/'build/benchmarks/ovo_full_20261110/upstream'))
    from ovo.entities.clip_generator import CLIPGenerator
    from ovo.entities.clips_merging import WeightsPredictorMerger
    from ovo.utils.clip_utils import load_clip_model
    cfg=read(p['scenes'][0]['job'])['config']['semantic']['clip'];torch.set_num_threads(2)
    assert torch.cuda.is_available();free,total=torch.cuda.mem_get_info();assert free>=18*1024**3
    torch.cuda.set_per_process_memory_fraction(.5);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    encoder=CLIPGenerator.__new__(CLIPGenerator);encoder.config=cfg;encoder.device='cuda';encoder.embed_type='learned';encoder.mask_res=cfg['mask_res']
    encoder.model,encoder.tokenizer,encoder.preprocess,encoder.clip_dim=load_clip_model(cfg['model_card'],False);encoder.model.eval();encoder.model.visual.cuda()
    weights=Path(cfg['weights_predictor_path']);merger=WeightsPredictorMerger(yaml.safe_load((weights/'hparams.yaml').read_text())['model']).eval()
    merger.load_state_dict(torch.load(weights/'model.pt',map_location='cpu',weights_only=False));merger.cuda();files={};timings={}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for s in p['scenes']:
            m=read(INPUT/s['key']/'manifest.json');byframe={r['frame_id']:r for r in m['records']};dest=OUT/'frontend'/s['key']/'features';dest.mkdir(exist_ok=False);times=[]
            for rank,r in enumerate(seal['scenes'][s['key']]['records']):
                rgb=np.asarray(Image.open(byframe[r['frame_id']]['rgb']).convert('RGB')).copy();image=torch.from_numpy(rgb).cuda().permute(2,0,1)
                z=np.load(r['file']);binary=np.unpackbits(z['masks_packed'],axis=2)[:,:,:rgb.shape[1]].astype(bool);pieces=[]
                torch.cuda.synchronize();start=time.monotonic()
                for first in range(0,len(binary),8):
                    triplets=encoder.extract_clip(image,torch.from_numpy(binary[first:first+8]).cuda(),return_all=True)
                    pieces.append(merger(triplets).float().cpu().numpy())
                torch.cuda.synchronize();elapsed=time.monotonic()-start;f=np.concatenate(pieces) if pieces else np.empty((0,1152),np.float32)
                assert np.isfinite(f).all() and (not len(f) or np.all(np.abs(np.linalg.norm(f,axis=1)-1)<.02))
                path=dest/f'{r["frame_id"]:06d}.npy';np.save(path,f);files[str(path)]=sha(path);times.append(dict(frame_id=r['frame_id'],seconds=elapsed))
                if rank%25==0:print('FEATURES',s['key'],rank,len(binary),flush=True)
            timings[s['key']]=times
    dump(OUT/'features_complete.json',dict(complete=True,files=files,timings=timings,masks_seal_sha256=sha(OUT/'masks_complete.json'),peak_allocated_bytes=torch.cuda.max_memory_allocated()))


def supports():
    from omni_semantic_map.sparse_support_components import largest_measured_component
    p=verify(True);maskseal=read(OUT/'masks_complete.json');assert not (OUT/'supports_complete.json').exists();files={};counts={}
    for f,h in maskseal['files'].items():assert sha(f)==h
    for s in p['scenes']:
        manifest=read(INPUT/s['key']/'manifest.json');xyz=np.load(manifest['geometry'])['xyz'];byframe={r['frame_id']:r for r in manifest['records']}
        folder=OUT/'frontend'/s['key']/'supports';folder.mkdir(exist_ok=False);raw_total=0;kept_total=0
        for r in maskseal['scenes'][s['key']]['records']:
            measured=np.load(byframe[r['frame_id']]['file']);ids=measured['visible_source_ids'];px=measured['pixel_xy'];m=np.load(r['file']);seg=m['segmentation'];labels=seg[px[:,1],px[:,0]];pieces=[];raw_counts=[]
            for j in range(r['masks']):
                raw=ids[labels==j];raw_counts.append(len(raw));pieces.append(raw[largest_measured_component(xyz[raw],.4,16)])
            joined=np.concatenate(pieces) if pieces else np.empty(0,np.int64);offsets=np.r_[0,np.cumsum([len(v) for v in pieces])]
            assert len(np.unique(joined))==len(joined) and np.isin(joined,ids).all()
            path=folder/f'{r["frame_id"]:06d}.npz';np.savez_compressed(path,point_ids=joined,offsets=offsets,raw_counts=raw_counts)
            files[str(path)]=sha(path);raw_total+=sum(raw_counts);kept_total+=len(joined)
        counts[s['key']]=dict(raw_supports=raw_total,connected_supports=kept_total);print('SUPPORTS',s['key'],counts[s['key']],flush=True)
    dump(OUT/'supports_complete.json',dict(complete=True,files=files,counts=counts,masks_seal_sha256=sha(OUT/'masks_complete.json')))


def predict():
    from omni_semantic_map.discovery_query_memory import DiscoveryQueryMemory
    from discovery_cg_backend import build_map
    p=verify(True);assert not (OUT/'predictions_complete.json').exists();maskseal=read(OUT/'masks_complete.json');featureseal=read(OUT/'features_complete.json');supportseal=read(OUT/'supports_complete.json');files={}
    for seal in [maskseal,featureseal,supportseal]:
        for f,h in seal['files'].items():assert sha(f)==h,f
    taxonomy=ROOT/'build/benchmarks/kitti360_development_v1/taxonomy';text=np.load(taxonomy/'text_features.npy');vocab=read(taxonomy/'vocabulary.json')
    for s in p['scenes']:
        key=s['key'];manifest=read(INPUT/key/'manifest.json');measured=manifest['records'];records=maskseal['scenes'][key]['records'];byframe={r['frame_id']:r for r in records};cells=np.load(INPUT/'query_cells'/key/'source_to_cell.npy')
        geometry=np.load(manifest['geometry']);featuretimes={r['frame_id']:r['seconds'] for r in featureseal['timings'][key]}
        # Shared read-only descriptors/class metadata feed the backend; selection sees only geometric IDs.
        feats=[];frames=[];areas=[];classes=[];scores=[]
        for r in records:
            z=np.load(r['file']);feats.append(np.load(OUT/'frontend'/key/'features'/f'{r["frame_id"]:06d}.npy'))
            frames.extend([r['frame_id']]*r['masks']);areas.extend(z['areas']);classes.extend(z['semantic_ids']);scores.extend(z['scores'])
        feats=np.concatenate(feats);frames=np.asarray(frames);areas=np.asarray(areas);classes=np.asarray(classes);scores=np.asarray(scores)
        def load_support(rec):return np.load(OUT/'frontend'/key/'supports'/f'{rec["frame_id"]:06d}.npz')
        for method in METHODS:
            memory=DiscoveryQueryMemory(method if method!='all_frames' else 'uniform');chosen=[];access=[];size=p['buffer_size'] if method!='all_frames' else 1
            for start in range(0,len(measured),size):
                buffer=measured[start:start+size];candidates=[]
                for r in buffer:
                    visible=np.load(r['file'])['visible_source_ids'];candidates.append(dict(frame_id=r['frame_id'],visible_unit_ids=np.unique(cells[visible])))
                selected=memory.choose(candidates,buffer[-1]['frame_id'])
                # First outcome access for this buffer occurs strictly after choose().
                rec=byframe[selected];support=load_support(rec);positive=np.unique(cells[support['point_ids']]);memory.commit(selected,positive)
                access.append(dict(decision_frame=buffer[-1]['frame_id'],selected_frame=selected,outcome_accessed_after_selection=True,positive_unit_ids=positive.tolist()))
                chosen.append(rec)
            expected=(len(measured)+size-1)//size;assert len(chosen)==expected
            folder=OUT/'maps'/key/method;run=build_map(folder,geometry,chosen,load_support,feats,frames,areas,classes,scores,text,vocab)
            cost=sum(r['image_encode_seconds']+r['prompt_decode_seconds']+featuretimes[r['frame_id']] for r in chosen)
            dump(folder/'queries.json',dict(events=memory.events,access_log=access,selected_frame_ids=[r['frame_id'] for r in chosen],
                image_queries=len(chosen),text_queries=2*len(chosen),measured_cached_frontend_cost_seconds=cost,
                timing_scope='Sum of actual cached GPU forward times for selected frames, not measured end-to-end online runtime. Excludes shared loading/IO; backend time separately reported.'))
            # Replay from geometry and only selected outcomes, without opening any
            # unselected mask, feature, class or support file.
            replay=DiscoveryQueryMemory(method if method!='all_frames' else 'uniform')
            for start,item in zip(range(0,len(measured),size),access):
                buffer=measured[start:start+size]
                candidates=[dict(frame_id=r['frame_id'],visible_unit_ids=np.unique(cells[np.load(r['file'])['visible_source_ids']])) for r in buffer]
                selected=replay.choose(candidates,buffer[-1]['frame_id']);assert selected==item['selected_frame']
                support=load_support(byframe[selected]);positive=np.unique(cells[support['point_ids']])
                assert positive.tolist()==item['positive_unit_ids'];replay.commit(selected,positive)
            assert replay.events==memory.events
            dump(folder/'policy_replay.json',dict(complete=True,decisions=len(access),identical_events=True,
                scope='Replays the selector from causal geometry and selected support outcomes only; not a claim of online wall time.'))
            for f in folder.iterdir():files[str(f)]=sha(f)
            print('MAPPED',key,method,'queries',len(chosen),'objects',run['objects'],flush=True)
    dump(OUT/'predictions_complete.json',dict(complete=True,files=files,protocol_sha256=sha(OUT/'protocol.json'),
        masks_seal_sha256=sha(OUT/'masks_complete.json'),features_seal_sha256=sha(OUT/'features_complete.json'),supports_seal_sha256=sha(OUT/'supports_complete.json')))


def matched_ids(matches):
    result=set()
    for window in matches.values():
        for category,objects in window['groundTruth'].items():
            for obj in objects:
                if obj['instance_id']<1000 or obj['vert_count']<100:continue
                for pred in obj.get('matched_pred',[]):
                    inter=pred['intersection'];union=obj['vert_count']+pred['vert_count']-inter
                    if union>0 and inter/union>.5:result.add(int(obj['instance_id']))
    return result


def score():
    from omni_semantic_map.kitti360_evaluation import evaluate
    from evaluate_kitti360_development import transfer,clean_json
    p=verify();seal=read(OUT/'predictions_complete.json');assert not (OUT/'summary.json').exists();results={};gates=[]
    for f,h in seal['files'].items():assert sha(f)==h,f
    evaluator=ROOT/'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py'
    for s in p['scenes']:
        key=s['key'];methods={};matches={};changes={}
        if s['reference']:
            assert sha(s['reference'])==p['reference_hashes'][s['reference']];ref=np.load(s['reference'])
        for method in METHODS:
            folder=OUT/'maps'/key/method;run=read(folder/'run.json');query=read(folder/'queries.json')
            methods[method]=dict(run=run,image_queries=query['image_queries'],text_queries=query['text_queries'],frontend_seconds=query['measured_cached_frontend_cost_seconds'])
            if not s['reference']:continue
            g=np.load(folder/'geometry.npz');projected=transfer(g['xyz'],g['instance'],ref['xyz'],.2);rows=read(folder/'classification.json')['objects'];prediction={r['object_id']:r['predictions']['mean'] for r in rows}
            result=evaluate(ref['train_instance'],ref['confidence'],projected,prediction,evaluator)
            dump(folder/'matches.json',clean_json(result['matches']));methods[method]['metrics']=clean_json(result['averages']);matches[method]=matched_ids(result['matches'])
            methods[method]['matched_gt_ids']=sorted(matches[method]);print('SCORED',key,method,result['averages']['allAp50%'],flush=True)
        if matches:
            for control in ['uniform','geometric']:
                changes[control]=dict(gained=sorted(matches['residual']-matches[control]),lost=sorted(matches[control]-matches['residual']))
            gate=all(methods['residual']['metrics']['allAp50%']-methods[c]['metrics']['allAp50%']>=.02 and changes[c]['gained'] and not changes[c]['lost'] for c in ['uniform','geometric']);gate=bool(gate);gates.append(gate)
        else:gate=None
        results[key]=dict(methods=methods,changes_vs_controls=changes,gate=gate)
    dump(OUT/'summary.json',clean_json(dict(complete=True,results=results,advance_gate=all(gates),development_only=True,robot_accuracy_available=False,
        protocol_sha256=sha(OUT/'protocol.json'),prediction_seal_sha256=sha(OUT/'predictions_complete.json'))))
    print('ADVANCE',all(gates),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','masks','features','supports','predict','score']);globals()[parser.parse_args().action]()
