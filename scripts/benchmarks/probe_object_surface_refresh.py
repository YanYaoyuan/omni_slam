"""Acquire genuinely new object masks on unresolved measured surface.

This is a bounded information probe, not a new full-map result or a claim that
cropping is novel. Both branches have identical prompts and readout domains.
"""
import argparse
from pathlib import Path
import time
import numpy as np
from paper_final_common import ROOT, read, dump, sha

B=ROOT/'build/benchmarks'
OUT=B/'object_surface_refresh_probe_v1'


def verify():
    p=read(OUT/'protocol.json')
    for f,h in {**p['input_hashes'],**p['source_hashes']}.items():assert sha(f)==h,f
    return p


def freeze():
    assert not OUT.exists()
    rows=[c for c in read(B/'joint_semantic_revision_v1/protocol.json')['contexts'] if c['kind']=='outdoor']
    inputs={}; parent=read(B/'partition_revision_v1/protocol.json')
    for c in rows:
        paths=[c['job'],c['field'],c['sem'],Path(c['full'])/'geometry.npz',Path(c['full'])/'old_to_full.npy',
               Path(c['ours'])/'geometry.npz',Path(c['ours'])/'classification.json',
               B/'full_source_visual_visibility_audit_v1'/c['key']/'manifest.json']
        for frame in read(c['job'])['frames']:paths.append(frame['rgb'])
        manifest=read(B/'full_source_visual_visibility_audit_v1'/c['key']/'manifest.json')
        for rec in manifest['records']:paths.append(rec['file'])
        for f in paths:inputs[str(f)]=sha(f)
    checkpoint=parent['checkpoint'];assert sha(checkpoint)==parent['checkpoint_sha256'];inputs[checkpoint]=parent['checkpoint_sha256']
    sources=[Path(__file__),*sorted((Path('/home/user/vbot/sam3')/'sam3').rglob('*.py'))]
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),contexts=rows,input_hashes=inputs,
        source_hashes={str(f):sha(f) for f in sources},checkpoint=checkpoint,
        hypothesis='Check if same physical prompts in a local crop yield different, useful surface evidence, before implementing history repair. Existing map identities plus unassigned real surface define the acquisition domain, not a pair of mostly unborn objects.',
        selection='Existing exported objects only. At least16 distinct20cm anchor cells and3 distinct anchor pixels in a frame. Crop doubles the largest projected anchor-box extent, centered on the box, clipped to image. Require image-area/crop-area>=4 and>=16 unassigned measured cells inside crop. For each object select largest unassigned-cell count, then anchor cells, area gain, earliest frame. Per scene choose4building and4car objects by that order, unique IDs; ties smallerID. If insufficient, report missing slots without replacement classes/scenes.',
        sources='Same existing full real-source visibility cache, causal source acquisition,5cm measured-depth gate. Anchors are old positive source IDs; current exported propagated sources count as assigned. Unknown candidates must be real visible sources with no current source assignment. No depth filling, new scans, label priors or GT selection.',
        prompts='Three positive points: unique anchor pixel nearest centroid, then farthest point, then farthest from selected set; lexicographic pixel tie. Same physical points in full RGB and crop. No text, negative prompt, mask history, confidence/radius search. Native interactiveSAM3 returns3masks, choose highest native quality in each branch. Record every mask and actual prompt satisfaction.',
        control='Full-frame point-prompt decoding versus local-crop decoding. Native1008 input, same SAM3 checkpoint, BF16 seed0, frozen model. Crop changes context and sampling resolution; do NOT attribute differences solely to resolution. Read both branches on the SAME visible measured-source domain inside the crop, preserving missingness.',
        budget='At most16sites,32nativeimageencodes and32interactive decodes. Zero training, extra images, new maps or AP. No prompt/scale/checkpoint variants after results. Public exposed windows only; robot and independent generalization remain pending for an effective shared mechanism.',
        analysis='Seal outputs before any optional GT diagnostic. First verify prompts, actual mask differences, changed old/source and unassigned support. Without measurable distinct evidence, stop rather than generate identical full maps. Local masks and counts are not accuracy or paper contribution.'))
    print('SURFACE REFRESH PROBE FROZEN',flush=True)


def points_for(pixels):
    xy=np.unique(pixels,axis=0);assert len(xy)>=3
    first=int(np.argmin(((xy-xy.mean(0))**2).sum(1)));chosen=[first]
    for _ in range(2):
        distance=((xy[:,None]-xy[chosen][None])**2).sum(2).min(1)
        chosen.append(int(np.argmax(distance)))
    return xy[chosen]


def prepare():
    p=verify();assert not (OUT/'prepared.json').exists();files={};results={}
    for c in p['contexts']:
        key=c['key'];job=read(c['job']);frames={r['frame_id']:r for r in job['frames']}
        cam=job['config']['cam'];w,h=cam['W'],cam['H']
        field=np.load(c['field']);mapping=np.load(Path(c['full'])/'old_to_full.npy');sem=np.load(c['sem'])
        owners=np.zeros(len(field['source_to_cell']),np.int64);owners[mapping]=sem['owners']
        anchors=owners.copy();base=np.load(Path(c['ours'])/'geometry.npz');added=base['added_full_source_ids']
        if len(added):owners[added]=base['instance'][-len(added):]
        classes={o['object_id']:o['predictions']['mean']['train_id'] for o in read(Path(c['ours'])/'classification.json')['objects']}
        best={};pool=[]
        manifest=read(B/'full_source_visual_visibility_audit_v1'/key/'manifest.json')
        for rec in manifest['records']:
            d=np.load(rec['file']);ids=d['visible_source_ids'];xy=d['pixel_xy'];cells=field['source_to_cell'][ids]
            for oid in np.unique(anchors[ids]):
                if oid<=0 or classes.get(int(oid)) not in [2,13]:continue
                take=anchors[ids]==oid;n=len(np.unique(cells[take]));ap=xy[take]
                if n<16 or len(np.unique(ap,axis=0))<3:continue
                lo,hi=ap.min(0),ap.max(0)+1;center=(lo+hi)/2;extent=int(max(hi-lo))
                low=np.maximum(np.floor(center-extent).astype(int),0)
                high=np.minimum(np.ceil(center+extent).astype(int),[w,h])
                area=int(np.prod(high-low));gain=w*h/area
                if gain<4:continue
                inside=((xy>=low)&(xy<high)).all(1);unknown=inside&(owners[ids]==0)
                unresolved=len(np.unique(cells[unknown]))
                if unresolved<16:continue
                row=dict(object_id=int(oid),train_id=int(classes[int(oid)]),frame_id=rec['frame_id'],rgb=frames[rec['frame_id']]['rgb'],
                         visibility=rec['file'],crop=[*map(int,low),*map(int,high)],anchor_cells=n,
                         unknown_cells=unresolved,area_gain=float(gain),points=points_for(ap).tolist())
                pool.append(row);rank=(unresolved,n,gain,-rec['frame_id'])
                if int(oid) not in best or rank>best[int(oid)][0]:best[int(oid)]=(rank,row)
        selected=[]
        for cls in [2,13]:
            candidates=[x for x in best.values() if x[1]['train_id']==cls]
            candidates.sort(key=lambda x:(*x[0],-x[1]['object_id']),reverse=True)
            selected.extend(r for _,r in candidates[:4])
        assert len({r['object_id'] for r in selected})==len(selected)
        for row in selected:
            d=np.load(row['visibility']);ids=d['visible_source_ids'];xy=d['pixel_xy']
            low=np.array(row['crop'][:2]);high=np.array(row['crop'][2:]);inside=((xy>=low)&(xy<high)).all(1)
            path=OUT/'inputs'/key/f'object_{row["object_id"]}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,source_ids=ids[inside],pixel_xy=xy[inside],owner=owners[ids[inside]],anchor_owner=anchors[ids[inside]],
                                cell=field['source_to_cell'][ids[inside]])
            row['input']=str(path);files[str(path)]=sha(path)
        path=OUT/'inputs'/key/'candidates.json';dump(path,dict(all_eligible=pool,selected=selected));files[str(path)]=sha(path)
        results[key]=dict(eligible_object_views=len(pool),eligible_objects=len(best),selected=selected,
                          unfilled_slots=8-len(selected))
        print('SURFACE REFRESH SELECTED',key,[(r['object_id'],r['frame_id'],r['unknown_cells'],round(r['area_gain'],2)) for r in selected],flush=True)
    dump(OUT/'prepared.json',dict(complete=True,files=files,results=results,gt_read=False,protocol_sha256=sha(OUT/'protocol.json')))


def predict():
    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    p=verify();prep=read(OUT/'prepared.json');assert prep['complete']
    for f,h in prep['files'].items():assert sha(f)==h
    assert not (OUT/'execution.json').exists(),'Inspect live job before resuming'
    torch.set_num_threads(4);torch.manual_seed(0);free,total=torch.cuda.mem_get_info();assert free>=18*1024**3
    torch.cuda.set_per_process_memory_fraction(.5)
    dump(OUT/'execution.json',dict(started_epoch=time.time(),protocol_sha256=sha(OUT/'protocol.json'),physical_gpu=0,free_bytes=free))
    loaded=[];native=torch.nn.Module.load_state_dict
    def check(self,*a,**kw):
        r=native(self,*a,**kw)
        if self.__class__.__name__=='Sam3Image':loaded.append(dict(missing=list(r.missing_keys),unexpected=list(r.unexpected_keys)))
        return r
    torch.nn.Module.load_state_dict=check
    try:model=build_sam3_image_model(checkpoint_path=p['checkpoint'],load_from_HF=False,device='cuda',compile=False,enable_inst_interactivity=True)
    finally:torch.nn.Module.load_state_dict=native
    assert loaded==[dict(missing=[],unexpected=[])]
    processor=Sam3Processor(model,resolution=1008,confidence_threshold=.5);files={};results={};start=time.monotonic();calls=0
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for key,context in prep['results'].items():
            results[key]=[]
            for row in context['selected']:
                with Image.open(row['rgb']) as im:image=im.convert('RGB')
                crop=image.crop(row['crop']);points=np.asarray(row['points'],np.float32);record=dict(object_id=row['object_id'],frame_id=row['frame_id'],branches={})
                for name,im,pts in [('full',image,points),('crop',crop,points-np.array(row['crop'][:2],np.float32))]:
                    torch.manual_seed(0);tick=time.monotonic();state=processor.set_image(im)
                    masks,scores,_=model.predict_inst(state,point_coords=pts,point_labels=np.ones(len(pts),np.int32),multimask_output=True)
                    assert masks.shape[1:]==(im.height,im.width) and np.isfinite(scores).all()
                    selected=int(np.argmax(scores));calls+=1
                    path=OUT/'predictions'/key/f'object_{row["object_id"]}_{name}.npz';path.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(path,masks_packed=np.packbits(masks.astype(bool),axis=2),scores=scores,selected=np.array(selected),
                                        points=pts,original_size=np.array(im.size))
                    files[str(path)]=sha(path);record['branches'][name]=dict(file=str(path),seconds=time.monotonic()-tick,selected=selected,scores=scores.tolist())
                    del state,masks
                results[key].append(record);print('SURFACE REFRESH MASKS',key,row['object_id'],flush=True)
    dump(OUT/'predictions_complete.json',dict(complete=True,files=files,results=results,model_calls=calls,image_encodes=calls,instance_decodes=calls,
         seconds=time.monotonic()-start,peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,loading=loaded,gt_read=False,new_maps=0,new_ap=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','prepare','predict']);globals()[parser.parse_args().action]()
