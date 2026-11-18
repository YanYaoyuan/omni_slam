"""Frozen Cityscapes-supervised semantic control on original outdoor images.

External model only: no Ours claim, no GT in inference, no map edits. Preserve
native logits and exact measured source/map samples for later shared controls.
"""
import argparse,time,runpy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from download_segformer_control import OUT as MODEL,REVISION
from audit_semantic_counterevidence import OUT as AUDIT
from capture_semantic_context import INPUT
from omni_semantic_map.sparse_lidar_support import project_measured_visibility

OUT=ROOT/'build/benchmarks/closed_vocabulary_semantics_v1'
BASE=ROOT/'build/benchmarks/signed_query_association_v1/maps'
RUNTIME=ROOT/'build/benchmarks/kitti360_reserved_v1/runtime'


def freeze():
    assert not OUT.exists();parent=read(AUDIT/'protocol.json');model=read(MODEL/'source_manifest.json');assert model['revision']==REVISION;inputs={}
    for name,entry in model['files'].items():f=MODEL/name;assert sha(f)==entry['sha256'];inputs[str(f)]=entry['sha256']
    inputs[str(MODEL/'source_manifest.json')]=sha(MODEL/'source_manifest.json')
    for scene in parent['scenes']:
        key=scene['key'];manifest=read(INPUT/key/'manifest.json');job=read(scene['job'])
        paths=[scene['job'],INPUT/key/'manifest.json',manifest['geometry'],BASE/key/'geometry_graph/geometry.npz',BASE/key/'geometry_graph/classification.json']
        paths += [r[field] for r in manifest['records'] for field in ['file','rgb']]
        paths += [r['measured'] for r in job['frames'] if r['frame_id'] in {v['frame_id'] for v in manifest['records']}]
        for f in paths:inputs[str(f)]=sha(f)
    paths=[Path(__file__),ROOT/'scripts/benchmarks/download_segformer_control.py',ROOT/'omni_semantic_map/omni_semantic_map/sparse_lidar_support.py']
    paths+=list((RUNTIME/'lib/python3.11/site-packages/transformers/models/segformer').glob('*.py'))
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),input_hashes=inputs,source_hashes={str(f):sha(f) for f in paths},scenes=parent['scenes'],references=parent['references'],model_revision=REVISION,
         model='nvidia/segformer-b5-finetuned-cityscapes-1024-1024. External Cityscapes-supervised control, not Ours,open-vocabulary or a KITTI360 original-method reproduction. No fine-tuning on these scenes.',
         inference='Transformers4.51.0 local model/config/processor. Strict state_dict from torch.load(weights_only=True). Native processor1024x1024 bilinear,RGB,ImageNet mean/std,do_reduce_labels=False. FP32/eval/seed0,TF32disabled,noautocast,noresize/crop/scale ensemble. Bilinear logits to original image H/W with align_corners=False,then19-classsoftmax. No confidence filtering or class subset.',
         sampling='All108originalimages,oneforwardperimage. Save native low-resolutionFP32logits,denseargmaxclassmap,exact recorded measured-visible source IDs/pixels with19probabilities,and original fused-map points passing original5cmabs/0relative sparse-depthgate with pixels/probabilities. Missing observations absent,never background.',
         scope='Shared extra evidence only; all83objects/windows exposed, inherited supervised parent. Not official leaderboard,novelty,real-robotvalidation or newmapgain. All outputs sealed before any GT diagnostic. Budget108forwards,noadditionalprompts,checkpointsearch,training,mapexportorAP. A map-level control needs a separate frozen protocol.'))
    print('FROZEN CLOSED VOCABULARY CONTROL',flush=True)


def verify():
    p=read(OUT/'protocol.json')
    for f,h in {**p['input_hashes'],**p['source_hashes']}.items():assert sha(f)==h,f
    return p


def predict():
    import torch,transformers
    from PIL import Image
    from transformers import SegformerConfig,SegformerImageProcessor,SegformerForSemanticSegmentation
    p=verify();assert not (OUT/'predictions_complete.json').exists();assert not (OUT/'execution.json').exists(),'Inspect prior execution before resuming'
    torch.manual_seed(0);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    free,_=torch.cuda.mem_get_info();assert free>=10*1024**3;torch.cuda.set_per_process_memory_fraction(.5)
    dump(OUT/'execution.json',dict(started_epoch=time.time(),protocol_sha256=sha(OUT/'protocol.json'),torch=torch.__version__,transformers=transformers.__version__,physical_gpu=0,free_bytes=free))
    config=SegformerConfig.from_pretrained(str(MODEL),local_files_only=True);proc=SegformerImageProcessor.from_pretrained(str(MODEL),local_files_only=True)
    assert proc.size=={'height':1024,'width':1024} and not proc.do_reduce_labels
    taxonomy=runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['labels'];names={}
    for label in taxonomy:
        if 0<=label.trainId<19:names.setdefault(label.trainId,label.name)
    assert config.id2label==names
    model=SegformerForSemanticSegmentation(config);state=torch.load(MODEL/'pytorch_model.bin',map_location='cpu',weights_only=True);loading=model.load_state_dict(state,strict=True);assert not loading.missing_keys and not loading.unexpected_keys;del state;model.to('cuda').eval()
    files={};results={};tick=time.monotonic();forwards=0
    with torch.inference_mode():
        for scene in p['scenes']:
            key=scene['key'];manifest=read(INPUT/key/'manifest.json');base=np.load(BASE/key/'geometry_graph/geometry.npz');job=read(scene['job']);frames={r['frame_id']:r for r in job['frames']};rows=[]
            for rank,rec in enumerate(manifest['records']):
                start=time.monotonic();rgb=Image.open(rec['rgb']).convert('RGB');w,h=rgb.size;batch=proc(images=rgb,return_tensors='pt');assert tuple(batch['pixel_values'].shape)==(1,3,1024,1024)
                logits=model(**{k:v.to('cuda') for k,v in batch.items()}).logits;forwards+=1;full=torch.nn.functional.interpolate(logits,size=(h,w),mode='bilinear',align_corners=False);prob=full.softmax(1)[0];label=full.argmax(1)[0]
                obs=np.load(rec['file']);xy=obs['pixel_xy'];source_probs=prob[:,xy[:,1],xy[:,0]].T.cpu().numpy();np.testing.assert_allclose(source_probs.sum(1),1.,rtol=0,atol=1e-6)
                raw=frames[rec['frame_id']];np.testing.assert_allclose(rec['camera_pose'],raw['pose'],rtol=0,atol=1e-10);depth=np.load(raw['measured'])['depth'];x,y,_,visible=project_measured_visibility(base['xyz'],np.asarray(raw['pose']),job['config']['cam'],depth,SimpleNamespace(depth_absolute_m=.05,depth_relative=0.));ids=np.flatnonzero(visible);map_probs=prob[:,y[ids],x[ids]].T.cpu().numpy();np.testing.assert_allclose(map_probs.sum(1),1.,rtol=0,atol=1e-6)
                labels=label.cpu().numpy().astype(np.uint8);np.testing.assert_array_equal(labels[xy[:,1],xy[:,0]],source_probs.argmax(1));np.testing.assert_array_equal(labels[y[ids],x[ids]],map_probs.argmax(1))
                f=OUT/'frames'/key/f'{rec["frame_id"]:06d}.npz';f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,source_ids=obs['visible_source_ids'],source_pixel_xy=xy,source_probabilities=source_probs,map_ids=ids,map_pixel_xy=np.column_stack([x[ids],y[ids]]),map_probabilities=map_probs,logits=logits[0].cpu().numpy(),class_map=labels,original_size=np.array([w,h]))
                files[str(f)]=sha(f);rows.append(dict(frame_id=rec['frame_id'],rgb=rec['rgb'],file=str(f),sources=len(xy),map_points=len(ids),seconds=time.monotonic()-start))
                del logits,full,prob,label,batch
                if (rank+1)%10==0 or rank+1==len(manifest['records']):print('SEMANTIC CONTROL',key,rank+1,len(manifest['records']),flush=True)
            results[key]=rows
    dump(OUT/'predictions_complete.json',dict(complete=True,protocol_sha256=sha(OUT/'protocol.json'),files=files,results=results,forwards=forwards,seconds=time.monotonic()-tick,peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,gt_read=False,new_maps=0,loading_missing=[],loading_unexpected=[]))
    print('CLOSED VOCABULARY CAPTURE SEALED',forwards,flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','predict']);globals()[a.parse_args().action]()
