"""Execute the frozen SAM2/SigLIP frontend with one model resident at a time.

Mask generation, SigLIP's three descriptors, learned fusion, and geometry are
separate processes. FP32 parameters, CUDA BF16 autocast and eight-mask batches
are preserved. CPU round trips store BF16 values losslessly as float32. This
is a resource adaptation; validate against the original cache before use.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import cv2
import numpy as np
from audit_evidence_retention import ROOT, BASE, OUT, CACHE, dump
from omni_semantic_map.flat_benchmark import sha256

OLD=ROOT/'build/benchmarks/ovo_full_20261110'
UP=OLD/'upstream'
sys.path.insert(0,str(UP))
PHASES=('masks','triplets','features','geometry')


def gpu_setup():
    import torch
    from ovo.utils.gen_utils import setup_seed
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(2);cv2.setNumThreads(1);setup_seed(0)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    return torch


def rgb(record):
    image=cv2.imread(record['rgb'])
    if image is None:raise FileNotFoundError(record['rgb'])
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


def verify_job(job,path):
    for file,h in job['input_sha256'].items():
        if sha256(Path(file))!=h:raise ValueError('changed input: '+file)
    artifacts=json.loads((OLD/'model_artifacts.json').read_text())
    for file,item in artifacts['files'].items():
        if sha256(Path(file))!=item['sha256']:raise ValueError('changed model: '+file)
    for file,h in job['source_sha256'].items():
        if sha256(ROOT/file)!=h:raise ValueError('changed source: '+file)


def process(job_path,phase):
    torch=gpu_setup();job=json.loads(job_path.read_text());verify_job(job,job_path)
    out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
    folder=out/phase;folder.mkdir(exist_ok=False)
    status=dict(phase=phase,complete=False,job_sha256=sha256(job_path),source_sha256=sha256(Path(__file__)),
        device=torch.cuda.get_device_name(0),memory_limit_bytes=6*1024**3,records=[])
    status_path=out/(phase+'_run.json');dump(status_path,status);start=time.monotonic()
    cfg=job['config'];records=[r for r in job['frames'] if r['semantic']]
    prior=dict(triplets='masks',features='triplets',geometry='features').get(phase)
    if prior:
        previous=json.loads((out/(prior+'_run.json')).read_text())
        assert previous['complete'] and previous['job_sha256']==sha256(job_path)
        for name,h in previous['files_sha256'].items():assert sha256(out/name)==h,name
    try:
        if phase=='masks':
            from ovo.entities.mask_generator import MaskGenerator
            model=MaskGenerator(cfg['semantic']['sam'],device='cuda')
        elif phase=='triplets':
            from ovo.entities.clip_generator import CLIPGenerator
            from ovo.utils.clip_utils import load_clip_model
            # The native extract_clip(return_all=True) path does not call the
            # merger. Keep the text encoder on CPU; fixed text vectors already
            # exist. Only the image encoder needs device memory in this phase.
            model=CLIPGenerator.__new__(CLIPGenerator)
            model.config=cfg['semantic']['clip'];model.device='cuda'
            model.embed_type='learned';model.mask_res=model.config['mask_res']
            model.model,model.tokenizer,model.preprocess,model.clip_dim=load_clip_model(model.config['model_card'],False)
            model.model.eval();model.model.visual.cuda()
        elif phase=='features':
            import yaml
            from ovo.entities.clips_merging import WeightsPredictorMerger
            weights=Path(cfg['semantic']['clip']['weights_predictor_path'])
            model=WeightsPredictorMerger(yaml.safe_load((weights/'hparams.yaml').read_text())['model']).eval()
            model.load_state_dict(torch.load(weights/'model.pt',map_location='cpu',weights_only=False))
            model.cuda()
        else:
            geometry(job,out,torch,status,status_path)
            status.update(complete=True,wall_s=time.monotonic()-start,
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            dump(status_path,status);return
        for record in records:
            began=time.monotonic();name=f"{record['frame_id']:06d}.npz"
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                if phase=='masks':
                    image=rgb(record);seg,masks=model.get_masks(image,record['frame_id'])
                    if seg.numel()==0:
                        seg=torch.full(image.shape[:2],-1,device='cuda',dtype=torch.int32)
                        masks=torch.zeros((0,*image.shape[:2]),device='cuda',dtype=torch.bool)
                    binary=masks.cpu().numpy();n=len(binary);segmentation=seg.cpu().numpy().astype(np.int16)
                    arrays=dict(segmentation=segmentation,masks_packed=np.packbits(binary,axis=2),
                        areas=binary.reshape(n,-1).sum(1).astype(np.int32) if n else np.empty(0,np.int32),
                        exclusive_areas=np.bincount(segmentation[segmentation>=0],minlength=n).astype(np.int32))
                elif phase=='triplets':
                    image=rgb(record);image_tensor=torch.from_numpy(image).cuda().permute(2,0,1)
                    with np.load(out/'masks'/name) as f:masks=np.unpackbits(f['masks_packed'],axis=2)[:,:,:image.shape[1]].astype(bool)
                    pieces=[];compute_dtype='torch.bfloat16'
                    for first in range(0,len(masks),8):
                        value=model.extract_clip(image_tensor,torch.from_numpy(masks[first:first+8]).cuda(),return_all=True)
                        assert value.dtype in (torch.bfloat16,torch.float32)
                        compute_dtype=str(value.dtype)
                        pieces.append(value.float().cpu().numpy())
                    arrays=dict(triplets=np.concatenate(pieces) if pieces else np.empty((0,3,1152),np.float32),compute_dtype=np.array(compute_dtype))
                else:
                    with np.load(out/'triplets'/name) as f:triplets=f['triplets'];compute_dtype=str(f['compute_dtype'])
                    dtype={'torch.bfloat16':torch.bfloat16,'torch.float32':torch.float32}[compute_dtype]
                    pieces=[]
                    for first in range(0,len(triplets),8):
                        value=model(torch.from_numpy(triplets[first:first+8]).to(device='cuda',dtype=dtype))
                        pieces.append(value.float().cpu().numpy())
                    features=np.concatenate(pieces) if pieces else np.empty((0,1152),np.float32)
                    assert np.isfinite(features).all()
                    if len(features):assert np.all(np.abs(np.linalg.norm(features,axis=1)-1)<.02)
                    arrays=dict(features=features)
            np.savez_compressed(folder/name,**arrays);torch.cuda.synchronize()
            status['records'].append(dict(frame_id=record['frame_id'],wall_s=time.monotonic()-began,file=name))
            dump(status_path,status)
            if len(status['records'])%10==0 or len(records)<10:print(phase,job['scene'],len(status['records']),len(records),flush=True)
        status.update(complete=True,wall_s=time.monotonic()-start,files_sha256={str(p.relative_to(out)):sha256(p) for p in folder.glob('*.npz')},
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        dump(status_path,status)
    except Exception as e:
        status.update(error=repr(e),traceback=traceback.format_exc(),wall_s=time.monotonic()-start,
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        dump(status_path,status);raise


def geometry(job,out,torch,status,status_path):
    from ovo.slam.vanilla_mapper import VanillaMapper
    from ovo.utils import geometry_utils as gu
    cfg=job['config'];cam=cfg['cam']
    K=torch.tensor([[cam['fx'],0,cam['cx']],[0,cam['fy'],cam['cy']],[0,0,1]],device='cuda',dtype=torch.float32)
    mapper=VanillaMapper(cfg,K);cache=out/'cache';cache.mkdir();(cache/'frames').mkdir()
    manifest=dict(complete=False,scene=job['scene'],no_association=True,no_semantic_gt=True,
        config=cfg,records=[],descriptor_batch_size=8,support_precision='CUDA bfloat16 autocast',
        geometry_scope='native VanillaMapper on fixed supplied RGB-D poses',job_sha256=status['job_sha256'])
    offset=0;poses=[]
    for record in job['frames']:
        image=rgb(record);depth=cv2.imread(record['depth'],cv2.IMREAD_UNCHANGED).astype(np.float32)/cam['depth_scale']
        pose_np=np.asarray(record['pose'],np.float32);frame=(record['frame_id'],image,depth,pose_np)
        mapper.track_camera(frame);pose=mapper.get_c2w(record['frame_id']);mapper.map(frame,pose)
        if not record['semantic']:continue
        name=f"{record['frame_id']:06d}.npz"
        with np.load(out/'masks'/name) as f:arrays={k:f[k] for k in f.files}
        with np.load(out/'features'/name) as f:arrays['features']=f['features']
        n=len(arrays['features']);xyz,ids,_=mapper.get_map();dt=torch.from_numpy(depth).cuda();seg=torch.from_numpy(arrays['segmentation']).cuda()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            corners=gu.compute_camera_frustum_corners(dt,pose,K);frustum=gu.compute_frustum_point_ids(xyz,corners,device='cuda')
            filtered=gu.depth_filter(dt) if cfg['semantic']['depth_filter'] else dt
            indices,pixels=gu.match_3d_points_to_2d_pixels(filtered,torch.linalg.inv(pose),xyz[frustum],K,cfg['semantic']['match_distance_th'])
            mid=seg[pixels[:,1],pixels[:,0]].cpu().numpy();pid=ids[frustum][indices].reshape(-1).cpu().numpy().astype(np.int32)
        keep=mid>=0;mid=mid[keep];pid=pid[keep];order=np.argsort(mid,kind='stable')
        arrays.update(point_ids=pid[order],offsets=np.r_[0,np.cumsum(np.bincount(mid,minlength=n))].astype(np.int64))
        assert len(np.unique(pid))==len(pid)
        path=cache/'frames'/name;np.savez_compressed(path,**arrays)
        manifest['records'].append(dict(frame_id=record['frame_id'],file=name,masks=n,observation_offset=offset,
            matched_supports=len(pid),map_points=len(xyz),sha256=sha256(path)))
        offset+=n;poses.append(pose_np);dump(cache/'manifest.json',manifest)
        if len(poses)%10==0:print('geometry',job['scene'],len(poses),flush=True)
    saved=mapper.get_map_dict();np.savez_compressed(cache/'geometry.npz',xyz=saved['xyz'].numpy(),colors=saved['color'].numpy())
    np.save(cache/'poses.npy',np.stack(poses))
    for name in ['text_features.npy','vocabulary.json']:shutil.copyfile(Path(job['reference_cache'])/name,cache/name)
    manifest.update(complete=True,frames=len(job['frames']),frames_cached=len(poses),observations=offset,map_points=len(saved['xyz']),
        files_sha256={p.name:sha256(p) for p in cache.iterdir() if p.is_file() and p.name!='manifest.json'})
    dump(cache/'manifest.json',manifest)
    status['files_sha256']={'cache/manifest.json':sha256(cache/'manifest.json')}


def validate_smoke(job_path):
    job=json.loads(job_path.read_text());assert job['kind']=='replica_smoke'
    out=Path(job['output'])/'cache';expected=Path(job['reference_cache']);m=json.loads((out/'manifest.json').read_text())
    ref=json.loads((expected/'manifest.json').read_text());by_frame={r['frame_id']:r for r in ref['records']};checks=[]
    for row in m['records']:
        reference=by_frame[row['frame_id']];a=np.load(out/'frames'/row['file']);b=np.load(expected/'frames'/reference['file'])
        exact={k:bool(np.array_equal(a[k],b[k])) for k in ['segmentation','masks_packed','areas','exclusive_areas','point_ids','offsets']}
        error=float(np.max(np.abs(a['features']-b['features'])))
        checks.append(dict(frame_id=row['frame_id'],exact=exact,feature_max_abs_error=error))
        assert all(exact.values()) and error<=1e-6,checks[-1]
    a=np.load(out/'geometry.npz');b=np.load(expected/'geometry.npz');n=len(a['xyz'])
    np.testing.assert_array_equal(a['xyz'],b['xyz'][:n]);np.testing.assert_array_equal(a['colors'],b['colors'][:n])
    result=dict(complete=True,checks=checks,geometry_prefix_exact=True,job_sha256=sha256(job_path),
        scope='two semantic frames and three geometry frames of previously viewed Replica office_0; proves staged frontend equivalence on this smoke, not all inputs')
    dump(OUT/'split_frontend_validation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--phase',choices=PHASES+('validate',),required=True);args=p.parse_args()
    if args.phase=='validate':validate_smoke(args.job)
    else:process(args.job,args.phase)
