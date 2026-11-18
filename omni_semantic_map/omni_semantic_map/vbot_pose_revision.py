"""Real archived LIVO pose-revision replay; corrected poses are NOT object GT.

Retains source observation IDs and re-extracts measured depth supports/features.
Compares before, position-only update, graph revision, and clean corrected replay.
"""
import argparse,copy,json,time
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from .flat_benchmark import sha256
from .objects import transform_points,project
from .sensor_features import camera_pose_from_anchor,pool_measured_tokens
from .witness_graph import WitnessConfig,WitnessGraph


def prepare(input_dir,maps,inventory,model_dir,weights,output,capture=None,device='cuda:0'):
    import cv2,torch,yaml,open3d as o3d
    from PIL import Image
    from transformers import Dinov2Model
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    # Reuse the original calibration, projection and support-selection routines.
    from build_semantic_map import depth_support
    from map_quality import trajectory_poses
    from revisit_geometry import poses_from_rows
    output.mkdir(parents=True,exist_ok=False);(output/'features').mkdir()
    doc=json.loads((input_dir/'objects.json').read_text());source=Path(doc['keyframe_source'])
    index=source/'keyframes.csv';assert sha256(index)==doc['keyframe_index_sha256']
    rows=np.genfromtxt(index,delimiter=',',names=True);native=poses_from_rows(rows)
    trajectory=np.loadtxt(source/'trajectory_final.tum');revision=np.load(maps/'keyframe_poses.npz')
    assert np.array_equal(rows['stamp'],revision['stamps'])
    result=json.loads((maps/'result.json').read_text());assert result['source_index_sha256']==sha256(index)
    params_path=source.parent/'livo.yaml';params=yaml.safe_load(params_path.read_text())['/**']['ros__parameters']['extrin_calib']
    il=np.eye(4);il[:3,:3]=np.array(params['extrinsic_R']).reshape(3,3);il[:3,3]=params['extrinsic_T']
    cl=np.eye(4);cl[:3,:3]=np.array(params['Rcl']).reshape(3,3);cl[:3,3]=params['Pcl'];ci=cl@np.linalg.inv(il)
    calib=next(x['calibration'] for x in json.loads(inventory.read_text()) if x['name']==doc['bag'])
    intrinsic=np.array(calib['K'],float).reshape(3,3);intrinsic[0]*=640/calib['width'];intrinsic[1]*=360/calib['height']
    torch.set_num_threads(4);cv2.setNumThreads(1)
    dino=Dinov2Model.from_pretrained(str(model_dir),local_files_only=True).to(device).eval().requires_grad_(False)
    car_mode=doc['bag']=='rosbag2245';detector=None
    if car_mode:
        detector=maskrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None,min_size=360,max_size=640)
        detector.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));detector=detector.to(device).eval()
    cap=np.genfromtxt(capture/'frames.csv',delimiter=',',names=True) if capture else None
    grouped=defaultdict(list);source_obs=[]
    for obj in doc['objects']:
        for ob in obj['observations']:
            o=copy.deepcopy(ob);o['source_observation_id']=len(source_obs);source_obs.append(o);grouped[o['keyframe_id']].append(o)
    hashes={str(p.resolve()):sha256(p) for p in [input_dir/'objects.json',index,source/'trajectory_final.tum',maps/'keyframe_poses.npz',maps/'result.json',params_path,inventory,model_dir/'model.safetensors']}
    if car_mode:hashes[str(weights.resolve())]=sha256(weights)
    mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    manifest=dict(complete=False,bag=doc['bag'],gt_available=False,source_observations=len(source_obs),
      source_frames=len(doc['frames']),map_revision_scope='real archived offline pose-graph before/after, applied as one batch after replay; not live loop latency',
      support_scope='static source detections fixed; car masks re-inferred from stored JPEG with original weights and matched to source boxes; landmark support uses original past-1s scans',
      feature_model='frozen DINOv2-small, 448x252 full-view resize, 18x32 patches',device=device,device_name=torch.cuda.get_device_name(device),
      source_sha256={p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name('sensor_features.py')]})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2));all_obs=[];frames=[];failures=[];before=[];after=[];recovery=[];started=time.monotonic()
    for k,record in enumerate(f for f in doc['frames'] if 'skipped' not in f):
        kid=record['keyframe_id'];framepath=input_dir/'frames'/f'keyframe_{kid:05d}.jpg'
        image=cv2.imread(str(framepath));assert image.shape==(360,640,3)
        hashes[str(framepath.resolve())]=sha256(framepath)
        camera=ci@np.linalg.inv(trajectory_poses(trajectory,[record['image_stamp']])[0])@native[kid]
        before.append(camera_pose_from_anchor(revision['before'][kid],camera));after.append(camera_pose_from_anchor(revision['after'][kid],camera))
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        tensor=torch.from_numpy(np.asarray(Image.fromarray(rgb).resize((448,252),Image.Resampling.BICUBIC)).copy()).permute(2,0,1)[None].float().to(device)/255
        with torch.inference_mode():tokens=dino(pixel_values=(tensor-mean)/std).last_hidden_state[0,1:].reshape(18,32,-1).cpu().numpy()
        if car_mode:
            with torch.inference_mode():pred=detector([torch.from_numpy(rgb.copy()).permute(2,0,1).float().to(device)/255])[0]
            pred={name:value.cpu().numpy() for name,value in pred.items()}
            path=source/'keyframes'/f'{kid}.pcd';hashes[str(path.resolve())]=sha256(path)
            points=np.asarray(o3d.io.read_point_cloud(str(path)).points)
        elif grouped[kid]:
            scans=cap[(cap['stamp']<=record['stamp']+1e-5)&(cap['stamp']>=record['stamp']-1)]
            arrays=[]
            for scan in scans:
                path=capture/f'{int(scan["id"])}.xyz';hashes[str(path.resolve())]=sha256(path);arrays.append(np.fromfile(path,np.float32).reshape(-1,3))
            points=transform_points(np.concatenate(arrays),np.linalg.inv(native[kid]))
        group=[]
        if grouped[kid]:pixels,depth,valid=project(points,camera,intrinsic)
        for original in grouped[kid]:
            mask=None;box=np.array(original['box'])
            if car_mode:
                boxes=pred['boxes'];low=np.maximum(boxes[:,:2],box[:2]);high=np.minimum(boxes[:,2:],box[2:]);inter=np.prod(np.maximum(high-low,0),axis=1);area=np.prod(boxes[:,2:]-boxes[:,:2],axis=1);iou=inter/(area+np.prod(box[2:]-box[:2])-inter+1e-8);iou[pred['labels']!=3]=0
                if len(iou)==0 or iou.max()<.5:
                    failures.append(dict(source_observation_id=original['source_observation_id'],reason='mask redetection does not match source box'));continue
                j=int(iou.argmax());mask=cv2.erode((pred['masks'][j,0]>.7).astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
            elif original['label']=='fire_hydrant':
                hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);mask=((hsv[:,:,0]<15)|(hsv[:,:,0]>150))&(hsv[:,:,1]>95)&(hsv[:,:,2]>45)
            selected=depth_support(points,pixels,depth,valid,box,mask)
            if selected is None:
                failures.append(dict(source_observation_id=original['source_observation_id'],reason='recovered measured support has insufficient points'));continue
            local=transform_points(points[selected],camera);uv=pixels[selected]
            descriptor,anchors,features=pool_measured_tokens(tokens,uv,local,image.shape)
            o={name:copy.deepcopy(original[name]) for name in ['label','score','stamp','image_stamp','appearance','box','source_observation_id']}
            o.update(observation_id=len(all_obs),keyframe_id=k,source_keyframe_id=kid,evidence=str(framepath.resolve()),
              local_position=transform_points([original['local_position']],camera)[0].tolist(),feature_file=f'features/{len(all_obs):06d}.npz')
            delta=float(np.linalg.norm(np.median(points[selected],axis=0)-original['local_position']))
            recovery.append(dict(observation_id=len(all_obs),source_observation_id=o['source_observation_id'],original_support=original['support_points'],recovered_support=len(selected),center_difference_m=delta))
            np.savez_compressed(output/o['feature_file'],descriptor=descriptor,local_xyz=anchors,local_features=features.astype(np.float16),support_xyz=local.astype(np.float32),pixels=uv.astype(np.float32))
            all_obs.append(o);group.append(o['observation_id'])
        frames.append(dict(keyframe_id=k,source_keyframe_id=kid,image_stamp=record['image_stamp'],observations=group))
        if k%50==0:print(doc['bag'],k,'observations',len(all_obs),'failures',len(failures),flush=True)
    np.savez_compressed(output/'poses.npz',before=np.array(before),after=np.array(after))
    (output/'observations.json').write_text(json.dumps(all_obs,indent=2));(output/'frames.json').write_text(json.dumps(frames,indent=2))
    (output/'support_recovery.json').write_text(json.dumps(dict(recovered=recovery,failures=failures),indent=2))
    (output/'input_sha256.json').write_text(json.dumps(hashes,indent=2))
    manifest.update(complete=True,frames=len(frames),observations=len(all_obs),failed_supports=len(failures),wall_s=time.monotonic()-started,
      files_sha256={str(p.relative_to(output)):sha256(p) for p in output.rglob('*') if p.is_file() and p.name!='manifest.json'})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2));return manifest


def pair_partition_change(a,b):
    def c2(n):return n*(n-1)//2
    together=sum(c2(n) for n in Counter(zip(map(int,a),map(int,b))).values())
    removed=sum(c2(n) for n in Counter(map(int,a)).values())-together
    added=sum(c2(n) for n in Counter(map(int,b)).values())-together
    return dict(separated_observation_pairs=removed,joined_observation_pairs=added)


def experiment(cache,output):
    meta=json.loads((cache/'manifest.json').read_text());assert meta['complete']
    for name,h in meta['files_sha256'].items():assert sha256(cache/name)==h,name
    for p,h in json.loads((cache/'input_sha256.json').read_text()).items():assert sha256(p)==h,p
    output.mkdir(parents=True,exist_ok=False);obs=json.loads((cache/'observations.json').read_text());frames=json.loads((cache/'frames.json').read_text());poses=np.load(cache/'poses.npz')
    for o in obs:
        with np.load(cache/o['feature_file']) as f:
            for key in ['descriptor','local_xyz','local_features','support_xyz']:o[key]=f[key].astype(np.float32)
    summary=dict(bag=meta['bag'],frames=len(frames),observations=len(obs),gt_available=False,scope='real archived before/after pose replay; partition changes and clean-replay agreement, NOT accuracy or GT recovery',methods={})
    delta=np.linalg.norm(poses['before'][:,:3,3]-poses['after'][:,:3,3],axis=1);summary['camera_pose_correction_m']=dict(median=float(np.median(delta)),max=float(delta.max()))
    for policy,appearance in [('hard',True),('off',True),('witness',True),('witness',False)]:
        name=policy if appearance else 'witness_no_appearance';cfg=WitnessConfig(appearance_weight=.6 if appearance else 0.)
        tracker=WitnessGraph(cfg,policy);clean=WitnessGraph(cfg,policy)
        for frame in frames:
            group=[obs[i] for i in frame['observations']];tracker.add_frame(group,poses['before']);clean.add_frame(group,poses['after'])
        original=tracker.labels();started=time.monotonic();updated=tracker.revise_poses(poses['after']);elapsed=time.monotonic()-started
        assert np.array_equal(updated,clean.labels())
        tracker.revise_poses(poses['before']);assert np.array_equal(original,tracker.labels())
        result=dict(objects_before=len(np.unique(original)),objects_after=len(np.unique(updated)),
          changed_numeric_ids=int(np.count_nonzero(original!=updated)),**pair_partition_change(original,updated),
          correction_wall_s=elapsed,exact_equal_to_clean_corrected_replay=True,roundtrip_restores_before=True,
          position_only_identity_partition_changed=False)
        summary['methods'][name]=result
        d=output/name;d.mkdir();np.savez_compressed(d/'assignments.npz',before=original,after=updated,clean=clean.labels())
        (d/'revision_events.json').write_text(json.dumps(tracker.events[-2:],indent=2))
        print(name,result,flush=True)
    summary['source_sha256']={p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name('witness_graph.py'),Path(__file__).with_name('evidence_association.py')]}
    summary['cache_manifest_sha256']=sha256(cache/'manifest.json');(output/'summary.json').write_text(json.dumps(summary,indent=2));return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run']);p.add_argument('--output',type=Path,required=True)
    for key in ['input','maps','inventory','model','weights','capture','cache']:p.add_argument('--'+key,type=Path)
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.action=='prepare':r=prepare(a.input,a.maps,a.inventory,a.model,a.weights,a.output,a.capture,a.device)
    else:r=experiment(a.cache,a.output)
    print(json.dumps({k:v for k,v in r.items() if k not in ['files_sha256']},indent=2))
if __name__=='__main__':main()
