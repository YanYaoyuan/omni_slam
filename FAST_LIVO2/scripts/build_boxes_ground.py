#!/usr/bin/env python3
"""Recover measured 3D supports, reprojectable boxes and verified ground."""
import argparse,copy,json,hashlib
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
import yaml
from build_semantic_map import depth_support
from map_quality import trajectory_poses
from revisit_geometry import poses_from_rows
from omni_semantic_map.objects import project,transform_points,summarize_object
from omni_semantic_map.geometry import observed_bounds,bounds_corners,local_ground


def voxel(points,size=.15):
    if len(points)==0:return np.empty((0,3))
    _,idx=np.unique(np.floor(points/size).astype(np.int32),axis=0,return_index=True)
    return points[np.sort(idx)]


def main():
    ap=argparse.ArgumentParser()
    for n in ['input','capture','segmentation','maps','inventory','output']:ap.add_argument('--'+n,type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    for n in ['frames','evidence']:(args.output/n).mkdir(exist_ok=True)
    cv2.setNumThreads(1)
    doc=copy.deepcopy(json.loads((args.input/'objects.json').read_text()));source=Path(doc['keyframe_source'])
    missing_masks=[f['keyframe_id'] for f in doc['frames'] if not (args.segmentation/f"keyframe_{f['keyframe_id']:05d}.npz").exists()]
    if missing_masks:raise ValueError(f'Complete image segmentation first; missing {len(missing_masks)} masks')
    rows=np.genfromtxt(source/'keyframes.csv',delimiter=',',names=True);native=poses_from_rows(rows)
    traj=np.loadtxt(source/'trajectory_final.tum');revision=np.load(args.maps/'keyframe_poses.npz');poses=revision['after']
    assert np.array_equal(rows['stamp'],revision['stamps'])
    cap=np.genfromtxt(args.capture/'frames.csv',delimiter=',',names=True)
    cfg=yaml.safe_load((source.parent/'livo.yaml').read_text())['/**']['ros__parameters'];e=cfg['extrin_calib']
    il=np.eye(4);il[:3,:3]=np.array(e['extrinsic_R']).reshape(3,3);il[:3,3]=e['extrinsic_T']
    cl=np.eye(4);cl[:3,:3]=np.array(e['Rcl']).reshape(3,3);cl[:3,3]=e['Pcl'];ci=cl@np.linalg.inv(il)
    entry=next(e for e in json.loads(args.inventory.read_text()) if e['name']==doc['bag']);c=entry['calibration']
    k=np.array(c['K']).reshape(3,3);k[0]*=640/c['width'];k[1]*=360/c['height']
    grouped=defaultdict(list)
    for obj in doc['objects']:
        for ob in obj['observations']:grouped[ob['keyframe_id']].append((obj['id'],ob))
    for i,ob in enumerate(doc['dynamic_observations']):grouped[ob['keyframe_id']].append((f'person_obs_{i:04d}',ob))
    frames=[];ground_all=[];box_count=0;missing=[]
    for number,frame in enumerate(doc['frames']):
        kid=frame['keyframe_id'];stamp=frame['stamp'];name=f'keyframe_{kid:05d}'
        image=cv2.imread(str(args.input/'frames'/(name+'.jpg')))
        sem=np.load(args.segmentation/(name+'.npz'));mask=sem['ground'].copy()
        # Avoid projected ground on observed people, including uncertain segmentation borders.
        for _,ob in grouped[kid]:
            if ob['label']=='person':
                x1,y1,x2,y2=np.rint(ob['box']).astype(int);mask[max(0,y1):min(360,y2),max(0,x1):min(640,x2)]=False
        capture_rows=cap[(cap['stamp']<=stamp+1e-5)&(cap['stamp']>=stamp-1)]
        world=np.concatenate([np.fromfile(args.capture/f'{int(r["id"])}.xyz',np.float32).reshape(-1,3) for r in capture_rows])
        fused=transform_points(world,np.linalg.inv(native[kid]))
        camera=ci@np.linalg.inv(trajectory_poses(traj,[frame['image_stamp']])[0])@native[kid]
        fp,fd,fv=project(fused,camera,k)
        uv=np.nan_to_num(fp,nan=-1).astype(int)
        inside=fv&(uv[:,0]>=0)&(uv[:,0]<640)&(uv[:,1]>=0)&(uv[:,1]<360)&(fd<20)
        seeds=np.zeros(len(fused),bool);seeds[inside]=mask[uv[inside,1],uv[inside,0]]
        ground,planes=local_ground(world,seeds,native[kid,:3,3],seed=kid)
        local=voxel(fused[ground],.12);ground_native=transform_points(local,native[kid]);ground_corrected=transform_points(local,poses[kid]);ground_all.append(ground_corrected)
        np.savez_compressed(args.output/'frames'/(name+'.npz'),ground_local=local.astype(np.float32),ground_native=ground_native.astype(np.float32),ground_corrected=ground_corrected.astype(np.float32))
        current=np.asarray(o3d.io.read_point_cloud(str(source/'keyframes'/f'{kid}.pcd')).points);cp,cd,cv=project(current,camera,k)
        boxes=[]
        for oid,ob in grouped[kid]:
            dynamic=ob['label']=='person';points,pixels,depth,valid=(current,cp,cd,cv) if dynamic else (fused,fp,fd,fv)
            color_mask=None
            if ob['label']=='fire_hydrant':
                hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);color_mask=((hsv[:,:,0]<15)|(hsv[:,:,0]>150))&(hsv[:,:,1]>95)&(hsv[:,:,2]>45)
            ids=depth_support(points,pixels,depth,valid,ob['box'],color_mask)
            if ids is None:missing.append([kid,oid]);continue
            assert np.linalg.norm(np.median(points[ids],axis=0)-ob['local_position'])<1e-5
            b=observed_bounds(points[ids]);ob['local_support_bounds']=b
            corners=bounds_corners(b['min'],b['max'])
            boxes.append(dict(id=oid,label=ob['label'],dynamic=dynamic,
                corners_native=transform_points(corners,native[kid]).tolist(),
                corners_corrected=transform_points(corners,poses[kid]).tolist(),
                support_points=len(ids)))
            box_count+=1
        # Full pixel segmentation is a candidate; green point cloud is geometrically checked.
        annotated=cv2.imread(str(args.input/'evidence'/(name+'.jpg')))
        annotated[mask]=(annotated[mask]*.65+np.array([65,220,90])*.35).astype(np.uint8)
        cv2.imwrite(str(args.output/'evidence'/(name+'.jpg')),annotated)
        frames.append(dict(keyframe_id=kid,stamp=stamp,image_stamp=frame['image_stamp'],boxes=boxes,
                           semantic_ground_points=int(seeds.sum()),verified_ground_points=len(local),planes=planes))
        if number%25==0:print(number,len(doc['frames']),'ground',len(local),'boxes',len(boxes),flush=True)
    for o in doc['objects']:summarize_object(o,poses)
    doc['geometry_definition']='Boxes enclose measured surface supports; dimensions can be partial, not full 3D object detection.'
    doc['ground_model']=json.loads((args.segmentation/'model.json').read_text())
    (args.output/'objects.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2))
    (args.output/'geometry_frames.json').write_text(json.dumps(frames,separators=(',',':')))
    points=voxel(np.concatenate(ground_all),.15)
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points));cloud.paint_uniform_color([.15,.80,.35]);o3d.io.write_point_cloud(str(args.output/'ground_corrected.ply'),cloud)
    summary=dict(frames=len(frames),observation_boxes=box_count,missing_supports=missing,
        static_records_with_boxes=sum(bool(o.get('box3d')) for o in doc['objects']),
        confirmed_with_boxes=sum(o['confirmed'] and bool(o.get('box3d')) for o in doc['objects']),
        ground_points=len(points),ground_frames=sum(f['verified_ground_points']>0 for f in frames),
        semantic_ground_points=sum(f['semantic_ground_points'] for f in frames),
        verified_ground_observations=sum(f['verified_ground_points'] for f in frames),
        frame='offline_livo_map',source_sha256=hashlib.sha256((args.input/'objects.json').read_bytes()).hexdigest())
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)


if __name__=='__main__':main()
