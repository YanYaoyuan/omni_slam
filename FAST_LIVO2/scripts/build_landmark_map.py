#!/usr/bin/env python3
"""Offline static landmarks and transient people over a full large-scene LIVO bag."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import open3d as o3d
import yaml

from build_semantic_map import images_at, depth_support
from landmark_detector import LandmarkDetector, LABELS, THRESHOLDS
from map_quality import trajectory_poses
from revisit_geometry import poses_from_rows
from omni_semantic_map.objects import associate, project


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['inventory','source','maps','model','output']:
        ap.add_argument('--'+name,required=True,type=Path)
    ap.add_argument('--bag',required=True)
    ap.add_argument('--period',type=float,default=3.0)
    ap.add_argument('--resume',action='store_true',help='Reuse saved predictions after an interrupted local run')
    args=ap.parse_args()
    if args.output.exists() and not args.resume:ap.error('output exists; retain prior experiment')
    cached={}
    if args.resume and (args.output/'predictions.jsonl').exists():
        for line in (args.output/'predictions.jsonl').read_text().splitlines():
            record=json.loads(line);cached[record['keyframe_id']]=record
        print('reuse cached detections:',len(cached),flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    for name in ['frames','evidence']:(args.output/name).mkdir(exist_ok=True)
    cv2.setNumThreads(1)
    model=LandmarkDetector(args.model)
    entry=next(e for e in json.loads(args.inventory.read_text()) if e['name']==args.bag)
    cfg=yaml.safe_load((args.source.parent/'livo.yaml').read_text())['/**']['ros__parameters']
    rows=np.atleast_1d(np.genfromtxt(args.source/'keyframes.csv',delimiter=',',names=True))
    native=poses_from_rows(rows)
    traj=np.loadtxt(args.source/'trajectory_final.tum')
    revision=np.load(args.maps/'keyframe_poses.npz');poses=revision['after']
    assert np.array_equal(rows['stamp'],revision['stamps'])
    assert np.array_equal(rows['id'],np.arange(len(rows)))
    index_hash=hashlib.sha256((args.source/'keyframes.csv').read_bytes()).hexdigest()
    assert json.loads((args.maps/'result.json').read_text())['source_index_sha256']==index_hash
    c=entry['calibration'];kraw=np.array(c['K']).reshape(3,3);k=kraw.copy()
    k[0]*=640/c['width'];k[1]*=360/c['height']
    rect=cv2.initUndistortRectifyMap(kraw,np.array(c['D']),np.eye(3),k,(640,360),cv2.CV_32FC1)
    e=cfg['extrin_calib'];il=np.eye(4);il[:3,:3]=np.array(e['extrinsic_R']).reshape(3,3);il[:3,3]=e['extrinsic_T']
    cl=np.eye(4);cl[:3,:3]=np.array(e['Rcl']).reshape(3,3);cl[:3,3]=e['Pcl'];ci=cl@np.linalg.inv(il)
    ids=[];last=-np.inf
    for i,row in enumerate(rows):
        if row['stamp']-last>=args.period:ids.append(i);last=row['stamp']
    objects=[];dynamic=[];frames=[];detections_count=Counter();depth_count=Counter();started=time.monotonic()
    records=(args.output/'predictions.jsonl').open('w')
    for sequence,image_stamp,image in images_at(entry,rows['stamp'][ids],rect):
        i=ids[sequence];row=rows[i];kid=int(row['id']);filename=f'keyframe_{kid:05d}.jpg'
        if abs(image_stamp-row['stamp'])>.04 or not traj[0,0]<=image_stamp<=traj[-1,0]:
            frames.append(dict(keyframe_id=kid,skipped='image_time'));continue
        cv2.imwrite(str(args.output/'frames'/filename),image)
        cloud=o3d.io.read_point_cloud(str(args.source/'keyframes'/f'{kid}.pcd'))
        points=np.asarray(cloud.points)
        image_pose=trajectory_poses(traj,[image_stamp])[0]
        camera_from_keyframe=ci@np.linalg.inv(image_pose)@native[i]
        pixels,depth,valid=project(points,camera_from_keyframe,k)
        if kid in cached:
            assert abs(cached[kid]['image_stamp']-image_stamp)<1e-6
            predictions=cached[kid]['detections']
        else:predictions=model(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        observations=[];moving=[];annotated=image.copy()
        for d in predictions:
            label=d['label'];score=d['score'];box=d['box'];detections_count[label]+=1
            x1,y1,x2,y2=map(int,box)
            color=(255,160,40) if label=='person' else (45,225,65)
            cv2.rectangle(annotated,(x1,y1),(x2,y2),color,2)
            cv2.putText(annotated,f'{label} {score:.2f}',(x1,max(y1-4,13)),cv2.FONT_HERSHEY_SIMPLEX,.42,color,1,cv2.LINE_AA)
            selected=depth_support(points,pixels,depth,valid,box)
            d['support_points']=0 if selected is None else len(selected)
            if selected is None:continue
            depth_count[label]+=1
            crop=image[y1:y2,x1:x2];hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
            hist=cv2.calcHist([hsv],[0,1,2],None,[8,4,4],[0,180,0,256,0,256]).ravel();hist/=max(hist.sum(),1)
            ob=dict(keyframe_id=kid,stamp=float(row['stamp']),image_stamp=image_stamp,
                    image_offset_ms=float(image_stamp-row['stamp'])*1000,label=label,score=score,
                    box=box,local_position=np.median(points[selected],axis=0).tolist(),
                    support_points=len(selected),appearance=hist.tolist(),evidence='evidence/'+filename)
            if label=='person':
                ob['persistence']='transient_only';moving.append(ob)
            else:observations.append(ob)
            for u,v in pixels[selected].astype(int):cv2.circle(annotated,(u,v),2,(0,0,255),-1)
        associate(objects,observations,poses)
        for ob in observations:
            cv2.putText(annotated,ob['object_id'],(int(ob['box'][0]),min(350,int(ob['box'][3])+12)),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,255,255),1,cv2.LINE_AA)
        dynamic.extend(moving)
        cv2.imwrite(str(args.output/'evidence'/filename),annotated)
        frames.append(dict(keyframe_id=kid,stamp=float(row['stamp']),image_stamp=image_stamp,
                           detections=len(predictions),supported_observations=len(observations),
                           transient_observations=len(moving)))
        records.write(json.dumps(dict(keyframe_id=kid,image_stamp=image_stamp,detections=predictions))+'\n');records.flush()
        if sequence%20==0:
            print(json.dumps(dict(frame=sequence+1,total=len(ids),elapsed_s=round(time.monotonic()-started),detections=dict(detections_count),confirmed=Counter(o['label'] for o in objects if o['confirmed']))),flush=True)
    records.close()
    document=dict(schema_version=1,bag=args.bag,frame='offline_livo_map',
        map_revision=hashlib.sha256((args.maps/'keyframe_poses.npz').read_bytes()).hexdigest(),
        keyframe_source=str(args.source.resolve()),keyframe_index_sha256=index_hash,
        source_map=str((args.maps/'map_loop_corrected.pcd').resolve()),
        detector='Grounding DINO tiny / local CPU / open vocabulary',
        model_source=json.loads((args.model/'source.json').read_text()),
        parameters=dict(period_s=args.period,labels=LABELS,score_thresholds=THRESHOLDS,
                        person_policy='transient observations only; excluded from static objects'),
        objects=objects,dynamic_observations=dynamic,frames=frames,
        detection_counts=dict(detections_count),depth_supported_counts=dict(depth_count),
        requested_frames=len(ids),wall_s=time.monotonic()-started,
        limitations=['Door means a detected door/entrance candidate, not a building identity.',
            'Open vocabulary detections are proposals; firefighting equipment needs visual confirmation.',
            'People are not accumulated as static landmarks or reidentified across the route.',
            'Repeated records are not ground-truth unique object counts.',
            'Surface locations from box-filtered LiDAR clusters can include background; no instance masks.',
            'Offline corrected map coordinates differ from the uncorrected incremental video map.'])
    (args.output/'objects.json').write_text(json.dumps(document,ensure_ascii=False,indent=2))
    print(json.dumps(dict(detections=detections_count,depth=depth_count,static_records=len(objects),confirmed=Counter(o['label'] for o in objects if o['confirmed']),seconds=document['wall_s'])),flush=True)


if __name__=='__main__':main()
