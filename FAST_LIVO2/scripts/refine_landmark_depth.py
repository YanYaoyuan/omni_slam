#!/usr/bin/env python3
"""Reuse detections; fuse only past LiDAR scans for static landmark depth.

Moving people use the current keyframe only. Hydrants additionally require red
paint pixels in this recorded scene, to avoid assigning surrounding vegetation
to the small equipment. This is a scene-specific check, not a universal colour
assumption about all firefighting equipment.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import open3d as o3d
import yaml
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2

from build_semantic_map import depth_support
from map_quality import trajectory_poses
from revisit_geometry import poses_from_rows
from omni_semantic_map.objects import associate, project, transform_points


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['input','capture','inventory','maps','output']:
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--person-weights',type=Path,required=True,help='Independent COCO person detector vetoes equipment boxes covering people')
    ap.add_argument('--review',type=Path,help='Explicit visual-review rejection ledger, kept in output provenance')
    args=ap.parse_args()
    review=json.loads(args.review.read_text()) if args.review else []
    if args.output.exists():ap.error('output exists')
    doc=json.loads((args.input/'objects.json').read_text());source=Path(doc['keyframe_source'])
    args.output.mkdir(parents=True);(args.output/'evidence').mkdir()
    torch.set_num_threads(4);cv2.setNumThreads(1)
    person_model=maskrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None,min_size=360,max_size=640)
    person_model.load_state_dict(torch.load(args.person_weights,map_location='cpu',weights_only=True));person_model.eval()
    # All copied images remain independent of the earlier evidence directory.
    shutil.copytree(args.input/'frames',args.output/'frames')
    rows=np.genfromtxt(source/'keyframes.csv',delimiter=',',names=True);native=poses_from_rows(rows)
    traj=np.loadtxt(source/'trajectory_final.tum');cap=np.genfromtxt(args.capture/'frames.csv',delimiter=',',names=True)
    poses=np.load(args.maps/'keyframe_poses.npz')['after']
    cfg=yaml.safe_load((source.parent/'livo.yaml').read_text())['/**']['ros__parameters'];e=cfg['extrin_calib']
    il=np.eye(4);il[:3,:3]=np.array(e['extrinsic_R']).reshape(3,3);il[:3,3]=e['extrinsic_T']
    cl=np.eye(4);cl[:3,:3]=np.array(e['Rcl']).reshape(3,3);cl[:3,3]=e['Pcl'];ci=cl@np.linalg.inv(il)
    entry=next(e for e in json.loads(args.inventory.read_text()) if e['name']==doc['bag']);c=entry['calibration']
    k=np.array(c['K']).reshape(3,3);k[0]*=640/c['width'];k[1]*=360/c['height']
    records=[json.loads(line) for line in (args.input/'predictions.jsonl').read_text().splitlines()]
    frames=[];objects=[];moving=[];counts=Counter();updated=[];accepted_counts=Counter();rejected=[]
    for number,record in enumerate(records):
        i=record['keyframe_id'];stamp=rows[i]['stamp'];image_stamp=record['image_stamp'];filename=f'keyframe_{i:05d}.jpg'
        image=cv2.imread(str(args.input/'frames'/filename));assert image is not None
        annotated=image.copy();camera=ci@np.linalg.inv(trajectory_poses(traj,[image_stamp])[0])@native[i]
        selected_scans=cap[(cap['stamp']<=stamp+1e-5)&(cap['stamp']>=stamp-1.0)]
        assert len(selected_scans)>0
        world=np.concatenate([np.fromfile(args.capture/f'{int(r["id"])}.xyz',np.float32).reshape(-1,3) for r in selected_scans])
        fused=transform_points(world,np.linalg.inv(native[i]));fp,fd,fv=project(fused,camera,k)
        current=np.asarray(o3d.io.read_point_cloud(str(source/'keyframes'/f'{i}.pcd')).points);cp,cd,cv=project(current,camera,k)
        observations=[];transient=[];accepted=[]
        hsv_full=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        red_mask=((hsv_full[:,:,0]<15)|(hsv_full[:,:,0]>150))&(hsv_full[:,:,1]>95)&(hsv_full[:,:,2]>45)
        person_boxes=None
        for d in record['detections']:
            label=d['label'];x1,y1,x2,y2=map(int,d['box']);dynamic=label=='person'
            reviewed=next((r for r in review if r['keyframe_id']==i and r['label']==label and r['decision']=='reject'),None)
            if reviewed:
                rejected.append(dict(keyframe_id=i,reason='visual review: '+reviewed['reason'],detection=d));continue
            if label.startswith('fire_'):
                red_ratio=float(red_mask[y1:y2,x1:x2].mean())
                reason=None
                if red_ratio<.08:reason='insufficient red-paint evidence in this scene'
                else:
                    if person_boxes is None:
                        tensor=torch.from_numpy(cv2.cvtColor(image,cv2.COLOR_BGR2RGB).copy()).permute(2,0,1).float()/255
                        with torch.inference_mode():prediction=person_model([tensor])[0]
                        person_boxes=prediction['boxes'][(prediction['labels']==1)&(prediction['scores']>.5)].numpy()
                    for a,b,c,e in person_boxes:
                        covered=max(0,min(c,x2)-max(a,x1))*max(0,min(e,y2)-max(b,y1))/max((x2-x1)*(y2-y1),1)
                        if covered>.5:reason='independent person detector veto';break
                if reason:
                    rejected.append(dict(keyframe_id=i,reason=reason,detection=d,red_ratio=red_ratio));continue
            accepted.append(d);accepted_counts[label]+=1
            pts,pix,depth,valid=(current,cp,cd,cv) if dynamic else (fused,fp,fd,fv)
            mask=None
            if label=='fire_hydrant':
                mask=red_mask
            ids=depth_support(pts,pix,depth,valid,d['box'],mask)
            d['support_points']=0 if ids is None else len(ids)
            color=(255,160,40) if dynamic else (45,225,65)
            cv2.rectangle(annotated,(x1,y1),(x2,y2),color,2)
            cv2.putText(annotated,f"{label} {d['score']:.2f}",(x1,max(y1-4,13)),cv2.FONT_HERSHEY_SIMPLEX,.42,color,1,cv2.LINE_AA)
            if ids is None:continue
            counts[label]+=1
            hsv=cv2.cvtColor(image[y1:y2,x1:x2],cv2.COLOR_BGR2HSV)
            hist=cv2.calcHist([hsv],[0,1,2],None,[8,4,4],[0,180,0,256,0,256]).ravel();hist/=max(hist.sum(),1)
            ob=dict(keyframe_id=i,stamp=float(stamp),image_stamp=image_stamp,image_offset_ms=(image_stamp-stamp)*1000,
                label=label,score=d['score'],box=d['box'],local_position=np.median(pts[ids],axis=0).tolist(),
                support_points=len(ids),appearance=hist.tolist(),evidence='evidence/'+filename,
                depth_method='current_keyframe' if dynamic else 'past_1s_frontend_scans',
                capture_scan_ids=[] if dynamic else selected_scans['id'].astype(int).tolist())
            if dynamic:ob['persistence']='transient_only';transient.append(ob)
            else:observations.append(ob)
            for u,v in pix[ids].astype(int):cv2.circle(annotated,(u,v),2,(0,0,255),-1)
        associate(objects,observations,poses)
        for ob in observations:
            cv2.putText(annotated,ob['object_id'],(int(ob['box'][0]),min(350,int(ob['box'][3])+12)),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,255,255),1,cv2.LINE_AA)
        moving.extend(transient)
        cv2.imwrite(str(args.output/'evidence'/filename),annotated)
        frames.append(dict(keyframe_id=i,stamp=float(stamp),image_stamp=image_stamp,detections=len(accepted),supported_observations=len(observations),transient_observations=len(transient)))
        record['detections']=accepted;updated.append(record)
        if number%50==0:print(number,len(records),dict(counts),flush=True)
    doc=copy.deepcopy(doc);doc['raw_detection_counts']=doc['detection_counts'];doc.update(objects=objects,dynamic_observations=moving,frames=frames,depth_supported_counts=dict(counts),detection_counts=dict(accepted_counts))
    doc['parameters']['static_depth']='past one second of original frontend scans, transformed to keyframe coordinates'
    doc['parameters']['hydrant_gate']='red-paint HSV mask for this scene; same minimum 8 supported points; no inferred depth when absent'
    doc['parameters']['fire_detection_filter']='at least 8% red paint for this scene; reject if independent COCO person box covers more than half of equipment proposal'
    doc['person_veto_weights_sha256']=hashlib.sha256(args.person_weights.read_bytes()).hexdigest()
    doc['visual_review']=review
    doc['capture_index_sha256']=hashlib.sha256((args.capture/'frames.csv').read_bytes()).hexdigest()
    (args.output/'objects.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2))
    (args.output/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in updated))
    (args.output/'rejected_equipment.json').write_text(json.dumps(rejected,indent=2))
    print('CONFIRMED',dict(Counter(o['label'] for o in objects if o['confirmed'])),'DEPTH',dict(counts),flush=True)


if __name__=='__main__':main()
