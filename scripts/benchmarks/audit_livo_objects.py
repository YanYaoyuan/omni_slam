"""Exhaustive retained-object evidence and geometry diagnostics, without GT.

Every retained object gets earliest / largest-area / latest original-frame
evidence. Reprojection agreement is a self-consistency diagnostic, not accuracy:
unmeasured occluders and errors shared by multiple masks remain possible.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
import cv2
import numpy as np
from audit_evidence_retention import ROOT,OUT,REPORT,dump
from omni_semantic_map.flat_benchmark import sha256


def freeze():
    path=OUT/'livo_object_audit_protocol.json';assert not path.exists()
    dump(path,dict(scope=__doc__,source_sha256=sha256(Path(__file__)),
        selection='All final mean-readout retained objects, no size/score filtering. For each: earliest and latest member frames, plus the baseline largest-area selected observation.',
        query_set=['car','pipe','person','fire extinguisher','door'],
        image_review='Assistant visual evidence check; supported / contradicted / ambiguous / insufficient. Not independent human GT. Empty query does not prove absence.',
        projection='Use measured image-epoch transforms and exclusive segmentation. Union all member masks within each distinct frame. Only in-frustum points contribute; no assumed depth or occlusion oracle.',
        identity='Report causal past-observation reassignments separately from final filtering, fragmentation candidates and semantic changes. Do not call these ground-truth ID switches.',
        limits='All point metrics are internal consistency, not surveyed accuracy, full-object dimensions, or ground-truth recall.'))
    dump(REPORT/'stage3/livo_object_audit_protocol.json',json.loads(path.read_text()))


def audit(job_path,cache_name,run_name,report_name):
    protocol=OUT/'livo_object_audit_protocol.json';frozen=json.loads(protocol.read_text());assert sha256(Path(__file__))==frozen['source_sha256']
    job=json.loads(job_path.read_text());root=Path(job['output']);cache=root/cache_name;run=root/run_name
    report=REPORT/'stage3'/report_name;report.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((cache/'manifest.json').read_text());status=json.loads((run/'run.json').read_text());assert manifest['complete'] and status['complete']
    assert sha256(cache/'manifest.json')==status['cache_manifest_sha256']
    for name,h in status['files_sha256'].items():assert sha256(run/name)==h
    rows=json.loads((run/'classification.json').read_text())['objects'];g=np.load(run/'geometry.npz')
    frame_lookup={r['frame_id']:r for r in job['frames']};observation_lookup={}
    for r in manifest['records']:
        for i in range(r['masks']):observation_lookup[r['observation_offset']+i]=(r,i)
    outputs=[];started=time.monotonic()
    for row in rows:
        if not row['predictions']['mean']['retained']:continue
        alias=min(row['observation_ids'])+1;points=g['xyz'][g['instance']==row['object_id']]
        groups=defaultdict(set)
        for oid in row['observation_ids']:
            r,local=observation_lookup[oid];groups[r['frame_id']].add(local)
        first=min(groups);last=max(groups);largest=observation_lookup[row['selected_observation_ids'][0]][0]['frame_id']
        seen=np.zeros(len(points),np.int32);agree=np.zeros(len(points),np.int32);per_frame=[];panels={}
        for frame_id,mask_ids in sorted(groups.items()):
            frame=frame_lookup[frame_id];record=observation_lookup[next(i for i in row['observation_ids'] if observation_lookup[i][0]['frame_id']==frame_id)][0]
            path=cache/'frames'/record['file'];assert sha256(path)==record['sha256']
            with np.load(path) as f:seg=f['segmentation'];object_mask=np.isin(seg,list(mask_ids))
            T=np.asarray(frame['camera_from_tracking'])@np.linalg.inv(np.asarray(frame['pose']))
            cam=points@T[:3,:3].T+T[:3,3];valid=np.isfinite(cam).all(1)&(cam[:,2]>.2);ids=np.flatnonzero(valid)
            pix=cam[ids]@np.asarray(frame['intrinsic']).T;uv=pix[:,:2]/pix[:,2,None]
            h,w=seg.shape;inside=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h);ids=ids[inside];uv=np.floor(uv[inside]).astype(np.int64)
            hit=object_mask[uv[:,1],uv[:,0]];seen[ids]+=1;agree[ids[hit]]+=1
            per_frame.append(dict(frame_id=frame_id,in_frustum_points=len(ids),inside_member_masks=int(hit.sum())))
            if frame_id in [first,largest,last]:
                rgb=cv2.imread(frame['rgb']);overlay=rgb.copy();overlay[object_mask]=(overlay[object_mask]*.55+np.array([0,200,255])*.45).astype(np.uint8)
                # Show final geometry, both agreement (green) and mismatch (red).
                for (x,y),ok in zip(uv,hit):cv2.circle(overlay,(int(x),int(y)),1,(20,240,20) if ok else (30,30,240),-1)
                panels[frame_id]=overlay
        canvas=[]
        for title,fid in [('earliest',first),('largest mask',largest),('latest',last)]:
            panel=panels[fid].copy();cv2.rectangle(panel,(0,0),(panel.shape[1],30),(18,18,18),-1)
            cv2.putText(panel,f'{title} | frame {fid} | {row["predictions"]["mean"]["name"]} ID {alias}',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA);canvas.append(panel)
        picture=report/f'object_{alias:06d}.jpg';assert cv2.imwrite(str(picture),np.concatenate(canvas,axis=1))
        repeated=seen>=2;fractions=agree[repeated]/seen[repeated]
        np.savez_compressed(report/f'object_{alias:06d}_consistency.npz',seen_views=seen,agree_views=agree)
        item=dict(persistent_alias=alias,evaluation_id=row['object_id'],name=row['predictions']['mean']['name'],points=len(points),
            distinct_member_frames=len(groups),first_frame=first,last_frame=last,
            temporal_span_s=(frame_lookup[last]['stamp_ns']-frame_lookup[first]['stamp_ns'])/1e9,
            observed_extent_m=(points.max(0)-points.min(0)).tolist(),surface_centroid_m=points.mean(0).tolist(),
            points_seen_in_two_views=int(repeated.sum()),fraction_of_points_agreeing_in_two_views=float(np.mean(agree>=2)),
            median_mask_agreement_fraction_for_multiview_points=float(np.median(fractions)) if len(fractions) else None,
            fraction_of_multiview_points_agreeing_in_majority=float(np.mean(fractions>=.5)) if len(fractions) else None,
            per_frame=per_frame,evidence_image=str(picture),visual_review='pending')
        outputs.append(item)
    # Nearby same-class overlap candidates flag possible duplicates, not proven errors.
    pairs=[]
    for i,a in enumerate(rows):
        if not a['predictions']['mean']['retained']:continue
        pa=g['xyz'][g['instance']==a['object_id']];alo,ahi=pa.min(0),pa.max(0)
        for b in rows[i+1:]:
            if not b['predictions']['mean']['retained'] or a['predictions']['mean']['name']!=b['predictions']['mean']['name']:continue
            pb=g['xyz'][g['instance']==b['object_id']];blo,bhi=pb.min(0),pb.max(0)
            intersection=np.maximum(np.minimum(ahi,bhi)-np.maximum(alo,blo),0).prod();union=np.prod(ahi-alo)+np.prod(bhi-blo)-intersection
            if intersection>0:pairs.append(dict(a=min(a['observation_ids'])+1,b=min(b['observation_ids'])+1,name=a['predictions']['mean']['name'],box_iou=float(intersection/union)))
    result=dict(complete=True,protocol_sha256=sha256(protocol),prediction_sha256=sha256(run/'run.json'),objects=outputs,
        same_class_box_overlap_candidates=pairs,wall_s=time.monotonic()-started,scope=__doc__,visual_review_complete=False)
    dump(report/'audit.json',result);print('AUDITED',len(outputs),'objects',report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--job',type=Path);p.add_argument('--cache-name',default='cache');p.add_argument('--run-name',default='cg');p.add_argument('--report-name');a=p.parse_args()
    if a.freeze:freeze()
    if a.job:audit(a.job,a.cache_name,a.run_name,a.report_name)
