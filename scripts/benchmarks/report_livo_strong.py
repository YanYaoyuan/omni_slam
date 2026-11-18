"""Inspect real LIVO objects, measured supports, causal IDs and offline costs."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from audit_evidence_retention import OUT, REPORT, dump
from omni_semantic_map.flat_benchmark import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--cache-name',default='cache');p.add_argument('--run-name',default='cg');p.add_argument('--report-name',default='livo');a=p.parse_args();job=json.loads(a.job.read_text())
    root=Path(job['output']);cache=root/a.cache_name;run=root/a.run_name;out=REPORT/'stage3'/a.report_name;out.mkdir(parents=True,exist_ok=True)
    capture=json.loads(Path(job['capture']).read_text());manifest=json.loads((cache/'manifest.json').read_text());status=json.loads((run/'run.json').read_text())
    assert capture['complete'] and manifest['complete'] and status['complete']
    assert status['cache_manifest_sha256']==sha256(cache/'manifest.json')
    for name,h in status['files_sha256'].items():assert sha256(run/name)==h
    classification=json.loads((run/'classification.json').read_text())['objects'];geometry=np.load(run/'geometry.npz')
    causal=json.loads((run/'causal_diagnostics.json').read_text());raw=np.load(cache/'geometry.npz')
    frames={r['frame_id']:r for r in job['frames']};objects=[]
    for row in classification:
        if not row['predictions']['mean']['retained']:continue
        oid=row['object_id'];points=geometry['xyz'][geometry['instance']==oid];alias=min(row['observation_ids'])+1
        center=points.mean(0);low=points.min(0);high=points.max(0);observation=row['selected_observation_ids'][0]
        rec=next(r for r in manifest['records'] if r['observation_offset']<=observation<r['observation_offset']+r['masks'])
        local=observation-rec['observation_offset'];frame=frames[rec['frame_id']];f=np.load(cache/'frames'/rec['file'])
        image=cv2.imread(frame['rgb']);mask=np.unpackbits(f['masks_packed'][local],axis=1)[:,:image.shape[1]].astype(bool)
        overlay=image.copy();overlay[mask]=(overlay[mask]*.55+np.array([0,200,255])*.45).astype(np.uint8)
        ids=f['point_ids'][f['offsets'][local]:f['offsets'][local+1]];xyz=raw['xyz'][ids]
        world_to_camera=np.asarray(frame['camera_from_tracking'])@np.linalg.inv(np.asarray(frame['pose']))
        cam=xyz@world_to_camera[:3,:3].T+world_to_camera[:3,3];pixels=cam@np.asarray(frame['intrinsic']).T;uv=pixels[:,:2]/pixels[:,2,None]
        valid=np.isfinite(uv).all(1)&(cam[:,2]>.2)&(uv[:,0]>=0)&(uv[:,0]<image.shape[1])&(uv[:,1]>=0)&(uv[:,1]<image.shape[0])
        for x,y in np.floor(uv[valid]).astype(int):cv2.circle(overlay,(x,y),1,(80,255,20),-1)
        label=row['predictions']['mean']['name'];title=f'{label} | ID {alias} | frame {rec["frame_id"]} | {len(ids)} measured supports'
        cv2.rectangle(overlay,(0,0),(overlay.shape[1],26),(18,18,18),-1);cv2.putText(overlay,title,(8,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
        image_path=out/f'object_{alias}_{label.replace(" ","_")}.jpg';assert cv2.imwrite(str(image_path),overlay)
        centers=[o['center'] for r in causal['frames'] for o in r['objects'] if o['persistent_alias']==alias]
        centers=np.asarray(centers);max_motion=float(np.linalg.norm(centers-center,axis=1).max()) if len(centers) else None
        objects.append(dict(evaluation_id=oid,persistent_alias=alias,name=label,center_m=center.tolist(),bounds_min_m=low.tolist(),bounds_max_m=high.tolist(),
            world_frame=frame['world_frame'],unique_frames=row['unique_frames'],observations=len(row['observation_ids']),points=len(points),
            cosine_score=row['predictions']['mean']['score'],evidence_frame=rec['frame_id'],evidence_observation_id=observation,
            evidence_image=str(image_path),original_image=frame['rgb'],original_packet=frame['packet'],
            max_causal_centroid_distance_to_final_m=max_motion,
            center_scope='Centroid and axis-aligned box of observed surfaces, not full object extent or surveyed position accuracy.'))
    queries={name:[o['persistent_alias'] for o in objects if o['name']==name] for name in ['car','pipe','person','fire extinguisher','door']}
    costs={phase:json.loads((root/(phase+'_run.json')).read_text()) for phase in ['masks','triplets','features']}
    times={r['frame_id']:0. for r in manifest['records']}
    for c in costs.values():
        for r in c['records']:times[r['frame_id']]+=r['wall_s']
    for frame,t in zip(manifest['records'],status['frame_times']):times[frame['frame_id']]+=t
    native_visual={r['frame_id'] for r in capture['native_keyframes'] if r['visual_update']}
    received=set(frames);full=job['scene']=='rosbag2245_full'
    resource_paths=[root/(phase+'_resources.json') for phase in costs]
    semantic_rss=max(json.loads(p.read_text())['completed_children_max_rss_bytes'] for p in resource_paths) if all(p.exists() for p in resource_paths) else None
    summary=dict(complete=True,scope='Actual native LIVO replay -> atomic measured observations -> staged frozen strong frontend -> CG/guard -> objects and queries; '+('full temporal recording with source RGB rate; keyframe semantic processing, not online throughput.' if full else '30-second prefix, not full bag or online throughput.'),
        job_sha256=sha256(a.job),capture_sha256=sha256(Path(job['capture'])),cache_sha256=sha256(cache/'manifest.json'),backend_sha256=sha256(run/'run.json'),
        source_duration_s=capture['source_duration_s'],capture_wall_s=capture['wall_s'],native_keyframes=len(capture['native_keyframes']),native_visual_keyframes=len(native_visual),
        received_frames=len(received),missing_visual_frame_ids=sorted(native_visual-received),rejected_packets=capture['rejections'],
        mask_observations=manifest['observations'],matched_supports=sum(r['matched_supports'] for r in manifest['records']),
        masks_without_support=sum(r['masks_without_support'] for r in manifest['records']),raw_map_points=manifest['map_points'],
        final_regions=status['objects'],retained_objects=objects,queries=queries,query_scope='Exact predicted-label lookup. Empty results mean no retained prediction, not proof the target is absent. Image audit pending.',
        causal_identity_reassignment_events=causal['changed_past_observation_events'],identity_scope='Changes to previously assigned observation IDs, not GT ID-switch accuracy. New fragments are not counted as switches.',
        processing=dict(phase_wall_s={k:v['wall_s'] for k,v in costs.items()},backend_wall_s=status['wall_s'],
            measured_per_frame_service_median_s=float(np.median(list(times.values()))),measured_per_frame_service_p95_s=float(np.percentile(list(times.values()),95)),
            service_scope='Sum of staged frontend per-frame wall costs and CG step costs; excludes model loading, queue wait, capture and finalization. Not measured online end-to-end latency.',
            gpu_peak_allocated_bytes=max(v['gpu_peak_allocated_bytes'] for v in costs.values()),gpu_peak_reserved_bytes=max(v['gpu_peak_reserved_bytes'] for v in costs.values()),
            capture_process_peak_rss_bytes=capture['recorder_peak_rss_bytes'],livo_children_peak_rss_bytes=capture['children_max_rss_bytes'],semantic_workers_peak_rss_bytes=semantic_rss),
        live_vocabulary_extension=job['live_ontology_extension'],sparse_support_adaptation=manifest.get('sparse_component_protocol_sha256'),limitations=[('Full-duration input, semantics on emitted visual keyframes rather than every RGB frame.' if full else 'Only a 30-second prefix; full recording validation remains.'),'No GT position or identity annotations.','Sparse z buffer cannot reject occlusion by unmeasured surfaces.','Model inference was staged after capture; online queue latency not yet measured.'])
    dump(out/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from itertools import product,combinations,cycle
    fig=plt.figure(figsize=(11,8),layout='constrained');ax=fig.add_subplot(111,projection='3d')
    # Display crop only; output geometry and measurements are unmodified.
    cloud=raw['xyz'];view=np.isfinite(cloud).all(1)&(np.linalg.norm(cloud[:,:2],axis=1)<35)&(cloud[:,2]>-2)&(cloud[:,2]<8)
    sample=np.flatnonzero(view)[::8];ax.scatter(*cloud[sample].T,c='#b9c3ce',s=.4,alpha=.25)
    for o,c in zip(objects,cycle(['#e4572e','#1679ba','#3b8c46','#9956a7','#b58a16'])):
        pts=geometry['xyz'][geometry['instance']==o['evaluation_id']];ax.scatter(*pts.T,c=c,s=3)
        corners=np.array(list(product(*zip(o['bounds_min_m'],o['bounds_max_m']))))
        for x,y in combinations(corners,2):
            if np.count_nonzero(x!=y)==1:ax.plot(*np.stack([x,y]).T,c=c,lw=1.5)
        ax.text(*o['bounds_max_m'],f"{o['name']} / ID {o['persistent_alias']}",color=c)
    traj=np.array([np.asarray(r['pose'])[:3,3] for r in job['frames']]);ax.plot(*traj.T,c='#242b35',lw=2,label='LIVO keyframes')
    ax.set(xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',title='Actual LIVO + SAM2 / SigLIP / CG\nObserved-surface boxes; odometry coordinates')
    ax.set_box_aspect((1,1,.5));ax.view_init(30,-50);ax.legend();fig.savefig(out/'object_map.png',dpi=160);plt.close(fig)
    print(json.dumps({k:summary[k] for k in ['received_frames','final_regions','queries','processing']},indent=2),flush=True)


if __name__=='__main__':main()
