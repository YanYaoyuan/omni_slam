"""Fixed independent-scene predictions. No reference/annotation loading."""
from pathlib import Path
import argparse
from dataclasses import asdict
import time
import cv2
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from omni_semantic_map.visibility_cleanup import CleanupConfig,frame_evidence,rejection_mask,statistical_outliers

OUT=ROOT/'build/benchmarks/visibility_independent_20261112'
VARIANTS=['sor','visibility1','visibility3','visibility1_no_depth']


def clean(cache,source,target,job_path):
    target.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    cfg=CleanupConfig();m=read(cache/'manifest.json');job=read(job_path)
    assert m['complete'];g=np.load(source/'geometry.npz');xyz=g['xyz'];labels=g['instance']
    rows=read(source/'classification.json')['objects'];kept=[r['object_id'] for r in rows if r['predictions']['mean']['retained']]
    source_ids=np.flatnonzero(np.isin(labels,kept));points=xyz[source_ids];identities=labels[source_ids]
    owner={int(i):r['object_id'] for r in rows for i in r['observation_ids']}
    assert len(owner)==sum(len(r['observation_ids']) for r in rows)
    cameras=np.load(cache/'poses.npy');frame_by_id={r['frame_id']:r for r in job['frames']}
    cam=m['config']['cam'];votes=np.zeros((4,len(points)),np.uint16);events=[]
    assert len(cameras)==len(m['records'])
    for index,rec in enumerate(m['records']):
        path=cache/'frames'/rec['file'];assert sha(path)==rec['sha256']
        depth_path=frame_by_id[rec['frame_id']]['depth'];assert sha(depth_path)==job['input_sha256'][depth_path]
        with np.load(path) as z:masks=np.unpackbits(z['masks_packed'],axis=2)[:,:,:cam['W']].astype(bool)
        owners=np.array([owner.get(rec['observation_offset']+i,0) for i in range(rec['masks'])])
        depth=cv2.imread(depth_path,cv2.IMREAD_UNCHANGED).astype(np.float32)/cam['depth_scale']
        tic=time.monotonic();p,n,pf,nf,visible=frame_evidence(points,identities,cameras[index],cam,depth,masks,owners,cfg)
        votes+=np.array([p,n,pf,nf],np.uint16)
        events.append(dict(frame_id=rec['frame_id'],positive=int(p.sum()),negative=int(n.sum()),visible=int(visible.sum()),seconds=time.monotonic()-tic))
    reject=dict(sor=statistical_outliers(points,identities,cfg),visibility1=rejection_mask(votes[0],votes[1],cfg,single_frame=True),
        visibility3=rejection_mask(votes[0],votes[1],cfg),visibility1_no_depth=rejection_mask(votes[2],votes[3],cfg,single_frame=True))
    stats={}
    for name,removed in reject.items():
        folder=target/name;folder.mkdir();retain=np.ones(len(xyz),bool);retain[source_ids[removed]]=False
        np.savez_compressed(folder/'geometry.npz',xyz=xyz[retain],instance=labels[retain])
        np.save(folder/'retained_source_indices.npy',np.flatnonzero(retain))
        (folder/'classification.json').write_bytes((source/'classification.json').read_bytes())
        stats[name]=dict(removed_points=int(removed.sum()),foreground_points=len(points),
            geometry_path=str(folder/'geometry.npz'),classification_path=str(folder/'classification.json'))
    np.savez_compressed(target/'evidence.npz',source_indices=source_ids,positive=votes[0],negative=votes[1],fov_positive=votes[2],fov_negative=votes[3])
    dump(target/'events.json',events)
    dump(target/'run.json',dict(complete=True,no_gt=True,methods=stats,seconds=time.monotonic()-start,config=asdict(cfg),
        source_geometry_sha256=sha(source/'geometry.npz'),source_classification_sha256=sha(source/'classification.json'),
        files_sha256={str(p.relative_to(target)):sha(p) for p in target.rglob('*') if p.is_file() and p.name!='run.json'}))
    print('CLEAN COMPLETE',target,stats,flush=True)


def run(scene,action):
    from visibility_independent_queue import verify
    verify();cache=OUT/'frontend'/scene/'cache';folder=OUT/'maps'/scene
    if action=='cg':
        from paper_final_mapping import run_cg
        run_cg(cache,folder/'cg')
    elif action=='cleanup':clean(cache,folder/'cg/history2',folder/'cleanup',OUT/'jobs'/(scene+'.json'))
    elif action=='maskclustering':
        from maskclustering_shared import run as mc
        mc(cache,folder/'maskclustering')
    elif action=='seal_scene':
        sources={'cg':folder/'cg/cg','history2':folder/'cg/history2','maskclustering':folder/'maskclustering'}
        sources.update({v:folder/'cleanup'/v for v in VARIANTS})
        records={}
        for name,source in sources.items():
            if not (source/'geometry.npz').exists() or not (source/'classification.json').exists():
                records[name]=dict(complete=False,error='Method failed or output missing; preserved without retry.')
                continue
            records[name]=dict(geometry_path=str(source/'geometry.npz'),classification_path=str(source/'classification.json'),
                complete=True,geometry_sha256=sha(source/'geometry.npz'),classification_sha256=sha(source/'classification.json'))
        complete=all(records[n]['complete'] for n in ['cg','history2']+VARIANTS)
        dump(OUT/'predictions'/(scene+'.json'),dict(complete=complete,scene=scene,methods=records,protocol_sha256=sha(OUT/'protocol.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['cg','cleanup','maskclustering','seal_scene']);p.add_argument('--scene',required=True)
    a=p.parse_args();run(a.scene,a.action)
