"""One bounded development follow-up using existing local DINO tokens.

Four scene outputs, not a threshold sweep. The baseline is the stronger frozen
simple-ratio result, not the weaker history2 map. Oracle diagnostics stay out.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import argparse
import time
import resource
import numpy as np
import cv2
from paper_final_common import ROOT, read, dump, sha
import run_reversible_revision as previous
from omni_semantic_map.appearance_partition import AppearanceConfig,partition,validate
from omni_semantic_map.reversible_split import apply_splits,rollback
from omni_semantic_map.visibility_cleanup import CleanupConfig

PARENT=previous.OUT
OUT=ROOT/'build/benchmarks/appearance_revision_20261113'
DINO=ROOT/'build/benchmarks/evidence_retention_20261111/public_dino_v1/features'
ARMS=['appearance_unchecked','appearance_validated']


def freeze():
    assert not (OUT/'protocol.json').exists()
    parent=read(PARENT/'protocol.json')
    scenes=deepcopy(parent['scenes'])
    for s in scenes:
        source=PARENT/'runs'/s['key']/'simple_ratio'
        s['geometry']=str(source/'geometry.npz');s['classification']=str(source/'classification.json')
        for k in ['geometry','classification']:s['immutable_hashes'][s[k]]=sha(s[k])
        s['dino_manifest_sha256']=sha(DINO/s['key']/'manifest.json')
    paths=[Path(__file__),ROOT/'omni_semantic_map/omni_semantic_map/appearance_partition.py',
           ROOT/'omni_semantic_map/test/test_appearance_partition.py',
           ROOT/'omni_semantic_map/omni_semantic_map/reversible_split.py',
           ROOT/'scripts/benchmarks/run_reversible_revision.py',
           ROOT/'omni_semantic_map/omni_semantic_map/visibility_cleanup.py']
    dump(OUT/'protocol.json',dict(scenes=scenes,source_sha256={str(p.relative_to(ROOT)):sha(p)for p in paths},
        created_epoch=time.time(),config=asdict(AppearanceConfig()),arms=ARMS,
        development_only=True,maximum_new_geometry_outputs=4,baseline='Previously scored simple_ratio, fixed for all new arms',
        reason='V1 pair-mask proposals fail to restore instances. Existing geometry has possible coverage, but key errors precede final merge and simultaneous two-mask support is scarce. Test cached patch-level appearance with separately matched per-part masks in later object-local views.',
        budget_amendment='Explicit new development stage after v1 results, not a bugfix allowance or independent test. Four full outputs, one configuration. No automatic retuning.',
        feature='Reconstruct the exact sampled DINOv2 patch bins from original pool procedure; combine overlapping cached normalized tokens without new inference. Unknown patch bins remain unobserved.',
        view_split='For each object, use its first 50% usable views to propose, last 35% to validate; 15% gap. At least 24 depth-visible points defines usable. No GT used.',
        hypothesis='Appearance grouping can expose a finer partition than available same-frame masks. Require held-out feature agreement and per-part mask purity/coverage across two distinct later views.',
        output='Only repartition surviving source points. Classes inherited; no class accuracy claim; exact rollback and observation lineage required.'))
    for p in paths:
        dest=OUT/'sources'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())


def recover_grid(raw_xyz,frame,tokens,pose,cam,resize):
    """Invert deterministic pool bin selection; no invented feature interpolation."""
    iw,ih=resize;tw,th=iw//14,ih//14
    grid=np.zeros((th*tw,384),np.float32);count=np.zeros(th*tw,np.int32)
    inv=np.linalg.inv(pose)
    for i in range(len(frame['offsets'])-1):
        ids=frame['point_ids'][frame['offsets'][i]:frame['offsets'][i+1]]
        camera=raw_xyz[ids]@inv[:3,:3].T+inv[:3,3]
        camera=camera[np.isfinite(camera).all(1)&(camera[:,2]>.2)]
        u=camera[:,0]*cam['fx']/camera[:,2]+cam['cx'];v=camera[:,1]*cam['fy']/camera[:,2]+cam['cy']
        inside=(u>=0)&(u<cam['W'])&(v>=0)&(v<cam['H'])
        bins=np.floor(v[inside]*th/cam['H']).astype(int)*tw+np.floor(u[inside]*tw/cam['W']).astype(int)
        unique=np.unique(bins)
        chosen=np.unique(np.linspace(0,len(unique)-1,min(24,len(unique))).astype(int))
        selected=unique[chosen]
        values=tokens['local_features'][tokens['offsets'][i]:tokens['offsets'][i+1]].astype(np.float32)
        assert len(selected)==len(values)
        known=count[selected]>0
        if known.any():
            np.testing.assert_allclose(grid[selected[known]]/count[selected[known],None],values[known],atol=2e-3,rtol=0)
        grid[selected]+=values;count[selected]+=1
    return grid/np.maximum(count[:,None],1),count>0,tw,th


def predict(key):
    protocol=previous.verify();scene=next(s for s in protocol['scenes']if s['key']==key)
    target=OUT/'runs'/key;target.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);start=time.perf_counter()
    for k in ['geometry','classification']:assert sha(scene[k])==scene['immutable_hashes'][scene[k]]
    source=np.load(scene['geometry']);xyz,labels=source['xyz'],source['instance'];rows=read(scene['classification'])['objects']
    views,_,foreground,_=previous.build_views(scene,xyz,labels,rows,CleanupConfig())
    timelines={}
    for vi,v in enumerate(views):
        objects,n=np.unique(labels[v.indices],return_counts=True)
        for oid,number in zip(objects,n):
            if oid>0 and number>=24:timelines.setdefault(int(oid),[]).append(vi)
    train_frames={o:set(t[:int(len(t)*.5)]) for o,t in timelines.items()}
    val_frames={o:set(t[int(len(t)*.65):]) for o,t in timelines.items()}
    assert all(not train_frames[o]&val_frames[o] for o in timelines)
    sums=[np.zeros((len(xyz),384),np.float32)for _ in range(2)]
    counts=[np.zeros(len(xyz),np.uint16)for _ in range(2)]
    cache=Path(scene['cache']);manifest=read(cache/'manifest.json');cam=manifest['config']['cam'];poses=np.load(cache/'poses.npy')
    raw_xyz=np.load(cache/'geometry.npz')['xyz'];dpath=DINO/key
    assert sha(dpath/'manifest.json')==scene['dino_manifest_sha256']
    dm=read(dpath/'manifest.json');drecs={r['frame_id']:r for r in dm['records']}
    coverage=[]
    for vi,(rec,view)in enumerate(zip(manifest['records'],views)):
        dr=drecs[rec['frame_id']];assert sha(dpath/dr['file'])==dr['sha256']
        with np.load(cache/'frames'/rec['file'])as z:frame={k:z[k]for k in ['point_ids','offsets']}
        with np.load(dpath/dr['file'])as z:tokens={k:z[k]for k in ['local_features','offsets']}
        grid,known,tw,th=recover_grid(raw_xyz,frame,tokens,poses[vi],cam,dm['resize'])
        inv=np.linalg.inv(poses[vi]);camera=xyz[view.indices]@inv[:3,:3].T+inv[:3,3]
        u=camera[:,0]*cam['fx']/camera[:,2]+cam['cx'];v=camera[:,1]*cam['fy']/camera[:,2]+cam['cy']
        bins=np.floor(v*th/cam['H']).astype(int)*tw+np.floor(u*tw/cam['W']).astype(int)
        valid=(bins>=0)&(bins<len(known));valid[valid]&=known[bins[valid]]
        ids=view.indices[valid];values=grid[bins[valid]]
        for phase,scope in enumerate([train_frames,val_frames]):
            select=np.array([vi in scope.get(int(o),set())for o in labels[ids]],bool)
            sums[phase][ids[select]]+=values[select];counts[phase][ids[select]]+=1
        coverage.append(dict(frame_id=rec['frame_id'],known_patch_bins=int(known.sum()),featured_points=len(ids)))
    proposals=[];decisions=[];cfg=AppearanceConfig(**protocol['config'])
    for oid in sorted(timelines):
        ids=foreground[labels[foreground]==oid]
        proposal=partition(xyz,ids,sums[0],counts[0],cfg)
        if proposal is None:continue
        first=min(train_frames[oid]);proposal.update(object_id=oid,frame_id=views[first].frame_id,
            camera_center=views[first].camera_center,center=xyz[ids].mean(0))
        decision=validate(proposal,labels,sums[1],counts[1],views,sorted(val_frames[oid]),cfg)
        proposals.append(proposal);decisions.append(decision)
    dump(target/'feature_coverage.json',coverage)
    dump(target/'proposals.json',[dict(object_id=p['object_id'],parts=[int((p['child']==k).sum())for k in [0,1]],
        prototype_gap=p['prototype_gap'],feature_coverage=p['feature_coverage'],
        proposal_frame_indices=sorted(train_frames[p['object_id']]),validation_frame_indices=sorted(val_frames[p['object_id']]),
        validation=d)for p,d in zip(proposals,decisions)])
    outputs={}
    for name in ARMS:
        accept=[True]*len(proposals)if name=='appearance_unchecked'else[d['accepted']for d in decisions]
        revised,lineage=apply_splits(labels,proposals,accept);np.testing.assert_array_equal(rollback(revised,lineage),labels)
        folder=target/name;folder.mkdir();np.savez_compressed(folder/'geometry.npz',xyz=xyz,instance=revised)
        np.save(folder/'retained_source_indices.npy',np.arange(len(xyz)))
        dump(folder/'classification.json',dict(objects=previous.classify_partition(rows,revised,views,lineage)))
        dump(folder/'lineage.json',[dict(e,changed_source_indices=e['changed_source_indices'].tolist())for e in lineage])
        outputs[name]=dict(split_objects=len(lineage),points=len(xyz),geometry_sha256=sha(folder/'geometry.npz'))
    dump(target/'run.json',dict(complete=True,no_gt_decoded=True,scene=key,proposals=len(proposals),
        accepted=sum(d['accepted']for d in decisions),methods=outputs,protocol_sha256=sha(OUT/'protocol.json'),
        total_seconds=time.perf_counter()-start,cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        artifacts={str(p.relative_to(target)):sha(p)for p in target.rglob('*')if p.is_file()}))
    print(key,outputs,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','predict','seal','evaluate']);parser.add_argument('--scene',choices=previous.KEYS);args=parser.parse_args()
    previous.OUT=OUT;previous.ARMS=ARMS
    if args.action=='freeze':freeze()
    elif args.action=='predict':predict(args.scene)
    elif args.action=='seal':previous.seal()
    else:
        previous.evaluate()
        summary=read(OUT/'summary.json')
        summary.update(new_geometry_outputs=4,baseline='simple_ratio frozen v1 output',
                       note='Development follow-up with cached DINO, not independent test.')
        dump(OUT/'summary.json',summary)
