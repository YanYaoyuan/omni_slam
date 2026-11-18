"""One frozen development configuration for view-dependent mask incompleteness."""
import argparse
from dataclasses import asdict
from pathlib import Path
import time
import cv2
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from omni_semantic_map.mask_recall_revision import RecallConfig,revise
from omni_semantic_map.visibility_cleanup import CleanupConfig,project_visibility

PARENT=ROOT/'build/benchmarks/reversible_revision_20261113'
OUT=ROOT/'build/benchmarks/mask_recall_revision_20261113'
ARMS=['pooled','per_view','leave_point_out']


def freeze():
    assert not (OUT/'protocol.json').exists()
    parent=read(PARENT/'protocol.json')
    sources=[Path(__file__),ROOT/'omni_semantic_map/omni_semantic_map/mask_recall_revision.py',
             ROOT/'omni_semantic_map/test/test_mask_recall_revision.py',
             ROOT/'omni_semantic_map/omni_semantic_map/visibility_cleanup.py']
    dump(OUT/'protocol.json',dict(scenes=parent['scenes'],config=asdict(RecallConfig()),
        visibility_config=parent['visibility_config'],arms=ARMS,created_epoch=time.time(),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        development_only=True,maximum_new_outputs=6,
        hypothesis='Model each mask as an incomplete observation of a source object. Estimate view recall and latent membership rather than interpreting every absence as an equally reliable negative.',
        controls='Pooled recall versus per-view recall versus per-view recall excluding the evaluated point from its own sufficient statistics. Same input, class/identity, priors, iteration limit, and likelihood threshold.',
        caveat='Bernoulli conditional independence is a model assumption. Membership scores are not calibrated confidence. Temporal correlations and wrong object identities remain limitations. This stage does not claim novelty or online performance.',
        selection='Two already exposed development scenes, after diagnostic analysis. No threshold sweep. All six outputs sealed before scoring. Simple ratio and SOR remain mandatory comparators.',
        decision='Assess complete-map PQ and gained/lost matches versus frozen simple ratio in both scenes. A stable benefit must survive independent tests and robustness checks before replacing Ours. No new learned or foundation-model inference.'))
    for p in sources:
        dest=OUT/'sources'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())


def verify():
    p=read(OUT/'protocol.json')
    for path,h in p['source_sha256'].items():assert sha(ROOT/path)==h,path
    return p


def predict(key):
    protocol=verify();scene=next(s for s in protocol['scenes'] if s['key']==key)
    target=OUT/'runs'/key;target.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();cv2.setNumThreads(1)
    for k in ['geometry','classification']:assert sha(scene[k])==scene['immutable_hashes'][scene[k]]
    g=np.load(scene['geometry']);xyz,labels=g['xyz'],g['instance'];rows=read(scene['classification'])['objects']
    retained=[r['object_id'] for r in rows if r['predictions']['mean']['retained']]
    fg=np.flatnonzero(np.isin(labels,retained));flabels=labels[fg]
    owner={o:r['object_id'] for r in rows for o in r['observation_ids']}
    cache=Path(scene['cache']);manifest=read(cache/'manifest.json');cam=manifest['config']['cam']
    for name,h in manifest['files_sha256'].items():assert sha(cache/name)==h
    poses=np.load(cache/'poses.npy');cfg=CleanupConfig(**protocol['visibility_config'])
    observations=np.zeros((len(poses),len(fg)),np.uint8)
    for fi,(rec,dep) in enumerate(zip(manifest['records'],scene['depths'])):
        path=cache/'frames'/rec['file'];assert sha(path)==rec['sha256'] and sha(dep['path'])==dep['sha256']
        assert rec['frame_id']==dep['frame_id']
        depth=cv2.imread(dep['path'],cv2.IMREAD_UNCHANGED).astype(np.float32)/cam['depth_scale']
        with np.load(path) as z:masks=np.unpackbits(z['masks_packed'],axis=2)[:,:,:cam['W']].astype(bool)
        ids=np.arange(rec['observation_offset'],rec['observation_offset']+len(masks))
        owners=np.array([owner.get(int(o),0) for o in ids])
        x,y,_,visible=project_visibility(xyz[fg],poses[fi],cam,depth,cfg)
        for oid in np.intersect1d(owners,retained):
            local=np.flatnonzero(visible&(flabels==oid))
            union=masks[owners==oid].any(0)
            observations[fi,local]=1+union[y[local],x[local]].astype(np.uint8)
        if fi%50==0:print(key,fi,'frames',flush=True)
    counts=np.load(ROOT/'build/benchmarks/ratio_evidence_audit_20261113'/key/'evidence.npz')['counts']
    np.testing.assert_array_equal((observations==2).sum(0),counts[0,fg])
    np.testing.assert_array_equal((observations>0).sum(0),counts[1,fg])
    np.savez_compressed(target/'observations.npz',observations=observations,source_indices=fg)
    results={}
    for mode in ARMS:
        tic=time.monotonic();reject=np.zeros(len(fg),bool);details=[]
        for oid in retained:
            local=np.flatnonzero(flabels==oid)
            result=revise(observations[:,local],mode,RecallConfig(**protocol['config']))
            reject[local]=result['reject']
            details.append(dict(object_id=int(oid),points=len(local),removed=int(result['reject'].sum()),
                iterations=result['iterations'],maximum_change=result['maximum_change'],
                view_recall=result['view_recall'].tolist()))
        keep=np.ones(len(xyz),bool);keep[fg]=~reject
        dest=target/mode;dest.mkdir()
        np.savez_compressed(dest/'geometry.npz',xyz=xyz[keep],instance=labels[keep])
        np.save(dest/'retained_source_indices.npy',np.flatnonzero(keep))
        (dest/'classification.json').write_bytes(Path(scene['classification']).read_bytes())
        dump(dest/'objects.json',details)
        results[mode]=dict(removed=int(reject.sum()),points=int(keep.sum()),kernel_seconds=time.monotonic()-tic)
    dump(target/'run.json',dict(complete=True,scene=key,no_gt_decoded=True,methods=results,
        protocol_sha256=sha(OUT/'protocol.json'),elapsed_seconds=time.monotonic()-start,
        artifacts={str(p.relative_to(target)):sha(p) for p in target.rglob('*') if p.is_file()}))
    print(key,results,flush=True)


def seal():
    p=verify();assert not (OUT/'predictions_complete.json').exists()
    runs={}
    for scene in p['scenes']:
        path=OUT/'runs'/scene['key']/'run.json';run=read(path);assert run['complete']
        for name,h in run['artifacts'].items():assert sha(path.parent/name)==h
        runs[scene['key']]=sha(path)
    dump(OUT/'predictions_complete.json',dict(complete=True,runs=runs,protocol_sha256=sha(OUT/'protocol.json'),created_epoch=time.time()))


def evaluate():
    from omni_semantic_map.mesh_instance_evaluation import transfer_labels
    from omni_semantic_map.full_ovo_evaluation import voxel_metrics
    from omni_semantic_map.instance_error_audit import audit_instances,filter_transition
    protocol=verify();gate=read(OUT/'predictions_complete.json');assert gate['complete']
    assert gate['protocol_sha256']==sha(OUT/'protocol.json') and not (OUT/'summary.json').exists()
    results={}
    for scene in protocol['scenes']:
        key=scene['key'];runpath=OUT/'runs'/key/'run.json';assert sha(runpath)==gate['runs'][key]
        for name,h in read(runpath)['artifacts'].items():assert sha(runpath.parent/name)==h
        assert sha(scene['reference'])==scene['immutable_hashes'][scene['reference']]
        ref=np.load(scene['reference']);valid=ref['valid'] if 'valid' in ref.files else np.ones(len(ref['xyz']),bool)
        paths={'simple_ratio':PARENT/'runs'/key/'simple_ratio',
               'history2':Path(scene['geometry']).parent,
               'sor':ROOT/'build/benchmarks/visibility_cleanup_20261112/runs'/key/'geometric_sor'}
        paths.update({a:OUT/'runs'/key/a for a in ARMS});methods={};before=None
        for name,path in paths.items():
            g=np.load(path/'geometry.npz');rows=read(path/'classification.json')['objects']
            kept=[r['object_id'] for r in rows if r['predictions']['mean']['retained']]
            labels=np.where(np.isin(g['instance'],kept),g['instance'],0)
            projected,_=transfer_labels(g['xyz'],labels,ref['xyz']);audit=audit_instances(ref['instance'][valid],projected[valid])
            if before is None:before=audit
            methods[name]=dict(score=voxel_metrics(ref['instance'][valid],projected[valid]),transition=filter_transition(before,audit))
            folder=OUT/'evaluation'/key;dump(folder/(name+'_audit.json'),audit)
            np.savez_compressed(folder/(name+'.npz'),instance=projected)
        results[key]=methods
        print(key,{m:round(v['score']['observed_voxel_instance_pq']*100,3) for m,v in methods.items()},flush=True)
    dump(OUT/'summary.json',dict(complete=True,development_only=True,new_outputs=6,results=results,protocol_sha256=sha(OUT/'protocol.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','predict','seal','evaluate']);p.add_argument('--scene')
    args=p.parse_args()
    if args.action=='predict':predict(args.scene)
    else:globals()[args.action]()
