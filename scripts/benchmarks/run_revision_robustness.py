"""Frozen sensitivity study on fixed maps, not end-to-end SLAM robustness.

Three corruption families, three seeds, two exposed scenes, two algorithms.
All 36 outputs are sealed before any new score is decoded.
"""
from pathlib import Path
import argparse
import time
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from paper_final_common import ROOT,read,dump,sha
from omni_semantic_map.mask_recall_revision import RecallConfig,revise
from omni_semantic_map.visibility_cleanup import CleanupConfig,project_visibility

PARENT=ROOT/'build/benchmarks/mask_recall_revision_20261113'
RATIO=ROOT/'build/benchmarks/reversible_revision_20261113'
OUT=ROOT/'build/benchmarks/revision_robustness_20261113'
KINDS=['mask_boundary','depth_noise','pose_noise']
SEEDS=[1101,1102,1103]
METHODS=['simple_ratio','mask_recall']


def freeze():
    assert not (OUT/'protocol.json').exists()
    parent=read(PARENT/'protocol.json')
    for path,h in parent['source_sha256'].items():assert sha(ROOT/path)==h
    files=[Path(__file__),ROOT/'omni_semantic_map/omni_semantic_map/mask_recall_revision.py',
           ROOT/'omni_semantic_map/omni_semantic_map/visibility_cleanup.py']
    dump(OUT/'protocol.json',dict(scenes=parent['scenes'],config=parent['config'],
        visibility_config=parent['visibility_config'],kinds=KINDS,seeds=SEEDS,methods=METHODS,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},created_epoch=time.time(),
        parent_protocol_sha256=sha(PARENT/'protocol.json'),development_only=True,maximum_new_outputs=36,
        perturbations={'mask_boundary':'Independently erode each input mask with a square radius uniformly sampled from 1,2,3 pixels; same perturbed masks for both algorithms.',
                       'depth_noise':'Independent zero-mean Gaussian noise, sigma 0.02 m per valid depth pixel. Existing unknown pixels remain unknown; nonpositive perturbed values become unknown.',
                       'pose_noise':'Right-multiply each camera-to-world transform by independent local-camera Gaussian translation (sigma 0.01 m per axis) and rotation vector (sigma 1 degree per axis).'},
        randomization='SeedSequence([seed, frame_id, family_index]); source points and original mask ownership fixed. All algorithms share each perturbed observation matrix.',
        scope='Refinement-observation sensitivity only: source map was constructed from original clean inputs. No claim about rebuilding SLAM or frontend under sensor corruption. Scene is the independent unit; seeds are repeated perturbations, not extra scenes.',
        decision='Report every condition and seed, paired differences from clean results, gained/lost matches and mask_recall versus simple_ratio. No retuning or choosing favorable perturbations after scoring. Independent Hypersim tests remain unopened.'))
    for p in files:
        dest=OUT/'sources'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())


def verify():
    p=read(OUT/'protocol.json')
    for path,h in p['source_sha256'].items():assert sha(ROOT/path)==h
    assert sha(PARENT/'protocol.json')==p['parent_protocol_sha256']
    return p


def perturb(depth,pose,masks,kind,seed,frame_id):
    rng=np.random.default_rng(np.random.SeedSequence([seed,int(frame_id),KINDS.index(kind)]))
    if kind=='mask_boundary':
        radii=rng.integers(1,4,len(masks))
        masks=np.array([cv2.erode(m.astype(np.uint8),np.ones((2*int(r)+1,)*2,np.uint8),
                     borderType=cv2.BORDER_CONSTANT,borderValue=0)>0 for m,r in zip(masks,radii)],bool).reshape(masks.shape)
    elif kind=='depth_noise':
        valid=np.isfinite(depth)&(depth>0);depth=depth.copy()
        depth[valid]=np.maximum(0,depth[valid]+rng.normal(0,.02,int(valid.sum())))
        assert not np.any(depth[~valid]>0)
    elif kind=='pose_noise':
        delta=np.eye(4);delta[:3,3]=rng.normal(0,.01,3)
        delta[:3,:3]=Rotation.from_rotvec(rng.normal(0,np.deg2rad(1),3)).as_matrix()
        pose=pose@delta
    else:raise ValueError(kind)
    return depth,pose,masks


def predict(key):
    protocol=verify();scene=next(s for s in protocol['scenes'] if s['key']==key)
    target=OUT/'runs'/key;target.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(1)
    start=time.monotonic()
    for k in ['geometry','classification']:assert sha(scene[k])==scene['immutable_hashes'][scene[k]]
    g=np.load(scene['geometry']);xyz,labels=g['xyz'],g['instance'];rows=read(scene['classification'])['objects']
    kept=[r['object_id'] for r in rows if r['predictions']['mean']['retained']]
    fg=np.flatnonzero(np.isin(labels,kept));flabels=labels[fg]
    owner={o:r['object_id'] for r in rows for o in r['observation_ids']}
    cache=Path(scene['cache']);manifest=read(cache/'manifest.json');cam=manifest['config']['cam']
    for name,h in manifest['files_sha256'].items():assert sha(cache/name)==h
    poses=np.load(cache/'poses.npy');cfg=CleanupConfig(**protocol['visibility_config'])
    conditions=[(k,s) for k in protocol['kinds'] for s in protocol['seeds']]
    observations={c:np.zeros((len(poses),len(fg)),np.uint8) for c in conditions}
    assert len(poses)==len(manifest['records'])==len(scene['depths'])
    events=[]
    for fi,(rec,dep) in enumerate(zip(manifest['records'],scene['depths'])):
        path=cache/'frames'/rec['file'];assert sha(path)==rec['sha256'] and sha(dep['path'])==dep['sha256']
        assert rec['frame_id']==dep['frame_id']
        depth=cv2.imread(dep['path'],cv2.IMREAD_UNCHANGED).astype(np.float32)/cam['depth_scale']
        with np.load(path) as z:masks=np.unpackbits(z['masks_packed'],axis=2)[:,:,:cam['W']].astype(bool)
        ids=np.arange(rec['observation_offset'],rec['observation_offset']+len(masks))
        owners=np.array([owner.get(int(o),0) for o in ids])
        for kind,seed in conditions:
            d,p,m=perturb(depth,poses[fi],masks,kind,seed,rec['frame_id'])
            x,y,_,visible=project_visibility(xyz[fg],p,cam,d,cfg)
            for oid in np.intersect1d(owners,kept):
                local=np.flatnonzero(visible&(flabels==oid));union=m[owners==oid].any(0)
                observations[kind,seed][fi,local]=1+union[y[local],x[local]].astype(np.uint8)
            codes=observations[kind,seed][fi]
            events.append(dict(frame_id=int(rec['frame_id']),kind=kind,seed=seed,
                               visible_owned_points=int((codes>0).sum()),own_hits=int((codes==2).sum())))
        if fi%50==0:print(key,fi,'frames',flush=True)
    dump(target/'frame_events.json',events)
    methods={}
    for (kind,seed),obs in observations.items():
        condition=f'{kind}_s{seed}';folder=target/condition;folder.mkdir()
        np.savez_compressed(folder/'observations.npz',observations=obs,source_indices=fg)
        for method in METHODS:
            tic=time.monotonic()
            if method=='simple_ratio':
                total=(obs>0).sum(0);reject=((obs==2).sum(0)<=.5*total)&(total>0);nonconverged=[]
            else:
                reject=np.zeros(len(fg),bool);nonconverged=[]
                for oid in kept:
                    local=np.flatnonzero(flabels==oid)
                    result=revise(obs[:,local],'leave_point_out',RecallConfig(**protocol['config']))
                    reject[local]=result['reject']
                    if result['maximum_change']>protocol['config']['convergence']:nonconverged.append(int(oid))
            keep=np.ones(len(xyz),bool);keep[fg]=~reject;dest=folder/method;dest.mkdir()
            np.savez_compressed(dest/'geometry.npz',xyz=xyz[keep],instance=labels[keep])
            np.save(dest/'retained_source_indices.npy',np.flatnonzero(keep))
            (dest/'classification.json').write_bytes(Path(scene['classification']).read_bytes())
            methods[condition+'/'+method]=dict(removed=int(reject.sum()),kernel_seconds=time.monotonic()-tic,nonconverged_objects=nonconverged)
    dump(target/'run.json',dict(complete=True,no_gt_decoded=True,methods=methods,
        elapsed_seconds=time.monotonic()-start,protocol_sha256=sha(OUT/'protocol.json'),
        artifacts={str(p.relative_to(target)):sha(p) for p in target.rglob('*') if p.is_file()}))
    print(key,'complete',len(methods),'outputs',flush=True)


def seal():
    p=verify();assert not (OUT/'predictions_complete.json').exists();runs={}
    for s in p['scenes']:
        path=OUT/'runs'/s['key']/'run.json';r=read(path);assert r['complete']
        for name,h in r['artifacts'].items():assert sha(path.parent/name)==h
        runs[s['key']]=sha(path)
    dump(OUT/'predictions_complete.json',dict(complete=True,runs=runs,protocol_sha256=sha(OUT/'protocol.json'),created_epoch=time.time()))


def evaluate():
    from omni_semantic_map.mesh_instance_evaluation import transfer_labels
    from omni_semantic_map.full_ovo_evaluation import voxel_metrics
    from omni_semantic_map.instance_error_audit import audit_instances,filter_transition
    protocol=verify();gate=read(OUT/'predictions_complete.json')
    assert gate['complete'] and gate['protocol_sha256']==sha(OUT/'protocol.json')
    assert not (OUT/'summary.json').exists();results={}
    clean=read(PARENT/'summary.json')['results']
    for s in protocol['scenes']:
        key=s['key'];target=OUT/'runs'/key;assert sha(target/'run.json')==gate['runs'][key]
        for name,h in read(target/'run.json')['artifacts'].items():assert sha(target/name)==h
        assert sha(s['reference'])==s['immutable_hashes'][s['reference']]
        ref=np.load(s['reference']);valid=ref['valid'] if 'valid' in ref.files else np.ones(len(ref['xyz']),bool)
        rows=read(s['classification'])['objects'];kept=[r['object_id'] for r in rows if r['predictions']['mean']['retained']]
        scene_results={}
        for kind in KINDS:
            condition_results=[]
            for seed in SEEDS:
                pair={};audits={}
                for method in METHODS:
                    folder=target/f'{kind}_s{seed}'/method;g=np.load(folder/'geometry.npz')
                    ids=np.load(folder/'retained_source_indices.npy');original=np.load(s['geometry'])
                    np.testing.assert_array_equal(original['xyz'][ids],g['xyz']);np.testing.assert_array_equal(original['instance'][ids],g['instance'])
                    labels=np.where(np.isin(g['instance'],kept),g['instance'],0)
                    projected,_=transfer_labels(g['xyz'],labels,ref['xyz'])
                    audit=audit_instances(ref['instance'][valid],projected[valid]);audits[method]=audit
                    base_method='simple_ratio' if method=='simple_ratio' else 'leave_point_out'
                    clean_audit=read(PARENT/'evaluation'/key/(base_method+'_audit.json'))
                    score=voxel_metrics(ref['instance'][valid],projected[valid])
                    pair[method]=dict(score=score,delta_clean_pq_points=100*(score['observed_voxel_instance_pq']-clean[key][base_method]['score']['observed_voxel_instance_pq']),
                                      transition_from_clean=filter_transition(clean_audit,audit))
                    dump(OUT/'evaluation'/key/f'{kind}_s{seed}'/(method+'_audit.json'),audit)
                condition_results.append(dict(seed=seed,methods=pair,
                    adaptive_vs_ratio=filter_transition(audits['simple_ratio'],audits['mask_recall']),
                    adaptive_minus_ratio_pq_points=100*(pair['mask_recall']['score']['observed_voxel_instance_pq']-pair['simple_ratio']['score']['observed_voxel_instance_pq'])))
            scene_results[kind]=condition_results
            print(key,kind,'paired PQ deltas',[round(v['adaptive_minus_ratio_pq_points'],3) for v in condition_results],flush=True)
        results[key]=scene_results
    dump(OUT/'summary.json',dict(complete=True,development_only=True,new_outputs=36,results=results,
        protocol_sha256=sha(OUT/'protocol.json'),scope=protocol['scope']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','predict','seal','evaluate']);p.add_argument('--scene');args=p.parse_args()
    if args.action=='predict':predict(args.scene)
    else:globals()[args.action]()
