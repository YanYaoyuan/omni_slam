"""Bounded full sampled-sequence association experiment; no learned-memory claim."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from paper_final_common import ROOT, FROZEN, PY, CGPY, read, dump, sha

OUT = ROOT/'build/benchmarks/vggt_fullmap_v1'
PROBE = ROOT/'build/benchmarks/vggt_object_probe_v1'
ARMS = ['siglip_history2', 'vggt_single', 'vggt_joint']


def freeze():
    assert not OUT.exists(), 'immutable experiment already exists'
    prior=read(ROOT/'build/benchmarks/evidence_retention_20261111/public_dino_v1/protocol.json')
    scenes=[]; hashes={}
    for key in ['replica_office_0','scenenn_014']:
        s=next(s for s in prior['scenes'] if s['key']==key)
        cache=Path(s['cache']); m=read(cache/'manifest.json'); images={r['frame_id']:r for r in s['images']}
        records=[]
        for r in m['records']:
            im=images[r['frame_id']]; path=cache/'frames'/r['file']
            assert sha(path)==r['sha256'] and sha(im['rgb'])==im['sha256']
            hashes[str(path)]=r['sha256']; hashes[im['rgb']]=im['sha256']
            records.append(dict(r,rgb=im['rgb']))
        for name,h in m['files_sha256'].items():
            assert sha(cache/name)==h;hashes[str(cache/name)]=h
        hashes[str(cache/'manifest.json')]=sha(cache/'manifest.json')
        scenes.append(dict(key=key,dataset=s['dataset'],cache=str(cache),records=records,
                           reference=s['reference'],reference_sha256=s['hashes'][s['reference']]))
    files=[Path(__file__),Path(__file__).with_name('vggt_fullmap_backend.py'),
           Path(__file__).with_name('paper_final_mapping.py'),Path(__file__).with_name('run_livo_history_graph.py'),
           FROZEN/'config/cg_rgbd.json',PROBE/'assets.json',PROBE/'code.json']
    files+=list((ROOT/'omni_semantic_map/omni_semantic_map').glob('*.py'))
    dump(OUT/'protocol.json',dict(scenes=scenes,arms=ARMS,input_hashes=hashes,
        source_hashes={str(f):sha(f) for f in files},window=4,layer=23,cuda_allocator_gib=8,
        primary='Observed-reference instance PQ, nearest_5cm; foreground from original SigLIP mean. TP/FP/FN and per-object error signatures.',
        controls='Same SAM masks, geometry, poses, seed0, CG thresholds, history2 and final processing. Only image association descriptor changes. Original SigLIP descriptors remain semantic readout and text_feature source.',
        context='Chronological disjoint windows of 4 sampled images; final short window retained. Single-image control uses same layer/pooling/checkpoint. Offline windowed features can see future images within a window. Cross-window descriptor consistency is not assumed.',
        budget='Two scenes x three arms = six maps. No threshold sweep, no scene replacement. All maps sealed before GT scoring.',
        scope='Exposed development scenes, entire existing 200-sample trajectories each; not every raw image, not independent validation, not official leaderboard. VGGT feature controls are not the untrained object-memory method.',
        decision='Advance to a new frozen independent protocol only if joint exceeds both single and SigLIP in mean PQ and has no net TP loss versus SigLIP. Otherwise inspect saved errors once; no automatic parameter sweep.',
        created_epoch=time.time()))
    print('FROZEN',[(s['key'],len(s['records'])) for s in scenes],flush=True)


def verify(inputs=False):
    p=read(OUT/'protocol.json')
    for f,h in p['source_hashes'].items(): assert sha(f)==h,f
    if inputs:
        for f,h in p['input_hashes'].items(): assert sha(f)==h,f
    return p


def extract():
    p=verify(True); assets=read(PROBE/'assets.json')
    assert sha(assets['checkpoint'])==assets['sha256']
    for f,h in read(PROBE/'code.json')['files'].items(): assert sha(PROBE/'upstream'/f)==h
    sys.path.insert(0,str(PROBE/'upstream'))
    import torch
    import torch.nn.functional as F
    from vggt.models.aggregator import Aggregator
    from vggt.utils.load_fn import load_and_preprocess_images
    torch.set_num_threads(2);torch.manual_seed(0)
    torch.cuda.set_per_process_memory_fraction(p['cuda_allocator_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory)
    weights=torch.load(assets['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
    model=Aggregator().eval()
    model.load_state_dict({k[11:]:v for k,v in weights.items() if k.startswith('aggregator.')},strict=True)
    del weights
    model.cached_layer_indices={p['layer']};model.requires_grad_(False);model.cuda()
    dest=OUT/'features';dest.mkdir(exist_ok=False);started=time.monotonic();files={}
    for s in p['scenes']:
        values={a:[] for a in ARMS[1:]};records=s['records'];covered=0
        for offset in range(0,len(records),p['window']):
            batch=records[offset:offset+p['window']]
            images=load_and_preprocess_images([r['rgb'] for r in batch]).cuda()
            gh,gw=images.shape[-2]//14,images.shape[-1]//14
            assert images.shape[-1]==518
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                layers,start=model(images[None]); joint=layers[p['layer']][0,:,start:].float().cpu()
            del layers
            for i,r in enumerate(batch):
                assert r['observation_offset']==covered
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    layers,start=model(images[i][None,None]); single=layers[p['layer']][0,0,start:].float().cpu()
                del layers
                with np.load(Path(s['cache'])/'frames'/r['file']) as z:
                    binary=np.unpackbits(z['masks_packed'],axis=2)[:,:,:z['segmentation'].shape[1]]
                assert len(binary)==r['masks']
                if len(binary):
                    weights=F.interpolate(torch.from_numpy(binary.astype(np.float32))[:,None],size=(gh,gw),mode='area')[:,0].reshape(len(binary),-1)
                    for arm,tokens in [('vggt_single',single),('vggt_joint',joint[i])]:
                        assert len(tokens)==gh*gw
                        pooled=F.normalize(weights@F.normalize(tokens,dim=-1)/weights.sum(1).clamp_min(1e-12)[:,None],dim=-1).numpy()
                        assert np.isfinite(pooled).all() and (np.linalg.norm(pooled,axis=1)>0).all()
                        values[arm].append(pooled)
                covered+=r['masks']
            del images,joint,single
            dump(dest/'progress.json',dict(scene=s['key'],frames=offset+len(batch),total=len(records),wall_s=time.monotonic()-started))
            print('FEATURES',s['key'],offset+len(batch),covered,flush=True)
        for arm in ARMS[1:]:
            path=dest/f"{s['key']}_{arm}.npy";np.save(path,np.concatenate(values[arm]));files[str(path)]=sha(path)
    dump(dest/'manifest.json',dict(complete=True,gt_accessed=False,files=files,
        protocol_sha256=sha(OUT/'protocol.json'),wall_s=time.monotonic()-started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),device=torch.cuda.get_device_name(0)))


def mapping(scene,arm):
    p=verify();s=next(s for s in p['scenes'] if s['key']==scene)
    assert arm in ARMS
    seal=read(OUT/'features/manifest.json');assert seal['complete'] and seal['protocol_sha256']==sha(OUT/'protocol.json')
    features=None
    if arm!='siglip_history2':
        path=OUT/'features'/f'{scene}_{arm}.npy';assert sha(path)==seal['files'][str(path)];features=np.load(path)
    from vggt_fullmap_backend import run_cg
    run_cg(Path(s['cache']),OUT/'maps'/scene/arm,features)


def seal():
    p=verify();files={}
    for s in p['scenes']:
        for arm in ARMS:
            folder=OUT/'maps'/s['key']/arm
            assert read(folder/'run.json')['complete']
            assert read(folder/'history2/run.json')['complete']
            for f in folder.rglob('*'):
                if f.is_file():files[str(f)]=sha(f)
    dump(OUT/'predictions_complete.json',dict(complete=True,files=files,protocol_sha256=sha(OUT/'protocol.json')))


def evaluate():
    p=verify();sealed=read(OUT/'predictions_complete.json')
    assert sealed['complete'] and sealed['protocol_sha256']==sha(OUT/'protocol.json')
    for f,h in sealed['files'].items():assert sha(f)==h
    from omni_semantic_map.mesh_instance_evaluation import transfer_labels
    from omni_semantic_map.full_ovo_evaluation import voxel_metrics
    from omni_semantic_map.instance_error_audit import audit_instances,filter_transition
    results=[]
    for s in p['scenes']:
        assert sha(s['reference'])==s['reference_sha256']
        ref=np.load(s['reference']);truth=ref['instance'];valid=ref['valid'] if 'valid' in ref else np.ones(len(truth),bool)
        methods={};audits={}
        for arm in ARMS:
            folder=OUT/'maps'/s['key']/arm/'history2';geo=np.load(folder/'geometry.npz')
            rows=read(folder/'classification.json')['objects']
            retained=[r['object_id'] for r in rows if r['predictions']['mean']['retained']]
            labels=np.where(np.isin(geo['instance'],retained),geo['instance'],0)
            pred,_=transfer_labels(geo['xyz'],labels,ref['xyz'],mode='nearest_5cm')
            audit=audit_instances(truth[valid],pred[valid]);audits[arm]=audit
            dump(OUT/'evaluation'/s['key']/f'{arm}_objects.json',audit)
            methods[arm]=dict(metrics=voxel_metrics(truth[valid],pred[valid]),predicted_errors=audit['predicted_counts'],gt_errors=audit['gt_counts'])
            if arm!=ARMS[0]:methods[arm]['vs_baseline']=filter_transition(audits[ARMS[0]],audit)
        results.append(dict(scene=s['key'],methods=methods))
        print('SCORED',s['key'],{k:v['metrics']['observed_voxel_instance_pq']*100 for k,v in methods.items()},flush=True)
    means={a:float(np.mean([s['methods'][a]['metrics']['observed_voxel_instance_pq'] for s in results])*100) for a in ARMS}
    tp={a:sum(s['methods'][a]['metrics']['tp'] for s in results) for a in ARMS}
    gate=means[ARMS[2]]>max(means[ARMS[0]],means[ARMS[1]]) and tp[ARMS[2]]>=tp[ARMS[0]]
    dump(OUT/'evaluation.json',dict(complete=True,results=results,mean_pq100=means,total_tp=tp,advance_gate=gate,
        limitation=p['scope'],prediction_seal_sha256=sha(OUT/'predictions_complete.json')))


def pipeline():
    p=verify();env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def run(python,args):
        dump(OUT/'status.json',dict(complete=False,stage=args,updated=time.time(),pid=os.getpid()))
        subprocess.run([str(python),str(Path(__file__).resolve()),*args],env=env,check=True,cwd=ROOT)
    try:
        run(PY,['extract'])
        for s in p['scenes']:
            for arm in ARMS:run(CGPY,['map','--scene',s['key'],'--arm',arm])
        run(CGPY,['seal']);run(CGPY,['evaluate'])
        dump(OUT/'status.json',dict(complete=True,stage='evaluated',updated=time.time()))
    except Exception as e:
        old=read(OUT/'status.json');dump(OUT/'status.json',dict(old,failed=True,error=repr(e),updated=time.time()));raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','extract','map','seal','evaluate','pipeline'])
    ap.add_argument('--scene');ap.add_argument('--arm');args=ap.parse_args()
    if args.action=='map':mapping(args.scene,args.arm)
    else:globals()[args.action]()
