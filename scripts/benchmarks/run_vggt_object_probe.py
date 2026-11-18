"""Frozen, GT-free multi-view features followed by a separate instance probe.

This is a representation diagnostic, not a new mapping result. Published map
and support data are immutable. Only exposed development scenes are used.
"""
import argparse
from pathlib import Path
import sys
import time
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from prepare_vggt_probe_assets import OUT


def freeze():
    assert not (OUT/'protocol.json').exists()
    source=ROOT/'build/benchmarks/evidence_retention_20261111/public_dino_v1'
    p=read(source/'protocol.json');scenes=[];hashes={}
    # Fixed chronological indices, chosen without checking the identities inside.
    for key in ['replica_office_0','scenenn_014']:
        s=next(s for s in p['scenes'] if s['key']==key)
        cache=Path(s['cache']);manifest=read(cache/'manifest.json')
        images={r['frame_id']:r for r in s['images']}
        records=[]
        for i in [10,12,14,16]:
            rec=manifest['records'][i];image=images[rec['frame_id']]
            dino=source/'features'/key/f"{rec['frame_id']:06d}.npz"
            frame=cache/'frames'/rec['file']
            for file in [frame,Path(image['rgb']),dino]:hashes[str(file)]=sha(file)
            assert hashes[str(frame)]==rec['sha256']
            assert hashes[image['rgb']]==image['sha256']
            records.append(dict(index=i,frame_id=rec['frame_id'],rgb=image['rgb'],cache_frame=str(frame),dino=str(dino),observation_offset=rec['observation_offset']))
        scenes.append(dict(key=key,cache=str(cache),records=records,
            labels=str(ROOT/'build/benchmarks/full_object_mainline_20261112/supervision/evaluation'/key/'point_targets.npz')))
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),scenes=scenes,input_sha256=hashes,
        source_sha256={str(Path(__file__)):sha(__file__)},code_sha256=sha(OUT/'code.json'),
        feature_conditions=['dino_small_archived','vggt_patch_encoder','vggt_single','vggt_joint'],
        layer=23,frames_per_window=4,max_windows=2,cuda_allocator_gib=8,
        extraction='Official image preprocessing crop mode, width518. Landscape frames; no crop. Native fractional mask-area pooling over patch lattice, per-token L2 then weighted mean L2. Same masks and pooling for all VGGT conditions. Preserve dense tokens for downstream heads.',
        controls='VGGT-single versus VGGT-joint share identical checkpoint, last layer, token dimension, image preprocessing and pooling. Archived DINO-small uses its older local pooling and is contextual only, not a controlled backbone ablation. VGGT patch encoder is from the geometry-trained checkpoint, not standalone original DINOv2 weights.',
        evaluation='After all features sealed, use cached measured-support proxy instances.>=16 labeled source points and >=70% majority purity; labels only for scoring. Cross-frame nearest-descriptor retrieval among all eligible masks in target frame. Report only queries with a true mate, plus same-class distractor counts where available. Also report coverage and no-mate queries. No threshold fitting.',
        scope='Two exposed development windows. No full-map or official benchmark claim. VGGT pretraining includes Replica/Hypersim; no foundation-unseen claim.',
        no_gt_in_extraction=True))
    print('FROZEN 2 windows 8 images',flush=True)


def verify():
    p=read(OUT/'protocol.json');a=read(OUT/'assets.json')
    for key in ['input_sha256','source_sha256']:
        for f,h in p[key].items():assert sha(f)==h,f
    assert sha(OUT/'code.json')==p['code_sha256']
    for f,h in read(OUT/'code.json')['files'].items():assert sha(OUT/'upstream'/f)==h,f
    assert sha(a['checkpoint'])==a['sha256']
    return p,a


def extract():
    p,a=verify();dest=OUT/'features';dest.mkdir(exist_ok=False)
    sys.path.insert(0,str(OUT/'upstream'))
    import torch
    import torch.nn.functional as F
    from vggt.models.aggregator import Aggregator
    from vggt.utils.load_fn import load_and_preprocess_images
    torch.set_num_threads(2);torch.manual_seed(0)
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(p['cuda_allocator_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory)
    state=torch.load(a['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
    model=Aggregator().eval()
    status=model.load_state_dict({k[len('aggregator.'):]:v for k,v in state.items() if k.startswith('aggregator.')},strict=True)
    assert not status.missing_keys and not status.unexpected_keys
    model.cached_layer_indices={p['layer']}
    model.requires_grad_(False);model.cuda();del state
    records=[];started=time.monotonic()
    captured=[]
    def capture(module,args,output):
        value=output['x_norm_patchtokens'] if isinstance(output,dict) else output
        captured.append(value.detach().float().cpu())
    hook=model.patch_embed.register_forward_hook(capture)
    for scene in p['scenes']:
        output=dest/scene['key'];output.mkdir()
        images=load_and_preprocess_images([r['rgb'] for r in scene['records']]).cuda()
        assert images.shape[-2]<=518 and images.shape[-1]==518
        gh,gw=images.shape[-2]//14,images.shape[-1]//14
        tic=time.monotonic()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            layers,start=model(images[None]);joint=layers[p['layer']][0,:,start:].float().cpu()
        patch=captured.pop();del layers
        singles=[]
        for im in images:
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                layers,start=model(im[None,None]);single=layers[p['layer']][0,0,start:].float().cpu()
            singles.append(single);captured.clear();del layers
        singles=torch.stack(singles);torch.cuda.synchronize()
        masks=[];features={k:[] for k in ['vggt_patch_encoder','vggt_single','vggt_joint','dino_small_archived']};counts=[]
        for i,r in enumerate(scene['records']):
            z=np.load(r['cache_frame']);binary=np.unpackbits(z['masks_packed'],axis=2)[:,:,:z['segmentation'].shape[1]]
            weights=F.interpolate(torch.from_numpy(binary.astype(np.float32))[:,None],size=(gh,gw),mode='area')[:,0].reshape(len(binary),-1)
            for name,tokens in [('vggt_patch_encoder',patch[i]),('vggt_single',singles[i]),('vggt_joint',joint[i])]:
                assert tokens.shape[0]==gh*gw
                pooled=weights@F.normalize(tokens,dim=-1)
                pooled=F.normalize(pooled/weights.sum(1).clamp_min(1e-12)[:,None],dim=-1)
                features[name].append(pooled.numpy())
            old=np.load(r['dino'])['mean'];assert len(old)==len(binary)
            features['dino_small_archived'].append(old)
            masks.append(weights.numpy().astype(np.float16));counts.append(len(binary))
        for name,values in features.items():
            value=np.concatenate(values)
            assert np.isfinite(value).all()
            np.save(output/(name+'.npy'),value)
        np.savez_compressed(output/'dense_tokens.npz',joint=joint.numpy().astype(np.float16),
            single=singles.numpy().astype(np.float16),patch=patch.numpy().astype(np.float16),
            mask_weights=np.concatenate(masks),offsets=np.r_[0,np.cumsum(counts)],grid=np.array([gh,gw]))
        records.append(dict(scene=scene['key'],counts=counts,shape=list(images.shape),wall_s=time.monotonic()-tic,
            files_sha256={str(f.relative_to(dest)):sha(f) for f in output.iterdir()}))
        dump(dest/'progress.json',dict(records=records));print('FEATURES',scene['key'],counts,flush=True)
        del joint,singles,patch,images
    hook.remove()
    dump(dest/'manifest.json',dict(complete=True,gt_accessed=False,protocol_sha256=sha(OUT/'protocol.json'),
        assets_sha256=sha(OUT/'assets.json'),records=records,wall_s=time.monotonic()-started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),device=torch.cuda.get_device_name(0)))


def evaluate():
    p,_=verify();manifest=read(OUT/'features/manifest.json')
    assert manifest['complete'] and manifest['protocol_sha256']==sha(OUT/'protocol.json')
    assert not (OUT/'evaluation.json').exists()
    for record in manifest['records']:
        for f,h in record['files_sha256'].items():assert sha(OUT/'features'/f)==h,f
    results=[]
    for scene in p['scenes']:
        labels=np.load(scene['labels'])['instance'];truth=[];frames=[];details=[]
        for i,r in enumerate(scene['records']):
            z=np.load(r['cache_frame'])
            for j,(lo,hi) in enumerate(zip(z['offsets'][:-1],z['offsets'][1:])):
                targets=labels[z['point_ids'][lo:hi]];valid=targets>0
                keys,counts=np.unique(targets[valid],return_counts=True)
                dominant=int(keys[counts.argmax()]) if len(keys) else 0
                purity=float(counts.max()/len(targets)) if len(keys) and len(targets) else 0
                keep=valid.sum()>=16 and purity>=.7
                truth.append(dominant if keep else 0);frames.append(i)
                details.append(dict(frame_id=r['frame_id'],mask=j,proxy_instance=dominant,purity=purity,eligible=bool(keep)))
        truth=np.array(truth);frames=np.array(frames);scores={}
        for name in p['feature_conditions']:
            f=np.load(OUT/'features'/scene['key']/(name+'.npy'))
            f=f/np.maximum(np.linalg.norm(f,axis=1,keepdims=True),1e-12)
            correct=total=no_mate=0;events=[]
            for i in np.flatnonzero(truth>0):
                for target_frame in range(4):
                    if target_frame==frames[i]:continue
                    candidates=np.flatnonzero((frames==target_frame)&(truth>0))
                    if not len(candidates) or not np.any(truth[candidates]==truth[i]):no_mate+=1;continue
                    chosen=int(candidates[np.argmax(f[candidates]@f[i])]);ok=bool(truth[chosen]==truth[i])
                    correct+=ok;total+=1
                    events.append(dict(query=int(i),target_frame=target_frame,chosen=chosen,correct=ok,candidates=len(candidates)))
            scores[name]=dict(correct=correct,total=total,accuracy=correct/total if total else None,
                              no_mate=no_mate,events=events)
        results.append(dict(scene=scene['key'],observations=len(truth),eligible=int((truth>0).sum()),
                            distinct_proxy_instances=len(np.unique(truth[truth>0])),label_sha256=sha(scene['labels']),
                            conditions=scores,observation_labels=details))
    dump(OUT/'evaluation.json',dict(complete=True,protocol_sha256=sha(OUT/'protocol.json'),
        features_sha256=sha(OUT/'features/manifest.json'),results=results,
        limitation='Conditional retrieval on masks with a visible true mate and pure measured-support proxy labels. Correlated mask/frame queries, not independent samples. No same-class breakdown implemented; no calibrated confidence or final-map gain.'))
    for s in results:print(s['scene'],{k:(v['correct'],v['total']) for k,v in s['conditions'].items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','extract','evaluate']);args=parser.parse_args()
    {'freeze':freeze,'extract':extract,'evaluate':evaluate}[args.action]()
