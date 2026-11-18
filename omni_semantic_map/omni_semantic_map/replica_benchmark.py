"""Frozen independent-scene diagnostic on the vMAP rendering of Replica.

Predicted masks only. Ground truth is opened solely by evaluate(), after mapping.
This is NOT the official Replica/vMAP reconstruction or ScanNet AP benchmark.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import io
import json
from pathlib import Path
import time
import zipfile

import cv2
import numpy as np
from PIL import Image
from .flat_benchmark import sha256, instance_metrics
from .flat_surface_evaluation import majority_labels
from .sensor_features import pool_measured_tokens
from .evidence_association import MemoryAssociator
from .witness_graph import WitnessConfig, WitnessGraph
from .objects import transform_points

SCENES = ('office_0', 'room_1')
FRAME_IDS = list(range(0, 2000, 10))
BACKGROUND = [0, 5, 12, 30, 31, 40, 60, 79, 92, 93, 95, 97, 98, 255]
ARCHIVE_SHA256 = 'e70fd6fd72849072d84fc42bd879dc244c11c6fb8c3f6bdb4403af42fb07a00f'
METHODS = ('geometry', 'hard', 'off', 'witness', 'witness_no_appearance')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def projection(depth):
    if depth.shape != (680, 1200):
        raise ValueError('expected native 1200x680 depth')
    v, u = np.mgrid[0:680:8, 0:1200:8]
    z = depth[v, u].astype(np.float32) / 1000.
    valid = np.isfinite(z) & (z >= .1) & (z <= 5.)
    return np.stack([(u-599.5)*z/600, (v-339.5)*z/600, z], -1), valid, u, v


def extract(archive, output):
    """Select exact members only; zipfile.read checks their CRC automatically."""
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(scenes=list(SCENES), trajectory='00', frame_ids=FRAME_IDS,
        frame_scope='200 fixed uniformly subsampled frames per scene, NOT full trajectories',
        archive=str(archive.resolve()), expected_archive_sha256=ARCHIVE_SHA256,
        graph_config=asdict(WitnessConfig()), methods=list(METHODS),
        detector='Mask R-CNN ResNet50 FPN v2, frozen COCO weights',
        detector_min_size=360, detector_max_size=640, score_threshold=.5,
        mask_threshold=.5, overlap_policy='highest detection score wins',
        min_support=8, native_depth_stride=8, depth_range_m=[.1, 5.],
        camera=dict(width=1200, height=680, fx=600., fy=600., cx=599.5, cy=339.5),
        dino='frozen DINOv2-small, full RGB resize 448x252; 24 local tokens',
        gt_background_classes=BACKGROUND, evaluation_voxel_m=.05,
        primary='class-agnostic observed-surface voxel PQ, IoU>0.5; not official mesh AP',
        secondary='sequence pixel PQ; observation-pair P/R at GT support purity>=0.7',
        poses='provided camera-to-world; no estimated SLAM localization',
        parameters_tuned_on_these_scenes=False, gt_used_by_mapper=False,
        source_sha256={p.name: sha256(p) for p in [Path(__file__),
            Path(__file__).with_name('witness_graph.py'),
            Path(__file__).with_name('evidence_association.py'),
            Path(__file__).with_name('sensor_features.py')]})
    # Lock choices BEFORE extracting RGB or evaluating any scene.
    dump(output/'protocol.json', protocol)
    with zipfile.ZipFile(archive) as z:
        hashes = {}
        for scene in SCENES:
            prefix = f'vmap/{scene}/imap/00/'
            dest = output/'data'/scene
            dest.mkdir(parents=True)
            members = ['traj_w_c.txt', 'render_config.yaml']
            members += [f'{t}/{t}_{i}.png' for i in FRAME_IDS for t in
                        ['rgb', 'depth', 'semantic_instance', 'semantic_class']]
            for member in members:
                target = dest/member
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(z.read(prefix+member))
                hashes[str(target.relative_to(output))] = sha256(target)
            target = dest/'info_semantic.json'
            target.write_bytes(z.read(f'vmap/{scene}/habitat/info_semantic.json'))
            hashes[str(target.relative_to(output))] = sha256(target)
            print('extracted', scene, len(members)+1, flush=True)
    dump(output/'extraction.json', dict(complete=True, member_crc_verified=True,
         full_archive_sha256_verified=False, files_sha256=hashes))


def prepare(base, scene, model_path, weights, device):
    import torch
    import torchvision
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    from transformers import Dinov2Model
    torch.set_num_threads(4); cv2.setNumThreads(1)
    source = base/'data'/scene; output = base/'cache'/scene
    output.mkdir(parents=True, exist_ok=False); (output/'features').mkdir(); (output/'frames').mkdir()
    detector = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None,
                                      min_size=360, max_size=640)
    detector.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    detector = detector.to(device).eval().requires_grad_(False)
    model = Dinov2Model.from_pretrained(str(model_path), local_files_only=True).to(device).eval().requires_grad_(False)
    poses = np.loadtxt(source/'traj_w_c.txt').reshape(-1, 4, 4)[FRAME_IDS]
    if (not np.isfinite(poses).all() or not np.allclose(poses[:, 3], [0,0,0,1])
        or not np.allclose(poses[:,:3,:3].transpose(0,2,1)@poses[:,:3,:3], np.eye(3), atol=1e-4)
        or not np.allclose(np.linalg.det(poses[:,:3,:3]), 1, atol=1e-4)):
        raise ValueError('invalid camera poses')
    np.save(output/'poses.npy', poses)
    mean = torch.tensor([.485,.456,.406], device=device)[None,:,None,None]
    std = torch.tensor([.229,.224,.225], device=device)[None,:,None,None]
    hashes = {str(p.resolve()): sha256(p) for p in [source/'traj_w_c.txt', weights,
        model_path/'model.safetensors', model_path/'config.json', base/'protocol.json']}
    meta = dict(complete=False, scene=scene, device=device, device_name=torch.cuda.get_device_name(device),
        torch=torch.__version__, torchvision=torchvision.__version__, gt_used=False,
        source_sha256=sha256(Path(__file__)))
    dump(output/'manifest.json', meta)
    obs = []; records = []; start = time.monotonic()
    for k, fid in enumerate(FRAME_IDS):
        rgbpath = source/'rgb'/f'rgb_{fid}.png'; depthpath = source/'depth'/f'depth_{fid}.png'
        for p in [rgbpath, depthpath]: hashes[str(p.resolve())] = sha256(p)
        image = Image.open(rgbpath).convert('RGB'); rgb = np.array(image)
        points, valid, u, v = projection(np.array(Image.open(depthpath)))
        if rgb.shape != (680,1200,3): raise ValueError('unexpected RGB size')
        tensor = torch.from_numpy(np.array(image.resize((448,252), Image.Resampling.BICUBIC))).permute(2,0,1)[None].float().to(device)/255
        with torch.inference_mode():
            tokens = model(pixel_values=(tensor-mean)/std).last_hidden_state[0,1:].reshape(18,32,-1).cpu().numpy()
            pred = detector([torch.from_numpy(rgb.copy()).permute(2,0,1).float().to(device)/255])[0]
        pred = {n: x.cpu().numpy() for n,x in pred.items()}
        owner = np.full(valid.shape, -1, np.int32); rejected = 0; indices = []
        # Make disjoint predicted supports independently of all GT.
        for j in np.argsort(-pred['scores'], kind='stable'):
            if pred['scores'][j] < .5: continue
            selected = (pred['masks'][j,0,v,u] > .5) & valid & (owner < 0)
            if selected.sum() < 8: rejected += 1; continue
            oid = len(obs); owner[selected] = oid
            local = points[selected].astype(np.float32)
            pixels = np.column_stack([u[selected],v[selected]]).astype(np.int16)
            descriptor, anchors, features = pool_measured_tokens(tokens, pixels, local, rgb.shape)
            file = f'features/{oid:06d}.npz'
            np.savez_compressed(output/file, descriptor=descriptor, local_xyz=anchors,
                local_features=features.astype(np.float16), support_xyz=local, pixels=pixels)
            obs.append(dict(observation_id=oid, keyframe_id=k, source_frame=fid,
                label=f"coco_{int(pred['labels'][j])}", score=float(pred['scores'][j]),
                local_position=np.median(local,axis=0).tolist(), appearance=[1.],
                box=pred['boxes'][j].tolist(), feature_file=file))
            indices.append(oid)
        np.savez_compressed(output/'frames'/f'{fid:04d}.npz', owner=owner, valid=valid)
        records.append(dict(source_frame=fid, observations=indices, rejected_small_support=rejected))
        if k % 25 == 0: print(scene,k,'observations',len(obs),'seconds',round(time.monotonic()-start,1),flush=True)
    dump(output/'observations.json', obs); dump(output/'frames.json', records)
    dump(output/'input_sha256.json', hashes)
    meta.update(complete=True, frames=len(records), observations=len(obs), wall_s=time.monotonic()-start,
        files_sha256={str(p.relative_to(output)):sha256(p) for p in output.rglob('*') if p.is_file() and p.name!='manifest.json'})
    dump(output/'manifest.json', meta)
    return meta


def run(base, scene):
    cache = base/'cache'/scene; output = base/'runs'/scene
    meta = json.loads((cache/'manifest.json').read_text())
    if not meta['complete']: raise ValueError('incomplete cache')
    for n,h in meta['files_sha256'].items():
        if sha256(cache/n) != h: raise ValueError('modified cache: '+n)
    for n,h in json.loads((cache/'input_sha256.json').read_text()).items():
        if sha256(n) != h: raise ValueError('modified input: '+n)
    output.mkdir(parents=True, exist_ok=False)
    obs = json.loads((cache/'observations.json').read_text()); frames = json.loads((cache/'frames.json').read_text())
    poses = np.load(cache/'poses.npy')
    for o in obs:
        with np.load(cache/o['feature_file']) as f:
            for n in ['descriptor','local_xyz','local_features','support_xyz']: o[n] = f[n].astype(np.float32)
    summary = {}
    for method in METHODS:
        start = time.monotonic(); config = WitnessConfig(appearance_weight=0. if method=='witness_no_appearance' else .6)
        tracker = MemoryAssociator('geometry', config) if method=='geometry' else WitnessGraph(config, 'witness' if method.startswith('witness') else method)
        causal = np.zeros(len(obs), np.int32)
        for k, frame in enumerate(frames):
            ids = frame['observations']
            causal[ids] = tracker.add_frame([obs[i] for i in ids], poses)
            if k % 50 == 0: print(scene,method,k,flush=True)
        np.savez_compressed(output/(method+'.npz'), causal=causal, final=tracker.labels())
        if method != 'geometry': dump(output/(method+'_events.json'), tracker.events)
        summary[method] = dict(wall_s=time.monotonic()-start, config=asdict(config),
            final_objects=len(np.unique(tracker.labels())), causal_emitted_ids=len(np.unique(causal)))
    dump(output/'summary.json', dict(complete=True, methods=summary, cache_manifest_sha256=sha256(cache/'manifest.json'),
        source_sha256={p.name:sha256(p) for p in [Path(__file__), Path(__file__).with_name('witness_graph.py'), Path(__file__).with_name('evidence_association.py')]}))


def pair_score(gt, pred):
    c2 = lambda n: n*(n-1)//2
    tp = sum(c2(n) for n in Counter(zip(map(int,gt),map(int,pred))).values())
    pp = sum(c2(n) for n in Counter(map(int,pred)).values()); gp = sum(c2(n) for n in Counter(map(int,gt)).values())
    return dict(precision=tp/pp if pp else None, recall=tp/gp if gp else None,
                f1=2*tp/(pp+gp) if pp+gp else None, observations=len(gt), gt_instances=len(set(gt)))


def truth_labels(instance, semantic):
    # Preserve valid raw instance zero by adding 1. Unknown/stuff becomes zero.
    return np.where(np.isin(semantic, BACKGROUND), 0, instance.astype(np.int32)+1)


def evaluate(base, scene):
    cache=base/'cache'/scene; source=base/'data'/scene; runs=base/'runs'/scene; output=base/'evaluation'/scene
    if not json.loads((runs/'summary.json').read_text())['complete']: raise ValueError('incomplete runs')
    output.mkdir(parents=True, exist_ok=False)
    obs=json.loads((cache/'observations.json').read_text()); poses=np.load(cache/'poses.npy')
    coordinates=[]; truths=[]; owners=[]; identity=np.full(len(obs),-1,np.int32); gt_hashes={}
    for k,fid in enumerate(FRAME_IDS):
        points,valid,u,v=projection(np.array(Image.open(source/'depth'/f'depth_{fid}.png')))
        ip=source/'semantic_instance'/f'semantic_instance_{fid}.png'; sp=source/'semantic_class'/f'semantic_class_{fid}.png'
        for p in [ip,sp]: gt_hashes[str(p.resolve())]=sha256(p)
        raw=np.array(Image.open(ip)); semantic=np.array(Image.open(sp)); truth=truth_labels(raw,semantic)[v,u]
        with np.load(cache/'frames'/f'{fid:04d}.npz') as f:
            if not np.array_equal(f['valid'],valid): raise ValueError('different evaluation domain')
            owner=f['owner'].copy()
        for oid in np.unique(owner[owner>=0]):
            labels,counts=np.unique(truth[owner==oid],return_counts=True); j=counts.argmax()
            if labels[j]>0 and counts[j]/counts.sum()>=.7: identity[oid]=labels[j]
        coordinates.append(np.floor(transform_points(points[valid],poses[k])/.05).astype(np.int32))
        truths.append(truth[valid]); owners.append(owner[valid])
    voxel,inverse=np.unique(np.concatenate(coordinates),axis=0,return_inverse=True)
    pixel_truth=np.concatenate(truths); owner=np.concatenate(owners)
    voxel_truth=majority_labels(inverse,pixel_truth,len(voxel)); xyz=(voxel.astype(float)+.5)*.05
    np.savez_compressed(output/'reference.npz',xyz=xyz,instance=voxel_truth)
    np.save(output/'observation_gt.npy',identity)
    results=dict(scene=scene, frames=len(FRAME_IDS), sampled_valid_pixels=len(owner), evaluation_voxels=len(voxel),
        primary='observed 5cm voxel PQ, class-agnostic, NOT official mesh AP',
        secondary_identity_scope='only detected observations with >=70% GT support purity', methods={})
    def score(a,b):
        pairs,counts=np.unique(np.column_stack([a,b]),axis=0,return_counts=True)
        return instance_metrics(Counter({tuple(map(int,p)):int(c) for p,c in zip(pairs,counts)}))
    for method in METHODS:
        with np.load(runs/(method+'.npz')) as assignments:
            for mode in ['causal','final']:
                labels=assignments[mode]; predicted=np.zeros(len(owner),np.int32); mask=owner>=0
                predicted[mask]=labels[owner[mask]]
                voxpred=majority_labels(inverse,predicted,len(voxel)); metrics=score(voxel_truth,voxpred)
                metrics['observed_voxel_instance_pq']=metrics.pop('sequence_pixel_instance_pq')
                eligible=identity>=0; name=method+'_'+mode
                results['methods'][name]=dict(voxel=metrics, pixel=score(pixel_truth,predicted), identity=pair_score(identity[eligible],labels[eligible]))
                if mode=='final': np.savez_compressed(output/(name+'.npz'),xyz=xyz,instance=voxpred)
    dump(output/'gt_sha256.json',gt_hashes); dump(output/'metrics.json',results)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['extract','prepare','run','evaluate']); p.add_argument('--base',type=Path,required=True)
    p.add_argument('--archive',type=Path); p.add_argument('--scene',choices=SCENES)
    p.add_argument('--model',type=Path); p.add_argument('--weights',type=Path); p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if a.action=='extract': extract(a.archive,a.base)
    elif a.action=='prepare': prepare(a.base,a.scene,a.model,a.weights,a.device)
    elif a.action=='run': run(a.base,a.scene)
    else:
        result=evaluate(a.base,a.scene)
        print(json.dumps({k:v['voxel']['observed_voxel_instance_pq'] for k,v in result['methods'].items()},indent=2))

if __name__=='__main__': main()
