"""Fixed native point-correspondence audit, including the robot sensor adapter.

Tracks are external measurements, not Ours or a map-quality result. No map is
mutated. Reference labels are unavailable to preparation and inference.
"""
import argparse
from pathlib import Path
import sys
import time

import numpy as np
from paper_final_common import ROOT, read, dump, sha

B = ROOT / 'build/benchmarks'
OUT = B / 'observation_correspondence_probe_v1'
UPSTREAM = B / 'vggt_object_probe_v1/upstream'


def projection(xyz, frame, job, kind):
    pose = np.asarray(frame['pose'] if kind == 'outdoor' else frame['camera_pose'])
    inv = np.linalg.inv(pose)
    cam = xyz @ inv[:3, :3].T + inv[:3, 3]
    if kind == 'outdoor':
        k = job['config']['cam']; focal = [k['fx'], k['fy']]; center = [k['cx'], k['cy']]
    else:
        k = np.asarray(frame['intrinsic']); focal = [k[0, 0], k[1, 1]]; center = k[:2, 2]
    uv = cam[:, :2] / cam[:, 2:3] * focal + center
    return uv, cam[:, 2]


def verify():
    p = read(OUT / 'protocol.json')
    for f, h in {**p['inputs'], **p['sources']}.items():
        assert sha(f) == h, f
    return p


def freeze():
    assert not OUT.exists()
    contexts = read(B / 'joint_semantic_revision_v1/protocol.json')['contexts']
    assets = read(B / 'vggt_object_probe_v1/assets.json')
    assert sha(assets['checkpoint']) == assets['sha256']
    inputs = {assets['checkpoint']: assets['sha256']}; clips = []
    for c in contexts:
        job = read(c['job']); byframe = {f['frame_id']: f for f in job['frames']}
        observations = read(c['observations'])['results'][c['key']]
        manifest = (B / 'full_source_visual_visibility_audit_v1' / c['key'] / 'manifest.json'
                    if c['kind'] == 'outdoor' else B / 'discovery_measured_inputs_v1' / c['key'] / 'manifest.json')
        views = {v['frame_id']: v for v in read(manifest)['records']}
        paths = [c['job'], c['sem'], c['observations'], manifest, Path(c['full']) / 'geometry.npz',
                 Path(c['full']) / 'old_to_full.npy', Path(c['ours']) / 'geometry.npz']
        for quarter in [1, 2, 3]:
            i = len(observations) * quarter // 4
            rows = observations[i:i+3]; assert len(rows) == 3
            frames = [byframe[r['frame_id']] for r in rows]
            selected_views = [views[f['frame_id']] for f in frames]
            clips.append(dict(key=c['key'], kind=c['kind'], quarter=quarter, frames=frames,
                              observations=rows, views=selected_views))
            paths.extend(r['file'] for r in selected_views)
            paths.extend(r['file'] for r in rows)
            paths.extend(f['rgb'] for f in frames)
        for f in paths: inputs[str(f)] = sha(f)
    sources = [Path(__file__), *UPSTREAM.joinpath('vggt').rglob('*.py')]
    dump(OUT / 'protocol.json', dict(created_epoch=time.time(), contexts=contexts, clips=clips, assets=assets,
        inputs=inputs, sources={str(f):sha(f) for f in sources},
        question='Does native image correspondence provide useful evidence beyond a projected LiDAR landmark, on the SAME source observations? This screens observation-to-surface binding, not descriptor similarity or SAM mask propagation.',
        selection='Three 3-frame clips per scene, starting at floor(observation_count*q/4), q=1,2,3. No quality/object/GT choice. Seed at exact cached measured-visible source pixels, one smallest-source-ID per16px native-input tile per current ownership stratum (assigned/unassigned). Uniformly spaced sorted tiles capped64 each; no deficit replacement. Same full-source visibility for outdoor, original measured robot visibility with identity old-to-full mapping.',
        methods='Geometry projection; frozen native VGGT aggregator+track_head; pyramidal LK on identical resized RGB converted to gray (21window,3levels,30iterations,.01epsilon). Forward3frames and reverse3frames, no geometric initialization for either tracker. Native official width518/rounded14height preprocessing, no crop/pad; half-pixel coordinate transform. Geometry and tracker coordinates returned to original RGB, preserve static-map convention.',
        validity='VGGT requires all3forward vis>.5,conf>.5 and all3reverse vis>.5,conf>.5, allframes in image, forward/back cycle<=2nativepixels. LK requires allforward/back statuses, allframes in image, cycle<=2nativepixels. Frozen descriptive validity, not calibrated error probability or proof of physical correspondence. Report all raw tracks as well.',
        budget='9clips,at most128seeds/clip,18VGGTforwardcalls,54imageencodes. Same original checkpoint,eval/BF16/seed0,strict aggregator+track head load. Zero training,SAMcalls,newimages,maps,AP. No thresholds,clip,checkpoint or scale variants.',
        readout='Seal tracks before reference labels. Reprojection residual is disagreement NOT ground truth track error. Read cached semantic class at projected versus tracked original pixels, restricted to same existing measured-depth-visible source domain and known reference labels within strict20cm. Report all valid and method-common domains, corrections and regressions, coverage, assigned/unassigned strata. Robot has noGT; geometry/cycle statistics only. No direct writeback from this audit.',
        decision='Native tracks must improve conditional semantic correctness over geometry AND LK on common valid measured domains in EACH public window, with at least one corrected source observation per window, before considering this native-track route for map revision. Even a pass is only necessary evidence, not an effective novel mechanism or independent benchmark. Failed screen closes this configuration; no sweep.',
        scope='Two already exposed public windows and same-adapter robot clips. Not independent scenes, full330frame robot validation, instance AP, paper novelty or S100 realtime. Ordinary tracking/tracklet association already exists (SceneVGGT2602.15899); novelty must reside in demonstrated object-map revision, not the external head.'))
    print('CORRESPONDENCE FROZEN', [(c['key'], [f['frame_id'] for f in c['frames']]) for c in clips], flush=True)


def prepare():
    from PIL import Image
    p = verify(); assert not (OUT / 'prepared.json').exists()
    results = []; files = {}
    for c in p['contexts']:
        full = np.load(Path(c['full']) / 'geometry.npz'); xyz = full['xyz']
        mapping = np.load(Path(c['full']) / 'old_to_full.npy')
        owner = np.zeros(len(xyz), np.int64); owner[mapping] = np.load(c['sem'])['owners']
        base = np.load(Path(c['ours']) / 'geometry.npz'); added = base['added_full_source_ids']
        if len(added): owner[added] = base['instance'][-len(added):]
        job = read(c['job'])
        for clip in [r for r in p['clips'] if r['key'] == c['key']]:
            w, h = Image.open(clip['frames'][0]['rgb']).size
            nh = round(h * (518 / w) / 14) * 14; assert nh <= 518
            scale = np.array([518 / w, nh / h])
            d = np.load(clip['views'][0]['file']); ids = d['visible_source_ids']; xy = d['pixel_xy']
            if c['kind'] == 'robot': ids = mapping[ids]
            native = (xy + .5) * scale - .5
            chosen = []
            for assigned in [False, True]:
                candidates = np.flatnonzero((owner[ids] > 0) == assigned)
                candidates = candidates[np.argsort(ids[candidates], kind='stable')]
                tiles = np.floor(native[candidates] / 16).astype(int)
                _, first = np.unique(tiles, axis=0, return_index=True)
                candidates = candidates[first]
                sample = np.linspace(0, len(candidates)-1, min(64, len(candidates)), dtype=int)
                chosen.extend(candidates[sample])
            ix = np.asarray(chosen, int); src = ids[ix]; pixels = xy[ix]
            assert len(src) and len(src) == len(np.unique(src))
            uv, z = [], []; visible = []
            for frame, view in zip(clip['frames'], clip['views']):
                q, depth = projection(xyz[src], frame, job, c['kind']); uv.append(q); z.append(depth)
                viewids = np.load(view['file'])['visible_source_ids']
                if c['kind'] == 'robot': viewids = mapping[viewids]
                visible.append(np.isin(src, viewids))
            uv = np.asarray(uv); z = np.asarray(z)
            integer = np.floor(uv[0] + (.5 if c['kind'] == 'outdoor' else 0)).astype(int)
            np.testing.assert_array_equal(integer, pixels)
            name = f'{c["key"]}_q{clip["quarter"]}'
            path = OUT / 'inputs' / (name + '.npz'); path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, source_ids=src, xyz=xyz[src], owner=owner[src], seed_xy=pixels,
                                seed_native=(pixels+.5)*scale-.5, projected_xy=uv, camera_z=z,
                                measured_visible=np.asarray(visible), scale=scale, size=[w,h], native_size=[518,nh])
            files[str(path)] = sha(path)
            results.append(dict(name=name, input=str(path), **clip, seeds=len(src), assigned=int((owner[src]>0).sum())))
            print('CORRESPONDENCE INPUT', name, len(src), 'assigned', int((owner[src]>0).sum()), flush=True)
    dump(OUT / 'prepared.json', dict(complete=True, results=results, files=files, gt_read=False))


def inside(track, size):
    return np.isfinite(track).all(-1) & (track >= 0).all(-1) & (track < np.asarray(size)).all(-1)


def predict():
    import cv2
    import torch
    sys.path.insert(0, str(UPSTREAM))
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    p = verify(); prep = read(OUT / 'prepared.json')
    for f, h in prep['files'].items(): assert sha(f) == h, f
    assert not (OUT / 'execution.json').exists(), 'Check live process before resume'
    torch.set_num_threads(4); cv2.setNumThreads(2); torch.manual_seed(0)
    free, total = torch.cuda.mem_get_info(); assert free >= 16*1024**3
    torch.cuda.set_per_process_memory_fraction(14*1024**3 / total)
    dump(OUT / 'execution.json', dict(started_epoch=time.time(), torch=torch.__version__, cv2=cv2.__version__, free_bytes=free))
    model = VGGT(enable_camera=False, enable_point=False, enable_depth=False, enable_track=True).eval()
    state = torch.load(p['assets']['checkpoint'], map_location='cpu', weights_only=True, mmap=True)
    retain = {k:v for k,v in state.items() if k.startswith(('aggregator.', 'track_head.'))}
    omitted = [k for k in state if k not in retain]
    assert all(k.startswith(('camera_head.', 'point_head.', 'depth_head.')) for k in omitted)
    loading = model.load_state_dict(retain, strict=True); assert not loading.missing_keys and not loading.unexpected_keys
    del retain, state
    model.requires_grad_(False).cuda(); files = {}; results = []; calls = 0; start = time.monotonic()
    for row in prep['results']:
        tick = time.monotonic(); d = np.load(row['input']); seed = d['seed_native'].astype(np.float32)
        images = load_and_preprocess_images([f['rgb'] for f in row['frames']])
        assert tuple(images.shape[-2:]) == tuple(d['native_size'][::-1])
        gray = [cv2.cvtColor(np.round(im.permute(1,2,0).numpy()*255).astype(np.uint8),cv2.COLOR_RGB2GRAY) for im in images]
        images = images.cuda()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(images, query_points=torch.from_numpy(seed).cuda()); calls += 1
        forward = out['track'][0].float().cpu().numpy(); vis = out['vis'][0].float().cpu().numpy(); conf = out['conf'][0].float().cpu().numpy()
        assert forward.shape == (3,len(seed),2) and np.isfinite(forward).all()
        endvalid = inside(forward[-1], d['native_size']); backward = np.full_like(forward, np.nan)
        bvis = np.zeros_like(vis); bconf = np.zeros_like(conf)
        if endvalid.any():
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                rev = model(images.flip(0), query_points=torch.from_numpy(forward[-1,endvalid]).cuda()); calls += 1
            backward[:,endvalid] = rev['track'][0].float().cpu().numpy()
            bvis[:,endvalid] = rev['vis'][0].float().cpu().numpy(); bconf[:,endvalid] = rev['conf'][0].float().cpu().numpy()
            del rev
        cycle = np.linalg.norm(backward[-1]-seed, axis=1)
        valid = (inside(forward,d['native_size']).all(0) & inside(backward,d['native_size']).all(0)
                 & (vis>.5).all(0) & (conf>.5).all(0) & (bvis>.5).all(0) & (bconf>.5).all(0) & (cycle<=2))
        opts = dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,30,.01))
        def lk_sequence(imgs, q):
            tr=[q.copy()]; status=[np.ones(len(q),bool)]; error=[np.zeros(len(q),np.float32)]
            for a,b in zip(imgs[:-1],imgs[1:]):
                new,s,e=cv2.calcOpticalFlowPyrLK(a,b,tr[-1].reshape(-1,1,2),None,**opts)
                tr.append(new[:,0]); status.append(s[:,0].astype(bool)); error.append(e[:,0])
            return np.array(tr),np.array(status),np.array(error)
        lk, ls, le = lk_sequence(gray,seed); lb,lbs,lbe = lk_sequence(gray[::-1],lk[-1])
        lcycle=np.linalg.norm(lb[-1]-seed,axis=1)
        lvalid=ls.all(0)&lbs.all(0)&inside(lk,d['native_size']).all(0)&inside(lb,d['native_size']).all(0)&(lcycle<=2)
        path=OUT/'predictions'/(row['name']+'.npz'); path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,vggt_native=forward,vggt_reverse=backward,vggt_vis=vis,vggt_conf=conf,
            vggt_reverse_vis=bvis,vggt_reverse_conf=bconf,vggt_cycle=cycle,vggt_valid=valid,
            lk_native=lk,lk_reverse=lb,lk_status=ls,lk_reverse_status=lbs,lk_error=le,lk_reverse_error=lbe,
            lk_cycle=lcycle,lk_valid=lvalid,
            vggt_xy=(forward+.5)/d['scale']-.5,lk_xy=(lk+.5)/d['scale']-.5)
        files[str(path)]=sha(path);torch.cuda.synchronize()
        results.append(dict(name=row['name'],file=str(path),seeds=len(seed),vggt_valid=int(valid.sum()),lk_valid=int(lvalid.sum()),
                            seconds=time.monotonic()-tick,shared_valid=int((valid&lvalid).sum())))
        print('CORRESPONDENCE PREDICTED',results[-1],flush=True)
        del out,images
    dump(OUT/'predictions_complete.json',dict(complete=True,files=files,results=results,model_calls=calls,
        image_encodes=3*calls,seconds=time.monotonic()-start,peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,
        strict_loading=True,inactive_omitted_tensors=len(omitted),gt_read=False,new_maps=0,new_ap=0))


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','predict'])
    globals()[ap.parse_args().action]()
