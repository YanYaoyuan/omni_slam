"""Eight-image depth-evidence capability test, not a mapping/AP experiment."""
import argparse,sys,time
from pathlib import Path
import numpy as np
from paper_final_common import ROOT,read,dump,sha

OUT=ROOT/'build/benchmarks/dense_depth_evidence_probe_v1'
ASSETS=ROOT/'build/benchmarks/vggt_object_probe_v1'
INPUT=ROOT/'build/benchmarks/discovery_measured_inputs_v1'
METHODS=['nearest','linear_inverse','vggt_global','vggt_local_shape']


def freeze():
    assert not OUT.exists();assets=read(ASSETS/'assets.json');assert sha(assets['checkpoint'])==assets['sha256']
    scenes=[];inputs={str(ASSETS/'assets.json'):sha(ASSETS/'assets.json')}
    for scene in read(INPUT/'protocol.json')['scenes']:
        if scene['kind']!='outdoor':continue
        m=read(INPUT/scene['key']/'manifest.json');job=read(scene['job']);mid=len(m['records'])//2;chosen=m['records'][mid-2:mid+2];byframe={f['frame_id']:f for f in job['frames']};records=[]
        for r in chosen:
            frame=byframe[r['frame_id']];records.append(dict(frame_id=r['frame_id'],rgb=r['rgb'],measured=frame['measured']))
            for f in [r['rgb'],frame['measured']]:inputs[f]=sha(f)
        for f in [scene['job'],INPUT/scene['key']/'manifest.json']:inputs[str(f)]=sha(f)
        scenes.append(dict(key=scene['key'],camera=job['config']['cam'],records=records))
    sources=[Path(__file__),ROOT/'omni_semantic_map/omni_semantic_map/depth_evidence_interval.py',ROOT/'omni_semantic_map/test/test_depth_evidence_interval.py']+list((ASSETS/'upstream/vggt').rglob('*.py'))
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),scenes=scenes,methods=METHODS,assets=assets,input_hashes=inputs,source_hashes={str(f):sha(f) for f in sources},
        selection='Each exposed outdoor window: original middle index minus2 through plus1, four chronological frames; no object/GT/quality selection. Eight RGB total.',
        model='Existing pinned VGGT-1B weights, aggregator and native depth head only, strictly loaded. Known metric camera/poses unchanged. Official crop preprocessing,width518,height rounded to14; require no actual crop/pad. BF16 aggregation under autocast, native depth head arithmetic. Two joint4frame calls, eval/seed0;12GiB allocator cap onGPU0. Native depth/confidence and bilinear-resized original-resolution fields saved before LiDAR-depth comparison.',
        split='32pixel checkerboard tiles. Even(x//32+y//32) are measured fitting anchors in every frame. Odd tiles in frame index1 calibrate intervals; odd tiles in frame indices0,2,3 are held-out tests. Only available measured pixels scored. Views and pixels correlated; not independent scenes or IID coverage guarantees.',
        controls='Nearest measured fitting-anchor depth; Delaunay-linear inverse-depth interpolation with nearest fallback outside convex hull; VGGT one global metric scale per4frame window fitted as median(measured/rawpredicted) over fitting anchors; VGGT local shape transfer=nearest_anchor_measured_depth*rawpred(pixel)/rawpred(anchor). Same fitting anchor pixel set for all methods. No test/calibration depth values enter completion. No parameter fit or mask/GT reference access.',
        intervals='For each method/window, calibration frame odd measured pixels: split nearest-anchor pixel distances into3equal-frequency bins; signed depth residual divided by(0.05+0.01*prediction),quantiles0.05/0.95 in each bin(min32samples). Apply unchanged to test frames. An interval is precise only if finite/positive and width<=2*(0.05+0.01*prediction). Test empirical coverage, precise fraction and correct-precise/all; a passing interval is only candidate depth agreement, not proven visibility.',
        separation='At test measured pixels additionally evaluate synthetic query depths z +/- max(1m,0.1z). These are counterfactual depth-separation checks, not real occlusion labels. They must not be represented as segmentation or visibility GT.',
        gate='Necessary capability screen only: in EACH window localVGGT has precise_fraction>=0.25, measured-depth coverage among precise intervals>=0.90, correct_precise/all strictly above nearest AND linear_inverse, and synthetic wrong-depth agreement<=0.05 among precise. Otherwise no dense visibility/map expansion and no calibration/threshold/window sweep. Passing would still require actual cross-view/occlusion validation and complete-map controls.',
        scope='No new maps, object GT access, AP, robot or independent evaluation. Depth predictions never create XYZ. No novelty claim for pretrained depth, interpolation or empirical intervals; this screens usable additional geometry evidence before any new mapping mechanism. Existing failed branches stay closed.'))
    print('FROZEN',[(s['key'],[r['frame_id'] for r in s['records']]) for s in scenes],flush=True)


def verify():
    p=read(OUT/'protocol.json')
    for f,h in {**p['input_hashes'],**p['source_hashes']}.items():assert sha(f)==h,f
    assert sha(p['assets']['checkpoint'])==p['assets']['sha256']
    return p


def capture():
    p=verify();assert not (OUT/'capture_complete.json').exists();assert not (OUT/'execution.json').exists()
    import torch
    import torch.nn.functional as F
    from PIL import Image
    sys.path.insert(0,str(ASSETS/'upstream'))
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    torch.manual_seed(0);torch.set_num_threads(4);assert torch.cuda.is_available()
    free,total=torch.cuda.mem_get_info();assert free>=14*1024**3
    torch.cuda.set_per_process_memory_fraction(12*1024**3/total)
    dump(OUT/'execution.json',dict(started_epoch=time.time(),free_bytes=free,torch=torch.__version__,device=torch.cuda.get_device_name(0),protocol_sha256=sha(OUT/'protocol.json')))
    model=VGGT(enable_camera=False,enable_point=False,enable_depth=True,enable_track=False).eval()
    state=torch.load(p['assets']['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
    retained={k:v for k,v in state.items() if k.startswith(('aggregator.','depth_head.'))}
    omitted=[k for k in state if k not in retained];assert all(k.startswith(('camera_head.','point_head.','track_head.')) for k in omitted)
    result=model.load_state_dict(retained,strict=True);assert not result.missing_keys and not result.unexpected_keys
    del state,retained;model.requires_grad_(False).cuda();files={};rows=[];tick=time.monotonic()
    for scene in p['scenes']:
        start=time.monotonic();images=load_and_preprocess_images([r['rgb'] for r in scene['records']]).cuda()
        camera=scene['camera'];expected_h=round(camera['H']*(518/camera['W'])/14)*14
        assert expected_h<=518 and tuple(images.shape[-2:])==(expected_h,518)
        for r in scene['records']:assert Image.open(r['rgb']).size==(camera['W'],camera['H'])
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):output=model(images)
        depth=output['depth'][0,...,0].float();confidence=output['depth_conf'][0].float()
        assert tuple(depth.shape)==(4,expected_h,518)
        full=F.interpolate(depth[:,None],size=(camera['H'],camera['W']),mode='bilinear',align_corners=False)[:,0]
        conf=F.interpolate(confidence[:,None],size=(camera['H'],camera['W']),mode='bilinear',align_corners=False)[:,0]
        assert torch.isfinite(depth).all() and torch.all(depth>0) and torch.isfinite(confidence).all()
        path=OUT/'capture'/f'{scene["key"]}.npz';path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,depth=depth.cpu().numpy(),confidence=confidence.cpu().numpy(),depth_original=full.cpu().numpy(),confidence_original=conf.cpu().numpy(),frame_ids=np.array([r['frame_id'] for r in scene['records']]))
        torch.cuda.synchronize();files[str(path)]=sha(path);rows.append(dict(key=scene['key'],input_shape=list(images.shape),seconds=time.monotonic()-start));print('DEPTH CAPTURE',rows[-1],flush=True)
        del images,output,depth,confidence,full,conf
    dump(OUT/'capture_complete.json',dict(complete=True,files=files,rows=rows,seconds=time.monotonic()-tick,peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,strict_loading=True,omitted_inactive_tensors=len(omitted),no_lidar_values_read=True,no_object_gt_read=True))


def complete():
    from scipy.spatial import cKDTree,Delaunay
    from scipy.interpolate import LinearNDInterpolator
    p=verify();c=read(OUT/'capture_complete.json');assert c['complete'] and not (OUT/'completion_complete.json').exists()
    for f,h in c['files'].items():assert sha(f)==h
    files={};stats={}
    for scene in p['scenes']:
        key=scene['key'];d=np.load(OUT/'capture'/f'{key}.npz');h,w=scene['camera']['H'],scene['camera']['W'];yy,xx=np.indices((h,w));query=np.column_stack([xx.ravel(),yy.ravel()]);fit=((xx//32+yy//32)%2==0).ravel();xyfit=query[fit];anchor_data=[];ratios=[]
        for i,rec in enumerate(scene['records']):
            # Only read fitting-tile values into completion variables.
            values=np.load(rec['measured'])['depth'][xyfit[:,1],xyfit[:,0]];valid=np.isfinite(values)&(values>0);xy=xyfit[valid];z=values[valid];raw=d['depth_original'][i]
            assert len(xy)>100;anchor_data.append((xy,z));ratios.append(z/raw[xy[:,1],xy[:,0]])
        scale=float(np.median(np.concatenate(ratios)));stats[key]=dict(global_scale=scale,anchors=[len(a[0]) for a in anchor_data])
        for i,rec in enumerate(scene['records']):
            xy,z=anchor_data[i];distance,near=cKDTree(xy).query(query,workers=1);nearest=z[near];raw=d['depth_original'][i].ravel()
            inverse=LinearNDInterpolator(Delaunay(xy),1/z)(query);valid=np.isfinite(inverse)&(inverse>0);linear=nearest.copy();linear[valid]=1/inverse[valid]
            anchor_raw=d['depth_original'][i,xy[near,1],xy[near,0]];local=nearest*raw/anchor_raw
            preds=dict(nearest=nearest,linear_inverse=linear,vggt_global=raw*scale,vggt_local_shape=local)
            for v in preds.values():assert np.isfinite(v).all() and np.all(v>0)
            path=OUT/'completion'/key/f'{rec["frame_id"]:06d}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,**{k:v.reshape(h,w).astype(np.float32) for k,v in preds.items()},anchor_distance=distance.reshape(h,w).astype(np.float32),anchor_xy=xy,anchor_depth=z)
            files[str(path)]=sha(path)
        print('DEPTH COMPLETED',key,stats[key],flush=True)
    dump(OUT/'completion_complete.json',dict(complete=True,files=files,stats=stats,capture_sha256=sha(OUT/'capture_complete.json'),held_out_depth_not_used=True,no_new_geometry=True))


def score():
    from omni_semantic_map.depth_evidence_interval import fit_intervals,intervals,candidate_depth_agreement
    p=verify();c=read(OUT/'completion_complete.json');assert c['complete'] and not (OUT/'summary.json').exists()
    for f,h in c['files'].items():assert sha(f)==h
    results={};gate=True
    for scene in p['scenes']:
        key=scene['key'];frames=[]
        for i,rec in enumerate(scene['records']):
            measured=np.load(rec['measured'])['depth'];yy,xx=np.indices(measured.shape);take=((xx//32+yy//32)%2==1)&np.isfinite(measured)&(measured>0)
            d=np.load(OUT/'completion'/key/f'{rec["frame_id"]:06d}.npz');frames.append(dict(index=i,frame_id=rec['frame_id'],z=measured[take],distance=d['anchor_distance'][take],pred={m:d[m][take] for m in METHODS},pixel_xy=np.column_stack([xx[take],yy[take]])))
        cal=frames[1];test=[r for r in frames if r['index']!=1];z=np.concatenate([r['z'] for r in test]);distance=np.concatenate([r['distance'] for r in test]);rows={}
        for method in METHODS:
            model=fit_intervals(cal['pred'][method],cal['z'],cal['distance']);pred=np.concatenate([r['pred'][method] for r in test]);lo,hi,precise=intervals(pred,distance,model);agree=candidate_depth_agreement(z,lo,hi,precise);gap=np.maximum(1.,.1*z)
            wrong=candidate_depth_agreement(z+gap,lo,hi,precise)|candidate_depth_agreement(z-gap,lo,hi,precise)
            absolute=np.abs(pred-z);relative=absolute/z
            rows[method]=dict(test_measurements=len(z),calibration_measurements=len(cal['z']),median_absolute_m=float(np.median(absolute)),p90_absolute_m=float(np.quantile(absolute,.9)),median_relative=float(np.median(relative)),
                raw_within_5cm_fraction=float(np.mean(absolute<=.05)),raw_within_original_relative_gate=float(np.mean(absolute<=.05+.01*z)),
                precise=int(precise.sum()),precise_fraction=float(precise.mean()),precise_coverage=float(agree.sum()/precise.sum()) if precise.any() else None,
                correct_precise_fraction=float(agree.mean()),synthetic_wrong_agreement_fraction=float(wrong.sum()/precise.sum()) if precise.any() else None,interval_model=model)
            dump(OUT/'calibration'/key/(method+'.json'),model)
            path=OUT/'evaluated'/key/(method+'.npz');path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,predicted=pred,measured=z,lower=lo,upper=hi,precise=precise,agree=agree,synthetic_wrong_agreement=wrong)
            print('DEPTH SCORE',key,method,{k:v for k,v in rows[method].items() if k!='interval_model'},flush=True)
        r=rows['vggt_local_shape'];ok=r['precise_fraction']>=.25 and r['precise_coverage'] is not None and r['precise_coverage']>=.9 and r['synthetic_wrong_agreement_fraction']<=.05 and r['correct_precise_fraction']>max(rows[m]['correct_precise_fraction'] for m in ['nearest','linear_inverse'])
        gate &= ok;results[key]=dict(methods=rows,capability_gate=bool(ok),calibration_frame=cal['frame_id'],test_frames=[t['frame_id'] for t in test],test_frame_offsets=np.r_[0,np.cumsum([len(t['z']) for t in test])].tolist())
    dump(OUT/'summary.json',dict(complete=True,results=results,capability_gate=bool(gate),no_object_gt_read=True,new_maps=0,new_ap=0,robot=False,independent_test=False,protocol_sha256=sha(OUT/'protocol.json'),completion_sha256=sha(OUT/'completion_complete.json')))
    print('CAPABILITY GATE',bool(gate),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','capture','complete','score']);globals()[ap.parse_args().action]()
