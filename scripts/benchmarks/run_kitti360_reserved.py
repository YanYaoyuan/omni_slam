"""Two reserved outdoor windows, frozen predictions before annotation access."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from paper_final_common import ROOT, FROZEN, PY, CGPY, read, dump, sha

OUT=ROOT/'build/benchmarks/kitti360_reserved_v1'
META=ROOT/'build/datasets/kitti360_prepare'
DATA=ROOT/'build/datasets/kitti360_windows_v1'
PRIOR=ROOT/'build/benchmarks/kitti360_connected_development_v1'
PROBE=ROOT/'build/benchmarks/vggt_object_probe_v1'
METHODS={'history2':('base','maps/cg/history2'), 'SOR':('base','maps/cleanup/sor'),
    'Ours_visibility':('base','maps/cleanup/visibility1'),
    'VGGT_single':('vggt_single','maps/cg/history2'),
    'VGGT_single_plus_visibility':('vggt_single','maps/cleanup/visibility1')}


def freeze():
    assert not OUT.exists()
    selection=read(META/'window_selection.json');windows=[w for w in selection['windows'] if w['role']=='reserved_validation']
    assert len(windows)==2 and [len(w['input_frames']) for w in windows]==[48,53]
    prior=read(PRIOR/'mapping_protocol.json')
    for f,h in prior['source_sha256'].items():assert sha(ROOT/f)==h
    scripts=['run_kitti360_reserved.py','prepare_kitti360_reserved.py','prepare_kitti360_frontend.py',
        'prepare_strong_jobs.py','split_strong_frontend.py','filter_livo_sparse_support.py',
        'kitti360_development_mapping.py','vggt_fullmap_backend.py','paper_final_mapping.py',
        'run_livo_history_graph.py','evaluate_kitti360_development.py']
    sources={str(ROOT/'scripts/benchmarks'/f):sha(ROOT/'scripts/benchmarks'/f) for f in scripts}
    sources.update({str(ROOT/f):h for f,h in prior['source_sha256'].items()})
    for f in (ROOT/'omni_semantic_map/omni_semantic_map').glob('*.py'):sources[str(f)]=sha(f)
    for f in [PROBE/'assets.json',PROBE/'code.json',META/'evalInstanceLevelSemanticLabeling.py',
        ROOT/'build/benchmarks/ovo_full_20261110/model_artifacts.json',META/'window_selection.json']:
        sources[str(f)]=sha(f)
    hashes={};reference_hashes={}
    for w in windows:
        for kind in ['image_00','velodyne']:
            manifest=DATA/f"{w['scene']}_{kind}_download.json";m=read(manifest);assert m['complete']
            assert m['selection_sha256']==sha(META/'window_selection.json')
            hashes[str(manifest)]=sha(manifest)
            for r in m['files']:
                path=ROOT/r['path'];assert sha(path)==r['sha256'];hashes[str(path)]=r['sha256']
        reference=DATA/'opaque_references'/w['member'];reference_hashes[str(reference)]=sha(reference)
        for f in (META/'extracted'/w['scene']).rglob('*.txt'):hashes[str(f)]=sha(f)
    for f in (META/'extracted/calibration').glob('*.txt'):hashes[str(f)]=sha(f)
    tax=ROOT/'build/benchmarks/kitti360_development_v1/taxonomy'
    for f in tax.iterdir():
        if f.is_file():hashes[str(f)]=sha(f)
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),windows=windows,source_hashes=sources,input_hashes=hashes,
        opaque_reference_hashes=reference_hashes,methods=METHODS,development_protocol_sha256=sha(PRIOR/'mapping_protocol.json'),
        mapping='Fixed existing outdoor adapter: first measured point per5cm voxel; measured-depth 5cm gate; largest component radius0.4m/min16; DBSCAN off; association overlap20cm; same frozen CG/history2/final processing. Static building/car taxonomy from19 competing classes.',
        features='Baseline SAM2.1/SigLIP learned merger. Candidate only changes association image descriptors to pretrained VGGT single-frame layer23 fractional-mask pooling; SigLIP remains semantic readout. Width518, no crop for landscape inputs. No joint-window arm, GT oracle or object-memory training.',
        scoring='Primary full released static-window confidence-weighted AP50 building/car, pinned KITTI-360 kernel with explicit local compatibility adapter. Export nearest complete-map point within20cm including nontarget label0. Confidence mass min100. Secondary observed-support domain fixed from all measured geometry. Report per-window, per-class and pooled-window AP50; upstream AP50:90 separate, not website AP50:95.',
        pooling='Prefix every prediction filename and all matched references with scene ID before pooling official matching records, to prevent cross-scene ID collisions.',
        scope='Two previously reserved official-validation windows sampled every5 available poses:48+53 inputs. Not full drives, not entire validation set, not official leaderboard. Distinct separated camera trajectories; foundation-pretraining overlap unknown. No moving-object tracking claim.',
        budget='Two windows x two association passes; only history2 emitted. Cleanup uses existing code; unused simple-ratio/MC/SOR candidate outputs archived but not selected for primary comparison. Five prespecified scored methods. No window replacement or threshold sweep.',
        gt_policy='Only opaque reference checksums before prediction seal. Freeze all101 input frames and methods before any reserved inference. Seal predictions from both scenes before any annotation decoding.',
        stop='Complete all fixed methods, score once, publish improvements and regressions. Do not tune on these results or automatically expand.',
        device='One GPU job at a time; frontend allocator6GiB, VGGT8GiB; no utilization gate required by current user authorization.'))
    print('FROZEN reserved windows',[(w['scene'],len(w['input_frames'])) for w in windows],flush=True)


def verify():
    p=read(OUT/'protocol.json')
    for f,h in p['source_hashes'].items():assert sha(f)==h,f
    return p


def job_path(scene):return OUT/scene/'jobs'/(scene+'.json')


def prepare(scene):
    p=verify();w=next(w for w in p['windows'] if w['scene']==scene)
    for f,h in p['input_hashes'].items():assert sha(f)==h,f
    from prepare_kitti360_reserved import prepare as prepare_inputs
    prepare_inputs(w,OUT/scene)
    root=OUT/scene;job=job_path(scene)
    (root/'connected').mkdir()
    shutil.copytree(ROOT/'build/benchmarks/kitti360_development_v1/taxonomy',root/'taxonomy')
    config=read(PRIOR/'mapping_protocol.json')
    config.update(job_sha256=sha(job),parent_reserved_protocol_sha256=sha(OUT/'protocol.json'),
        scope='Frozen reserved-window run. All predictions precede GT decoding.')
    dump(root/'mapping_protocol.json',config)


def geometry(scene):
    verify();import kitti360_development_mapping as base
    base.OUT=OUT/scene;base.JOB=job_path(scene);base.geometry()


def configure_arm(scene,arm):
    root=OUT/scene/arm;root.mkdir(exist_ok=True)
    job=read(job_path(scene));job.update(output=str(OUT/scene/'connected'),kind='kitti360_reserved_connected')
    target=root/'job.json';assert not target.exists();dump(target,job)
    config=read(OUT/scene/'mapping_protocol.json');config['job_sha256']=sha(target)
    dump(root/'mapping_protocol.json',config)
    return root,target


def extract(scene):
    verify();assets=read(PROBE/'assets.json');assert sha(assets['checkpoint'])==assets['sha256']
    for f,h in read(PROBE/'code.json')['files'].items():assert sha(PROBE/'upstream'/f)==h
    sys.path.insert(0,str(PROBE/'upstream'))
    import torch
    import torch.nn.functional as F
    from vggt.models.aggregator import Aggregator
    from vggt.utils.load_fn import load_and_preprocess_images
    torch.set_num_threads(2);torch.manual_seed(0)
    torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
    state=torch.load(assets['checkpoint'],map_location='cpu',weights_only=True,mmap=True);model=Aggregator().eval()
    model.load_state_dict({k[11:]:v for k,v in state.items() if k.startswith('aggregator.')},strict=True);del state
    model.cached_layer_indices={23};model.requires_grad_(False);model.cuda()
    cache=OUT/scene/'connected/cache';manifest=read(cache/'manifest.json');job=read(job_path(scene))
    dest=OUT/scene/'vggt_features';dest.mkdir(exist_ok=False);values=[];offset=0;begin=time.monotonic()
    for i,(r,rec) in enumerate(zip(job['frames'],manifest['records'])):
        assert r['frame_id']==rec['frame_id'] and rec['observation_offset']==offset
        im=load_and_preprocess_images([r['rgb']]).cuda();gh,gw=im.shape[-2]//14,im.shape[-1]//14
        assert im.shape[-1]==518 and im.shape[-2]<518
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            layers,start=model(im[None]);tokens=layers[23][0,0,start:].float().cpu()
        del layers,im
        z=np.load(cache/'frames'/rec['file']);binary=np.unpackbits(z['masks_packed'],axis=2)[:,:,:z['segmentation'].shape[1]]
        assert len(binary)==rec['masks']
        if len(binary):
            weights=F.interpolate(torch.from_numpy(binary.astype(np.float32))[:,None],size=(gh,gw),mode='area')[:,0].reshape(len(binary),-1)
            f=F.normalize(weights@F.normalize(tokens,dim=-1)/weights.sum(1).clamp_min(1e-12)[:,None],dim=-1).numpy()
            assert np.isfinite(f).all() and (np.linalg.norm(f,axis=1)>0).all();values.append(f)
        offset+=rec['masks']
        if i%10==0:print('VGGT',scene,i,len(job['frames']),flush=True)
    np.save(dest/'features.npy',np.concatenate(values))
    dump(dest/'manifest.json',dict(complete=True,no_gt=True,observations=offset,wall_s=time.monotonic()-begin,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),features_sha256=sha(dest/'features.npy'),
        cache_sha256=sha(cache/'manifest.json'),protocol_sha256=sha(OUT/'protocol.json')))


def mapping(scene,arm):
    verify();assert arm in ['base','vggt_single'];root,target=configure_arm(scene,arm)
    import kitti360_development_mapping as base
    import run_livo_history_graph as native
    from vggt_fullmap_backend import run_cg
    from omni_semantic_map.replay_fusion import ConceptGraphsBackend
    sys.path.insert(0,str(ROOT/'build/benchmarks/expanded_semantic_20261109/concept-graphs'))
    from conceptgraph.slam import mapping as native_mapping,utils
    original_init=ConceptGraphsBackend.__init__;original_overlap=native_mapping.compute_overlap_matrix_2set
    original_final_overlap=utils.compute_overlap_matrix;original_finalize=native.finalize
    def initialize(self,cfg,seed=0):
        cfg=deepcopy(cfg);cfg.dbscan_remove_noise=False;original_init(self,cfg,seed=seed)
    def overlap_config(cfg):
        cfg=deepcopy(cfg);cfg.downsample_voxel_size=.2;return cfg
    ConceptGraphsBackend.__init__=initialize
    native_mapping.compute_overlap_matrix_2set=lambda cfg,*args:original_overlap(overlap_config(cfg),*args)
    utils.compute_overlap_matrix=lambda cfg,*args:original_final_overlap(overlap_config(cfg),*args)
    native.finalize=base.outdoor_finalize
    features=None
    if arm=='vggt_single':
        folder=OUT/scene/'vggt_features';m=read(folder/'manifest.json');assert m['complete'] and sha(folder/'features.npy')==m['features_sha256']
        features=np.load(folder/'features.npy')
    try:run_cg(OUT/scene/'connected/cache',root/'maps/cg',features)
    finally:
        ConceptGraphsBackend.__init__=original_init;native_mapping.compute_overlap_matrix_2set=original_overlap
        utils.compute_overlap_matrix=original_final_overlap;native.finalize=original_finalize
    base.OUT=root;base.JOB=target;base.cleanup()


def seal():
    p=verify();files={}
    for w in p['windows']:
        root=OUT/w['scene']
        for arm in ['base','vggt_single']:
            assert read(root/arm/'maps/cg/run.json')['complete'] and read(root/arm/'maps/cleanup/run.json')['complete']
            for f in (root/arm).rglob('*'):
                if f.is_file():files[str(f)]=sha(f)
        for folder in [root/'connected/cache',root/'vggt_features',root/'frontend',root/'inputs',root/'jobs']:
            for f in folder.rglob('*'):
                if f.is_file():files[str(f)]=sha(f)
    dump(OUT/'predictions_complete.json',dict(complete=True,no_reference_labels_decoded=True,
        protocol_sha256=sha(OUT/'protocol.json'),files=files,created_epoch=time.time()))


def evaluate_all():
    p=verify();sealed=read(OUT/'predictions_complete.json')
    assert sealed['complete'] and sealed['protocol_sha256']==sha(OUT/'protocol.json')
    for f,h in sealed['files'].items():assert sha(f)==h
    import evaluate_kitti360_development as e
    from omni_semantic_map.kitti360_evaluation import kernel
    results=[]
    for w in p['windows']:
        scene=w['scene'];root=OUT/scene;ref=DATA/'opaque_references'/w['member']
        assert sha(ref)==p['opaque_reference_hashes'][str(ref)]
        e.OUT=root/'evaluation';assert not (e.OUT/'protocol.json').exists()
        methods={k:str(root/a/b) for k,(a,b) in METHODS.items()}
        hashes={str(ref):sha(ref),str(job_path(scene)):sha(job_path(scene)),str(OUT/'predictions_complete.json'):sha(OUT/'predictions_complete.json')}
        for folder in methods.values():
            for name in ['geometry.npz','classification.json']:hashes[str(Path(folder)/name)]=sha(Path(folder)/name)
        dump(e.OUT/'protocol.json',dict(reference=str(ref),job=str(job_path(scene)),
            cache=str(root/'frontend'/scene/'cache'),methods=methods,files_sha256=hashes,
            source_sha256=p['source_hashes'],export_radius_m=.2,observed_support_radius_m=.2,
            parent_protocol_sha256=sha(OUT/'protocol.json'),scope=p['scope'],annotation_access_after_all_predictions=True))
        e.run()
        r=read(e.OUT/'summary.json');r.update(reserved_validation=True,independent_test=False,
            independence_scope='First method evaluation on reserved windows; pretraining overlap and geometry overlap not certified.')
        dump(e.OUT/'summary.json',r);results.append(dict(scene=scene,result=r))
    args,ns=kernel(META/'evalInstanceLevelSemanticLabeling.py');pooled={}
    def rename(value,scene):
        if isinstance(value,list):return [rename(v,scene) for v in value]
        if isinstance(value,dict):return {k:(scene+'/'+v if k=='filename' else rename(v,scene)) for k,v in value.items()}
        return value
    for method in METHODS:
        pooled[method]={}
        for domain in ['full_window','observed_support']:
            matches={}
            for w in p['windows']:
                scene=w['scene'];entry=read(OUT/scene/'evaluation'/f'{method}_{domain}_matches.json')['window']
                matches[scene]=rename(entry,scene)
            ap=ns['evaluateMatches'](matches,args);pooled[method][domain]=e.clean_json(ns['computeAverages'](ap,args))
            dump(OUT/'pooled_matches'/f'{method}_{domain}.json',matches)
    dump(OUT/'summary.json',dict(complete=True,reserved_windows=results,pooled=pooled,
        official_server_score=False,protocol_sha256=sha(OUT/'protocol.json'),
        prediction_seal_sha256=sha(OUT/'predictions_complete.json'),scope=p['scope']))
    print('POOLED AP50', {m:v['full_window']['allAp50%'] for m,v in pooled.items()},flush=True)


def pipeline():
    p=verify();env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def call(python,script,args):
        dump(OUT/'status.json',dict(complete=False,pid=os.getpid(),stage=args,script=script,updated=time.time()))
        subprocess.run([str(python),script,*args],cwd=ROOT,env=env,check=True)
    script=str(Path(__file__).resolve())
    try:
        for w in p['windows']:
            scene=w['scene'];call(CGPY,script,['prepare','--scene',scene])
            for phase in ['masks','triplets','features']:
                call(PY,'scripts/benchmarks/split_strong_frontend.py',['--job',str(job_path(scene)),'--phase',phase])
            call(CGPY,script,['geometry','--scene',scene])
            call(CGPY,'scripts/benchmarks/filter_livo_sparse_support.py',[
                '--cache',str(OUT/scene/'frontend'/scene/'cache'),'--output',str(OUT/scene/'connected/cache'),
                '--livo-config','build/experiments/semantic_20261106/managed_config/livo.yaml',
                '--cg-config',str(FROZEN/'config/cg_rgbd.json')])
            call(PY,script,['extract','--scene',scene])
            for arm in ['base','vggt_single']:call(CGPY,script,['mapping','--scene',scene,'--arm',arm])
        call(CGPY,script,['seal']);call(CGPY,script,['evaluate_all'])
        dump(OUT/'status.json',dict(complete=True,stage='evaluated',updated=time.time()))
    except Exception as error:
        old=read(OUT/'status.json');dump(OUT/'status.json',dict(old,failed=True,error=repr(error),updated=time.time()));raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','geometry','extract','mapping','seal','evaluate_all','pipeline'])
    ap.add_argument('--scene');ap.add_argument('--arm');a=ap.parse_args()
    if a.action=='mapping':mapping(a.scene,a.arm)
    elif a.action in ['prepare','geometry','extract']:globals()[a.action](a.scene)
    else:globals()[a.action]()
