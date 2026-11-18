"""Frozen SceneNN input adapter and separate observed-surface evaluation.

GT mesh/XML are opened only by quality evaluation and input alignment QA.
"""
import argparse,json,hashlib,time,gzip,pickle
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from .flat_benchmark import sha256,instance_metrics
from .flat_surface_evaluation import majority_labels
from .replay_fusion import FusionObservation,ReplayableObjectMap,ConceptGraphsBackend
from .local_fusion import LocalReplayObjectMap
from .dependency_fusion import DependencyReplayObjectMap
from .adaptive_fusion import AdaptiveReplayObjectMap
from .local_fusion_benchmark import RssSampler

BASE=Path('build/benchmarks/scenenn_20261110')
DATA=Path('/media/user/50D2BD30D2BD1AE4/WorkSpace/omni_slam_benchmarks/SceneNN')
CFG=Path('build/benchmarks/expanded_semantic_20261109/conceptgraphs_runs/office_0/mapping_config.json')
def dump(p,x):p.write_text(json.dumps(x,indent=2))
def inputs(scene):
 protocol=json.loads((BASE/'protocol.json').read_text());ids=protocol['frame_ids'][scene]
 source=DATA/('frames_016' if scene=='016' else 'sparse_'+scene)
 lines=(DATA/'main'/scene/'trajectory.log').read_text().splitlines()
 raw=np.array([[list(map(float,row.split())) for row in lines[i+1:i+5]] for i in range(0,len(lines),5)])[ids]
 assert np.isfinite(raw).all() and np.allclose(raw[:,3],[0,0,0,1])
 poses=raw.copy();u,_,v=np.linalg.svd(raw[:,:3,:3]);poses[:,:3,:3]=u@v
 assert np.all(np.linalg.det(poses[:,:3,:3])>0) and np.max(np.linalg.norm(poses-raw,axis=(1,2)))<.001
 return protocol,ids,source,raw,poses

def project(depth):
 assert depth.shape==(480,640)
 v,u=np.mgrid[0:480:8,0:640:8];z=depth[v,u]/1000.
 return np.stack([(u-320)*z/544.47329,(v-240)*z/544.47329,z],-1),(z>=.1)&(z<=5),u,v

def mesh(scene):
 with (DATA/'main'/scene/(scene+'.ply')).open('rb') as f:
  header=[]
  while True:
   s=f.readline().decode().strip();header.append(s)
   if s=='end_header':break
  n=int(next(s for s in header if s.startswith('element vertex')).split()[-1])
  dtype=np.dtype([(k,'<f4') for k in ['x','y','z','nx','ny','nz']]+[('label','<u4'),('r','u1'),('g','u1'),('b','u1')])
  a=np.fromfile(f,dtype=dtype,count=n)
 return np.column_stack([a[k] for k in ['x','y','z']]),a['label'].astype(np.int64)

def qa(scene):
 from scipy.spatial import cKDTree
 protocol,ids,source,raw,poses=inputs(scene);xyz,_=mesh(scene);tree=cKDTree(xyz);checks=[]
 for j in np.linspace(0,199,10,dtype=int):
  points,valid,_,_=project(np.array(Image.open(source/'depth'/f'depth{ids[j]+1:05d}.png')))
  world=points[valid]@poses[j,:3,:3].T+poses[j,:3,3];dist=tree.query(world)[0]
  checks.append(dict(frame=ids[j],median_m=float(np.median(dist)),p90_m=float(np.percentile(dist,90)),within_5cm=float(np.mean(dist<.05))))
 result=dict(scene=scene,checks=checks,rotation_projection_max_frobenius=float(np.linalg.norm(poses-raw,axis=(1,2)).max()),scope='QA only, no pose/offset fit')
 dump(BASE/(scene+'_qa.json'),result);print(result)

def prepare(scene,device):
 import torch,open_clip
 from torchvision.models.detection import maskrcnn_resnet50_fpn_v2,MaskRCNN_ResNet50_FPN_V2_Weights
 from conceptgraph.utils.model_utils import compute_clip_features
 from types import SimpleNamespace
 torch.set_num_threads(2);torch.cuda.set_device(device)
 protocol,ids,source,raw,poses=inputs(scene);out=BASE/'cache'/scene;out.mkdir(parents=True,exist_ok=False);(out/'frames').mkdir();(out/'features').mkdir()
 weights=Path('build/experiments/semantic_20261106/models/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth');clipweights=Path('build/benchmarks/expanded_semantic_20261109/models/open_clip_pytorch_model.bin')
 detector=maskrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None,min_size=360,max_size=640)
 detector.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));detector=detector.to(device).eval().requires_grad_(False)
 model,_,preprocess=open_clip.create_model_and_transforms('ViT-H-14',pretrained=str(clipweights),device=device);model=model.eval().requires_grad_(False);tokenizer=open_clip.get_tokenizer('ViT-H-14');classes=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1.meta['categories']
 obs=[];frames=[];hashes={str(p):sha256(p) for p in [weights,clipweights,BASE/'protocol.json',CFG,DATA/'main'/scene/'trajectory.log',DATA/'main/intrinsic/asus.ini']};start=time.monotonic()
 dump(out/'manifest.json',dict(complete=False,scene=scene))
 for k,fid in enumerate(ids):
  rgbpath=source/'image'/f'image{fid+1:05d}.png';depthpath=source/'depth'/f'depth{fid+1:05d}.png'
  rgb=np.array(Image.open(rgbpath).convert('RGB'));depth=np.array(Image.open(depthpath));points,valid,u,v=project(depth)
  for p in [rgbpath,depthpath]:hashes[str(p)]=sha256(p)
  with torch.inference_mode():pred=detector([torch.from_numpy(rgb.copy()).permute(2,0,1).float().to(device)/255])[0]
  pred={n:x.cpu().numpy() for n,x in pred.items()};owner=np.full(valid.shape,-1,np.int32);chosen=[];oids=[]
  for j in np.argsort(-pred['scores'],kind='stable'):
   if pred['scores'][j]<.5:continue
   sel=(pred['masks'][j,0,v,u]>.5)&valid&(owner<0)
   if sel.sum()<8:continue
   oid=len(obs);owner[sel]=oid;oids.append(oid);chosen.append(j)
   obs.append(dict(observation_id=oid,frame_id=k,category=int(pred['labels'][j]),box=pred['boxes'][j].tolist(),score=float(pred['scores'][j])))
  if chosen:
   det=SimpleNamespace(xyxy=pred['boxes'][chosen],class_id=pred['labels'][chosen])
   with torch.inference_mode():_,im,txt=compute_clip_features(rgb,det,model,preprocess,tokenizer,classes,device)
   for j,oid in enumerate(oids):
    sel=owner==oid
    np.savez_compressed(out/'features'/f'{oid:06d}.npz',points=points[sel],colors=rgb[v[sel],u[sel]]/255.,image=im[j],text=txt[j])
  np.savez_compressed(out/'frames'/f'{k:04d}.npz',points=points[valid],owner=owner[valid],pixels=np.column_stack([u[valid],v[valid]]))
  frames.append(dict(source_frame=fid,observations=oids,rgb=str(rgbpath),depth=str(depthpath)))
  if k%25==0:print(scene,k,len(obs),round(time.monotonic()-start,1),flush=True)
 np.savez_compressed(out/'poses.npz',raw=raw,poses=poses);dump(out/'observations.json',obs);dump(out/'frames.json',frames);dump(out/'input_hashes.json',hashes)
 dump(out/'manifest.json',dict(complete=True,scene=scene,frames=200,observations=len(obs),gt_used=False,device=device,wall_s=time.monotonic()-start,adapter_sha256=sha256(Path(__file__)),files_sha256={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}))

def load(scene):
 cache=BASE/'cache'/scene;meta=json.loads((cache/'manifest.json').read_text());assert meta['complete']
 for name,h in meta['files_sha256'].items():assert sha256(cache/name)==h,name
 for name,h in json.loads((cache/'input_hashes.json').read_text()).items():assert sha256(Path(name))==h,name
 obs=json.loads((cache/'observations.json').read_text());frames=json.loads((cache/'frames.json').read_text());records=[]
 for k,f in enumerate(frames):
  group=[]
  for oid in f['observations']:
   with np.load(cache/'features'/f'{oid:06d}.npz') as a:group.append(FusionObservation(oid,k,obs[oid]['category'],a['points'],a['colors'],a['image'],a['text']))
  records.append(group)
 return cache,records,np.load(cache/'poses.npz')['poses'],len(obs)

def factory(method):
 from omegaconf import OmegaConf
 cfg=OmegaConf.create(json.loads(CFG.read_text()))
 if method=='native':return ReplayableObjectMap(lambda:ConceptGraphsBackend(cfg))
 if method=='cached_full':return LocalReplayObjectMap(cfg,'full')
 if method=='dependency':return DependencyReplayObjectMap(cfg)
 return AdaptiveReplayObjectMap(cfg,policy='auto' if method=='adaptive' else 'checkpoint')
def labels(state,n):
 a=state.labels();return np.array([a[i] for i in range(n)],np.int32)
def perturb(poses,case,scale):
 before=poses.copy();first=180 if case=='tail10' else 0;last=20 if case=='head10' else 200;amplitude=.05 if scale=='small' else .2
 for k in range(first,last):
  frac=(k-first+1)/(last-first);angle=np.deg2rad(3)*frac*amplitude/.2;delta=np.eye(4);delta[:2,:2]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]];delta[0,3]=amplitude*frac;before[k]=delta@poses[k]
 return before

def reference(scene):
 import torch
 torch.set_num_threads(1);cache,records,poses,n=load(scene);out=BASE/'reference'/scene;out.mkdir(parents=True,exist_ok=False);state=factory('native')
 for k,group in enumerate(records):state.add_frame(k,group,poses[k])
 np.save(out/'labels.npy',labels(state,n))
 with gzip.open(out/'objects.pkl.gz','wb') as f:pickle.dump(state.export_objects(),f)
 dump(out/'run.json',dict(complete=True,observations=n,objects=len(state.export_objects())))

def run(scene,method,case,scale):
 import torch,gc,os
 torch.set_num_threads(1);cache,records,poses,n=load(scene);out=BASE/'runs'/scene/(case+'_'+scale)/method;out.mkdir(parents=True,exist_ok=False)
 source={str(p):sha256(p) for p in [Path(__file__),CFG,*Path(__file__).parent.glob('*fusion.py')]};dump(out/'run.json',dict(complete=False))
 before=perturb(poses,case,scale);state=factory(method);start=time.perf_counter()
 for k,g in enumerate(records):state.add_frame(k,g,before[k])
 ingest=time.perf_counter()-start;frozen=labels(state,n);causal=state.causal_labels();old=state._backend;oldposes=state._poses
 expected=np.load(BASE/'reference'/scene/'labels.npy')
 with gzip.open(BASE/'reference'/scene/'objects.pkl.gz','rb') as f:expected_objects=pickle.load(f)
 trials=[]
 for rep in range(3):
  state._backend=old;state._poses=oldposes;state.revision=0;state.events=[];gc.collect()
  with RssSampler() as memory:
   start=time.perf_counter();state.revise_poses(dict(enumerate(poses)));corrected=labels(state,n);wall=time.perf_counter()-start
  np.testing.assert_array_equal(corrected,expected);assert state.causal_labels()==causal
  actual=state.export_objects();assert len(actual)==len(expected_objects)
  for a,e in zip(actual,expected_objects):
   assert a['observation_ids']==e['observation_ids']
   for key in ['points','colors','box','image_feature','text_feature']:np.testing.assert_allclose(a[key],e[key],atol=1e-10,rtol=0)
  trials.append(dict(wall_s=wall,rss_peak_bytes=memory.peak,exact=True,stats=state.events[-1]['update_stats']))
 np.savez_compressed(out/'labels.npz',frozen=frozen,corrected=corrected,causal=np.array([causal[i] for i in range(n)]))
 for name,h in source.items():assert sha256(Path(name))==h
 dump(out/'run.json',dict(complete=True,scene=scene,method=method,case=case,scale=scale,median_update_s=float(np.median([t['wall_s'] for t in trials])),ingest_s=ingest,trials=trials,source_sha256=source,cpu_affinity=sorted(os.sched_getaffinity(0))))
 print('COMPLETE',scene,case,scale,method,flush=True)

def evaluate(scene):
 from scipy.spatial import cKDTree
 import xml.etree.ElementTree as ET
 cache=BASE/'cache'/scene;poses=np.load(cache/'poses.npz')['poses'];xyz,gt=mesh(scene);root=ET.parse(DATA/'main'/scene/(scene+'.xml')).getroot()
 bg=[0]+[int(x.attrib['id']) for x in root.iter('label') if x.attrib.get('text','').lower().startswith(('wall','floor','ceiling','unlabeled'))]
 gt[np.isin(gt,bg)]=0;tree=cKDTree(xyz);worlds=[];owners=[];truths=[];matched=[]
 for k in range(200):
  with np.load(cache/'frames'/f'{k:04d}.npz') as f:world=f['points']@poses[k,:3,:3].T+poses[k,:3,3];owner=f['owner'].copy()
  dist,index=tree.query(world);valid=dist<=.05;worlds.append(world[valid]);owners.append(owner[valid]);truths.append(gt[index[valid]]);matched.append([int(valid.sum()),len(valid)])
 vox,inv=np.unique(np.floor(np.concatenate(worlds)/.05).astype(np.int32),axis=0,return_inverse=True);truth=majority_labels(inv,np.concatenate(truths),len(vox));owner=np.concatenate(owners)
 paths={'clean_reference':BASE/'reference'/scene/'labels.npy'}
 for case in ['head10','tail10','global']:
  for scale in ['small','large']:paths[case+'_'+scale]=BASE/'runs'/scene/(case+'_'+scale)/'cached_full'/'labels.npz'
 result=dict(scene=scene,scope='GT-matched observed surface 5cm class-agnostic instance PQ, not official mesh AP; background excluded from objects',matched_points=sum(v[0] for v in matched),valid_points=sum(v[1] for v in matched),voxels=len(vox),methods={})
 for name,p in paths.items():
  candidates={'final':np.load(p)} if p.suffix=='.npy' else dict(np.load(p))
  for kind,lab in candidates.items():
   pred=np.zeros(len(owner),np.int64);valid=owner>=0;pred[valid]=lab[owner[valid]];vp=majority_labels(inv,pred,len(vox));pairs,counts=np.unique(np.column_stack([truth,vp]),axis=0,return_counts=True)
   result['methods'][name+'_'+kind]=instance_metrics(Counter({tuple(map(int,a)):int(n) for a,n in zip(pairs,counts)}))
 out=BASE/'evaluation'/scene;out.mkdir(parents=True,exist_ok=True);dump(out/'metrics.json',result)
 np.savez_compressed(out/'domain.npz',xyz=(vox+.5)*.05,truth=truth,inverse=inv,owner=owner)
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['qa','prepare','reference','run','evaluate']);p.add_argument('--scene',required=True,choices=['005','011','016']);p.add_argument('--device',default='cuda:0');p.add_argument('--method',default='adaptive');p.add_argument('--case',default='tail10');p.add_argument('--scale',default='large');a=p.parse_args()
 if a.action=='prepare':prepare(a.scene,a.device)
 elif a.action=='run':run(a.scene,a.method,a.case,a.scale)
 else:globals()[a.action](a.scene)
