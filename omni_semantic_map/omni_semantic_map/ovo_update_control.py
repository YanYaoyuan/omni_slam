"""OVO official post-correction kernel on a shared frozen initial object map.

Not full OVO: no OVO tracking/SAM/learned feature frontend. Exact upstream
Instance3D and update_map bodies are loaded through AST to avoid importing
unused neural networks. Input object descriptors come from the shared CG map;
the same descriptor is registered for every contributing keyframe. Post-merge
feature updates are native but do not enter another update in this control.
"""
import ast,json,time,heapq,typing,hashlib
from pathlib import Path
from functools import partial
from collections import deque
import numpy as np
import torch
from . import scenenn_benchmark as bench
from .flat_benchmark import sha256
UPSTREAM=Path('build/benchmarks/adaptive_fusion_20261110/upstream_ovo')

def native_classes():
 env=dict(np=np,torch=torch,heapq=heapq,partial=partial,**{k:getattr(typing,k) for k in ['Any','Dict','List','Tuple']})
 def execute(path,names):
  tree=ast.parse(path.read_text());tree.body=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
  exec(compile(tree,str(path),'exec'),env)
 execute(UPSTREAM/'ovo/entities/instance3d.py',['l1_medoid','cossim_medoid','avg_pooling','camfusion','Instance3D'])
 env['Instance3D'].set_fusion('l1_medoid');env['Instance3D'].n_top_kf=10
 import importlib.util
 spec=importlib.util.spec_from_file_location('native_ovo_instance_utils',UPSTREAM/'ovo/utils/instance_utils.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);env['instance_utils']=module
 tree=ast.parse((UPSTREAM/'ovo/entities/ovo.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='OVO')
 node.body=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name in ['update_map','update_objects_clip','complete_semantic_info']]
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(UPSTREAM/'ovo/entities/ovo.py'),'exec'),env)
 return env['OVO'],env['Instance3D']

def initialize(records,objects):
 cls,instance=native_classes();state=cls.__new__(cls);state.objects={};state.keyframes={'frame_id':list(range(len(records))),'ins_descriptors':{}};state.keyframes_queue=deque();state.th_centroid=1.5;state.th_cossim=.81;state.th_points=.1
 byid={o.observation_id:o for g in records for o in g};points=[];frames=[];point_labels=[];owners=[];offset=0
 for obj in objects:
  oid=int(obj['object_id']);x=instance(oid);feat=torch.tensor(obj['image_feature'],dtype=torch.float32).reshape(1,-1);x.clip_feature=feat.clone()
  for i in obj['observation_ids']:
   o=byid[i];n=len(o.points);points.append(o.points);frames.extend([o.frame_id]*n);point_labels.extend([oid]*n);owners.extend([i]*n)
   x.update(list(range(offset,offset+n)),o.frame_id,n);offset+=n
   state.keyframes['ins_descriptors'].setdefault(o.frame_id,{})[oid]=feat.clone()
  state.objects[oid]=x
 if not points:return state,np.empty((0,3)),np.array([],int),torch.empty(0,dtype=torch.int64),np.array([],int)
 return state,np.concatenate(points),np.array(frames),torch.tensor(point_labels,dtype=torch.int64),np.array(owners)

def correct(state,points,frame_ids,point_labels,owners,poses,nobs):
 world=np.einsum('nij,nj->ni',poses[frame_ids,:3,:3],points)+poses[frame_ids,:3,3]
 data=(torch.tensor(world,dtype=torch.float32),torch.arange(len(world),dtype=torch.int64),point_labels)
 got=state.update_map(data,{k:True for k in range(len(poses))}).numpy();result=np.zeros(nobs,np.int32)
 for oid in np.unique(owners):
  values=np.unique(got[owners==oid]);assert len(values)==1;result[oid]=values[0]
 return result

def run(scene,case,scale):
 torch.set_num_threads(1);cache,records,poses,n=bench.load(scene);before=bench.perturb(poses,case,scale);mapping=bench.factory('native')
 for k,g in enumerate(records):mapping.add_frame(k,g,before[k])
 frozen=bench.labels(mapping,n);expected=np.load(bench.BASE/'runs'/scene/(case+'_'+scale)/'cached_full'/'labels.npz')['frozen'];np.testing.assert_array_equal(frozen,expected)
 initial=mapping.export_objects();trials=[]
 for _ in range(3):
  state,points,frames,point_labels,owners=initialize(records,initial)
  start=time.perf_counter();result=correct(state,points,frames,point_labels,owners,poses,n);elapsed=time.perf_counter()-start
  trials.append(elapsed)
 out=bench.BASE/'ovo_control'/scene/(case+'_'+scale);out.mkdir(parents=True,exist_ok=False);np.save(out/'labels.npy',result)
 bench.dump(out/'run.json',dict(complete=True,scene=scene,case=case,scale=scale,scope=__doc__,initial_objects=len(initial),final_objects=len(state.objects),initial_labels_equal=True,median_update_s=float(np.median(trials)),trials_s=trials,upstream_commit='e0d16f7a459ea5451540c50ac264b00aca3aeb85',source_sha256={str(p):sha256(p) for p in [Path(__file__),UPSTREAM/'ovo/entities/ovo.py',UPSTREAM/'ovo/entities/instance3d.py',UPSTREAM/'ovo/utils/instance_utils.py']}))
 print('COMPLETE OVO CONTROL',scene,case,scale,flush=True)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--case',required=True);p.add_argument('--scale',required=True);a=p.parse_args();run(a.scene,a.case,a.scale)
