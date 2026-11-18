import sys,tempfile
from pathlib import Path
sys.path[:0]=['scripts/benchmarks','omni_semantic_map'];from run_object_repair_v2 import repair,control,ROOT,dump,SupportGraphConfig
sys.path.insert(0,str(ROOT/'build/benchmarks/expanded_semantic_20261109/concept-graphs'))
import numpy as np,torch
torch.set_num_threads(1)
# Two contiguous measured support regions, first seen under one mask, then split.
x=np.arange(0,.4,.0125);y=np.arange(0,.1,.0125);z=np.arange(0,.05,.0125)
xyz=np.array(np.meshgrid(x,y,z,indexing='ij')).reshape(3,-1).T.astype(np.float32);n=len(xyz);left=np.flatnonzero(xyz[:,0]<.2);right=np.flatnonzero(xyz[:,0]>=.2)
with tempfile.TemporaryDirectory(prefix='repair_v2_smoke_') as tmp:
 p=Path(tmp);cache=p/'cache';(cache/'frames').mkdir(parents=True)
 records=[];features=[];frames=[];areas=[];offset=0
 for frame in range(5):
  pieces=[np.arange(n)] if frame==0 else [left,right];ids=np.concatenate(pieces);offsets=np.r_[0,np.cumsum([len(a) for a in pieces])]
  name=f'{frame:06d}.npz';np.savez_compressed(cache/'frames'/name,point_ids=ids,offsets=offsets)
  records.append(dict(file=name,masks=len(pieces),observation_offset=offset,frame_id=frame));offset+=len(pieces)
  features.extend([[1.,0.]]*len(pieces));frames.extend([frame]*len(pieces));areas.extend([len(a) for a in pieces])
 data=(cache,dict(records=records),dict(xyz=xyz,colors=np.ones((n,3),np.uint8)*128),np.array(features,np.float32),np.array(frames),np.array(areas),np.array([[1.,0.],[0.,1.]],np.float32),dict(names=['chair','table'],replica_class_ids=[14,20]),np.tile(np.eye(4),(5,1,1)))
 entry=dict(key='synthetic_no_GT',config=str(ROOT/'docs/paper/frozen_v1/config/cg_rgbd.json'))
 for name in ['constraint_control','repair','no_revoke']:
  target=p/name;target.mkdir()
  result=control(entry,target,data) if name=='constraint_control' else repair(entry,target,data,name=='repair',SupportGraphConfig())
  assert len(result['frame_times_s'])==5
  print('ADAPTER SMOKE PASS',name,'violations',result['confirmed_constraint_violations'],flush=True)
