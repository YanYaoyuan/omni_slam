"""Isolated execution of the frozen four-window review extension.

Never modify old experiment roots, thresholds, model weights or reservations.
Stages require immutable seals. All 28 maps precede reference decoding.
"""
import argparse,copy,inspect,shutil,time
from pathlib import Path
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from run_intrinsic_public_holdout import isolated_adapter
B=ROOT/'build/benchmarks';OUT=B/'reviewer_extension_20261117';DATA=ROOT/'build/datasets/reviewer_extension_20261117';PARENT=B/'revision_assessment_v1';META=ROOT/'build/datasets/kitti360_prepare'
METHODS=['strong_baseline','unconditional','history_only','witness_history','nearest_witness_history','full_authorization','MaskClustering_shared_core']

def seal(path):
 d=read(path);assert d['complete'],path
 for f,h in d.get('files',{}).items():assert sha(f)==h,f
 return d

def registration():
 r=read(OUT/'protocol.json');assert sha(DATA/'window_selection.json')==r['selection_sha256'];assert sha(r['scientific_parent'])==r['scientific_parent_sha256']
 for f,h in r['sources'].items():assert sha(f)==h,f
 p=copy.deepcopy(read(PARENT/'protocol.json'));p['methods']=METHODS;p['scenes']=[]
 for w in read(DATA/'window_selection.json')['windows']:
  key='kitti360_'+w['scene'].split('_')[-2];p['scenes'].append(dict(key=key,kind='outdoor',window=w,job=str(OUT/'prepared'/key/'jobs'/(w['scene']+'.json')),opaque_reference=str(DATA/'data/opaque_references'/w['member'])))
 p.update(prompts=['car','building'],semantic_ids=[26,11],sam3_threshold=.5,sam3_resolution=1008,checkpoint=p['sam3_checkpoint'],checkpoint_sha256=p['sam3_sha256'],rotations=read(B/'intrinsic_surface_ownership_v1/protocol.json')['rotations'])
 return p

def verify(data=False):
 p=registration();e=read(OUT/'execution_protocol.json')
 for f,h in {**e['sources'],**e['inputs']}.items():assert sha(f)==h,f
 if data and (OUT/'prepared_complete.json').exists():seal(OUT/'prepared_complete.json')
 return p

def start(name):
 verify();path=OUT/'stages'/(name+'.json');assert not path.exists(),f'Stage exists; inspect terminal/recover explicitly: {path}';dump(path,dict(stage=name,started_epoch=time.time(),execution_protocol_sha256=sha(OUT/'execution_protocol.json')))

def dataset_audit():
 import zlib
 from PIL import Image
 from omni_semantic_map.kitti360_input import load_poses,load_left_calibration,load_times
 p=registration();status=read(DATA/'data/download_status.json');assert status['complete'] and not status['failed'];assert not (DATA/'verification.json').exists();files={};count=0
 for path in (DATA/'data').glob('*_download.json'):
  d=read(path);assert d['complete'] and d['selection_sha256']==sha(DATA/'window_selection.json')
  for x in d['files']:
   f=ROOT/x['path'];assert sha(f)==x['sha256'];blob=f.read_bytes();assert len(blob)==x['bytes'] and zlib.crc32(blob)&0xffffffff==x['crc32'];files[str(f)]=x['sha256'];count+=1
 assert count==492
 cal=load_left_calibration(META/'extracted/calibration');trans=cal['pose_from_cam']@np.linalg.inv(cal['rectification'])@cal['rectified_from_velodyne'];old=read(ROOT/'build/datasets/kitti360_reserved_v4/geometry_separation_audit.json')['boxes'];boxes={k+'/prior':v for k,v in old.items()};results={};pairs=[]
 for scene in p['scenes']:
  w=scene['window'];seq=w['scene'];poses=load_poses(META/'extracted'/seq/'poses.txt');it=load_times(META/'extracted'/seq/'image_00/timestamps.txt');st=load_times(META/'extracted'/seq/'velodyne_points/timestamps.txt');lo=np.full(3,np.inf);hi=-lo;returns=0;offset=[]
  for fid in w['input_frames']:
   path=DATA/'data/data_3d_raw'/seq/'velodyne_points/data'/f'{fid:010d}.bin';raw=np.fromfile(path,dtype='<f4').reshape(-1,4)[:,:3];assert np.isfinite(raw).all();raw=raw[np.linalg.norm(raw,axis=1)>0].astype(float);t=poses[fid]@trans;xyz=raw@t[:3,:3].T+t[:3,3];lo=np.minimum(lo,xyz.min(0));hi=np.maximum(hi,xyz.max(0));returns+=len(raw);offset.append(float((st[fid]-it[fid])/1e6))
   with Image.open(DATA/'data/data_2d_raw'/seq/'image_00/data_rect'/f'{fid:010d}.png') as im:assert im.size==(cal['width'],cal['height']);im.verify()
  box=np.array([lo,hi]);key=scene['key']+'/extension'
  for name,row in boxes.items():
   y=np.array(row['aabb']);dist=float(np.linalg.norm(np.maximum(np.maximum(box[0]-y[1],y[0]-box[1]),0)));pairs.append(dict(a=key,b=name,aabb_lower_bound_m=dist,separated=dist>.2))
  boxes[key]=dict(aabb=box.tolist(),frames=61,raw_returns=returns);results[scene['key']]=dict(frames=61,raw_returns=returns,scan_minus_image_ms=dict(min=min(offset),max=max(offset)))
 proven=all(v['separated'] for v in pairs);dump(DATA/'geometry_separation_audit.json',dict(complete=True,separation_proven=proven,boxes=boxes,pairs=pairs,reference_labels_decoded=False,source_sha256=sha(__file__)))
 dump(DATA/'verification.json',dict(complete=True,files_sha256=files,files=count,results=results,separation_proven=proven,reference_labels_decoded=False,rgb_content_viewed=False));assert proven,'No replacement: raw overlap needs explicit audit before independence claims'
 print('DATA AUDIT',count,'files, separation min',min(x['aabb_lower_bound_m'] for x in pairs),flush=True)

def freeze():
 p=registration();assert not (OUT/'execution_protocol.json').exists();v=read(DATA/'verification.json');assert v['complete'] and v['separation_proven'];assert read(OUT/'gpu1_preflight.json')['complete'];e=read(PARENT/'execution_protocol.json');sources=dict(e['sources']);inputs={str(OUT/'protocol.json'):sha(OUT/'protocol.json'),str(DATA/'verification.json'):sha(DATA/'verification.json'),str(DATA/'geometry_separation_audit.json'):sha(DATA/'geometry_separation_audit.json')}
 # Frozen scientific and model dependencies retained; new roots are registered separately.
 for f,h in e['sources'].items():assert sha(f)==h,f
 for f,h in e['inputs'].items():assert sha(f)==h,f
 inputs.update(e['inputs']);inputs.update(v['files_sha256'])
 for f in [Path(__file__),ROOT/'scripts/benchmarks/review_maskclustering_control.py',ROOT/'scripts/benchmarks/review_causal_revision.py']:sources[str(f)]=sha(f)
 for f in [DATA/'window_selection.json',OUT/'gpu1_preflight.json',OUT/'runtime_golden.json',OUT/'semantic_runtime_golden.json',OUT/'execution_validation/complete.json']:inputs[str(f)]=sha(f)
 assert read(OUT/'runtime_golden.json')['complete'] and read(OUT/'semantic_runtime_golden.json')['complete'] and read(OUT/'execution_validation/complete.json')['complete']
 dump(OUT/'execution_protocol.json',dict(complete=True,sources=sources,inputs=inputs,physical_gpu=1,cuda_visible_devices='1',no_parameter_changes=True,methods=METHODS,scope='Isolated roots and seven registered exports. All28maps sealed before any new reference decoding. Four same-sequence spatial holdouts, not official validation-only.'))
 gold=read(PARENT/'input_golden.json');assert gold['source_sha256']==sha(ROOT/'scripts/benchmarks/prepare_batch_transport_assessment.py');dump(OUT/'input_golden.json',gold)
 for sub in ['raw','ownership_axis','ownership_intrinsic','semantics']:
  d=OUT/sub;d.mkdir(exist_ok=False);dump(d/'protocol.json',p)
 print('EXECUTION SEALED',flush=True)

def runner():
 n=isolated_adapter('run_revision_assessment.py');n.OUT=OUT;n.INPUT=OUT/'measured';n.FULL=OUT/'full_measured';n.OBS=OUT/'observations';n.COVER=OUT/'cover';n.AUTH=OUT/'authorization';n.verify=verify;n.verify_inputs=lambda:verify(True);n.stage_start=start
 native_adapter=n.backbone_adapter
 def annotated_adapter(filename):
  m=native_adapter(filename)
  if filename=='capture_closed_vocabulary_semantics.py':
   original_dump=m.dump
   def scoped_dump(path,data):
    if Path(path).name=='execution.json':data=dict(data,physical_gpu=1,cuda_visible_devices='1')
    return original_dump(path,data)
   m.dump=scoped_dump
  return m
 n.backbone_adapter=annotated_adapter
 assert n.visibility.__globals__ is n.__dict__
 return n

def prepare():
 start('prepare');n=isolated_adapter('prepare_batch_transport_assessment.py');n.OUT=OUT;n.DATA=DATA;n.INPUT=OUT/'measured';n.FULL=OUT/'full_measured';n.verify=verify;n.prepare()
 a=isolated_adapter('prepare_revision_assessment.py');a.OUT=OUT;a.DATA=DATA;a.INPUT=n.INPUT;a.FULL=n.FULL;a.verify=lambda:verify(True)
 # Audit uses input adapter hash only as provenance; provide explicit reused-source seal.
 dump(OUT/'input_adapter_protocol.json',dict(complete=True,source_sha256=sha(ROOT/'scripts/benchmarks/prepare_batch_transport_assessment.py'),isolated_roots=True,prior_golden=str(PARENT/'input_golden.json')));a.audit()

def finalize():
 from omni_semantic_map.batch_stable_transport import nearest_owner
 start('finalize');n=runner();seal(OUT/'predictions_complete.json');seal(n.COVER/'maps_complete.json');seal(n.AUTH/'maps_complete.json');files={};results={}
 for c in n.contexts():
  key=c['key'];base=dict(np.load(Path(c['ours'])/'geometry.npz'));parent=dict(np.load(n.COVER/'maps'/key/'cover_history/geometry.npz'));full=dict(np.load(n.AUTH/'maps'/key/'ours/geometry.npz'));d=np.load(n.AUTH/'prepared'/key/'write_decisions.npz');f=np.load(n.AUTH/'prepared'/key/'field.npz');nold=len(base['xyz']);history=dict(parent);history['instance']=np.r_[full['instance'][:nold],parent['instance'][nold:]];geometries=dict(strong_baseline=base,unconditional=parent,history_only=history,full_authorization=full)
  source=parent['added_full_source_ids'];cells=np.load(OUT/'fields'/key/'field.npz')['source_to_cell'];nodes=f['cell_atom'][cells[source]];owners=nearest_owner(f['nearest_owner'],f['owner_distance']);near=np.full(len(source),-1,np.int64);near[d['direct']]=owners[nodes[d['direct']]]
  for method in ['witness_history','nearest_witness_history']:
   keep=d['direct']&d['witness'];labels=parent['instance'][nold:]
   if method=='nearest_witness_history':keep&=near>=0;labels=near
   geometries[method]=dict(xyz=np.r_[base['xyz'],parent['xyz'][nold:][keep]],instance=np.r_[full['instance'][:nold],labels[keep]],added_full_source_ids=source[keep])
  results[key]={}
  for method,g in geometries.items():
   dest=OUT/'comparison_maps'/key/method;dest.mkdir(parents=True,exist_ok=False);np.testing.assert_array_equal(g['xyz'][:nold],base['xyz']);assert np.isfinite(g['xyz']).all();np.savez_compressed(dest/'geometry.npz',**g);shutil.copy2(Path(c['ours'])/'classification.json' if method=='strong_baseline' else n.AUTH/'maps'/key/'ours/classification.json',dest/'classification.json');row=dict(points=len(g['xyz']),old_point_id_changes=int(np.count_nonzero(g['instance'][:nold]!=base['instance'])),nonempty_ids=len(set(g['instance'])-{0}));dump(dest/'run.json',row);results[key][method]=row
   for fpath in dest.iterdir():files[str(fpath)]=sha(fpath)
 dump(OUT/'internal_maps_complete.json',dict(complete=True,files=files,results=results,final_maps=24,reference_labels_decoded=False));print('24 INTERNAL MAPS SEALED',flush=True)

def external():
 start('external');seal(OUT/'internal_maps_complete.json');n=isolated_adapter('review_maskclustering_control.py');n.SRC=OUT;n.OUT=OUT/'external';n.SCENES=[s['key'] for s in verify()['scenes']];n.verify=verify;n.OUT.mkdir(exist_ok=False);shutil.copy2(OUT/'protocol.json',n.OUT/'protocol.json');n.maps();files=dict(seal(OUT/'internal_maps_complete.json')['files']);results=read(OUT/'internal_maps_complete.json')['results']
 for key in n.SCENES:
  src=n.OUT/'maps'/key;dest=OUT/'comparison_maps'/key/'MaskClustering_shared_core';shutil.copytree(src,dest);results[key]['MaskClustering_shared_core']=read(dest/'run.json')
  for f in dest.iterdir():files[str(f)]=sha(f)
 dump(OUT/'comparison_maps_complete.json',dict(complete=True,files=files,results=results,final_maps=28,reference_labels_decoded=False));print('ALL28MAPS SEALED',flush=True)

def score():
 start('score');sealed=seal(OUT/'comparison_maps_complete.json');assert sealed['final_maps']==28
 n=isolated_adapter('evaluate_revision_assessment.py');n.OUT=OUT;n.FULL=OUT/'full_measured';n.COVER=OUT/'cover';n.AUTH=OUT/'authorization';n.verify=verify;n.METHODS=METHODS
 inputs=dict(sealed['files']);inputs[str(OUT/'comparison_maps_complete.json')]=sha(OUT/'comparison_maps_complete.json')
 for s in verify()['scenes']:
  for f in [s['opaque_reference'],s['job']]:inputs[f]=sha(f)
 sources={str(ROOT/'scripts/benchmarks/evaluate_revision_assessment.py'):sha(ROOT/'scripts/benchmarks/evaluate_revision_assessment.py'),str(Path(__file__)):sha(__file__)}
 dump(OUT/'scoring_protocol.json',dict(complete=True,inputs=inputs,sources=sources,methods=METHODS,modes=n.MODES,reference_labels_decoded=False,scope='Four spatial holdout windows; 28 sealed maps before any new reference decoding. Reuse fixed scoring kernel, no new acceptance-gate selection. All outcomes retained.'))
 code=inspect.getsource(n.score)
 for a,b in [('assert count == 24','assert count == 84'),('finite_windows == 2','finite_windows == 4'),('new_final_maps=8, new_public_windows=2','new_final_maps=28, new_public_windows=4'),('prospective_gate=gate','descriptive_parent_criterion_on_four_windows=gate, independent_new_sequences=False, spatial_holdout=True')]:
  assert code.count(a)==1,a;code=code.replace(a,b)
 exec(compile(code,str(__file__)+':score_counts','exec'),n.__dict__);n.score()


def run(action):
 if action in ['dataset_audit','freeze','prepare','finalize','external','score']:return globals()[action]()
 n=runner()
 if action=='visibility':return n.visibility()
 if action in ['atoms','cover_prepare','cover_maps','identity_prepare','identity_maps']:return n.revision(action)
 b=n.backbone()
 if action in ['masks','features','supports']:return b.frontend(action)
 return getattr(b,'maps' if action=='strong_maps' else action)()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action');args=p.parse_args();run(args.action)
