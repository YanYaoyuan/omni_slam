"""GPU1 runtime regression on exposed cached frames before any new-window inference."""
import argparse,copy,time
from pathlib import Path
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from run_intrinsic_public_holdout import isolated_adapter
from run_reviewer_extension import OUT,PARENT
PROBE=OUT/'runtime_probe'

def setup():
 assert not PROBE.exists();p=copy.deepcopy(read(PARENT/'protocol.json'));p.update(prompts=['car','building'],semantic_ids=[26,11],sam3_threshold=.5,sam3_resolution=1008)
 for s in p['scenes']:
  m=read(PARENT/'measured'/s['key']/'manifest.json');m['records']=[m['records'][0],m['records'][-1]];dump(PROBE/'measured'/s['key']/'manifest.json',m)
 dump(PROBE/'protocol.json',p)

def run(action):
 if action=='setup':return setup()
 p=read(PROBE/'protocol.json');n=isolated_adapter('run_discovery_query_benchmark.py');n.OUT=PROBE;n.INPUT=PROBE/'measured';n.verify=lambda data=False:p;getattr(n,action)();results={}
 if action=='masks':
  for k,row in read(PROBE/'masks_complete.json')['scenes'].items():
   for r in row['records']:
    got=np.load(r['file']);old=PARENT/'frontend'/k/'masks'/Path(r['file']).name;want=np.load(old)
    for key in got.files:np.testing.assert_array_equal(got[key],want[key])
    results[k+'/'+str(r['frame_id'])]=dict(all_arrays_exact=True,reference_sha256=sha(old))
 else:
  for k,rows in read(PROBE/'features_complete.json')['timings'].items():
   for r in rows:
    f=f'{r["frame_id"]:06d}.npy';got=np.load(PROBE/'frontend'/k/'features'/f);old=PARENT/'frontend'/k/'features'/f;want=np.load(old);np.testing.assert_array_equal(got,want);results[k+'/'+str(r['frame_id'])]=dict(all_arrays_exact=True,reference_sha256=sha(old))
 dump(PROBE/(action+'_golden.json'),dict(complete=True,results=results,source_sha256=sha(__file__),physical_gpu=1,new_window_frames=0))
 if (PROBE/'masks_golden.json').exists() and (PROBE/'features_golden.json').exists():dump(OUT/'runtime_golden.json',dict(complete=True,mask_and_feature_arrays_exact=True,exposed_frames=4,new_window_inference=False,checks={x:sha(PROBE/(x+'_golden.json')) for x in ['masks','features']},source_sha256=sha(__file__)))
 print('RUNTIME GOLDEN',action,results,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['setup','masks','features']);run(p.parse_args().action)
