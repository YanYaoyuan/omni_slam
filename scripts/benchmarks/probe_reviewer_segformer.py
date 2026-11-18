"""Replay the frozen dense semantic capture on the four exposed runtime frames."""
from pathlib import Path
import numpy as np
from paper_final_common import read,dump,sha
from probe_reviewer_gpu_runtime import OUT,PARENT,PROBE
from run_intrinsic_public_holdout import isolated_adapter
n=isolated_adapter('capture_closed_vocabulary_semantics.py');n.OUT=PROBE/'semantics';n.INPUT=PROBE/'measured';n.BASE=PARENT/'baseline';p=read(PROBE/'protocol.json');n.verify=lambda:p
original=n.dump
def annotated(path,data):
 if Path(path).name=='execution.json':data=dict(data,physical_gpu=1,cuda_visible_devices='1')
 return original(path,data)
n.dump=annotated;n.OUT.mkdir(exist_ok=False);dump(n.OUT/'protocol.json',p);n.predict();results={}
for k,rows in read(n.OUT/'predictions_complete.json')['results'].items():
 for r in rows:
  got=np.load(r['file']);f=PARENT/'semantics/frames'/k/Path(r['file']).name;want=np.load(f)
  for name in got.files:np.testing.assert_array_equal(got[name],want[name])
  results[k+'/'+str(r['frame_id'])]=dict(all_arrays_exact=True,reference_sha256=sha(f))
dump(OUT/'semantic_runtime_golden.json',dict(complete=True,results=results,new_window_frames=0,physical_gpu=1,source_sha256=sha(__file__)))
print('SEMANTIC GOLDEN PASSED',flush=True)
