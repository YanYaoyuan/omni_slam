"""One bounded GPU1 pipeline; stop on any error, never tune/retry silently."""
import os,sys,time,subprocess,json,fcntl
from pathlib import Path
from paper_final_common import ROOT,read,dump,sha
from run_reviewer_extension import OUT,DATA
GPU=ROOT/'build/benchmarks/kitti360_reserved_v1/runtime/bin/python';CPU=ROOT/'build/benchmarks/expanded_semantic_20261109/cg_venv/bin/python';DRIVER=ROOT/'scripts/benchmarks/run_reviewer_extension.py'
STAGES=['dataset_audit','freeze','prepare','masks','features','supports','base_maps','candidates','intrinsic','raw','geometry','anchors','semantics','aggregate','fields','strong_maps','visibility','atoms','cover_prepare','cover_maps','identity_prepare','identity_maps','finalize','external','score']
GPU_STAGES={'masks','features','raw','semantics'}

def main():
 lock=(OUT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'queue_status.json').exists(),'Inspect prior status; no silent restart'
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HOME=str(ROOT/'build/benchmarks/ovo_full_20261110/hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MPLCONFIGDIR=str(OUT/'mpl_cache'),PYTHONUNBUFFERED='1')
 gpupath=':'.join(map(str,[ROOT/'build/benchmarks/sam3_probe_runtime/deps',ROOT/'build/benchmarks/ovo_full_20261110/venv311/lib/python3.11/site-packages',Path('/home/user/vbot/sam3'),ROOT/'omni_semantic_map',ROOT/'scripts/benchmarks']))
 cpupath=':'.join(map(str,[ROOT/'omni_semantic_map',ROOT/'scripts/benchmarks']))
 state=dict(complete=False,failed=False,pid=os.getpid(),physical_gpu=1,started_epoch=time.time(),stage='waiting_inputs',completed_stages=[],driver_sha256=sha(DRIVER),queue_sha256=sha(__file__),no_parameter_search=True)
 def save():state['updated_epoch']=time.time();dump(OUT/'queue_status.json',state)
 save()
 try:
  deadline=time.monotonic()+7200
  while True:
   d=read(DATA/'data/download_status.json')
   if d.get('failed'):raise RuntimeError('Download failed; inspect download_status.json; no new inference')
   ready=d.get('complete') and all((OUT/f).exists() and read(OUT/f).get('complete') for f in ['runtime_golden.json','semantic_runtime_golden.json','execution_validation/complete.json'])
   if ready:break
   if time.monotonic()>deadline:raise TimeoutError('Inputs not ready after2h; no inference')
   state['download_completed']=d.get('completed',[]);save();time.sleep(5)
  for action in STAGES:
   assert sha(DRIVER)==state['driver_sha256'],'Driver changed after queue start'
   state['stage']=action;state['stage_started_epoch']=time.time();save();log=OUT/'logs'/(action+'.log');python=GPU if action in GPU_STAGES else CPU;current=dict(env,PYTHONPATH=gpupath if action in GPU_STAGES else cpupath)
   with log.open('w') as output:
    proc=subprocess.Popen([str(python),str(DRIVER),action],cwd=ROOT,env=current,stdout=output,stderr=subprocess.STDOUT);state['child_pid']=proc.pid;save();code=proc.wait()
   state['child_exit_code']=code;save()
   if code:raise RuntimeError(f'{action} exited{code}; inspect{log}')
   state['completed_stages'].append(action);save()
  state.update(complete=True,stage='complete',finished_epoch=time.time(),summary=str(OUT/'summary.json'));save()
 except BaseException as e:
  state.update(failed=True,error=repr(e),finished_epoch=time.time());save();raise

if __name__=='__main__':main()
