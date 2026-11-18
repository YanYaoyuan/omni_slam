"""Correct the sandbox-only GPU conclusion; preserve the original observation."""
import json,subprocess,time,os,sys
from pathlib import Path
import torch
from paper_final_common import ROOT,dump,sha
OUT=ROOT/'build/benchmarks/reviewer_extension_20261117'
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1';assert torch.cuda.is_available() and torch.cuda.device_count()==1
x=torch.arange(1024,device='cuda');assert x.sum().item()==523776;torch.cuda.synchronize()
dump(OUT/'gpu1_preflight.json',dict(complete=True,epoch=time.time(),physical_gpu=1,cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],local_cuda_index=0,gpu_name=torch.cuda.get_device_name(0),torch=torch.__version__,python=sys.version,free_total_bytes=torch.cuda.mem_get_info(),nvidia_smi=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu','--format=csv'],text=True),original_preflight_sha256=sha(OUT/'preflight.json'),correction='Original check ran inside device-restricted sandbox; host GPU1 is available. It was wrong to infer host GPU unavailability. Original record preserved. Use approved host execution with CUDA_VISIBLE_DEVICES=1.',cuda_kernel_test=True,source_sha256=sha(__file__)))
print('GPU1 PREFLIGHT PASSED',flush=True)
