"""Deterministic native implementation of the whole-region objective.

The algebraic gain cancels noncommon category terms. Floating reduction order
can differ from the Python reference at numerical ties; no bitwise parity
claim. Tests replay each selected gain and greedy optimality independently.
"""
import ctypes
import hashlib
from pathlib import Path
import subprocess
import numpy as np


SOURCE=Path(__file__).with_suffix('.cpp')


def library(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    target=directory/('libcensored_'+digest[:16]+'.so')
    if not target.exists():
        subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(SOURCE),'-o',str(target)],check=True)
    lib=ctypes.CDLL(str(target))
    i64=np.ctypeslib.ndpointer(dtype=np.int64,ndim=1,flags='C_CONTIGUOUS')
    f64=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')
    lib.partition_regions.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_int64,i64,i64,
        ctypes.c_int64,i64,i64,i64,f64,i64,i64,i64,i64,f64,ctypes.POINTER(ctypes.c_int64),
        f64,ctypes.POINTER(ctypes.c_char),ctypes.c_int]
    lib.partition_regions.restype=ctypes.c_int
    return lib


def partition_native(count,categories,left,right,atom,view,token,mass,lib):
    raw=[categories,left,right,atom,view,token]
    if not all(np.asarray(v).ndim==1 and np.issubdtype(np.asarray(v).dtype,np.integer) for v in raw):
        raise ValueError('One-dimensional integer arrays required')
    categories,left,right,atom,view,token=[np.ascontiguousarray(v,np.int64) for v in raw]
    mass=np.ascontiguousarray(mass,float)
    if left.shape!=right.shape or not(atom.shape==view.shape==token.shape==mass.shape):
        raise ValueError('Aligned observations and graph required')
    if count<0 or count>=2**31 or len(categories)>=2**31:
        raise ValueError('Native index range exceeded')
    labels=np.empty(count,np.int64);aa=np.empty(count,np.int64);bb=np.empty(count,np.int64)
    gains=np.empty(count,float);size=ctypes.c_int64();metrics=np.empty(5,float);error=ctypes.create_string_buffer(512)
    status=lib.partition_regions(count,len(categories),len(left),left,right,len(atom),atom,view,token,mass,
        categories,labels,aa,bb,gains,ctypes.byref(size),metrics,error,len(error))
    if status:raise ValueError(error.value.decode())
    trace=[(int(a),int(b),float(g)) for a,b,g in zip(aa[:size.value],bb[:size.value],gains[:size.value])]
    return labels,dict(merges=trace,initial_score=float(metrics[0]),final_score=float(metrics[1]),
        score_gain=float(metrics[2]),maximum_remaining_gain=float(metrics[3]),
        gain_evaluations=int(metrics[4]),regions=count-size.value,
        objective_ledger_residual=float(metrics[1]-metrics[0]-metrics[2]),backend='native_algebraic_gain')
