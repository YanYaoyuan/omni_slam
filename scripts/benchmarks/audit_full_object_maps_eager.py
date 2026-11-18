"""Run the unchanged frozen audit with each NPZ decompressed once per open.

The original audit repeatedly accesses NpzFile members inside token loops;
NumPy decompresses those members on every access. This execution-only adapter
changes I/O caching, not arrays, label rules, predictions, metrics or checks.
"""
from pathlib import Path
import time
import numpy as np
import audit_full_object_maps as original


class LoadedArchive(dict):
    @property
    def files(self):return list(self)
    def __enter__(self):return self
    def __exit__(self,*args):return False


def main():
    load=np.load; count=0
    def eager(*args,**kwargs):
        nonlocal count
        value=load(*args,**kwargs)
        if not isinstance(value,np.lib.npyio.NpzFile):return value
        try:
            result=LoadedArchive((key,value[key]) for key in value.files);count+=1
        finally:value.close()
        return result
    # Check actual exported schema with both readers before adapting execution.
    row=original.read(original.OUT/'supervision_summary.json')['scenes'][0]
    manifest=original.read(row['manifest']);path=Path(row['manifest']).parent/manifest['records'][0]['file']
    cached=eager(path)
    with load(path) as direct:
        assert cached.files==direct.files
        for key in direct.files:np.testing.assert_array_equal(cached[key],direct[key])
    start=time.monotonic();np.load=eager
    try:original.audit()
    finally:np.load=load
    original.dump(original.OUT/'verification_runtime.json',dict(complete=True,
        adaptation='Eager NPZ materialization; unchanged frozen audit and exact arrays.',
        original_attempt='Interrupted before completion due to repeated decompression inside per-token loops; no model/label/score changes.',
        source_sha256=original.sha(Path(__file__)),npz_opens=count,
        schema_array_equivalence_checked=True,wall_s=time.monotonic()-start,
        audit_sha256=original.sha(original.OUT/'audit.json')))


if __name__=='__main__':main()
