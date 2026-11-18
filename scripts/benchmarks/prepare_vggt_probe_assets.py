"""Fetch one pinned official VGGT checkpoint and code, without installation."""
import hashlib
import io
from pathlib import Path
import shutil
import tarfile
import time
import urllib.request
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from paper_final_common import ROOT, dump

OUT=ROOT/'build/benchmarks/vggt_object_probe_v1'
CODE='a288dd0f14786c93483e45524328726ab7b1b4ce'
MODEL='860abec7937da0a4c03c41d3c269c366e82abdf9'
WEIGHT_SHA='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
SIZE=5026874952


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    code=OUT/'upstream'
    if not code.exists():
        url=f'https://codeload.github.com/facebookresearch/vggt/tar.gz/{CODE}'
        with urllib.request.urlopen(url,timeout=90) as r:
            archive=tarfile.open(fileobj=r,mode='r|gz')
            for item in archive:
                parts=Path(item.name).parts[1:]
                if not parts or not item.isfile():continue
                name=Path(*parts)
                if '..' in parts:raise ValueError('Unsafe path')
                if not (parts[0]=='vggt' or name.name in ['LICENSE.txt','LICENSE','README.md','requirements.txt','pyproject.toml']):continue
                target=code/name;target.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(item) as src,target.open('wb') as dest:shutil.copyfileobj(src,dest)
        dump(OUT/'code.json',dict(commit=CODE,url=url,files={str(p.relative_to(code)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code.rglob('*') if p.is_file()}))
    target=OUT/'model.pt';part=OUT/'model.pt.part'
    if not target.exists():
        offset=part.stat().st_size if part.exists() else 0
        progress=OUT/'download_chunks.json';chunk_size=4*1024**2
        if progress.exists():
            state=json.loads(progress.read_text());assert state['model']==MODEL and state['chunk_size']==chunk_size
            completed=set(state['completed'])
        else:completed=set(range(offset//chunk_size))
        remaining=SIZE-sum(min(chunk_size,SIZE-i*chunk_size) for i in completed)
        if shutil.disk_usage(OUT).free < remaining+2*1024**3:raise RuntimeError('Need checkpoint plus2GiB reserve')
        url=f'https://huggingface.co/facebook/VGGT-1B/resolve/{MODEL}/model.pt'
        def save():dump(progress,dict(model=MODEL,chunk_size=chunk_size,completed=sorted(completed)))
        save()
        fd=os.open(part,os.O_RDWR|os.O_CREAT,0o644)
        def fetch(i):
            first=i*chunk_size;last=min(SIZE,first+chunk_size)-1
            for attempt in range(5):
                try:
                    request=urllib.request.Request(url+f'?chunk={i}',headers={'Range':f'bytes={first}-{last}'})
                    with urllib.request.urlopen(request,timeout=25) as r:
                        if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {first}-{last}/{SIZE}':raise RuntimeError('Invalid content range')
                        data=r.read(last-first+2)
                    if len(data)!=last-first+1:raise RuntimeError('Incomplete range')
                    cursor=0
                    while cursor<len(data):cursor+=os.pwrite(fd,data[cursor:],first+cursor)
                    return i
                except Exception:
                    if attempt==4:raise
                    time.sleep(min(2**attempt,8))
        total=(SIZE+chunk_size-1)//chunk_size
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures=[pool.submit(fetch,i) for i in range(total) if i not in completed]
                for future in as_completed(futures):
                    completed.add(future.result());save()
                    if len(completed)%64==0:print('CHUNKS',len(completed),'/',total,flush=True)
            os.fsync(fd)
        finally:os.close(fd)
        assert len(completed)==total
        if part.stat().st_size!=SIZE:raise RuntimeError('Incomplete checkpoint')
        h=hashlib.sha256()
        with part.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
        assert h.hexdigest()==WEIGHT_SHA,'Official LFS SHA256 mismatch'
        part.rename(target)
    dump(OUT/'assets.json',dict(complete=True,code_commit=CODE,model_revision=MODEL,
        checkpoint=str(target),size_bytes=SIZE,sha256=WEIGHT_SHA,
        source='https://huggingface.co/facebook/VGGT-1B',code='https://github.com/facebookresearch/vggt'))
    print('ASSETS COMPLETE',flush=True)


if __name__=='__main__':main()
