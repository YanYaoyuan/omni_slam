"""Acquire pinned public SAM-V release weights; reuse the user's VGGT file."""
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'build/benchmarks/samv_external_v1'
SAMV_REVISION = '192c8e21f6ffba23148bbac2c457511a0b3e8604'
SAMV_SHA = '0e3858ffe2a04b739243a62a38453df3dc1721e0762a5a0be214286acb0af90a'
SAM_SHA = 'a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def download(name, url, expected=None, size=None, prefix=None):
    path=OUT/'weights'/name;path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        value=digest(path)
        assert (expected is None or value==expected) and (prefix is None or value.startswith(prefix))
        assert size is None or path.stat().st_size==size
        return dict(path=str(path),bytes=path.stat().st_size,sha256=value,url=url,reused=True)
    assembled=path.with_suffix(path.suffix+'.assembled')
    if assembled.exists():
        # The filename's4b8939 is MD5, not SHA256. Preserve the failed check log;
        # require the full published SHA256 before accepting downloaded bytes.
        assert expected is not None and digest(assembled)==expected
        assert size is None or assembled.stat().st_size==size
        assembled.rename(path)
        return dict(path=str(path),bytes=path.stat().st_size,sha256=expected,url=url,reused=True,
                    verification_recovery='FilenameMD5prefixwasincorrectlytreatedasSHA256;fullpublishedSHA256verifiedbeforeacceptance')
    part=path.with_suffix(path.suffix+'.part')
    if part.exists():
        # Existing transfer was confirmed terminal before enabling this path.
        # Require exact server ranges; never append a full 200 response.
        start=part.stat().st_size
        req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{start}','User-Agent':'omni_slam-research/1.0'})
        with urllib.request.urlopen(req,timeout=45) as response:
            assert response.status==206
            content_range=response.headers['Content-Range']
            assert content_range.startswith(f'bytes {start}-{start}/')
            total_size=int(content_range.split('/')[1])
        chunks=path.parent/(name+'.ranges');chunks.mkdir(exist_ok=True)
        ranges=[(lo,min(lo+64*1024**2,total_size)-1) for lo in range(start,total_size,64*1024**2)]
        def fetch(bounds):
            lo,hi=bounds;dest=chunks/f'{lo:012d}-{hi:012d}'
            if dest.exists():
                assert dest.stat().st_size==hi-lo+1
                return dest
            request=urllib.request.Request(url,headers={'Range':f'bytes={lo}-{hi}','User-Agent':'omni_slam-research/1.0'})
            pending=dest.with_suffix('.partial')
            with urllib.request.urlopen(request,timeout=90) as response, pending.open('wb') as target:
                assert response.status==206 and response.headers['Content-Range']==f'bytes {lo}-{hi}/{total_size}'
                for block in iter(lambda:response.read(1024**2),b''):target.write(block)
            assert pending.stat().st_size==hi-lo+1
            pending.rename(dest);print('RANGE_VERIFIED',name,lo,hi,flush=True)
            return dest
        with ThreadPoolExecutor(max_workers=4) as pool:completed=list(pool.map(fetch,ranges))
        assembled=path.with_suffix(path.suffix+'.assembled')
        with assembled.open('wb') as target:
            for source in [part]+completed:
                with source.open('rb') as f:
                    for block in iter(lambda:f.read(8*1024**2),b''):target.write(block)
        assert assembled.stat().st_size==total_size
        value=digest(assembled)
        assert expected is None or value==expected
        assert prefix is None or value.startswith(prefix)
        assert size is None or total_size==size
        assembled.rename(path)
        print('VERIFIED_RESUMED',name,total_size,value,flush=True)
        return dict(path=str(path),bytes=total_size,sha256=value,url=url,reused=False,resumed_prefix_bytes=start)
    h=hashlib.sha256();total=0;last=time.monotonic()
    request=urllib.request.Request(url,headers={'User-Agent':'omni_slam-research/1.0'})
    with urllib.request.urlopen(request,timeout=45) as response, part.open('xb') as target:
        expected_length=response.headers.get('Content-Length')
        for block in iter(lambda:response.read(8*1024**2),b''):
            target.write(block);h.update(block);total+=len(block)
            if time.monotonic()-last>20:
                print(name,'downloaded_MiB',round(total/1024**2),flush=True);last=time.monotonic()
    value=h.hexdigest()
    assert expected_length is None or total==int(expected_length)
    assert size is None or total==size
    assert expected is None or value==expected
    assert prefix is None or value.startswith(prefix)
    part.rename(path)
    print('VERIFIED',name,total,value,flush=True)
    return dict(path=str(path),bytes=total,sha256=value,url=url,reused=False)


def main():
    assert not (OUT/'assets.json').exists()
    rows={}
    rows['samv_stage2']=download('sam_v_stage2.pth',
        f'https://huggingface.co/Frank-Gong123/SAM-V/resolve/{SAMV_REVISION}/sam_v_stage2.pth',SAMV_SHA,29948278)
    rows['sam_vit_h']=download('sam_vit_h_4b8939.pth',
        'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',expected=SAM_SHA,size=2564550879)
    rows['sam_vit_h']['checksum_crosscheck']='https://huggingface.co/sam-hq-team/sam-hq-training/blob/main/pretrained_checkpoint/sam_vit_h_4b8939.pth'
    vggt=json.loads((ROOT/'build/benchmarks/vggt_object_probe_v1/assets.json').read_text())
    assert digest(Path(vggt['checkpoint']))==vggt['sha256']
    rows['vggt']=dict(path=vggt['checkpoint'],sha256=vggt['sha256'],bytes=vggt['size_bytes'],reused=True)
    result=dict(complete=True,models=rows,samv_model_revision=SAMV_REVISION,
        code_commit='33fab24c1d0d21ac56f4e471abf30c6de3e7018b',created_epoch=time.time(),
        source_sha256=digest(Path(__file__)),no_inference=True)
    (OUT/'assets.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
