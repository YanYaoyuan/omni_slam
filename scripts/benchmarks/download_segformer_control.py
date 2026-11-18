"""Download a fixed public semantic control; no repository code execution."""
import hashlib,json,time,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'build/models/segformer_b5_cityscapes'
REPO='nvidia/segformer-b5-finetuned-cityscapes-1024-1024'
REVISION='2c6f153e4c23c229e2fa2b188eb250607e030cd8'
WEIGHT_SHA='bdefc7b5bf4b548d347e6bcbe3ab1a0ff4370047211ae9e56c9e53dfb1f78033'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true',timeout=20) as response:meta=json.load(response)
    assert meta['sha']==REVISION
    entries={f['rfilename']:f for f in meta['siblings']};files={}
    for name in ['config.json','preprocessor_config.json','README.md','pytorch_model.bin']:
        dest=OUT/name;entry=entries[name];url=f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}?download=true'
        if not dest.exists():
            partial=dest.with_suffix(dest.suffix+'.partial');assert not partial.exists(),'Inspect incomplete download before retrying'
            with urllib.request.urlopen(url,timeout=45) as response, partial.open('wb') as stream:
                while True:
                    data=response.read(1024*1024)
                    if not data:break
                    stream.write(data)
            assert partial.stat().st_size==entry['size']
            data=partial.read_bytes();digest=hashlib.sha256(data).hexdigest()
            if 'lfs' in entry:assert digest==entry['lfs']['sha256']==WEIGHT_SHA
            else:assert hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()==entry['blobId']
            partial.replace(dest)
        data=dest.read_bytes();digest=hashlib.sha256(data).hexdigest();assert len(data)==entry['size']
        if name=='pytorch_model.bin':assert digest==WEIGHT_SHA
        files[name]=dict(sha256=digest,bytes=len(data),url=url)
        print('VERIFIED',name,len(data),flush=True)
    (OUT/'source_manifest.json').write_text(json.dumps(dict(repository=REPO,revision=REVISION,download_epoch=time.time(),files=files,scope='Public Cityscapes supervised external semantic control; not Ours or KITTI360 benchmark result.'),indent=2)+'\n')
    print('CONTROL MODEL READY',OUT,flush=True)


if __name__=='__main__':main()
