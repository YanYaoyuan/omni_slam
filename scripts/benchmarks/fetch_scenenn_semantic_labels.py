"""Fetch only six official CVPR18 ZIP members using bounded HTTP ranges."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import urllib.request
import zlib
from audit_evidence_retention import OUT, dump

URL='https://hkust-vgd.ust.hk/scenenn/home/cvpr18/data/scenenn_seg_76.zip'


def fetch(start,end):
    with urllib.request.urlopen(urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}'}),timeout=60) as r:
        assert r.status==206 and r.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
        b=r.read();assert len(b)==end-start+1;return b


def main():
    folder=OUT/'official_annotation_audit';entries=json.loads((folder/'cvpr18_zip_contents.json').read_text())
    wanted=[x for x in entries if any(x['name'].endswith('_'+s+'.hdf5') for s in ['014','015','021'])]
    assert len(wanted)==6
    output=folder/'cvpr18';output.mkdir(exist_ok=False);manifest=dict(complete=False,url=URL,files=[])
    for entry in wanted:
        head=fetch(entry['offset'],entry['offset']+29);assert head[:4]==b'PK\x03\x04'
        n,e=struct.unpack('<HH',head[26:30]);start=entry['offset']+30+n+e;size=entry['compressed'];step=4*1024**2
        ranges=[(i,min(i+step,start+size)-1) for i in range(start,start+size,step)]
        with ThreadPoolExecutor(4) as pool:compressed=b''.join(pool.map(lambda x:fetch(*x),ranges))
        assert entry['method']==8;raw=zlib.decompress(compressed,-15)
        assert len(raw)==entry['size'] and zlib.crc32(raw)==entry['crc']
        path=output/Path(entry['name']).name;path.write_bytes(raw)
        manifest['files'].append(dict(**entry,path=str(path),sha256=hashlib.sha256(raw).hexdigest()))
        dump(output/'download.json',manifest);print('DOWNLOADED',path.name,len(raw),flush=True)
    manifest['complete']=True;dump(output/'download.json',manifest)


if __name__=='__main__':main()
