"""Fetch fixed raw sequences; never inspect final-test semantic values here."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import struct
import urllib.request
import zlib
from paper_final_common import ROOT,RUN,FROZEN,STORAGE,OLD,SCENES,read,sha,dump,verify_execution,run_scenes
from category_holdout_data import fetch

URL='https://hkust-vgd.ust.hk/scenenn/'


def initialize():
    RUN.mkdir(parents=True,exist_ok=True)
    assert read(FROZEN/'protocol.json')['final_test_scenes']==SCENES
    for name in ['data','frontend']:
        target=STORAGE/name;target.mkdir(parents=True,exist_ok=True)
        link=RUN/name
        if not link.exists():link.symlink_to(target,target_is_directory=True)
        assert link.resolve()==target
    path=RUN/'selection.json'
    selection=dict(scenes=SCENES,paper_protocol_sha256=sha(FROZEN/'protocol.json'),
        paper_manifest_sha256=sha(FROZEN/'manifest.json'),storage=str(STORAGE),
        no_replacement=True,no_gt_in_inference=True,
        selection='Exact six IDs fixed by paper-v1 before this execution; annotations only decoded after all predictions.')
    if path.exists():assert read(path)==selection
    else:dump(path,selection)


def download():
    initialize();records=[]
    names=['main/intrinsic/asus.ini']+[f'main/{s}/{f}' for s in SCENES for f in [s+'.ply',s+'.xml','trajectory.log']]+[f'main/oni/{s}.oni' for s in run_scenes()]
    for name in names:
        p=RUN/'data'/name;p.parent.mkdir(parents=True,exist_ok=True)
        with urllib.request.urlopen(urllib.request.Request(URL+name,method='HEAD'),timeout=45) as resp:
            size=int(resp.headers['Content-Length'])
        assert size<8*1024**3 and shutil.disk_usage(STORAGE).free>size+10*1024**3
        if not p.exists():
            part=p.with_suffix(p.suffix+'.part')
            # A partial file is only resumed at a completed range boundary.
            step=8*1024**2;done=part.stat().st_size if part.exists() else 0
            assert done<=size and (done%step==0 or done==size)
            with part.open('ab') as f:
                ranges=[(i,min(i+step,size)-1) for i in range(done,size,step)]
                for first in range(0,len(ranges),4):
                    with ThreadPoolExecutor(4) as pool:
                        for raw in pool.map(lambda pair:fetch(URL+name,*pair),ranges[first:first+4]):f.write(raw);f.flush()
                    dump(RUN/'download_progress.json',dict(file=name,completed_bytes=f.tell(),size=size))
            assert part.stat().st_size==size;part.replace(p)
        assert p.stat().st_size==size
        records.append(dict(path=str(p),url=URL+name,bytes=size,sha256=sha(p)))
        dump(RUN/'download.json',dict(complete=False,files=records));print('DOWNLOADED',name,size,flush=True)
    entries=read(FROZEN/'config/final_test_annotation_members.json')['members']
    archive=URL+'home/cvpr18/data/scenenn_seg_76.zip'
    for e in entries:
        p=RUN/'data/annotation'/Path(e['name']).name;p.parent.mkdir(exist_ok=True)
        if not p.exists():
            head=fetch(archive,e['offset'],e['offset']+29);assert head[:4]==b'PK\x03\x04'
            n,k=struct.unpack('<HH',head[26:30]);start=e['offset']+30+n+k
            with ThreadPoolExecutor(4) as pool:
                compressed=b''.join(pool.map(lambda pair:fetch(archive,*pair),[(i,min(i+4*1024**2,start+e['compressed'])-1) for i in range(start,start+e['compressed'],4*1024**2)]))
            raw=zlib.decompress(compressed,-15);assert len(raw)==e['size'] and zlib.crc32(raw)==e['crc']
            p.write_bytes(raw)
        assert p.stat().st_size==e['size']
        records.append(dict(path=str(p),url=archive,bytes=p.stat().st_size,sha256=sha(p)))
        print('ANNOTATION BYTES',p.name,'not decoded',flush=True)
    dump(RUN/'download.json',dict(complete=True,files=records,selection_sha256=sha(RUN/'selection.json'),semantic_values_decoded=False,
        data_amendment_sha256=sha(RUN/'data_amendment.json'),unavailable=read(RUN/'data_amendment.json')['unscored_scenes']))


def prepare():
    verify_execution()
    import category_holdout_data as original
    original.DEST=RUN
    original.SCENES=run_scenes()
    # The existing preparer needs the sampling/input schema, not new method choices.
    spec=read(RUN/'selection.json');spec.update(
        source_sha256=read(OLD/'category_holdout_v1/selection.json')['source_sha256'])
    original_read=original.read
    def patched_read(path):
        if Path(path)==RUN/'selection.json':return spec
        return original_read(path)
    original.read=patched_read
    # Finish all decoding before inference; the underlying code never opens semantic GT.
    original.prepare()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['download','prepare']);args=p.parse_args()
    if args.action=='download':download()
    else:prepare()
