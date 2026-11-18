"""Fetch the preselected SceneNN sequences, without decoding semantic labels."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import shutil
import struct
import urllib.request
import zlib
from paper_final_common import ROOT, read, dump, sha
from category_holdout_data import fetch

OUT=ROOT/'build/benchmarks/visibility_independent_20261112'
URL='https://hkust-vgd.ust.hk/scenenn/'


def download():
    selection=read(OUT/'selection.json');availability=read(OUT/'availability.json')
    assert [r['scene'] for r in availability]==selection['selected_scenes']
    files=[f for s in availability for f in s['files']]
    assert all(f.get('status')==200 for f in files)
    name='main/intrinsic/asus.ini'
    with urllib.request.urlopen(urllib.request.Request(URL+name,method='HEAD'),timeout=30) as response:
        files.insert(0,dict(name=name,bytes=int(response.headers['Content-Length'])))
    assert sum(f['bytes'] for f in files)<6*1024**3
    records=[]
    for entry in files:
        name=entry['name'];size=entry['bytes'];path=OUT/'data'/name;path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            assert shutil.disk_usage(OUT).free>size+8*1024**3, 'insufficient reserve for derived caches'
            part=path.with_suffix(path.suffix+'.part');step=8*1024**2
            done=part.stat().st_size if part.exists() else 0
            assert done<=size and (done%step==0 or done==size)
            pairs=[(i,min(size,i+step)-1) for i in range(done,size,step)]
            with part.open('ab') as output:
                for start in range(0,len(pairs),4):
                    with ThreadPoolExecutor(4) as pool:
                        for raw in pool.map(lambda bounds:fetch(URL+name,*bounds),pairs[start:start+4]):
                            output.write(raw);output.flush()
                    dump(OUT/'download_progress.json',dict(file=name,downloaded_bytes=output.tell(),total_bytes=size))
            assert part.stat().st_size==size;part.replace(path)
        assert path.stat().st_size==size
        records.append(dict(path=str(path),url=URL+name,bytes=size,sha256=sha(path)))
        dump(OUT/'download.json',dict(complete=False,files=records));print('DOWNLOADED',name,size,flush=True)
    archive=URL+'home/cvpr18/data/scenenn_seg_76.zip'
    for entry in selection['annotation_members']:
        path=OUT/'data/annotation'/Path(entry['name']).name;path.parent.mkdir(exist_ok=True)
        if not path.exists():
            head=fetch(archive,entry['offset'],entry['offset']+29);assert head[:4]==b'PK\x03\x04'
            n,k=struct.unpack('<HH',head[26:30]);begin=entry['offset']+30+n+k
            with ThreadPoolExecutor(4) as pool:
                compressed=b''.join(pool.map(lambda bounds:fetch(archive,*bounds),
                    [(i,min(i+4*1024**2,begin+entry['compressed'])-1) for i in range(begin,begin+entry['compressed'],4*1024**2)]))
            raw=zlib.decompress(compressed,-15)
            assert len(raw)==entry['size'] and zlib.crc32(raw)==entry['crc']
            temporary=path.with_suffix('.part');temporary.write_bytes(raw);temporary.replace(path)
        assert path.stat().st_size==entry['size']
        records.append(dict(path=str(path),url=archive,bytes=path.stat().st_size,sha256=sha(path)))
        print('ANNOTATION BYTES',path.name,'not decoded',flush=True)
    dump(OUT/'download.json',dict(complete=True,files=records,selection_sha256=sha(OUT/'selection.json'),semantic_values_decoded=False))


def prepare(scene=None):
    import category_holdout_data as original
    original.DEST=OUT
    selected=read(OUT/'selection.json')['selected_scenes']
    assert scene is None or scene in selected
    wanted=[scene] if scene else selected
    original.SCENES=[s for s in wanted if not (OUT/'jobs'/(s+'.json')).exists()]
    # A scene can be decoded while later scenes download. Do not claim the
    # global download is complete: construct an explicitly scoped manifest.
    if scene is not None:
        current=read(OUT/'download.json')
        names={f'main/{scene}/{scene}.ply',f'main/{scene}/{scene}.xml',
               f'main/{scene}/trajectory.log',f'main/oni/{scene}.oni','main/intrinsic/asus.ini'}
        subset=[r for r in current['files'] if str(Path(r['path']).relative_to(OUT/'data')) in names]
        assert len(subset)==len(names),'scene input download not complete'
        scoped=dict(complete=True,scene=scene,files=subset,global_download_complete=current['complete'])
        dump(OUT/'scene_downloads'/(scene+'.json'),scoped)
        previous=original.read
        original.read=lambda path: scoped if Path(path)==OUT/'download.json' else previous(path)
    original.prepare()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['download','prepare']);parser.add_argument('--scene');args=parser.parse_args()
    download() if args.action=='download' else prepare(args.scene)
