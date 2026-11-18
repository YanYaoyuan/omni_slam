"""Bounded access to official Hypersim ZIPs using verified HTTP byte ranges.

Indexing never downloads a whole scene. Data extraction validates ZIP CRC and
records file hashes. Scene/frame selection is explicit, not score dependent.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import time
import urllib.request
import zipfile

BASE='https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/'


class RangeFile(io.RawIOBase):
    def __init__(self,url,budget):
        self.url=url;self.offset=0;self.used=0;self.budget=budget
        req=urllib.request.Request(url,headers={'Range':'bytes=0-0','Accept-Encoding':'identity'})
        with urllib.request.urlopen(req,timeout=40) as r:
            if r.status!=206:raise ValueError('Server must support partial content')
            match=re.fullmatch(r'bytes 0-0/(\d+)',r.headers.get('Content-Range',''))
            if not match:raise ValueError('Unexpected Content-Range')
            self.size=int(match[1]);assert len(r.read(2))==1
        self.used=1
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.offset
    def seek(self,offset,whence=0):
        target=offset if whence==0 else self.offset+offset if whence==1 else self.size+offset
        if target<0:raise ValueError('Negative seek')
        self.offset=target;return target
    def read(self,n=-1):
        n=min(self.size-self.offset,n if n>=0 else self.size-self.offset)
        if n<=0:return b''
        if n>32*1024**2 or self.used+n>self.budget:raise ValueError('Range download budget exceeded')
        start=self.offset;end=start+n-1
        req=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req,timeout=40) as r:
                    if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{self.size}':
                        raise ValueError('Incorrect partial content response')
                    data=r.read(n+1)
                    if len(data)!=n:raise ValueError('Incomplete or oversized range')
                break
            except (TimeoutError,OSError):
                if attempt==2:raise
                time.sleep(1)
        self.offset+=n;self.used+=n;return data


def main():
    parser=argparse.ArgumentParser();parser.add_argument('scene')
    parser.add_argument('--out',type=Path,default=Path('build/datasets/hypersim_prepare'))
    parser.add_argument('--frames',default='0,20,40,60,80')
    parser.add_argument('--position-check',action='store_true',help='Also fetch world positions for the first selected frame')
    parser.add_argument('--hdr',action='store_true',help='Include lossless HDR color and render-entity masks for tonemapping')
    parser.add_argument('--reserved',action='store_true',help='Use all previously reserved public test frames, with split verification')
    parser.add_argument('--extract',action='store_true');parser.add_argument('--budget-mib',type=int,default=512)
    args=parser.parse_args()
    if not re.fullmatch(r'ai_\d{3}_\d{3}',args.scene):raise ValueError('Invalid scene name')
    frames={int(v) for v in args.frames.split(',')}
    if args.reserved:
        metadata=Path('build/datasets/hypersim_prepare')
        reservation=json.loads((metadata/'reserved_test_scenes.json').read_text())
        assert hashlib.sha256((metadata/'splits.csv').read_bytes()).hexdigest()==reservation['split_sha256']
        frames=set(reservation['scenes'][args.scene])
        public={int(r['frame_id']) for r in csv.DictReader((metadata/'splits.csv').open())
                if r['scene_name']==args.scene and r['camera_name']=='cam_00'
                and r['included_in_public_release']=='True' and r['split_partition_name']=='test'}
        assert frames==public and frames
    args.out.mkdir(parents=True,exist_ok=True)
    f=RangeFile(BASE+args.scene+'.zip',args.budget_mib*1024**2)
    with zipfile.ZipFile(f) as z:
        infos=z.infolist()
        index=[dict(name=i.filename,size=i.file_size,compressed_size=i.compress_size,crc=i.CRC) for i in infos]
        (args.out/(args.scene+'_index.json')).write_text(json.dumps(index,indent=2))
        selected=[]
        for i in infos:
            name=i.filename
            if i.is_dir():continue
            if '/_detail/cam_00/' in name or name.endswith('/_detail/metadata_scene.csv'):
                selected.append(i);continue
            if args.position_check and '/scene_cam_00_geometry_hdf5/' in name and name.endswith(f'/frame.{min(frames):04d}.position.hdf5'):
                selected.append(i);continue
            hdr_match=re.search(r'/frame\.(\d+)\.(color\.hdf5|render_entity_id\.hdf5)$',name)
            if args.hdr and hdr_match and int(hdr_match[1]) in frames and '/scene_cam_00_' in name:
                selected.append(i);continue
            match=re.search(r'/frame\.(\d+)\.(tonemap\.jpg|depth_meters\.hdf5|semantic_instance\.hdf5|semantic\.hdf5)$',name)
            if match and int(match[1]) in frames and '/scene_cam_00_' in name:selected.append(i)
        print(args.scene,'entries',len(infos),'selected',len(selected),'uncompressed',sum(i.file_size for i in selected),flush=True)
        files=[]
        if args.extract:
            for i in selected:
                dest=args.out/'data'/i.filename
                if not dest.resolve().is_relative_to((args.out/'data').resolve()):raise ValueError('Unsafe ZIP path')
                if i.file_size>64*1024**2:raise ValueError('Unexpectedly large single file')
                if dest.exists():
                    data=dest.read_bytes()
                    if len(data)!=i.file_size or zipfile.crc32(data)&0xffffffff!=i.CRC:raise ValueError('Existing file differs')
                else:
                    data=z.read(i);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
                files.append(dict(path=str(dest),sha256=hashlib.sha256(data).hexdigest()))
                print(i.filename,flush=True)
        record=dict(scene=args.scene,url=f.url,zip_size=f.size,range_bytes=f.used,frames=sorted(frames),
                    extracted=args.extract,hdr=args.hdr,reserved=args.reserved,files=files,selected=[i.filename for i in selected],
                    index_sha256=hashlib.sha256((args.out/(args.scene+'_index.json')).read_bytes()).hexdigest())
        (args.out/(args.scene+('_download.json' if args.extract else '_selection.json'))).write_text(json.dumps(record,indent=2))


if __name__=='__main__':main()
