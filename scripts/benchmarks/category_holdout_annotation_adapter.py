"""Document missing official 030 category annotations before any prediction.

All three selected sequences remain in mapping; only 025 and 032 are scorable
under the frozen CVPR18 domain. Do not substitute XML or another sequence.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import struct
import zlib
from audit_evidence_retention import OUT,dump
from public_dino_protocol import read
from omni_semantic_map.flat_benchmark import sha256
import category_holdout_data as data


def amendment():
    path=data.DEST/'annotation_amendment.json';assert not path.exists()
    dump(path,dict(reason='Official cached CVPR18 central directory has coords/seg members for025/032, none for030.',
        selected_scenes=data.SCENES,scored_scenes=['025','032'],unscored_scenes={'030':'missing CVPR18 category annotation; keep mapping, no comparable category score'},
        replacement_scenes=[],no_model_or_metric_change=True,source=str(Path(__file__).resolve()),sha256=sha256(__file__),
        original_selection_sha256=sha256(data.DEST/'selection.json'),
        directory_sha256=sha256(OUT/'official_annotation_audit/cvpr18_zip_contents.json')))


def annotations():
    record=read(data.DEST/'annotation_amendment.json');assert sha256(__file__)==record['sha256']
    download=read(data.DEST/'download.json');assert len(download['files'])==13 and not download['complete']
    for f in download['files']:assert sha256(f['path'])==f['sha256']
    entries=read(OUT/'official_annotation_audit/cvpr18_zip_contents.json')
    wanted=[e for e in entries if any(e['name'].endswith('_'+s+'.hdf5') for s in data.SCENES)];assert len(wanted)==4
    url=data.URL+'home/cvpr18/data/scenenn_seg_76.zip'
    for e in wanted:
        path=data.DEST/'data/annotation'/Path(e['name']).name;path.parent.mkdir(exist_ok=True)
        assert not path.exists()
        head=data.fetch(url,e['offset'],e['offset']+29);assert head[:4]==b'PK\x03\x04'
        n,k=struct.unpack('<HH',head[26:30]);start=e['offset']+30+n+k
        with ThreadPoolExecutor(4) as pool:
            comp=b''.join(pool.map(lambda pair:data.fetch(url,*pair),[(i,min(i+4*1024**2,start+e['compressed'])-1) for i in range(start,start+e['compressed'],4*1024**2)]))
        raw=zlib.decompress(comp,-15);assert len(raw)==e['size'] and zlib.crc32(raw)==e['crc'];path.write_bytes(raw)
        download['files'].append(dict(path=str(path),url=url,sha256=sha256(path),bytes=len(raw)));print('ANNOTATION',path.name,flush=True)
    download.update(complete=True,annotation_amendment_sha256=sha256(data.DEST/'annotation_amendment.json'),category_annotation_scenes=record['scored_scenes'])
    dump(data.DEST/'download.json',download)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--annotations',action='store_true');p.add_argument('--prepare',action='store_true');p.add_argument('--evaluate',action='store_true');a=p.parse_args()
    if a.freeze:amendment()
    if a.annotations:annotations()
    if a.prepare:data.prepare()
    if a.evaluate:
        import category_holdout_finish as finish
        audit=read(data.DEST/'annotation_amendment.json');assert sha256(__file__)==audit['sha256']
        def evaluation_read(path):
            result=read(path)
            if Path(path)==data.DEST/'selection.json':result=dict(result,scenes=audit['scored_scenes'])
            return result
        finish.read=evaluation_read;finish.evaluation()
        dump(data.DEST/'unscored_scenes.json',audit['unscored_scenes'])
