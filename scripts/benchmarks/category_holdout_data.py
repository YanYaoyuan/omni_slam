"""Predeclare untouched SceneNN sequences, download, decode and freeze frontend jobs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import struct
import subprocess
import time
import urllib.request
import zlib
import numpy as np
from PIL import Image
from audit_evidence_retention import ROOT,OUT,CACHE,dump
from public_dino_protocol import read
from omni_semantic_map.flat_benchmark import sha256
from omni_semantic_map.scene_coordinate_basis import canonicalize_world

DEST=OUT/'category_holdout_v1'
SCENES=['025','030','032']
URL='https://hkust-vgd.ust.hk/scenenn/'


def freeze():
    assert not DEST.exists()
    parent=read(OUT/'public_category_evidence_v2/protocol.json')
    extra=[Path(__file__).resolve(),ROOT/'omni_semantic_map/omni_semantic_map/consensus_category.py',
        ROOT/'omni_semantic_map/omni_semantic_map/scene_coordinate_basis.py',
        ROOT/'scripts/benchmarks/split_strong_frontend.py',ROOT/'scripts/benchmarks/retention_holdout_backend.py',
        ROOT/'scripts/benchmarks/run_public_history_graph.py',ROOT/'scripts/benchmarks/category_holdout_finish.py']
    dump(DEST/'selection.json',dict(scenes=SCENES,excluded_viewed=['005','011','014','015','016','021'],
        selection='Next three lowest-numbered unused official main sequences; chosen before RGB, semantic labels or method scores.',
        official_index=URL+'main/',development_summary_sha256=sha256(OUT/'public_category_evidence_v2/summary.json'),
        methods=['mean','area','consensus_sqrt_area'],candidate='consensus_sqrt_area',
        foreground='Freeze original unique-frame mean foreground before category-only update; secondary refiltering reported separately.',
        mapping='Original RGB-D CG DBSCAN on / overlap .025m; two-query-frame history bridge; fixed vocabulary, SAM2.1 HieraL + SigLIP384 learned frontend.',
        sampling='400 uniform geometry frames, 200 semantic frames; no result-based frame/scene selection.',
        coordinates='Previously frozen first-pose world-basis normalization, residual<=.001, no GT fitting.',
        evaluation='Same corrected CVPR18 observed voxel domain and fixed NYU ontology mapping; class-aware pooled-instance PQ diagnostic, not official macro PQ.',
        nyu_prediction_mapping=parent['nyu_prediction_mapping'],source_sha256={**parent['source_sha256'],**{str(p):sha256(p) for p in extra}},
        independent_of_category_selection=True,no_tuning=True,
        limits='Unseen sequence validation within the same dataset family, not a universal generalization test. Keep all failures and regressions.'))


def fetch(url,start=None,end=None):
    for attempt in range(4):
        try:
            headers={} if start is None else {'Range':f'bytes={start}-{end}'}
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=45) as r:
                if start is not None:assert r.status==206 and r.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
                b=r.read()
            if start is not None:assert len(b)==end-start+1
            return b
        except Exception:
            if attempt==3:raise
            time.sleep(2)


def download():
    spec=read(DEST/'selection.json')
    for p,h in spec['source_sha256'].items():assert sha256(p)==h
    names=['main/intrinsic/asus.ini']+[f'main/{s}/{f}' for s in SCENES for f in [s+'.ply',s+'.xml','trajectory.log']]+[f'main/oni/{s}.oni' for s in SCENES]
    records=[]
    for name in names:
        path=DEST/'data'/name;path.parent.mkdir(parents=True,exist_ok=True)
        with urllib.request.urlopen(urllib.request.Request(URL+name,method='HEAD'),timeout=30) as r:size=int(r.headers['Content-Length'])
        assert size<2*1024**3
        if not path.exists():
            temporary=path.with_suffix(path.suffix+'.part')
            with temporary.open('wb') as f:
                step=8*1024**2;ranges=[(i,min(i+step,size)-1) for i in range(0,size,step)]
                # Ordered consumption bounds memory to a small number of chunks.
                for batch in range(0,len(ranges),4):
                    with ThreadPoolExecutor(4) as pool:
                        for data in pool.map(lambda pair:fetch(URL+name,*pair),ranges[batch:batch+4]):f.write(data)
            assert temporary.stat().st_size==size;temporary.replace(path)
        assert path.stat().st_size==size
        records.append(dict(path=str(path),url=URL+name,sha256=sha256(path),bytes=size))
        dump(DEST/'download.json',dict(complete=False,files=records));print('DOWNLOADED',name,size,flush=True)
    # Fetch annotation ZIP members using the previously archived central directory;
    # no semantic values are decoded until all candidate predictions are finished.
    archive=URL+'home/cvpr18/data/scenenn_seg_76.zip'
    entries=read(OUT/'official_annotation_audit/cvpr18_zip_contents.json')
    wanted=[e for e in entries if any(e['name'].endswith('_'+s+'.hdf5') for s in SCENES)]
    assert len(wanted)==6
    for e in wanted:
        path=DEST/'data/annotation'/Path(e['name']).name;path.parent.mkdir(exist_ok=True)
        if not path.exists():
            head=fetch(archive,e['offset'],e['offset']+29);assert head[:4]==b'PK\x03\x04'
            n,k=struct.unpack('<HH',head[26:30]);start=e['offset']+30+n+k
            with ThreadPoolExecutor(4) as pool:
                compressed=b''.join(pool.map(lambda pair:fetch(archive,*pair),[(i,min(i+4*1024**2,start+e['compressed'])-1) for i in range(start,start+e['compressed'],4*1024**2)]))
            raw=zlib.decompress(compressed,-15);assert len(raw)==e['size'] and zlib.crc32(raw)==e['crc'];path.write_bytes(raw)
        records.append(dict(path=str(path),url=archive,sha256=sha256(path),bytes=path.stat().st_size))
        print('DOWNLOADED',path.name,flush=True)
    dump(DEST/'download.json',dict(complete=True,files=records,selection_sha256=sha256(DEST/'selection.json')))


def prepare():
    from prepare_strong_jobs import sources
    download=read(DEST/'download.json');assert download['complete']
    for r in download['files']:assert sha256(r['path'])==r['sha256']
    binary=OUT/'tools/playback400';assert binary.exists()
    ref=CACHE/'office_0';original=read(ref/'manifest.json')['config']
    calibration={k:float(v) for k,v in (line.split() for line in (DEST/'data/main/intrinsic/asus.ini').read_text().splitlines())}
    for s in SCENES:
        source=DEST/'data'/('sample400_'+s);trajectory=DEST/'data/main'/s/'trajectory.log'
        lines=trajectory.read_text().splitlines();assert len(lines)%5==0
        ids0=np.array([int(lines[i].split()[0]) for i in range(0,len(lines),5)]);np.testing.assert_array_equal(ids0,np.arange(len(ids0)))
        raw=np.array([[list(map(float,l.split())) for l in lines[i+1:i+5]] for i in range(0,len(lines),5)])
        result=canonicalize_world(raw);ids=np.array([j*(len(raw)-1)//399 for j in range(400)]);assert len(set(ids))==400
        assert not source.exists()
        with (DEST/(s+'_decode.log')).open('x') as log:
            subprocess.run([str(binary),str(DEST/'data/main/oni'/(s+'.oni')),str(source),str(len(raw))],stdout=log,stderr=subprocess.STDOUT,check=True)
        times=np.loadtxt(source/'timestamp.txt');assert len(times)>=len(raw)
        np.testing.assert_array_equal(times[:,0],np.arange(1,len(times)+1))
        np.savez_compressed(source/'poses.npz',source_ids=ids,raw=raw[ids],poses=result['poses'][ids],canonical_from_raw=result['canonical_from_raw'])
        cfg=read(ref/'manifest.json')['config'];cfg['cam'].update(H=480,W=640,fx=calibration['fx'],fy=calibration['fy'],cx=calibration['cx'],cy=calibration['cy'],depth_scale=1000.)
        records=[];hashes={str(ref/name):sha256(ref/name) for name in ['manifest.json','text_features.npy','vocabulary.json']}
        for i,fid in enumerate(ids):
            rgb=source/'image'/f'image{fid+1:05d}.png';depth=source/'depth'/f'depth{fid+1:05d}.png'
            for p in [rgb,depth]:
                with Image.open(p) as im:assert im.size==(640,480)
                hashes[str(p)]=sha256(p)
            records.append(dict(frame_id=int(fid),rgb=str(rgb),depth=str(depth),pose=result['poses'][fid].tolist(),semantic=i%2==0))
        assert len(list((source/'image').glob('*.png')))==400 and len(list((source/'depth').glob('*.png')))==400
        dump(source/'manifest.json',dict(complete=True,no_gt=True,source_frames=len(raw),geometry_frames=400,semantic_frames=200,
            decoder_sha256=sha256(binary),max_remaining_rotation_projection=result['max_remaining_rotation_projection'],
            canonical_from_raw=result['canonical_from_raw'].tolist(),maximum_rgb_depth_time_difference_ms=float(np.abs(times[ids,1]-times[ids,2]).max()/1000)))
        hashes[str(source/'poses.npz')]=sha256(source/'poses.npz')
        job=DEST/'jobs'/(s+'.json');assert not job.exists()
        dump(job,dict(kind='scenenn',scene=s,output=str(DEST/'frontend'/s),config=cfg,reference_cache=str(ref),frames=records,
            input_sha256=hashes,source_sha256=sources(),selection_sha256=sha256(DEST/'selection.json'),no_gt_in_inference=True,peak_cuda_allocator_limit_gib=6))
        print('PREPARED',s,len(raw),'->400/200',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--download',action='store_true');p.add_argument('--prepare',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    if a.download:download()
    if a.prepare:prepare()
