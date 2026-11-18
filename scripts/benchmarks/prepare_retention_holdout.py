"""Freeze unused scenes and fetch official SceneNN inputs; no model/GT scoring."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request
from audit_evidence_retention import ROOT, OUT, REPORT, BASE, dump
from omni_semantic_map.flat_benchmark import sha256

SCENES=('014','015','021')
DATA=Path('/media/user/50D2BD30D2BD1AE4/WorkSpace/omni_slam_benchmarks/SceneNN_retention_20261111')
OFFICIAL='https://hkust-vgd.ust.hk/scenenn/'


def freeze():
    path=OUT/'stage3_selection.json'
    if path.exists():raise FileExistsError(path)
    stage2=json.loads((REPORT/'stage2/summary.json').read_text());assert stage2['complete']
    model=ROOT/'omni_semantic_map/omni_semantic_map/evidence_readout.py'
    dump(path,dict(stage=3,scenes=SCENES,excluded_used_scenes=['005','011','016'],
        selection='Three lowest-numbered unused scenes in official main index; selected without viewing RGB, labels or method scores.',
        official_index=OFFICIAL+'main/',data_root=str(DATA),
        methods=['mean','area_density'],bandwidth_degrees=30.0,
        developmental_delta_pq100=100*stage2['methods']['area_density']['delta_vs_mean'],
        frozen_readout_source_sha256=sha256(model),stage2_protocol_sha256=sha256(OUT/'stage2_protocol.json'),
        cg_config=str(BASE.parents[0]/'expanded_semantic_20261109/conceptgraphs_runs/office_0/mapping_config.json'),
        cg_config_sha256=sha256(BASE.parents[0]/'expanded_semantic_20261109/conceptgraphs_runs/office_0/mapping_config.json'),
        frontend='same frozen SAM2.1 Hiera L and SigLIP-384 learned merger; no text/feature changes',
        sampling='400 uniformly spaced geometry frames over full provided trajectory; semantic on even sampled indices (200 frames)',
        calibration='official ASUS depth/RGB intrinsics; ONI official registration/decoder; input alignment checked without fitting',
        evaluation='paired class-agnostic observed 5cm instance PQ, strict IoU>0.5, shared geometry domain, assigned 1NN <=5cm; not official benchmark',
        gt_background='same previous SceneNN policy: unlabeled, wall, floor, ceiling by XML text prefix; GT only in evaluation',
        model_selection='Do not tune on these scenes; retain every selected scene, failed run and regression. Any input exclusion requires technical evidence and explicit report.',
        live_validation='LIVO atomic frames with measured sparse supports, strong frontend and same CG/readout branches; pending',
        status='configuration_selected_data_pending'))
    dump(REPORT/'stage3/selection.json',json.loads(path.read_text()))
    print('FROZEN',path,flush=True)


def download():
    protocol=json.loads((OUT/'stage3_selection.json').read_text())
    assert protocol['scenes']==list(SCENES)
    DATA.mkdir(parents=True,exist_ok=True)
    names=['main/intrinsic/asus.ini']+[f'main/{s}/{name}' for s in SCENES for name in [f'{s}.ply',f'{s}.xml','trajectory.log']]+[f'main/oni/{s}.oni' for s in SCENES]
    sizes={}
    for name in names:
        with urllib.request.urlopen(urllib.request.Request(OFFICIAL+name,method='HEAD'),timeout=40) as r:sizes[name]=int(r.headers['Content-Length'])
    assert sum(sizes.values()) < 30*1024**3, 'unexpected input size; inspect before downloading'
    manifest=dict(complete=False,scenes=SCENES,files=[],total_bytes=sum(sizes.values()),selection_sha256=sha256(OUT/'stage3_selection.json'))
    dump(OUT/'holdout_download.json',manifest)
    for name in names:
        dest=DATA/name;dest.parent.mkdir(parents=True,exist_ok=True);size=sizes[name]
        if not dest.exists():
            step=8*1024*1024;parts=DATA/'.ranges'/name;parts.mkdir(parents=True,exist_ok=True)
            jobs=[(start,min(start+step,size)-1) for start in range(0,size,step)]
            def fetch(job):
                start,end=job;target=parts/str(start)
                if target.exists() and target.stat().st_size==end-start+1:return
                for attempt in range(4):
                    try:
                        req=urllib.request.Request(OFFICIAL+name,headers={'Range':f'bytes={start}-{end}'})
                        with urllib.request.urlopen(req,timeout=45) as r:
                            if r.status==200 and start==0 and end==size-1:pass
                            elif r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':raise ValueError('server ignored range')
                            temporary=target.with_suffix('.part')
                            with temporary.open('wb') as f:
                                while chunk:=r.read(1024*1024):f.write(chunk)
                        assert temporary.stat().st_size==end-start+1
                        temporary.replace(target);return
                    except Exception:
                        if attempt==3:raise
                        time.sleep(2)
            with ThreadPoolExecutor(8) as pool:list(pool.map(fetch,jobs))
            assembled=dest.with_suffix(dest.suffix+'.assembling')
            with assembled.open('wb') as f:
                for start,_ in jobs:
                    with (parts/str(start)).open('rb') as part:
                        while chunk:=part.read(1024*1024):f.write(chunk)
            assert assembled.stat().st_size==size;assembled.replace(dest)
        assert dest.stat().st_size==size
        manifest['files'].append(dict(url=OFFICIAL+name,path=str(dest),bytes=size,sha256=sha256(dest)))
        dump(OUT/'holdout_download.json',manifest)
        print('DOWNLOADED',name,size,flush=True)
    manifest['complete']=True
    dump(DATA/'manifest.json',manifest);dump(OUT/'holdout_download.json',manifest)
    print('COMPLETE',len(names),sum(sizes.values()),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--download',action='store_true');args=parser.parse_args()
    if args.freeze:freeze()
    if args.download:download()
