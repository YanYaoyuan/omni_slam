"""Reserve/download two new validation windows without inspecting RGB or GT.

Dataset preparation only. A separate frozen predictor protocol is required
before evaluation. Failed discovery-query candidates will not be expanded here.
"""
import argparse
from pathlib import Path
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from omni_semantic_map.kitti360_input import load_poses

META=ROOT/'build/datasets/kitti360_prepare'
DEST=ROOT/'build/datasets/kitti360_unseen_v2'


def reserve():
    assert not (DEST/'window_selection.json').exists()
    old=read(META/'window_selection.json')['windows'];chosen=[]
    paths=[META/'window_selection.json',META/'splits/2013_05_28_drive_val.txt',META/'semantic_archive_index.json']
    entries={r['name']:r for r in read(paths[-1])['entries']};pose_cache={}
    def separation(x,y):
        x,y=np.asarray(x),np.asarray(y)
        return float(np.linalg.norm(np.maximum(np.maximum(x[0,:2]-y[1,:2],y[0,:2]-x[1,:2]),0)))
    for member in sorted(paths[1].read_text().splitlines()):
        scene=Path(member).parts[-3]
        if scene in {s['scene'] for s in old+chosen}:continue
        path=META/'extracted'/scene/'cam0_to_world.txt'
        if scene not in pose_cache:pose_cache[scene]=load_poses(path);paths.append(path)
        poses=pose_cache[scene];start,end=map(int,Path(member).stem.split('_'));frames=[f for f in poses if start<=f<=end]
        if len(frames)<100:continue
        xyz=np.asarray([poses[f][:3,3] for f in frames]);box=[xyz.min(0).tolist(),xyz.max(0).tolist()]
        distances=[separation(box,s['trajectory_aabb']) for s in old+chosen]
        if min(distances)<=200:continue
        chosen.append(dict(scene=scene,member=member,start=start,end=end,posed_frames=frames,
            input_frames=sorted(set(frames[::5]+frames[-1:])),trajectory_aabb=box,trajectory_separation_m=distances,
            gt_bytes=entries[member]['bytes'],role='reserved_validation'))
        if len(chosen)==2:break
    assert len(chosen)==2
    dump(DEST/'window_selection.json',dict(windows=chosen,already_exposed_windows=[dict(scene=s['scene'],start=s['start'],end=s['end']) for s in old],
        selection='First two lexicographic official validation windows in previously unused sequences; >=100 posed frames; XY trajectory AABBs separated >200m from all exposed and other selected windows. Uses only split/archive/pose metadata.',
        sampling='Same existing adapter: every fifth available rectified left-camera pose in full window, plus last. Not processing all raw frames.',
        protocol_status='Dataset reservation only. No predictor, method selection or score claimed. Do not run failed query-memory branch here. Freeze successful method and controls separately before decoding reference labels.',
        independence='New sequences and camera trajectories; actual observed geometry overlap must still be checked. No RGB viewed or reference labels decoded by this script.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},selection_code_sha256=sha(__file__)))
    # The existing verified range downloader consumes the unmodified archive index.
    dump(DEST/'semantic_archive_index.json',read(META/'semantic_archive_index.json'))
    print('RESERVED',[(s['scene'],s['start'],s['end'],len(s['input_frames'])) for s in chosen],flush=True)


def download():
    import sys
    import download_kitti360_windows as native
    selection=read(DEST/'window_selection.json');assert sha(__file__)==selection['selection_code_sha256']
    for f,h in selection['source_sha256'].items():assert sha(ROOT/f)==h,f
    assert read(DEST/'semantic_archive_index.json')==read(META/'semantic_archive_index.json')
    native.BASE=DEST;native.OUT=DEST/'data'
    sys.argv=[__file__,'--workers','2'];native.main()
    dump(DEST/'download_provenance.json',dict(complete=True,wrapper_sha256=sha(__file__),
        downloader_sha256=sha(native.__file__),selection_sha256=sha(DEST/'window_selection.json'),
        decoded_reference_labels=False,viewed_rgb=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['reserve','download']);globals()[parser.parse_args().action]()
