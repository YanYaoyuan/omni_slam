"""Reserve genuinely unused sequences using metadata, then download opaque data.

No RGB viewing, reference decoding or method evaluation. Existing exposed
0000/0002/0003/0004/0005/0006/0007 are excluded in their entirety, not renamed holdouts.
"""
import argparse
from pathlib import Path
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from omni_semantic_map.kitti360_input import load_poses

META=ROOT/'build/datasets/kitti360_prepare'
DEST=ROOT/'build/datasets/kitti360_reserved_v4'
EXPOSED=[META/'window_selection.json',ROOT/'build/benchmarks/intrinsic_public_holdout_v1/metadata/window_selection.json',
         ROOT/'build/datasets/kitti360_reserved_v3/window_selection.json']


def reserve():
    assert not (DEST/'window_selection.json').exists()
    old=[w for path in EXPOSED for w in read(path)['windows']]
    excluded={w['scene'] for w in old}
    assert excluded=={f'2013_05_28_drive_{i:04d}_sync' for i in [0,2,3,4,5,6,7]}
    paths=[*EXPOSED,META/'splits/2013_05_28_drive_val.txt',META/'semantic_archive_index.json']
    entries={r['name']:r for r in read(paths[-1])['entries']};chosen=[];poses={};considered=[]
    def separation(a,b):
        a,b=np.asarray(a),np.asarray(b)
        return float(np.linalg.norm(np.maximum(np.maximum(a[0,:2]-b[1,:2],b[0,:2]-a[1,:2]),0)))
    for member in sorted((META/'splits/2013_05_28_drive_val.txt').read_text().splitlines()):
        scene=Path(member).parts[-3]
        if scene in excluded or scene in {w['scene'] for w in chosen}:continue
        path=META/'extracted'/scene/'cam0_to_world.txt'
        if scene not in poses:poses[scene]=load_poses(path);paths.append(path)
        start,end=map(int,Path(member).stem.split('_'));frames=sorted(f for f in poses[scene] if start<=f<=end)
        if len(frames)<100:considered.append(dict(member=member,reason='fewer_than100posed_frames'));continue
        xyz=np.array([poses[scene][f][:3,3] for f in frames]);box=[xyz.min(0).tolist(),xyz.max(0).tolist()]
        distances=[separation(box,w['trajectory_aabb']) for w in old+chosen]
        if min(distances)<=200:considered.append(dict(member=member,reason='trajectory_AABB_separation_not_over200m',minimum_m=min(distances)));continue
        chosen.append(dict(scene=scene,member=member,start=start,end=end,posed_frames=frames,
            input_frames=sorted(set(frames[::5]+frames[-1:])),trajectory_aabb=box,trajectory_separation_m=distances,
            gt_bytes=entries[member]['bytes'],role='reserved_unseen_validation'))
        if len(chosen)==2:break
    assert len(chosen)==2,'No valid metadata-only pair; do not relax criteria automatically'
    dump(DEST/'window_selection.json',dict(windows=chosen,excluded_sequences=sorted(excluded),
        already_exposed_windows=[dict(scene=w['scene'],start=w['start'],end=w['end'],trajectory_aabb=w['trajectory_aabb']) for w in old],
        considered=considered,selection='First two lexicographic official validation windows in unused sequences,>=100availableposedframes,XYtrajectoryAABBseparation>200m from every exposed and newly selected window. Entire exposed sequences excluded.',
        sampling='Every fifth available pose plus last, unchanged from previous adapter. Not all raw frames or full official benchmark.',
        protocol_status='Data reservation only under the user-authorized new outdoor assessment objective. No model inference, RGB inspection or reference decoding until a separately frozen method and controls. Does not promote a failed development candidate or change any existing gate. Acquisition may proceed independently of method development.',
        independence='No RGB viewed or labels decoded. Actual measurement-space overlap must still be checked before an independent-test claim. Official validation data, not a hidden official test server.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},selection_code_sha256=sha(__file__)))
    dump(DEST/'semantic_archive_index.json',read(META/'semantic_archive_index.json'))
    print('RESERVED V4',[(w['scene'],w['start'],w['end'],len(w['input_frames'])) for w in chosen],flush=True)


def download():
    import sys
    import download_kitti360_windows as native
    s=read(DEST/'window_selection.json');assert sha(__file__)==s['selection_code_sha256']
    for f,h in s['source_sha256'].items():assert sha(ROOT/f)==h,f
    assert read(DEST/'semantic_archive_index.json')==read(META/'semantic_archive_index.json')
    native.BASE=DEST;native.OUT=DEST/'data';sys.argv=[__file__,'--workers','2'];native.main()
    dump(DEST/'download_provenance.json',dict(complete=True,wrapper_sha256=sha(__file__),
        downloader_sha256=sha(native.__file__),selection_sha256=sha(DEST/'window_selection.json'),
        decoded_reference_labels=False,viewed_rgb=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['reserve','download']);globals()[ap.parse_args().action]()
