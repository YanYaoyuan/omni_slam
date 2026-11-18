"""Decode only the preselected 400 frames per unused SceneNN trajectory."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
from prepare_retention_holdout import SCENES, DATA
from audit_evidence_retention import ROOT, OUT, REPORT, dump
from omni_semantic_map.flat_benchmark import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--scenes',nargs='+',default=list(SCENES));p.add_argument('--resume-decoded',action='store_true');args=p.parse_args()
    assert set(args.scenes)<=set(SCENES)
    downloaded=json.loads((OUT/'holdout_download.json').read_text())
    tools=OUT/'tools';tools.mkdir(exist_ok=True)
    original=ROOT/'build/benchmarks/scenenn_20261110/tools/PlaybackSparse.cpp'
    source=tools/'Playback400.cpp';binary=tools/'playback400'
    text=original.read_text();needle='for(int j=0;j<200;j++) selected.insert(1+j*(pose_count-1)/199);'
    assert text.count(needle)==1
    source.write_text(text.replace(needle,'for(int j=0;j<400;j++) selected.insert(1+j*(pose_count-1)/399);'))
    subprocess.run(['g++','-O2','-I/usr/include/ni',str(source),'-o',str(binary),'-lOpenNI','-lfreeimage','-lboost_filesystem','-lboost_system'],check=True)
    entries={f['path']:f for f in downloaded['files']}
    for scene in args.scenes:
        output=DATA/('sample400_'+scene)
        if output.exists() and not args.resume_decoded:raise FileExistsError(output)
        if (output/'manifest.json').exists():raise FileExistsError('input already finalized')
        oni=DATA/'main/oni'/(scene+'.oni');trajectory=DATA/'main'/scene/'trajectory.log'
        assert str(oni) in entries and str(trajectory) in entries, 'requested scene not fully downloaded'
        for path in [oni,trajectory]:assert sha256(path)==entries[str(path)]['sha256']
        lines=trajectory.read_text().splitlines();assert len(lines)%5==0
        source_ids=np.array([int(lines[i].split()[0]) for i in range(0,len(lines),5)])
        np.testing.assert_array_equal(source_ids,np.arange(len(source_ids)))
        raw=np.array([[list(map(float,line.split())) for line in lines[i+1:i+5]] for i in range(0,len(lines),5)])
        assert len(raw)>=400 and np.isfinite(raw).all() and np.allclose(raw[:,3],[0,0,0,1])
        ids=np.array([j*(len(raw)-1)//399 for j in range(400)],np.int64)
        assert len(set(ids))==400
        if output.exists():
            assert args.resume_decoded and 'Recorded:' in (OUT/(scene+'_decode.log')).read_text(), 'decoder completion required before resuming metadata validation'
        else:
            with (OUT/(scene+'_decode.log')).open('w') as log:
                subprocess.run([str(binary),str(oni),str(output),str(len(raw))],stdout=log,stderr=subprocess.STDOUT,check=True)
        times=np.loadtxt(output/'timestamp.txt');assert len(times)>=len(raw),(scene,len(times),len(raw))
        np.testing.assert_array_equal(times[:,0],np.arange(1,len(times)+1))
        selected=raw[ids];poses=selected.copy();u,_,v=np.linalg.svd(selected[:,:3,:3]);poses[:,:3,:3]=u@v
        repair=float(np.max(np.linalg.norm(poses-selected,axis=(1,2))))
        assert (np.linalg.det(poses[:,:3,:3])>0).all()
        expected_rgb=[output/'image'/f'image{i+1:05d}.png' for i in ids]
        expected_depth=[output/'depth'/f'depth{i+1:05d}.png' for i in ids]
        assert set((output/'image').glob('*.png'))==set(expected_rgb)
        assert set((output/'depth').glob('*.png'))==set(expected_depth)
        hashes={}
        for rgb,depth in zip(expected_rgb,expected_depth):
            with Image.open(rgb) as image:assert image.size==(640,480)
            with Image.open(depth) as image:assert image.size==(640,480) and np.array(image).max()>0
            for path in [rgb,depth]:hashes[str(path.relative_to(output))]=sha256(path)
        # Parsing may finish for an affine provided world, but never expose
        # independent per-frame SVD repairs as usable metric poses in that case.
        pose_arrays=dict(source_ids=ids,raw=selected)
        if repair<.001:pose_arrays['poses']=poses
        np.savez_compressed(output/'poses.npz',**pose_arrays)
        hashes['poses.npz']=sha256(output/'poses.npz');hashes['timestamp.txt']=sha256(output/'timestamp.txt')
        result=dict(scene=scene,complete=True,source_frames=len(raw),decoded_video_frames=len(times),
            trailing_video_frames_without_pose=len(times)-len(raw),geometry_frames=400,semantic_frames=200,
            source_frame_ids=ids.tolist(),semantic_sample_indices=list(range(0,400,2)),
            index_policy='zero-index trajectory block i corresponds to synchronized decoded frame i+1; no fitted offset',
            maximum_rgb_depth_time_difference_ms=float(np.abs(times[ids,1]-times[ids,2]).max()/1000),
            max_rotation_orthonormalization_frobenius=repair,
            requires_world_basis_normalization=repair>=.001,
            original_decoder_sha256=sha256(original),decoder_source_sha256=sha256(source),decoder_binary_sha256=sha256(binary),
            file_sha256=hashes,scope='input integrity only; geometric alignment QA still required; no model or semantic-label evaluation')
        dump(output/'manifest.json',result);dump(REPORT/'stage3'/(scene+'_input_manifest.json'),result)
        print('DECODED',scene,len(raw),'->400 geometry/200 semantic',flush=True)


if __name__=='__main__':main()
