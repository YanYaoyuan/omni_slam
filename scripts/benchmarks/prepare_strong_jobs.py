"""Declare immutable staged frontend jobs before touching held-out RGB."""
import argparse
import json
from pathlib import Path
import numpy as np
from audit_evidence_retention import ROOT, BASE, OUT, CACHE, dump
from prepare_retention_holdout import SCENES,DATA
from omni_semantic_map.flat_benchmark import sha256


def sources():
    paths=[ROOT/'scripts/benchmarks/split_strong_frontend.py']
    up=ROOT/'build/benchmarks/ovo_full_20261110/upstream'
    paths += [up/p for p in ['ovo/entities/mask_generator.py','ovo/entities/clip_generator.py','ovo/entities/clips_merging.py',
        'ovo/utils/segment_utils.py','ovo/utils/clip_utils.py','ovo/utils/geometry_utils.py','ovo/slam/vanilla_mapper.py']]
    return {str(p.relative_to(ROOT)):sha256(p) for p in paths}


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['smoke','heldout'],required=True);p.add_argument('--tag',default='v1');args=p.parse_args()
    reference=CACHE/'office_0';old=json.loads((reference/'manifest.json').read_text())
    selected=json.loads((OUT/'stage3_selection.json').read_text())
    scenes=['replica_smoke'] if args.kind=='smoke' else list(SCENES)
    for scene in scenes:
        cfg=json.loads(json.dumps(old['config']));input_hashes={str(reference/name):sha256(reference/name) for name in ['manifest.json','text_features.npy','vocabulary.json']}
        if args.kind=='smoke':
            source=reference.parents[2]/'OVO_Replica_full/office_0'
            frames=[0,5,10];poses=np.loadtxt(source/'traj.txt').reshape(-1,4,4)
            colors=sorted((source/'results').glob('frame*.jpg'));depths=sorted((source/'results').glob('depth*.png'))
            records=[dict(frame_id=f,rgb=str(colors[f]),depth=str(depths[f]),pose=poses[f].tolist(),semantic=(i%2==0)) for i,f in enumerate(frames)]
            output=OUT/'split_frontend'/args.tag/'replica_smoke'
        else:
            assert json.loads((OUT/'split_frontend_validation.json').read_text())['complete']
            source=DATA/('sample400_'+scene);meta=json.loads((source/'manifest.json').read_text());assert meta['complete']
            coordinate=OUT/'metric_inputs'/scene/'poses.npz';basis=json.loads(coordinate.with_name('manifest.json').read_text())
            assert sha256(coordinate)==basis['output_sha256'] and basis['no_gt']
            values=np.load(coordinate);frames=values['source_ids'];poses=values['poses']
            records=[dict(frame_id=int(f),rgb=str(source/'image'/f'image{f+1:05d}.png'),depth=str(source/'depth'/f'depth{f+1:05d}.png'),pose=poses[i].tolist(),semantic=(i%2==0)) for i,f in enumerate(frames)]
            calibration={k:float(v) for k,v in (line.split() for line in (DATA/'main/intrinsic/asus.ini').read_text().splitlines())}
            cfg['cam'].update(H=int(calibration['color_height']),W=int(calibration['color_width']),fx=calibration['fx'],fy=calibration['fy'],cx=calibration['cx'],cy=calibration['cy'],depth_scale=1000.)
            input_hashes.update({str(coordinate):sha256(coordinate),str(source/'manifest.json'):sha256(source/'manifest.json'),str(OUT/'stage3_coordinate_protocol.json'):sha256(OUT/'stage3_coordinate_protocol.json')})
            output=DATA/'strong_frontend'/args.tag/scene
        for r in records:
            for key in ['rgb','depth']:input_hashes[r[key]]=sha256(Path(r[key]))
        path=OUT/'jobs'/args.tag/(scene+'.json')
        assert not path.exists() and not output.exists()
        dump(path,dict(kind='replica_smoke' if args.kind=='smoke' else 'scenenn',scene=scene,output=str(output),config=cfg,
            reference_cache=str(reference),frames=records,input_sha256=input_hashes,source_sha256=sources(),
            selection_sha256=sha256(OUT/'stage3_selection.json'),
            staging='SAM masks -> native SigLIP return_all triplets -> native learned merger -> native geometry; lossless CPU storage, eight-mask batches',
            no_gt_in_inference=True,peak_cuda_allocator_limit_gib=6))
        print('JOB',path,flush=True)


if __name__=='__main__':main()
