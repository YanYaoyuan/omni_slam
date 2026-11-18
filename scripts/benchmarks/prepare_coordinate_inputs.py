"""Freeze and apply a pose-only world-basis normalization to all held-out scenes."""
import argparse
import json
from pathlib import Path
import numpy as np
from prepare_retention_holdout import SCENES, DATA
from audit_evidence_retention import ROOT, OUT, REPORT, dump
from omni_semantic_map.flat_benchmark import sha256
from omni_semantic_map.scene_coordinate_basis import canonicalize_world


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--run',action='store_true');args=p.parse_args()
    target=OUT/'stage3_coordinate_protocol.json';source=ROOT/'omni_semantic_map/omni_semantic_map/scene_coordinate_basis.py'
    if args.freeze:
        assert not target.exists()
        dump(target,dict(stage=3,input_revision=2,scenes=SCENES,
            reason='SceneNN 021 provided rotation blocks have singular values approximately 1.042,1.000,0.963; left Gram is nearly constant (max variation .000254), consistent with a fixed affine world basis. No model scores have been computed.',
            rule='W=(R0 R0.T)^(-1/2) from first supplied pose only; left-multiply every pose AND reference mesh xyz by W; project residual rotations to SO(3) only if Frobenius residual <=.001.',
            use_all_scenes=True,uses_gt_for_transform=False,uses_pose_or_timestamp_fitting=False,
            tolerance=.001,keep_raw=True,
            indexing='uniform400 over provided trajectory; synchronized decoder index=sourceID+1; trailing video frames without provided poses are recorded and unused',
            mapping_config_unchanged=True,semantic_model_and_readout_unchanged=True,
            prior_selection_sha256=sha256(OUT/'stage3_selection.json'),
            source_sha256={str(x.relative_to(ROOT)):sha256(x) for x in [source,Path(__file__)]}))
        dump(REPORT/'stage3/coordinate_protocol.json',json.loads(target.read_text()))
    if not args.run:return
    protocol=json.loads(target.read_text())
    for file,h in protocol['source_sha256'].items():assert sha256(ROOT/file)==h
    downloaded=json.loads((OUT/'holdout_download.json').read_text());assert downloaded['complete']
    hashes={x['path']:x['sha256'] for x in downloaded['files']}
    for scene in SCENES:
        path=DATA/'main'/scene/'trajectory.log';assert sha256(path)==hashes[str(path)]
        lines=path.read_text().splitlines();raw=np.array([[list(map(float,line.split())) for line in lines[i+1:i+5]] for i in range(0,len(lines),5)])
        result=canonicalize_world(raw,protocol['tolerance']);ids=np.array([j*(len(raw)-1)//399 for j in range(400)])
        dest=OUT/'metric_inputs'/scene;dest.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(dest/'poses.npz',source_ids=ids,raw=raw[ids],poses=result['poses'][ids],canonical_from_raw=result['canonical_from_raw'])
        metadata={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in result.items() if k!='poses'}
        metadata.update(scene=scene,complete=True,source_poses=len(raw),sampled_poses=len(ids),no_gt=True,
            input_sha256=sha256(path),protocol_sha256=sha256(target),output_sha256=sha256(dest/'poses.npz'))
        dump(dest/'manifest.json',metadata);dump(REPORT/'stage3'/(scene+'_coordinate_basis.json'),metadata)
        print('CANONICAL',scene,'remaining rotation projection',metadata['max_remaining_rotation_projection'],flush=True)


if __name__=='__main__':main()
