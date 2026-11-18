"""Geometry-only input alignment QA, with no pose/offset fitting or scoring."""
import argparse
import json
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from prepare_retention_holdout import SCENES, DATA
from audit_evidence_retention import OUT, REPORT, dump
from omni_semantic_map.flat_benchmark import sha256


def mesh_xyz(path):
    with path.open('rb') as f:
        lines=[]
        while True:
            raw=f.readline()
            if not raw:raise ValueError('unterminated PLY header')
            line=raw.decode('ascii').strip();lines.append(line)
            if line=='end_header':break
        assert 'format binary_little_endian 1.0' in lines
        n=int(next(s for s in lines if s.startswith('element vertex ')).split()[-1])
        start=next(i for i,s in enumerate(lines) if s.startswith('element vertex '))+1
        types={'float':'<f4','float32':'<f4','double':'<f8','uint':'<u4','uint32':'<u4','int':'<i4','uchar':'u1','uint8':'u1'}
        columns=[]
        for line in lines[start:]:
            if not line.startswith('property '):break
            _,kind,name=line.split();columns.append((name,types[kind]))
        vertices=np.fromfile(f,dtype=np.dtype(columns),count=n)
        if len(vertices)!=n:raise ValueError('truncated vertex data')
    return np.column_stack([vertices[k] for k in ('x','y','z')])


def main():
    p=argparse.ArgumentParser();p.add_argument('--scenes',nargs='+',default=list(SCENES));p.add_argument('--canonical',action='store_true');args=p.parse_args()
    calibration={}
    for line in (DATA/'main/intrinsic/asus.ini').read_text().splitlines():
        key,value=line.split();calibration[key]=float(value)
    summary=[]
    for scene in args.scenes:
        source=DATA/('sample400_'+scene);manifest=json.loads((source/'manifest.json').read_text());assert manifest['complete']
        for name,h in manifest['file_sha256'].items():assert sha256(source/name)==h
        pose_path=OUT/'metric_inputs'/scene/'poses.npz' if args.canonical else source/'poses.npz'
        poses=np.load(pose_path);ids=poses['source_ids'];matrix=poses['poses']
        mesh=mesh_xyz(DATA/'main'/scene/(scene+'.ply'))
        if args.canonical:
            basis=poses['canonical_from_raw'];mesh=mesh@basis[:3,:3].T+basis[:3,3]
        tree=cKDTree(mesh)
        records=[]
        for j in np.linspace(0,len(ids)-1,10,dtype=int):
            depth=np.array(Image.open(source/'depth'/f'depth{ids[j]+1:05d}.png'))
            v,u=np.mgrid[0:480:8,0:640:8];z=depth[v,u]/1000
            keep=(z>=.1)&(z<=5)
            points=np.column_stack([((u-calibration['cx'])*z/calibration['fx'])[keep],((v-calibration['cy'])*z/calibration['fy'])[keep],z[keep]])
            world=points@matrix[j,:3,:3].T+matrix[j,:3,3]
            distances=tree.query(world)[0]
            if not len(distances):raise ValueError('no valid depth support')
            records.append(dict(sample_index=int(j),source_frame=int(ids[j]),valid_samples=len(distances),
                median_m=float(np.median(distances)),p90_m=float(np.quantile(distances,.9)),within_5cm=float((distances<=.05).mean())))
        result=dict(scene=scene,complete=True,scope=__doc__,checks=records,calibration=calibration,
            fit_applied=False,semantic_mesh_labels_read=False,canonical_world_basis=args.canonical,
            sampled_depth_domain='stride8, depth .1..5 m for QA only',
            median_of_medians_m=float(np.median([r['median_m'] for r in records])),
            any_frame_median_over_5cm=any(r['median_m']>.05 for r in records))
        suffix='_metric' if args.canonical else ''
        dump(REPORT/'stage3'/(scene+'_alignment'+suffix+'.json'),result);summary.append(result)
        print('QA',scene,result['median_of_medians_m'],'flag',result['any_frame_median_over_5cm'],flush=True)
    dump(OUT/('holdout_input_qa'+suffix+'.json'),dict(complete=True,scenes=summary,scope='input alignment, not model evaluation'))


if __name__=='__main__':main()
