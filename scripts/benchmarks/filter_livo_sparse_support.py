"""Live-only sampling adaptation; preserve frozen public benchmark runs."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import yaml
from audit_evidence_retention import ROOT, dump
from omni_semantic_map.flat_benchmark import sha256
from omni_semantic_map.sparse_support_components import largest_measured_component


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--livo-config',type=Path,required=True);p.add_argument('--cg-config',type=Path,required=True);a=p.parse_args()
    old=json.loads((a.cache/'manifest.json').read_text());assert old['complete'] and old['no_association']
    config=yaml.safe_load(a.livo_config.read_text())['/**']['ros__parameters'];leaf=float(config['preprocess']['filter_size_surf'])
    cg=json.loads(a.cg_config.read_text());minimum=int(cg['min_points_threshold']);radius=2*leaf
    a.output.mkdir(exist_ok=False);(a.output/'frames').mkdir()
    protocol=dict(scope=__doc__,source_cache_sha256=sha256(a.cache/'manifest.json'),radius_m=radius,minimum_points=minimum,
        rule='Per original exclusive mask, retain largest connected component in measured 3D points; reject component below unchanged CG minimum support count.',
        derivation='Radius is twice configured LIVO surface voxel leaf, fixed before running filtered results. No object-specific size prior or GT.',
        source_sha256={str(q):sha256(q) for q in [Path(__file__),ROOT/'omni_semantic_map/omni_semantic_map/sparse_support_components.py',a.livo_config,a.cg_config]},
        limitations='Largest component can select background when its support dominates; thin/small objects may be rejected. Not a complete occlusion model.')
    dump(a.output/'support_protocol.json',protocol)
    for name,h in old['files_sha256'].items():
        assert sha256(a.cache/name)==h
        shutil.copyfile(a.cache/name,a.output/name)
    xyz=np.load(a.cache/'geometry.npz')['xyz'];records=[];changes=[]
    for rec in old['records']:
        assert sha256(a.cache/'frames'/rec['file'])==rec['sha256']
        with np.load(a.cache/'frames'/rec['file']) as f:arrays={k:f[k] for k in f.files}
        pieces=[];offsets=[0]
        for k in range(rec['masks']):
            ids=arrays['point_ids'][arrays['offsets'][k]:arrays['offsets'][k+1]]
            keep=largest_measured_component(xyz[ids],radius,minimum);pieces.append(ids[keep]);offsets.append(offsets[-1]+len(keep))
            changes.append(dict(frame_id=rec['frame_id'],mask_id=k,before=len(ids),after=len(keep)))
        arrays['point_ids']=np.concatenate(pieces) if pieces else np.empty(0,np.int64);arrays['offsets']=np.array(offsets,np.int64)
        path=a.output/'frames'/rec['file'];np.savez_compressed(path,**arrays)
        records.append(dict(rec,matched_supports=len(arrays['point_ids']),masks_without_support=int(np.count_nonzero(np.diff(arrays['offsets'])==0)),sha256=sha256(path)))
    manifest=dict(old,records=records,sparse_component_protocol_sha256=sha256(a.output/'support_protocol.json'),
        geometry_scope=old['geometry_scope']+'; live-only largest measured component support validation')
    manifest['files_sha256']={q.name:sha256(q) for q in a.output.iterdir() if q.is_file() and q.name!='manifest.json'}
    dump(a.output/'manifest.json',manifest);dump(a.output/'support_changes.json',dict(complete=True,records=changes,
        before=sum(r['before'] for r in changes),after=sum(r['after'] for r in changes)))
    print('FILTERED',sum(r['before'] for r in changes),'->',sum(r['after'] for r in changes),'radius',radius,flush=True)


if __name__=='__main__':main()
