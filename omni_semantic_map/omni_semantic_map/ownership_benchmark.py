"""Visibility ownership experiments using frozen Replica detector caches."""
import argparse,json,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from PIL import Image
from . import replica_benchmark as replica
from .flat_benchmark import sha256
from .ownership_graph import OwnershipGraph,OwnershipConfig
from .competitive_graph import CompetitiveGraph,CompetitiveConfig

VARIANTS=('ownership_hard','ownership_witness','ownership_no_appearance')
COMPETITIVE=('competitive','competitive_no_veto','competitive_no_rank')


def run(base,scene,competitive=False):
    cache=base/'cache'/scene;output=base/'runs'/scene
    meta=json.loads((cache/'manifest.json').read_text())
    if not meta['complete']:raise ValueError('incomplete cache')
    for n,h in meta['files_sha256'].items():
        if sha256(cache/n)!=h:raise ValueError('modified cache '+n)
    for n,h in json.loads((cache/'input_sha256.json').read_text()).items():
        if sha256(n)!=h:raise ValueError('modified input '+n)
    obs=json.loads((cache/'observations.json').read_text());frames=json.loads((cache/'frames.json').read_text());poses=np.load(cache/'poses.npy')
    views=[]
    for f in frames:
        fid=f['source_frame']
        with np.load(cache/'frames'/f'{fid:04d}.npz') as d:
            views.append(dict(owner=d['owner'].copy(),valid=d['valid'].copy(),
                depth=np.array(Image.open(base/'data'/scene/'depth'/f'depth_{fid}.png'))[::8,::8]/1000.,
                stride=8,intrinsic=np.array([[600.,0,599.5],[0,600.,339.5],[0,0,1.]])))
    for o in obs:
        with np.load(cache/o['feature_file']) as f:
            for key in ['descriptor','local_xyz','local_features','support_xyz']:o[key]=f[key].astype(np.float32)
    summary_path=output/'summary.json';summary=json.loads(summary_path.read_text())
    variants=COMPETITIVE if competitive else VARIANTS
    for name in variants:
        if (output/(name+'.npz')).exists():raise FileExistsError(name)
    for name in variants:
        if competitive:
            config=CompetitiveConfig(rank_by_ownership=not name.endswith('no_rank'),separation_veto=not name.endswith('no_veto'))
            t=CompetitiveGraph(views,config)
        else:
            config=OwnershipConfig(appearance_weight=0. if name.endswith('no_appearance') else .6)
            t=OwnershipGraph(views,config,'hard' if name.endswith('hard') else 'witness')
        causal=np.zeros(len(obs),np.int32);start=time.monotonic()
        for k,frame in enumerate(frames):
            ids=frame['observations'];causal[ids]=t.add_frame([obs[i] for i in ids],poses)
            if k%50==0:print(scene,name,k,flush=True)
        np.savez_compressed(output/(name+'.npz'),causal=causal,final=t.labels())
        replica.dump(output/(name+'_events.json'),t.events)
        evidence=t.directional if competitive else t.edge_evidence
        replica.dump(output/(name+'_edge_evidence.json'),[dict(observations=list(pair),**e) for pair,e in evidence.items()])
        summary['methods'][name]=dict(wall_s=time.monotonic()-start,config=asdict(config),
            final_objects=len(np.unique(t.labels())),causal_emitted_ids=len(np.unique(causal)),
            accepted_edges=len(t.edges),rejected_edges=0 if competitive else sum(not e['accepted'] for e in t.edge_evidence.values()))
    summary['ownership_source_sha256']={p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name('ownership_graph.py'),Path(__file__).with_name('competitive_graph.py')]}
    replica.dump(summary_path,summary)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['run','evaluate'])
    p.add_argument('--base',type=Path,required=True);p.add_argument('--scene',required=True)
    p.add_argument('--competitive',action='store_true')
    a=p.parse_args()
    if a.action=='run':run(a.base,a.scene,a.competitive)
    else:
        replica.METHODS=tuple(json.loads((a.base/'runs'/a.scene/'summary.json').read_text())['methods'])
        result=replica.evaluate(a.base,a.scene)
        print(json.dumps({k:v['voxel']['observed_voxel_instance_pq'] for k,v in result['methods'].items()},indent=2))

if __name__=='__main__':main()
