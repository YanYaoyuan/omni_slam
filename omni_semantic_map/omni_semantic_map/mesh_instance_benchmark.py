"""Frozen SceneNN prediction re-evaluation on full mesh vertices.

Primary: class-agnostic ScanNet AP kernel, all named thing instances, >=100
vertices, single-neighbor transfer limited to 5cm. Full mesh is primary;
input-observed vertices and unbounded transfer are separate sensitivities.
No learned model, map identity, old prediction or old score is modified.
"""
import argparse
import gzip
import json
from pathlib import Path
import pickle
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree

from . import scenenn_benchmark as bench
from .flat_benchmark import sha256
from .mesh_instance_evaluation import ScanNetAP, UPSTREAM, transfer_labels

BASE = Path('build/benchmarks/mesh_instance_20261110')
TRANSFERS = ('nearest_5cm','nearest_unbounded','knn5_unbounded')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def prepare(scene):
    """Freeze exported prediction coordinates before opening semantic GT."""
    cache, records, poses, n = bench.load(scene)
    out = BASE/'predictions'/scene; out.mkdir(parents=True, exist_ok=False)
    world, owners = [], []
    for k, group in enumerate(records):
        for o in group:
            world.append(o.points@poses[k,:3,:3].T+poses[k,:3,3])
            owners.append(np.full(len(o.points),o.observation_id,np.int32))
    world, owners = np.concatenate(world), np.concatenate(owners)
    obs = json.loads((cache/'observations.json').read_text())
    scores = np.array([o['score'] for o in obs])
    np.savez_compressed(out/'raw_supports.npz', points=world, owner=owners, scores=scores)
    ref = bench.BASE/'reference'/scene
    labels = {'replay_raw':np.load(ref/'labels.npy')}
    sources = [cache/'manifest.json',ref/'labels.npy',ref/'objects.pkl.gz']
    with gzip.open(ref/'objects.pkl.gz','rb') as f:
        objects = pickle.load(f)
    points = np.concatenate([o['points'] for o in objects])
    point_ids = np.concatenate([np.full(len(o['points']),o['object_id'],np.int32) for o in objects])
    confidences = {int(o['object_id']):float(scores[np.unique(o['observation_ids'])].mean()) for o in objects}
    np.savez_compressed(out/'replay_native_geometry.npz',points=points,labels=point_ids)
    for case in ('head10','tail10','global'):
        for scale in ('small','large'):
            key=case+'_'+scale
            fp=bench.BASE/'runs'/scene/key/'cached_full/labels.npz'
            op=bench.BASE/'ovo_control'/scene/key/'labels.npy'
            with np.load(fp) as f:
                labels['frozen_raw/'+key]=f['frozen'].copy()
            labels['ovo_raw/'+key]=np.load(op); sources.extend([fp,op])
    np.savez_compressed(out/'assignments.npz',**labels)
    dump(out/'manifest.json',dict(complete=True,scene=scene,
         native_confidence=confidences,source_hashes={str(p):sha256(p) for p in sources},
         files_sha256={p.name:sha256(p) for p in out.glob('*.npz')}))


def domain(scene):
    xyz, original = bench.mesh(scene)
    xml = bench.DATA/'main'/scene/(scene+'.xml')
    elements=list(ET.parse(xml).getroot().iter('label'))
    names={int(x.attrib['id']):x.attrib.get('text','').strip() for x in elements}
    things={i:name for i,name in names.items() if name and name.lower()!='unlabeled'
            and not name.lower().startswith(('floor','wall','ceiling'))}
    truth=np.where(np.isin(original,list(things)),original,0)
    # Observation coverage uses all sampled valid depth points, including
    # undetected/background surfaces. No prediction-dependent domain shrinking.
    cache=bench.BASE/'cache'/scene; poses=np.load(cache/'poses.npz')['poses']; supports=[]
    hashes={str(xml):sha256(xml),str(bench.DATA/'main'/scene/(scene+'.ply')):sha256(bench.DATA/'main'/scene/(scene+'.ply'))}
    for k in range(len(poses)):
        p=cache/'frames'/f'{k:04d}.npz'; hashes[str(p)]=sha256(p)
        with np.load(p) as f:
            supports.append(f['points']@poses[k,:3,:3].T+poses[k,:3,3])
    distance=cKDTree(np.concatenate(supports)).query(xyz,workers=1)[0]
    observed=distance<=.05
    return xyz,truth,observed,things,hashes


def evaluate(scene):
    pred=BASE/'predictions'/scene; manifest=json.loads((pred/'manifest.json').read_text())
    assert manifest['complete']
    for name,h in manifest['files_sha256'].items():
        assert sha256(pred/name)==h
    for path,h in manifest['source_hashes'].items():
        assert sha256(Path(path))==h
    out=BASE/'evaluation'/scene; out.mkdir(parents=True,exist_ok=False)
    with np.load(pred/'raw_supports.npz') as f:
        points,owners,scores=f['points'],f['owner'],f['scores']
    with np.load(pred/'assignments.npz') as f:
        assignments={k:f[k].copy() for k in f.files}
    with np.load(pred/'replay_native_geometry.npz') as f:
        native_points,native_labels=f['points'],f['labels']
    xyz,truth,observed,names,hashes=domain(scene)
    np.savez_compressed(out/'domain.npz',xyz=xyz,truth=truth,observed=observed)
    counts={int(g):int((truth==g).sum()) for g in np.unique(truth) if g>0}
    observed_counts={int(g):int((truth[observed]==g).sum()) for g in counts}
    metadata=dict(vertices=len(truth),observed_vertices=int(observed.sum()),
        named_gt=[dict(id=g,name=names[g],vertices=c,observed_vertices=observed_counts[g]) for g,c in counts.items()],
        full_gt_ge100=sum(c>=100 for c in counts.values()),
        observed_gt_ge100=sum(c>=100 for c in observed_counts.values()),source_hashes=hashes)
    dump(out/'domain.json',metadata)
    kernel=ScanNetAP(); results={}; matched={}
    for name in ['replay_native_geometry',*assignments]:
        if name=='replay_native_geometry':
            cloud,ids=native_points,native_labels
            confidence={int(k):v for k,v in manifest['native_confidence'].items()}
        else:
            lab=assignments[name]; cloud,ids=points,lab[owners]
            confidence={int(g):float(scores[lab==g].mean()) for g in np.unique(lab) if g>0}
        for mode in TRANSFERS:
            vertex_ids,distance=transfer_labels(cloud,ids,xyz,mode=mode)
            key=name+'/'+mode; target=out/'labels'/name; target.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(target/(mode+'.npz'),labels=vertex_ids)
            for scope,selector in [('full',np.ones(len(truth),bool)),('observed',observed)]:
                values=vertex_ids[selector]; groups=np.unique(values[values>0])
                match=kernel.match(truth[selector],[values==g for g in groups],
                                   [confidence[int(g)] for g in groups],scene)
                metric=kernel.score({scene:match})
                record=key+'/'+scope
                matched[record]=match
                results[record]=dict(**metric,predicted_vertices=int((values>0).sum()),
                    predicted_instances=len(groups),eligible_predictions=len(match['pred']['object']),
                    nearest_distance_median=float(np.median(distance)) if np.isfinite(distance).all() else None,
                    extrapolated_beyond_5cm_vertices=int(((distance>.05)&(vertex_ids>0)&selector).sum()))
            print('SCORED',scene,name,mode,flush=True)
    with gzip.open(out/'matches.pkl.gz','wb') as f:
        pickle.dump(matched,f)
    dump(out/'metrics.json',dict(complete=True,scene=scene,domain=metadata,results=results,
        source_sha256={str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name('mesh_instance_evaluation.py'),*UPSTREAM.glob('*')]}))
    print('COMPLETE',scene,flush=True)


def aggregate():
    kernel=ScanNetAP(); scenes=('005','011','016'); data={}
    for s in scenes:
        directory=BASE/'evaluation'/s
        assert json.loads((directory/'metrics.json').read_text())['complete']
        with gzip.open(directory/'matches.pkl.gz','rb') as f:
            data[s]=pickle.load(f)
    keys=list(data[scenes[0]]); assert all(set(data[s])==set(keys) for s in scenes)
    dump(BASE/'pooled_metrics.json',{key:kernel.score({s:data[s][key] for s in scenes}) for key in keys})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','evaluate','aggregate']);p.add_argument('--scene')
    args=p.parse_args()
    if args.action=='aggregate':aggregate()
    else:globals()[args.action](args.scene)
