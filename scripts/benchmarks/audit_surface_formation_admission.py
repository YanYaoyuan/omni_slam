"""Posthoc necessary coverage of the common pre-partition source pool."""
import numpy as np
from scipy.spatial import cKDTree
from paper_final_common import read,dump,sha
from run_surface_formation import OUT,BASE,INPUT,verify


def main():
    p=verify();results={}
    for s in p['scenes']:
        key=s['key'];d=np.load(OUT/'prepared'/key/'graph.npz');keys=d['atomkeys'];sid=d['source_ids'];atom=d['source_atoms']
        source=np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz'];base=np.load(BASE/key/'geometry_graph/geometry.npz')
        rows=read(BASE/key/'geometry_graph/classification.json')['objects'];ref=np.load(s['reference']);gt=ref['train_instance'];w=ref['confidence'].astype(float);mass=np.bincount(gt,weights=w);out=[]
        for tid in [2,13]:
            ids=[r['object_id'] for r in rows if r['predictions']['mean']['train_id']==tid]
            free=(keys[atom,4]==0)&(keys[atom,0]==tid)
            cloud=np.concatenate([base['xyz'][np.isin(base['instance'],ids)],source[sid[free]]])
            take=gt//1000==tid
            dist=cKDTree(cloud).query(ref['xyz'][take],distance_upper_bound=.2,workers=1)[0]
            covered=np.bincount(gt[take],weights=w[take]*(dist<.2),minlength=len(mass))
            for g in np.flatnonzero((np.arange(len(mass))//1000==tid)&(mass>=100)):
                out.append(dict(gt_id=int(g),coverage=float(covered[g]/mass[g])))
        results[key]=out
        print(key,{str(t):sum(r['gt_id']//1000==t and r['coverage']>.5 for r in out) for t in [2,13]},flush=True)
    # Reproduce the initial read-only diagnosis before recording reusable code.
    if (OUT/'admission_coverage.json').exists():assert read(OUT/'admission_coverage.json')['results']==results
    dump(OUT/'admission_coverage.json',dict(complete=True,diagnostic_only=True,results=results,source_sha256=sha(__file__),
        protocol_sha256=sha(OUT/'protocol.json'),scope='Maximum same-class candidate cloud: original frozen fused points plus ALL admitted unowned sources, before graph partition/birth gate. Perfect spatial assignment/background rejection permitted. Coverage is necessary only, not AP or achievable recall. No new maps/scoring.'))


if __name__=='__main__':main()
