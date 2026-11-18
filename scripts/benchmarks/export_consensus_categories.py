"""Export verified category-only candidate maps; preserve original foreground."""
from pathlib import Path
import numpy as np
from audit_evidence_retention import ROOT,OUT,dump
from public_dino_protocol import read,verify
from omni_semantic_map.flat_benchmark import sha256
from omni_semantic_map.shared_observations import observation_table
from omni_semantic_map.consensus_category import classify


def main():
    source=OUT/'public_category_evidence_v2';spec=read(source/'protocol.json');verify(spec)
    dest=OUT/'consensus_category_candidate_v1';dest.mkdir(exist_ok=False)
    verified=0;counts={}
    for e in spec['scenes']:
        key=e['key'];cache=Path(e['cache']);features,frames,areas=observation_table(cache)
        text=np.load(cache/'text_features.npy');text/=np.linalg.norm(text,axis=1,keepdims=True);vocab=read(cache/'vocabulary.json')
        old=read(Path(e['fixed_map'])/'classification.json')['objects'];pred={r['object_id']:r for r in read(source/'runs'/key/'predictions.json')['objects']}
        geometry=np.load(Path(e['fixed_map'])/'geometry.npz');objects=[];rows=[]
        for r in old:
            cat=classify(r['observation_ids'],features,frames,areas,text,vocab)
            expected=pred[r['object_id']]['methods']['consensus_sqrt_area']
            for k in ['name','replica_class_id','fallback']:assert cat[k]==expected[k]
            for k in ['score','margin','weights']:np.testing.assert_allclose(cat[k],expected[k],atol=1e-12,rtol=0)
            row=dict(r,category_update=cat,geometry_retained=r['predictions']['mean']['retained']);rows.append(row);verified+=1
            if row['geometry_retained']:
                xyz=geometry['xyz'][geometry['instance']==r['object_id']];lo,hi=xyz.min(0),xyz.max(0)
                objects.append(dict(object_id=r['object_id'],category=cat['name'],replica_class_id=cat['replica_class_id'],
                    baseline_category=r['predictions']['mean']['name'],center=((lo+hi)/2).tolist(),size=(hi-lo).tolist(),
                    bounds_min=lo.tolist(),bounds_max=hi.tolist(),observed_points=len(xyz),
                    selected_observation_ids=cat['selected_observation_ids'],weights=cat['weights']))
        target=dest/key
        dump(target/'classification.json',dict(objects=rows,active_category='category_update',retention='geometry_retained'))
        dump(target/'object_map.json',dict(objects=objects,geometry=str(Path(e['fixed_map'])/'geometry.npz'),
            geometry_sha256=sha256(Path(e['fixed_map'])/'geometry.npz'),
            coordinate_system='Original supplied-pose dataset coordinates; unchanged from native_bridge_2',
            selection='Preserve original mean-based foreground; new category never changes inclusion.',
            bounds='AABB of observed object points, not a trained 3D box detector or completed object extent.'))
        counts[key]=len(objects)
    sources=[Path(__file__).resolve(),ROOT/'omni_semantic_map/omni_semantic_map/consensus_category.py',ROOT/'omni_semantic_map/omni_semantic_map/category_evidence.py']
    dump(dest/'manifest.json',dict(complete=True,no_gt=True,development_candidate=True,
        source_protocol_sha256=sha256(source/'protocol.json'),verified_objects=verified,foreground_counts=counts,
        source_sha256={str(p):sha256(p) for p in sources},
        outputs={str(p.relative_to(dest)):sha256(p) for p in dest.glob('*/*.json')},
        promotion='Selected from viewed development results; exported for review, not activated in ROS runtime.'))
    print('EXPORTED',verified,counts,flush=True)


if __name__=='__main__':main()
