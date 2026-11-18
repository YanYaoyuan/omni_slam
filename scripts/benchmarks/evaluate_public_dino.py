"""Post-hoc instance and category metrics on frozen public reference domains."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

from public_dino_protocol import DEST,PARENT,read,verify
from audit_evidence_retention import ROOT,OUT,dump
from omni_semantic_map.flat_benchmark import sha256
from omni_semantic_map.full_ovo_evaluation import voxel_metrics
from omni_semantic_map.mesh_instance_evaluation import transfer_labels
from omni_semantic_map.instance_error_audit import audit_instances,filter_transition
from omni_semantic_map.semantic_instance_diagnostic import semantic_metrics


def classes_for_scene(entry,gt_ids):
    assert sha256(entry['semantic_metadata'])==entry['semantic_metadata_sha256']
    if entry['dataset']=='Replica':
        objects=read(entry['semantic_metadata'])['objects']
        mapping={int(o['id'])+1:int(o['class_id']) for o in objects if int(o['class_id'])>=0}
        provenance=dict(source=entry['semantic_metadata'],sha256=entry['semantic_metadata_sha256'])
    else:
        import h5py
        from evaluate_retention_holdout import mesh
        old=read(OUT/'stage3_cvpr18_domain_protocol.json')['files_sha256'];scene=entry['scene']
        coord=OUT/'official_annotation_audit/cvpr18'/f'scenenn_coords_{scene}.hdf5'
        meshpath=Path(read(OUT/'stage3_selection.json')['data_root'])/'main'/scene/(scene+'.ply')
        for p in [coord,meshpath,Path(entry['semantic_metadata'])]:assert sha256(p)==old[str(p)]
        with h5py.File(coord) as f:coords=f['coords'][:].reshape(-1,3)
        with h5py.File(entry['semantic_metadata']) as f:labels=f['label'][:].reshape(-1)
        coords,first,inv=np.unique(coords,axis=0,return_index=True);unique_labels=labels[first]
        assert np.array_equal(unique_labels[inv],labels)
        xyz,instances=mesh(meshpath);dist,idx=cKDTree(xyz).query(coords,workers=1);assert np.all(dist==0)
        counts=defaultdict(Counter)
        pairs,num=np.unique(np.column_stack([instances[idx],unique_labels]),axis=0,return_counts=True)
        for (iid,cls),n in zip(pairs,num):counts[int(iid)][int(cls)]+=int(n)
        mapping={};purities={}
        for iid in gt_ids:
            if not counts[int(iid)]:continue
            cls,n=max(counts[int(iid)].items(),key=lambda x:(x[1],-x[0]));purity=n/sum(counts[int(iid)].values())
            purities[int(iid)]=dict(dominant_class=cls,purity=purity,annotation_vertices=sum(counts[int(iid)].values()))
            if cls>0 and purity>=.7:mapping[int(iid)]=cls
        provenance=dict(sources={str(p):sha256(p) for p in [coord,meshpath,Path(entry['semantic_metadata'])]},instance_annotation_purities=purities)
    return {i:mapping[i] for i in map(int,gt_ids) if i in mapping},provenance


def evaluate(key):
    spec=read(DEST/'protocol.json');verify(spec);entry=next(e for e in spec['scenes'] if e['key']==key)
    run=DEST/'runs'/key;state=read(run/'run.json');assert state['complete'] and state['no_gt']
    assert state['protocol_sha256']==sha256(DEST/'protocol.json')
    for variant in state['variants']:
        for name,h in variant['files_sha256'].items():assert sha256(run/variant['name']/name)==h
    assert sha256(entry['reference'])==entry['hashes'][entry['reference']]
    reference=np.load(entry['reference']);vertices,truth=reference['xyz'],reference['instance']
    valid=reference['valid'] if 'valid' in reference.files else np.ones(len(truth),bool)
    gids=np.unique(truth[valid&(truth>0)]);gt_classes,provenance=classes_for_scene(entry,gids)
    sem_valid=valid&((truth==0)|np.isin(truth,list(gt_classes)))
    out=DEST/'evaluation'/key;out.mkdir(parents=True,exist_ok=False)
    dump(out/'semantic_reference.json',dict(gt_classes=gt_classes,provenance=provenance,gt_instances=len(gids),
        category_gt_instances=len(gt_classes),void_class_instances=sorted(set(map(int,gids))-set(gt_classes)),
        original_valid_voxels=int(valid.sum()),category_valid_voxels=int(sem_valid.sum())))
    methods={};base_audit=None
    for name in spec['variants']:
        folder=(PARENT/'runs'/key/name) if name in ['baseline','native_bridge_2'] else run/name
        geometry=np.load(folder/'geometry.npz');rows=read(folder/'classification.json')['objects']
        kept=[r['object_id'] for r in rows if r['predictions']['mean']['retained']]
        labels=np.where(np.isin(geometry['instance'],kept),geometry['instance'],0)
        projected,_=transfer_labels(geometry['xyz'],labels,vertices,mode='nearest_5cm')
        score=voxel_metrics(truth[valid],projected[valid]);audit=audit_instances(truth[valid],projected[valid])
        if name in ['baseline','native_bridge_2']:
            old=read(PARENT/'evaluation'/key/'metrics.json')['methods'][name]
            assert score['observed_voxel_instance_pq']==old['observed_voxel_instance_pq']
            np.testing.assert_array_equal(projected,np.load(PARENT/'evaluation'/key/(name+'.npz'))['instance'])
        if name=='baseline':base_audit=audit
        pred_classes={r['object_id']:(r['predictions']['mean']['replica_class_id'] if entry['dataset']=='Replica' else
                         spec['nyu_prediction_mapping'].get(r['predictions']['mean']['name'],-1)) for r in rows}
        semantic=semantic_metrics(truth[sem_valid],projected[sem_valid],gt_classes,pred_classes)
        score.update(predicted_error_counts=audit['predicted_counts'],gt_error_counts=audit['gt_counts'],
                     matched_gt_transition=filter_transition(base_audit,audit))
        methods[name]=dict(instance=score,semantic=semantic)
        np.savez_compressed(out/(name+'.npz'),instance=projected)
        dump(out/(name+'_audit.json'),audit)
    dump(out/'metrics.json',dict(complete=True,key=key,dataset=entry['dataset'],methods=methods,
        category_domain=dict(gt_instances=len(gids),annotated_instances=len(gt_classes),valid_voxel_fraction=float(sem_valid.sum()/valid.sum())),
        protocol_sha256=sha256(DEST/'protocol.json'),run_sha256=sha256(run/'run.json'),files_sha256={p.name:sha256(p) for p in out.iterdir()}))
    print('EVALUATED DINO',key,{n:(round(100*v['instance']['observed_voxel_instance_pq'],4),round(100*v['semantic']['class_aware_pq'],4)) for n,v in methods.items()},flush=True)


def summarize():
    spec=read(DEST/'protocol.json');rows=[read(DEST/'evaluation'/e['key']/'metrics.json') for e in spec['scenes']]
    assert all(r['complete'] for r in rows);datasets={}
    for dataset in ['Replica','SceneNN']:
        subset=[r for r in rows if r['dataset']==dataset];methods={}
        for name in spec['variants']:
            values=[r['methods'][name] for r in subset]
            delta=np.array([r['methods'][name]['instance']['observed_voxel_instance_pq']-r['methods']['native_bridge_2']['instance']['observed_voxel_instance_pq'] for r in subset])*100
            matches=sum(v['semantic']['geometry_matches'] for v in values);correct=sum(v['semantic']['correct_class_matches'] for v in values)
            methods[name]=dict(instance_pq100=float(np.mean([v['instance']['observed_voxel_instance_pq'] for v in values])*100),
                class_aware_pq100=float(np.mean([v['semantic']['class_aware_pq'] for v in values])*100),
                delta_vs_history_pq100=float(delta.mean()),wins=int((delta>1e-8).sum()),ties=int((abs(delta)<=1e-8).sum()),losses=int((delta< -1e-8).sum()),
                correct_class_matches=correct,geometry_matches=matches,matched_class_accuracy=correct/matches if matches else None,
                totals={k:sum(v['instance'][k] for v in values) for k in ['tp','fp','fn']},
                predicted_error_counts=dict(sum((Counter(v['instance']['predicted_error_counts']) for v in values),Counter())))
        datasets[dataset]=methods
    dump(DEST/'summary.json',dict(complete=True,datasets=datasets,scenes=rows,protocol_sha256=sha256(DEST/'protocol.json')))
    print(datasets,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene');p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.scene:evaluate(a.scene)
    if a.summarize:summarize()
