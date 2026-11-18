"""Map all preselected unseen inputs, then evaluate frozen category candidates."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import sys
import numpy as np
from audit_evidence_retention import ROOT,OUT,dump
from public_dino_protocol import read,verify
from omni_semantic_map.flat_benchmark import sha256

DEST=OUT/'category_holdout_v1'
METHODS=['mean','area','consensus_sqrt_area']


def mapping():
    from retention_holdout_backend import run
    import run_public_history_graph as history
    from omni_semantic_map.shared_observations import observation_table,select_views
    from omni_semantic_map.consensus_category import classify
    from omni_semantic_map.category_evidence import readout
    from omni_semantic_map.full_ovo_evaluation import BACKGROUND
    selection=read(DEST/'selection.json');verify(selection)
    cfg=read(OUT/'stage3_selection.json')['cg_config'];entries=[]
    for s in selection['scenes']:
        cache=DEST/'frontend'/s/'cache';baseline=DEST/'baseline'/s
        run(cache,baseline)
        entry=dict(key='scenenn_'+s,dataset='SceneNN',scene=s,cache=str(cache),config=cfg,
            baseline_geometry=str(baseline/'geometry.npz'),baseline_classification=str(baseline/'classification.json'),
            baseline_readout='mean',reference=str(DEST/'evaluation'/s/'reference.npz'),baseline_evaluation='not_scored_before_mapping')
        entry['hashes']={str(p):sha256(p) for p in [Path(cfg),baseline/'geometry.npz',baseline/'classification.json',cache/'manifest.json']}
        entries.append(entry)
    hroot=DEST/'history';dump(hroot/'protocol.json',dict(scenes=entries,source_sha256=selection['source_sha256']))
    history.DEST=hroot
    for e in entries:
        history.run(e['key'])
        cache=Path(e['cache']);features,frames,areas=observation_table(cache)
        text=np.load(cache/'text_features.npy');text/=np.linalg.norm(text,axis=1,keepdims=True);vocab=read(cache/'vocabulary.json')
        original=hroot/'runs'/e['key']/'native_bridge_2';rows=read(original/'classification.json')['objects'];updates=[]
        for r in rows:
            selected=np.array(select_views(r['observation_ids'],frames,areas,top_k=0));methods={}
            candidate=classify(r['observation_ids'],features,frames,areas,text,vocab)
            for method in METHODS:
                if method=='consensus_sqrt_area':p=candidate
                else:
                    out=readout(features[selected],areas[selected],areas[selected],text,method=method);order=np.argsort(-out['scores'],kind='stable');best=int(order[0])
                    p=dict(name=vocab['names'][best],replica_class_id=int(vocab['replica_class_ids'][best]),score=float(out['scores'][best]),
                        weights=out['weights'].tolist(),selected_observation_ids=selected.tolist())
                methods[method]=dict(p,retained=p['replica_class_id'] not in BACKGROUND)
            assert methods['mean']['replica_class_id']==r['predictions']['mean']['replica_class_id']
            updates.append(dict(object_id=r['object_id'],base_retained=r['predictions']['mean']['retained'],methods=methods))
        dump(DEST/'predictions'/(e['scene']+'.json'),dict(complete=True,no_gt=True,objects=updates,geometry=str(original/'geometry.npz'),geometry_sha256=sha256(original/'geometry.npz'),selection_sha256=sha256(DEST/'selection.json')))
    dump(DEST/'predictions_complete.json',dict(complete=True,files={s:sha256(DEST/'predictions'/(s+'.json')) for s in selection['scenes']}))


def evaluation():
    from scipy.spatial import cKDTree
    from evaluate_retention_holdout import mesh
    from omni_semantic_map.flat_surface_evaluation import majority_labels
    from omni_semantic_map.mesh_instance_evaluation import transfer_labels
    from omni_semantic_map.full_ovo_evaluation import voxel_metrics
    from omni_semantic_map.semantic_instance_diagnostic import semantic_metrics
    sys.path.insert(0,str(OUT/'tools/hdf5_reader'));import h5py
    spec=read(DEST/'selection.json');verify(spec);gate=read(DEST/'predictions_complete.json');assert gate['complete']
    for s,h in gate['files'].items():assert sha256(DEST/'predictions'/(s+'.json'))==h
    download=read(DEST/'download.json');assert download['complete']
    for f in download['files']:assert sha256(f['path'])==f['sha256']
    results=[]
    for s in spec['scenes']:
        state=read(DEST/'predictions'/(s+'.json'));assert state['no_gt'] and sha256(state['geometry'])==state['geometry_sha256']
        xyz,instances=mesh(DEST/'data/main'/s/(s+'.ply'))
        with h5py.File(DEST/'data/annotation'/f'scenenn_coords_{s}.hdf5') as f:coords=f['coords'][:].reshape(-1,3)
        with h5py.File(DEST/'data/annotation'/f'scenenn_seg_{s}.hdf5') as f:labels=f['label'][:].reshape(-1)
        coords,first,inv=np.unique(coords,axis=0,return_index=True,return_inverse=True);cls=labels[first]
        assert np.array_equal(cls[inv],labels)
        distance,index=cKDTree(xyz).query(coords,workers=2);assert np.all(distance==0)
        counts=defaultdict(Counter)
        pairs,n=np.unique(np.column_stack([instances[index],cls]),axis=0,return_counts=True)
        for (iid,c),k in zip(pairs,n):counts[int(iid)][int(c)]+=int(k)
        categories={};purities={}
        for iid,c in counts.items():
            cl,k=max(c.items(),key=lambda item:(item[1],-item[0]));purity=k/sum(c.values());purities[iid]=dict(class_id=cl,purity=purity)
            if cl>0 and purity>=.7:categories[iid]=cl
        basis=np.load(DEST/'data'/('sample400_'+s)/'poses.npz')['canonical_from_raw']
        xyz=xyz@basis[:3,:3].T+basis[:3,3];coords=coords@basis[:3,:3].T+basis[:3,3]
        distance,index=cKDTree(coords).query(xyz,workers=2);semantic=np.where(distance<=.05,cls[index],0)
        mesh_truth=np.where(np.isin(semantic,[1,2,22]),0,instances);mesh_truth[(semantic==0)|(instances==0)]=-1
        points=np.load(DEST/'frontend'/s/'cache/geometry.npz')['xyz'];distance,index=cKDTree(xyz).query(points,workers=2);near=distance<=.05
        vox,inv=np.unique(np.floor(points[near]/.05).astype(np.int64),axis=0,return_inverse=True)
        truth=majority_labels(inv,mesh_truth[index[near]],len(vox));valid=truth>=0;vertices=(vox+.5)*.05
        semvalid=valid&((truth==0)|np.isin(truth,list(categories)))
        geom=np.load(state['geometry']);rows=state['objects'];keep=[r['object_id'] for r in rows if r['base_retained']]
        projected,_=transfer_labels(geom['xyz'],np.where(np.isin(geom['instance'],keep),geom['instance'],0),vertices,mode='nearest_5cm')
        instance=voxel_metrics(truth[valid],projected[valid]);methods={};changes=[];baseline=None
        for method in METHODS:
            predclasses={r['object_id']:spec['nyu_prediction_mapping'].get(r['methods'][method]['name'],-1) for r in rows}
            fixed=semantic_metrics(truth[semvalid],projected[semvalid],categories,predclasses)
            if baseline is None:baseline={m['gt_id']:m for m in fixed['matches']}
            assert [(m['gt_id'],m['predicted_id'],m['iou']) for m in fixed['matches']]==[(m['gt_id'],m['predicted_id'],m['iou']) for m in baseline.values()]
            for m in fixed['matches']:
                b=baseline[m['gt_id']]
                if b['class_correct']!=m['class_correct']:changes.append(dict(method=method,gt_id=m['gt_id'],object_id=m['predicted_id'],before=b,after=m))
            newkeep=[r['object_id'] for r in rows if r['methods'][method]['retained']]
            filtered,_=transfer_labels(geom['xyz'],np.where(np.isin(geom['instance'],newkeep),geom['instance'],0),vertices,mode='nearest_5cm')
            methods[method]=dict(fixed=fixed,refiltered_instance=voxel_metrics(truth[valid],filtered[valid]),refiltered_semantic=semantic_metrics(truth[semvalid],filtered[semvalid],categories,predclasses))
        out=DEST/'evaluation'/s;out.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(out/'reference.npz',xyz=vertices,instance=truth,valid=valid)
        np.savez_compressed(out/'fixed_prediction.npz',instance=projected)
        dump(out/'category_reference.json',dict(classes=categories,purities=purities))
        result=dict(complete=True,scene=s,methods=methods,changes=changes,fixed_instance=instance,
            domain=dict(raw_points=len(points),within_mesh_5cm=float(near.mean()),observed_voxels=len(vox),valid_voxel_fraction=float(valid.mean()),category_valid_voxel_fraction=float(semvalid.sum()/valid.sum()),
            gt_instances=len(np.unique(truth[truth>0])),category_gt_instances=len(np.unique(truth[semvalid&(truth>0)]))),prediction_sha256=sha256(DEST/'predictions'/(s+'.json')))
        dump(out/'metrics.json',result);results.append(result)
        print('HOLDOUT',s,{m:(v['fixed']['correct_class_matches'],100*v['fixed']['class_aware_pq']) for m,v in methods.items()},flush=True)
    summary={}
    for method in METHODS:
        delta=np.array([100*(r['methods'][method]['fixed']['class_aware_pq']-r['methods']['mean']['fixed']['class_aware_pq']) for r in results])
        summary[method]=dict(class_pq100=float(np.mean([r['methods'][method]['fixed']['class_aware_pq'] for r in results])*100),
            delta_pq100=float(delta.mean()),correct=sum(r['methods'][method]['fixed']['correct_class_matches'] for r in results),matches=sum(r['methods'][method]['fixed']['geometry_matches'] for r in results),
            wins=int((delta>1e-8).sum()),ties=int((abs(delta)<=1e-8).sum()),losses=int((delta< -1e-8).sum()),
            gains=sum(c['after']['class_correct'] for r in results for c in r['changes'] if c['method']==method),
            regressions=sum(c['before']['class_correct'] for r in results for c in r['changes'] if c['method']==method))
    dump(DEST/'summary.json',dict(complete=True,selection_sha256=sha256(DEST/'selection.json'),methods=summary,scenes=results))
    print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mapping',action='store_true');p.add_argument('--evaluate',action='store_true');a=p.parse_args()
    if a.mapping:mapping()
    if a.evaluate:evaluation()
