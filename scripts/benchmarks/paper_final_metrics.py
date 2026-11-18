"""Frozen observed-domain metrics plus explicitly separated mechanism diagnostics."""
from collections import Counter,defaultdict
import numpy as np
from paper_final_common import OLD,RUN,read,dump
from omni_semantic_map.full_ovo_evaluation import voxel_metrics
from omni_semantic_map.semantic_instance_diagnostic import semantic_metrics


def complete_metrics(truth,prediction,gt_classes=None,pred_classes=None):
    ins=voxel_metrics(truth,prediction);tp,fp,fn=[ins[k] for k in ['tp','fp','fn']]
    ins.update(precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None)
    if not tp:ins['sq']=None
    if not ins['gt_visible_instances'] and not ins['predicted_instances']:
        ins['observed_voxel_instance_pq']=None;ins['rq']=None
    if gt_classes is None:return ins
    sem=semantic_metrics(truth,prediction,gt_classes,pred_classes)
    gids=np.unique(truth[truth>0]);pids=np.unique(prediction[prediction>0])
    classes=sorted({int(gt_classes[int(i)]) for i in gids});perclass={}
    gclass=np.array([gt_classes.get(int(i),0) for i in truth]);pclass=np.array([pred_classes.get(int(i),-1) if i>0 else 0 for i in prediction])
    for c in classes:
        ng=sum(gt_classes[int(i)]==c for i in gids);npred=sum(pred_classes.get(int(i),-1)==c for i in pids)
        matches=[m for m in sem['matches'] if m['class_correct'] and m['gt_class']==c]
        n=len(matches);s=sum(m['iou'] for m in matches);den=.5*(ng+npred)
        intersection=int(((gclass==c)&(pclass==c)).sum());union=int(((gclass==c)|(pclass==c)).sum())
        perclass[str(c)]=dict(gt_instances=ng,predicted_instances=npred,tp=n,fp=npred-n,fn=ng-n,
            pq=s/den if den else None,sq=s/n if n else None,rq=n/den if den else None,
            voxel_intersection=intersection,voxel_union=union,iou=intersection/union if union else None)
    sem.update(per_class=perclass,macro_pq=float(np.mean([r['pq'] for r in perclass.values()])) if classes else None,
        foreground_miou=float(np.mean([r['iou'] for r in perclass.values()])) if classes else None,
        macro_class_set=classes,scope='observed annotated domain; not official full-mesh leaderboard')
    return dict(instance=ins,semantic=sem)


def geometry_signatures(truth,prediction):
    pairs,counts=np.unique(np.column_stack([truth,prediction]),axis=0,return_counts=True)
    gt=Counter();pred=Counter();overlaps=defaultdict(dict)
    for (g,p),n in zip(pairs,counts):
        if g>0:gt[int(g)]+=int(n)
        if p>0:pred[int(p)]+=int(n)
        if g>0 and p>0:overlaps[int(p)][int(g)]=int(n)
    fragments={str(g):sum(row.get(g,0)/n>=.1 for row in overlaps.values()) for g,n in gt.items()}
    mixed=[]
    for p,row in overlaps.items():
        significant=[g for g,n in row.items() if n/pred[p]>=.1]
        if len(significant)>=2:mixed.append(p)
    return dict(mixed_predicted_ids=mixed,mixed_object_rate=len(mixed)/len(pred) if pred else None,
        mixed_definition='at least two foreground GT IDs each cover >=10% of predicted object on valid domain',
        gt_fragment_counts=fragments,fragmented_gt=sum(v>1 for v in fragments.values()),
        fragmentation_definition='count predictions covering >=10% of each GT; raw overlaps retained',
        raw_overlap_counts={str(p):{str(g):n for g,n in row.items()} for p,row in overlaps.items()},
        gt_voxels=dict(gt),predicted_voxels=dict(pred))


def unique_annotation(coords,labels):
    """Conflicting labels at an identical official coordinate are void, never first-wins."""
    coords,first,inv=np.unique(coords,axis=0,return_index=True,return_inverse=True)
    cls=labels[first].copy()
    conflicts=np.unique(inv[cls[inv]!=labels])
    cls[conflicts]=0
    return coords,cls,len(conflicts)


def reference(scene,root=RUN):
    """Same original SceneNN CVPR18 domain construction as viewed validation."""
    import sys
    from scipy.spatial import cKDTree
    from evaluate_retention_holdout import mesh
    from omni_semantic_map.flat_surface_evaluation import majority_labels
    sys.path.insert(0,str(OLD/'tools/hdf5_reader'));import h5py
    xyz,instances=mesh(root/'data/main'/scene/(scene+'.ply'))
    with h5py.File(root/'data/annotation'/f'scenenn_coords_{scene}.hdf5') as f:coords=f['coords'][:].reshape(-1,3)
    with h5py.File(root/'data/annotation'/f'scenenn_seg_{scene}.hdf5') as f:labels=f['label'][:].reshape(-1)
    coords,cls,conflicting_coordinates=unique_annotation(coords,labels)
    distance,index=cKDTree(xyz).query(coords,workers=2);assert np.all(distance==0)
    pairs,n=np.unique(np.column_stack([instances[index],cls]),axis=0,return_counts=True);counts=defaultdict(Counter)
    for (iid,c),k in zip(pairs,n):counts[int(iid)][int(c)]+=int(k)
    categories={};purities={}
    for iid,c in counts.items():
        cl,k=max(c.items(),key=lambda item:(item[1],-item[0]));purity=k/sum(c.values());purities[iid]=dict(class_id=cl,purity=purity)
        if cl>0 and purity>=.7:categories[iid]=cl
    basis=np.load(root/'data'/('sample400_'+scene)/'poses.npz')['canonical_from_raw']
    xyz=xyz@basis[:3,:3].T+basis[:3,3];coords=coords@basis[:3,:3].T+basis[:3,3]
    distance,index=cKDTree(coords).query(xyz,workers=2);semantic=np.where(distance<=.05,cls[index],0)
    mesh_truth=np.where(np.isin(semantic,[1,2,22]),0,instances);mesh_truth[(semantic==0)|(instances==0)]=-1
    points=np.load(root/'frontend'/scene/'cache/geometry.npz')['xyz'];distance,index=cKDTree(xyz).query(points,workers=2);near=distance<=.05
    vox,inv=np.unique(np.floor(points[near]/.05).astype(np.int64),axis=0,return_inverse=True)
    truth=majority_labels(inv,mesh_truth[index[near]],len(vox));valid=truth>=0;vertices=(vox+.5)*.05
    semvalid=valid&((truth==0)|np.isin(truth,list(categories)))
    support_truth=np.full(len(points),-1,np.int64);support_truth[near]=truth[inv]
    domain=dict(raw_points=len(points),within_mesh_5cm=float(near.mean()),observed_voxels=len(vox),
        conflicting_annotation_coordinates_voided=conflicting_coordinates,
        valid_voxels=int(valid.sum()),category_valid_voxels=int(semvalid.sum()),valid_voxel_fraction=float(valid.mean()),
        category_valid_voxel_fraction=float(semvalid.sum()/valid.sum()) if valid.any() else None,
        gt_instances=len(np.unique(truth[truth>0])),category_gt_instances=len(np.unique(truth[semvalid&(truth>0)])))
    return dict(xyz=vertices,truth=truth,valid=valid,semvalid=semvalid,categories=categories,purities=purities,
                support_truth=support_truth,domain=domain)


def graph_diagnostics(cache,maproot,support_truth):
    """GT attached only post prediction; never feeds graph construction."""
    supports=[];m=read(cache/'manifest.json');observation_labels=[];reasons=Counter()
    def label(ids):
        labels=support_truth[ids];labels=labels[labels>=0]
        if len(labels)<16:return dict(gt=None,reason='fewer_than_16_valid_supports',valid_supports=len(labels))
        values,counts=np.unique(labels,return_counts=True);best=int(counts.argmax());purity=float(counts[best]/counts.sum())
        gt=int(values[best]);reason='foreground' if purity>=.7 and gt>0 else 'background' if purity>=.7 else 'mixed'
        return dict(gt=gt if reason=='foreground' else None,reason=reason,purity=purity,valid_supports=len(labels))
    for rec in m['records']:
        f=np.load(cache/'frames'/rec['file'])
        for i in range(rec['masks']):
            ids=f['point_ids'][f['offsets'][i]:f['offsets'][i+1]];supports.append(ids)
            record=label(ids);observation_labels.append(record);reasons[record['reason']]+=1
    nodes=read(maproot/'nodes.json');nodegt={}
    for node in nodes:
        ids=np.unique(np.concatenate([supports[i] for i in node['members']]))
        nodegt[node['alias']]=label(ids)
    variants={}
    for name in ['cg','history1','history2','history2_no_veto']:
        graph=read(maproot/name/'graph.json');edges=[]
        for e in graph['decisions']:
            a=nodegt[e['a']]['gt'];b=nodegt[e['b']]['gt']
            edges.append(dict(**e,gt_a=a,gt_b=b,correct=None if a is None or b is None else a==b))
        joined=[x for x in edges if x['reason']=='joined'];known=[x for x in joined if x['correct'] is not None]
        merges=read(maproot/name/'final_native_merges.json')
        variants[name]=dict(edges=edges,joined=len(joined),judgable_joined=len(known),
            joined_precision=sum(x['correct'] for x in known)/len(known) if known else None,
            joined_gt_coverage=len(known)/len(joined) if joined else None,
            wrong_joined=sum(not x['correct'] for x in known),
            coframe_rejections=sum(x['reason']=='coframe_conflict' for x in edges),
            post_native_merge_conflict_pairs=sum(len(x['coframe_conflicts']) for x in merges),
            native_final_merge_count=len(merges))
    return dict(complete=True,observation_GT_labels=observation_labels,observation_GT_status=dict(reasons),
                node_GT_labels=nodegt,variants=variants,
                assignment='unique measured support points on same observed voxel domain; >=16 valid supports and >=.7 dominant foreground purity')
