"""Portable execution of the already frozen geometry_graph REFERENCE control.

Not a new method. Shared measured-support graphs only avoid repeating identical
radius queries; component membership and ordering must match the old routine.
"""
import copy
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from omni_semantic_map.query_hypothesis_evidence import membership
from omni_semantic_map.signed_query_association import positive_components


def frame_components(points,inside,radius=.4,minimum=16):
    points=np.asarray(points,float);inside=np.asarray(inside,bool)
    if inside.ndim!=2 or inside.shape[1]!=len(points):raise ValueError('Query support shape mismatch')
    edges=cKDTree(points).query_pairs(radius,output_type='ndarray')
    graph=coo_matrix((np.ones(len(edges),np.uint8),(edges[:,0],edges[:,1])),shape=(len(points),len(points))).tocsr()
    cache={};results=[]
    for mask in inside:
        indices=np.flatnonzero(mask)
        if len(indices)<minimum:results.append(np.empty(0,np.int64));continue
        token=np.packbits(mask).tobytes()
        if token not in cache:
            _,labels=connected_components(graph[indices][:,indices],directed=False);counts=np.bincount(labels);winner=int(counts.argmax())
            cache[token]=indices[labels==winner] if counts[winner]>=minimum else np.empty(0,np.int64)
        results.append(cache[token])
    return results


def export_reference(base,classification,source,sets,metadata,cells):
    n=len(metadata);matrix=membership([np.unique(cells[s]) for s in sets],int(cells.max())+1)
    overlap=(matrix@matrix.T).tocoo();a,b=overlap.row,overlap.col;size=np.diff(matrix.indptr);semantics=np.array([m['semantic_id'] for m in metadata])
    same=(a<b)&(semantics[a]==semantics[b]);a,b,inter=a[same],b[same],overlap.data[same];union=size[a]+size[b]-inter
    groups=positive_components(n,a,b,inter/np.maximum(1,union)>.5)
    by_id={o['object_id']:o for o in classification['objects']};old_ids=[m['old_object_id'] for m in metadata if m['old_object_id']>=0];locked=np.unique(np.r_[np.concatenate([sets[j] for j,m in enumerate(metadata) if m['old_object_id']>=0]),base['added_source_ids']])
    assignments={};proposals=[];next_id=max(old_ids)+1
    for members in groups:
        cls=np.unique(semantics[members]);assert len(cls)==1;sem=int(cls[0]);old=sorted(metadata[j]['old_object_id'] for j in members if metadata[j]['old_object_id']>=0);raw=[j for j in members if metadata[j]['hypothesis_index']>=0];frames=sorted(set(int(metadata[j]['seed_frame']) for j in raw))
        if old:
            oid=old[0];score=max(by_id[i]['predictions']['mean']['score'] for i in old)
            for i in old:assignments[i]=oid
        else:
            if len(frames)<3 or max((metadata[j]['score'] for j in raw),default=0)<=.1:continue
            oid=next_id;next_id+=1;score=float(np.mean([max(metadata[j]['score'] for j in raw if metadata[j]['seed_frame']==f) for f in frames]))
        ids=np.unique(np.concatenate([sets[j] for j in members]));proposals.append(dict(object_id=oid,semantic_id=sem,score=score,old_ids=old,node_ids=members,frames=frames,source_ids=ids))
    assert set(assignments)==set(old_ids);used=np.zeros(len(source),bool);used[locked]=True;new_xyz=[];new_instance=[];new_sources=[];records=[];group_records=[]
    for proposal in sorted(proposals,key=lambda q:(-q['score'],q['object_id'])):
        ids=proposal.pop('source_ids');free=ids[~used[ids]]
        if not proposal['old_ids'] and len(free)<16:continue
        used[free]=True;oid=proposal['object_id'];new_xyz.append(source[free]);new_instance.append(np.full(len(free),oid,np.int64));new_sources.append(free);proposal['added_points']=len(free);group_records.append(proposal)
        if len(proposal['old_ids'])==1:record=copy.deepcopy(by_id[proposal['old_ids'][0]])
        else:
            sem=proposal['semantic_id'];observations=sorted(set(v for i in proposal['old_ids'] for v in by_id[i].get('observation_ids',[])));record=dict(object_id=oid,observation_ids=observations,selected_observation_ids=observations,predictions=dict(mean=dict(name='car' if sem==26 else 'building',semantic_id=sem,train_id={11:2,26:13}[sem],score=proposal['score'],retained=True)))
        record['source_node_ids']=proposal['node_ids'];record['merged_old_ids']=proposal['old_ids'];records.append(record)
    old_instance=np.array([assignments[int(v)] for v in base['instance']],np.int64);added=np.concatenate(new_sources) if new_sources else np.empty(0,np.int64);points=np.concatenate([base['xyz']]+new_xyz);instance=np.concatenate([old_instance]+new_instance)
    assert len(np.unique(added))==len(added) and not np.isin(added,locked).any();np.testing.assert_array_equal(points[:len(base['xyz'])],base['xyz']);np.testing.assert_array_equal(points[len(base['xyz']):],source[added])
    export=copy.deepcopy(classification);export['objects']=sorted(records,key=lambda q:q['object_id'])
    stat=dict(objects=len(records),new_objects=sum(not r['old_ids'] for r in group_records),old_ids_merged=sum(len(r['old_ids'])-1 for r in group_records if r['old_ids']),old_point_ids_changed=int((old_instance!=base['instance']).sum()),added_points=len(added))
    return dict(xyz=points,instance=instance,added_source_ids=added),export,dict(groups=group_records,old_id_assignment=assignments,merge_trace=[]),stat
