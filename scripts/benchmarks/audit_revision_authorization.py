"""Independent view-deletion, local graph and complete-export checks."""
from pathlib import Path
import itertools
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from paper_final_common import ROOT,read,dump,sha
from run_revision_authorization import OUT,PARENT,OBS,NEW,verify
from omni_semantic_map.batch_stable_transport import native_library,nearest_groups,nearest_owner


def run():
    p=verify();seal=read(OUT/'maps_complete.json');assert seal['complete'] and not (OUT/'delivery_audit.json').exists()
    prep=read(OUT/'prepared.json')
    for f,h in {**seal['files'],**prep['files']}.items():assert sha(f)==h,f
    prior=read(ROOT/'build/benchmarks/object_revision_provenance_v1/summary.json');results={};maps=deletions=pair_deletions=0
    for c in p['contexts']:
        key=c['key'];base=dict(np.load(Path(c['ours'])/'geometry.npz'));parent=dict(np.load(PARENT/'maps'/key/'cover_history/geometry.npz'));nold=len(base['xyz'])
        field=np.load(c['field']);atoms=np.load(OBS/'prepared'/key/'atoms.npz');part=np.load(PARENT/'prepared'/key/'components.npz');pf=np.load(PARENT/'transport'/key/'cover/field.npz')
        data=dict(np.load(OUT/'prepared'/key/'field.npz'));group=read(OUT/'prepared'/key/'groups.json');evidence=read(OUT/'prepared'/key/'evidence.json')
        # Read diagnostic evidence only in this audit; inference did not read it.
        common=0
        for name,e in evidence.items():
            values=np.array([r['shared']-r['separate'] for r in e['observations']]);worst=min([float(values.sum())]+[float(np.delete(values,i).sum()) for i in range(len(values))])
            assert abs(worst-e['worst_after_one_deletion'])<1e-12
            assert e['approved']==(worst>1e-10);pair_deletions+=len(values)
            a,b=name.split('_');old=prior['results'][key]['pair_observations'].get(name,prior['results'][key]['pair_observations'].get(b+'_'+a))
            if old is not None:
                assert len(old)==len(e['observations'])
                for x,y in zip(old,e['observations']):
                    assert x['frame_id']==y['frame_id']
                    np.testing.assert_allclose([x['shared'],x['exclusive']],[y['shared'],y['separate']],atol=1e-12,rtol=0)
                common+=1
        for g in group['groups'].values():
            assert g['members'][0]==g['recipient'] and set(g['members'])==set(g['accepted'])|{g['recipient']}
            for a,b in itertools.combinations(g['members'],2):assert evidence[f'{min(a,b)}_{max(a,b)}']['approved']
        # Rebuild local edge geometry instead of trusting stored graph weights.
        n=len(atoms['cell_ids']);edges=atoms['local_edges'];a,b=edges//n,edges%n
        centers=field['centers'][atoms['cell_ids']];distance=np.linalg.norm(centers[a]-centers[b],axis=1)
        graph=csr_matrix((np.r_[distance,distance],(np.r_[a,b],np.r_[b,a])),shape=(n,n));graph.sort_indices()
        np.testing.assert_array_equal(graph.indptr,data['graph_indptr']);np.testing.assert_array_equal(graph.indices,data['graph_indices']);np.testing.assert_array_equal(graph.data,data['graph_weights'])
        seeds=data['seeds'];labels=np.unique(seeds[:,1]);all_distance=[]
        for label in labels:
            sources=np.unique(seeds[seeds[:,1]==label,0]);all_distance.append(dijkstra(graph,directed=True,indices=sources,min_only=True))
        matrix=np.asarray(all_distance);ordered=np.sort(matrix,axis=0);nearest=ordered[:2].T
        if len(labels)==1:nearest=np.column_stack((ordered[0],np.full(n,np.inf)))
        np.testing.assert_allclose(nearest,data['owner_distance'],atol=1e-7,rtol=0)
        unambiguous=np.isfinite(nearest[:,0])&(nearest[:,0]<nearest[:,1]-1e-10)
        expected=labels[np.argmin(matrix[:,unambiguous],axis=0)]
        np.testing.assert_array_equal(expected,data['nearest_owner'][unambiguous,0])
        lib=native_library(OUT/'native');counts=np.bincount(seeds[:,2]);frames=sorted(np.flatnonzero(counts),key=lambda x:(-counts[x],x))[:2];rows=[]
        for frame in frames:
            remain=seeds[seeds[:,2]!=frame];own,dist=nearest_groups(graph,remain[:,0],remain[:,1],lib)
            dec=nearest_owner(own,dist);known=data['certified_owner']>=0
            np.testing.assert_array_equal(dec[known],data['certified_owner'][known]);deletions+=1
            rows.append(dict(frame_rank=int(frame),removed_seeds=int(counts[frame]),certified_cells_checked=int(known.sum()),changed=0))
        # Exact cell footprints from old XYZ, independent of prepared oldcell.
        dtype=np.dtype([('x',np.int64),('y',np.int64),('z',np.int64)])
        fk=np.ascontiguousarray(field['cells'],np.int64).view(dtype).ravel();bk=np.ascontiguousarray(np.floor(base['xyz']/.2),np.int64).view(dtype).ravel()
        pos=np.searchsorted(fk,bk);valid=pos<len(fk);valid[valid]&=fk[pos[valid]]==bk[valid]
        oldcell=np.full(nold,-1,np.int64);oldcell[valid]=pos[valid];np.testing.assert_array_equal(oldcell,part['old_cell'])
        lo=np.full(len(fk),np.iinfo(np.int64).max,np.int64);hi=np.full(len(fk),-1,np.int64)
        np.minimum.at(lo,pos[valid],base['instance'][valid]);np.maximum.at(hi,pos[valid],base['instance'][valid])
        np.testing.assert_array_equal(np.where((lo==hi)&(lo>0),lo,0),data['pure_old_cell_id'])
        # Every labelled anchor has an actual mask-witnessed cell and frame.
        witness=set(map(tuple,data['witnesses'][:,[0,2]].tolist()))
        for node,label,frame in seeds:
            assert (int(node),int(frame)) in witness
            component=str(int(part['component'][node]));oid=int(data['pure_old_cell_id'][atoms['cell_ids'][node]])
            if str(oid) in group['aliases'][component]:expected=group['aliases'][component][str(oid)]
            else:assert not group['members'][component];expected=group['groups'][component]['recipient']
            assert label==expected
        source=parent['added_full_source_ids'];nodes=data['cell_atom'][field['source_to_cell'][source]];direct=nodes>=0
        direct[direct]&=part['cover_accepted'][part['component'][nodes[direct]]]
        direct[direct]&=pf['root_owner'][part['root'][nodes[direct]]]>0
        witnessed=np.zeros(len(source),bool);witnessed[direct]=data['witness_node'][nodes[direct]]
        local=np.full(len(source),-1,np.int64);local[direct]=data['certified_owner'][nodes[direct]]
        hist=base['instance'].copy()
        for component,g in group['groups'].items():
            take=(data['old_component']==int(component))&np.isin(base['instance'],g['accepted'])&(parent['instance'][:nold]==g['recipient'])
            ii=np.flatnonzero(take);take[ii]&=data['certified_owner'][data['cell_atom'][oldcell[ii]]]==g['recipient'];hist[take]=g['recipient']
        full=np.load(Path(c['full'])/'geometry.npz')['xyz'];classes=read(PARENT/'maps'/key/'cover_history/classification.json')
        for method in NEW:
            if method=='identity_only':keep=~direct|(local>=0);newlabel=parent['instance'][nold:].copy();newlabel[direct]=local[direct];prefix=hist
            elif method=='write_only':keep=direct&witnessed;newlabel=parent['instance'][nold:];prefix=parent['instance'][:nold]
            else:keep=direct&witnessed&(local>=0);newlabel=local;prefix=base['instance'] if method=='current_only' else hist
            g=np.load(OUT/'maps'/key/method/'geometry.npz')
            np.testing.assert_array_equal(g['xyz'][:nold],base['xyz']);np.testing.assert_array_equal(g['instance'][:nold],prefix)
            np.testing.assert_array_equal(g['added_full_source_ids'],source[keep]);np.testing.assert_array_equal(g['xyz'][nold:],full[source[keep]])
            np.testing.assert_array_equal(g['instance'][nold:],newlabel[keep]);assert read(OUT/'maps'/key/method/'classification.json')==classes;maps+=1
        current=np.load(OUT/'maps'/key/'current_only/geometry.npz');ours=np.load(OUT/'maps'/key/'ours/geometry.npz')
        np.testing.assert_array_equal(current['xyz'],ours['xyz']);np.testing.assert_array_equal(current['instance'][nold:],ours['instance'][nold:])
        results[key]=dict(common_pair_evidence_replayed=common,all_group_pairs_approved=True,all_atom_group_distances_scipy_checked=n,
            actual_frame_removals=rows,exact_old_cell_mapping_verified=True,all_seeds_have_mask_witness=True,
            complete_map_replays=4,current_ablation_only_old_labels=True)
        print('AUTHORIZED MAP AUDIT',key,results[key],flush=True)
    dump(OUT/'delivery_audit.json',dict(complete=True,results=results,complete_map_replays=maps,actual_frame_removals=deletions,
        explicit_pair_view_deletions=pair_deletions,source_sha256=sha(__file__),maps_seal_sha256=sha(OUT/'maps_complete.json'),
        semantic_accuracy_not_proven=True))


if __name__=='__main__':run()
