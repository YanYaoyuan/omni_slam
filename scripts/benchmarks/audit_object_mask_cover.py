"""Raw-mask explanation replay, real-map history audit and source deletion."""
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from paper_final_common import ROOT,read,dump,sha
from run_object_mask_cover import OUT,PARENT,RAW,verify
from omni_semantic_map.batch_stable_transport import native_library,nearest_groups,nearest_owner,certify


def run():
    p=verify();seal=read(OUT/'maps_complete.json');assert seal['complete'] and not (OUT/'delivery_audit.json').exists()
    for f,h in seal['files'].items():assert sha(f)==h,f
    prepared=read(OUT/'prepared.json')
    for f,h in prepared['files'].items():assert sha(f)==h,f
    rawseal=read(RAW);results={};n_maps=n_deletions=n_explanations=0
    for c in p['contexts']:
        key=c['key'];atoms=dict(np.load(PARENT/'prepared'/key/'atoms.npz'));n=len(atoms['cell_ids'])
        part=dict(np.load(OUT/'prepared'/key/'components.npz'));e=atoms['local_edges'];left,right=e//n,e%n
        nc,cc=connected_components(csr_matrix((np.ones(2*len(e)),(np.r_[left,right],np.r_[right,left])),shape=(n,n)),directed=False)
        roots=np.full(nc,n);np.minimum.at(roots,cc,np.arange(n))
        np.testing.assert_array_equal(part['root'],roots[cc])
        field=np.load(c['field']);sourcecell=field['source_to_cell'];celln=len(field['cells'])
        cellcomp=np.full(celln,-1,np.int32);cellcomp[atoms['cell_ids']]=cc
        rawby={r['frame_id']:r for r in rawseal['results'][key]};records=read(OUT/'prepared'/key/'explanations.json')
        sums={m:np.zeros(nc) for m in ['single','cover']};support={m:np.zeros(nc,int) for m in sums};checked=0
        for rec in read(PARENT/'prepared'/key/'views.json'):
            rows=[r for r in records if r['frame_id']==rec['frame_id']]
            if not rows:continue
            v=np.load(rec['file']);raw=np.load(rawby[rec['frame_id']]['file']);px=v['pixel_xy']
            weights=1./np.bincount(v['cells'],minlength=celln)[v['cells']]
            comp=cellcomp[v['cells']];cache={}
            for row in rows:
                component=row['component'];rid=11 if part['component_class'][component]==2 else 26
                qcount=int(((raw['semantic_ids']==rid)&(raw['scores']>0)).sum());member=comp==component
                for method in sums:
                    response=row[method];queries=response['native_rows'];assert len(queries)==len(set(queries))
                    union=np.zeros(len(px),bool);cost=0.
                    for q in queries:
                        assert raw['semantic_ids'][q]==rid and raw['scores'][q]>0
                        if q not in cache:
                            unpacked=np.unpackbits(raw['masks_packed'][q],axis=1)
                            cache[q]=unpacked[px[:,1],px[:,0]].astype(bool)
                        union |= cache[q];cost+=np.log2(1+qcount)-np.log2(float(raw['scores'][q]))
                    inside=float(weights[union&member].sum());outside=float(weights[union&~member].sum())
                    np.testing.assert_allclose([inside,outside,cost,inside-outside-cost],
                        [response['inside'],response['outside'],response['cost'],response['utility']],rtol=0,atol=1e-7)
                    sums[method][component]+=inside-outside-cost;support[method][component]+=inside-outside-cost>1e-10
                    checked+=1
        for m in sums:
            np.testing.assert_allclose(sums[m],part[m+'_utility'],atol=1e-7,rtol=0)
            np.testing.assert_array_equal(support[m],part[m+'_support_frames'])
            np.testing.assert_array_equal((sums[m]>np.log2(nc+1))&(support[m]>=3),part[m+'_accepted'])
        n_explanations+=checked
        base=dict(np.load(Path(c['ours'])/'geometry.npz'));full=np.load(Path(c['full'])/'geometry.npz')['xyz']
        oldrecords={r['object_id']:r['predictions']['mean'] for r in read(Path(c['ours'])/'classification.json')['objects']}
        duplicate=cKDTree(base['xyz']).query(full,workers=1)[0]>.05
        graph=csr_matrix((field['graph_weights'],field['graph_indices'],field['graph_indptr']),shape=(celln,celln))
        lib=native_library(OUT/'map_native');deletions=[]
        for mode in ['geometry','single','cover']:
            data=dict(np.load(OUT/'transport'/key/mode/'field.npz'));dec=data['certified_owner']
            np.testing.assert_array_equal(data['duplicate_free'],duplicate)
            np.testing.assert_array_equal(certify(data['nearest_owner'],data['owner_distance'],data['nearest_token'],data['token_distance'],data['token_groups'][:,0]),dec)
            accepted=np.ones(nc,bool) if mode=='geometry' else part[mode+'_accepted']
            assert np.all(data['root_owner'][part['roots'][~accepted]]==0)
            expected_cellowner=np.zeros(celln,np.int64);expected_cellowner[atoms['cell_ids']]=data['root_owner'][part['root']]
            np.testing.assert_array_equal(data['cell_direct_owner'],expected_cellowner)
            lineage=read(OUT/'transport'/key/mode/'lineage.json');oldids=set();linked=set()
            for row in lineage['aligned']:
                r,o=row['root'],row['old_id'];assert r not in linked and o not in oldids;linked.add(r);oldids.add(o)
                assert oldrecords[o]['train_id']==atoms['train_class'][r]
                assert row['overlap_points']==int(((part['old_point_proposed_region']==r)&(base['instance']==o)).sum())>0
            for row in lineage['births']:
                r=row['root'];assert part['source_count'][r]>=16 and part['observed_frames'][r]>=3
                assert row['object_id'] not in oldrecords and 0<row['score']<=1
            keep=data['old_cell']>=0;keep[keep]&=(expected_cellowner[data['old_cell'][keep]]>0)&(dec[data['old_cell'][keep]]==expected_cellowner[data['old_cell'][keep]])
            np.testing.assert_array_equal(keep,data['old_edit_eligible'])
            old=base['instance'].copy();old[keep]=expected_cellowner[data['old_cell'][keep]]
            added=np.flatnonzero(duplicate&(dec[sourcecell]>0));names=['cover_current','cover_history'] if mode=='cover' else [mode+'_history']
            for name in names:
                g=np.load(OUT/'maps'/key/name/'geometry.npz');count=len(base['xyz'])
                np.testing.assert_array_equal(g['xyz'][:count],base['xyz'])
                np.testing.assert_array_equal(g['instance'][:count],base['instance'] if name=='cover_current' else old)
                np.testing.assert_array_equal(g['added_full_source_ids'],added)
                np.testing.assert_array_equal(g['xyz'][count:],full[added]);np.testing.assert_array_equal(g['instance'][count:],dec[sourcecell[added]])
                categories={r['object_id']:r['predictions']['mean'] for r in read(OUT/'maps'/key/name/'classification.json')['objects']}
                assert set(np.unique(g['instance']))<=(set(categories)|{0})
                assert all(categories[o]==v for o,v in oldrecords.items());n_maps+=1
            # Actual removal of the observation-frame batch with most seeds.
            counts=np.bincount(data['seeds'][:,2]);frame=int(np.argmax(counts));seeds=data['seeds'][data['seeds'][:,2]!=frame]
            own,dist=nearest_groups(graph,seeds[:,0],seeds[:,1],lib);without=nearest_owner(own,dist);known=dec>=0
            np.testing.assert_array_equal(without[known],dec[known]);n_deletions+=1
            deletions.append(dict(mode=mode,frame_rank=frame,deleted_seeds=int(counts[frame]),certified_labels_checked=int(known.sum()),changed=0))
        single=np.load(OUT/'maps'/key/'single_history/geometry.npz');cover=np.load(OUT/'maps'/key/'cover_history/geometry.npz')
        same_geometry=np.array_equal(single['xyz'],cover['xyz']);same_labels=np.array_equal(single['instance'],cover['instance'])
        same_scores=read(OUT/'maps'/key/'single_history/classification.json')==read(OUT/'maps'/key/'cover_history/classification.json')
        if np.array_equal(part['single_accepted'],part['cover_accepted']):assert same_geometry and same_labels and same_scores
        current=np.load(OUT/'maps'/key/'cover_current/geometry.npz');np.testing.assert_array_equal(current['xyz'],cover['xyz'])
        np.testing.assert_array_equal(current['instance'][len(base['xyz']):],cover['instance'][len(base['xyz']):])
        results[key]=dict(raw_explanations_checked=checked,connected_components_recomputed=True,
            direct_history_and_measured_additions_replayed=True,frame_removals=deletions,
            single_vs_cover_same_geometry=same_geometry,single_vs_cover_same_labels=same_labels,single_vs_cover_same_classes_scores=same_scores,
            cover_history_ablation_only_old_labels=True)
        print('COVER RAW AND MAP AUDIT',key,results[key],flush=True)
    dump(OUT/'delivery_audit.json',dict(complete=True,results=results,raw_explanation_checks=n_explanations,
        complete_map_replays=n_maps,actual_frame_removals=n_deletions,
        source_sha256=sha(__file__),maps_seal_sha256=sha(OUT/'maps_complete.json'),
        semantic_accuracy_not_established_by_checks=True))


if __name__=='__main__':run()
