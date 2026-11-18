"""Independent export replay plus actual frame-batch deletion checks."""
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from paper_final_common import read,dump,sha
from evaluate_censored_partition import OUT,verify_maps
from omni_semantic_map.batch_stable_transport import native_library,nearest_groups,nearest_owner,certify


def main():
    p=verify_maps();sealed=read(OUT/'maps_complete.json');assert sealed['complete']
    assert not (OUT/'delivery_audit.json').exists()
    for f,h in sealed['files'].items():assert sha(f)==h,f
    results={};checks=0
    for c in p['contexts']:
        tick=time.monotonic();key=c['key'];base=dict(np.load(Path(c['ours'])/'geometry.npz'))
        full=np.load(Path(c['full'])/'geometry.npz')['xyz'];field=np.load(c['field']);nc=len(field['cells']);sourcecell=field['source_to_cell']
        graph=csr_matrix((field['graph_weights'],field['graph_indices'],field['graph_indptr']),shape=(nc,nc))
        atoms=dict(np.load(OUT/'prepared'/key/'atoms.npz'))
        oldrecords={r['object_id']:r['predictions']['mean'] for r in read(Path(c['ours'])/'classification.json')['objects']}
        lib=native_library(OUT/'map_native');rows={}
        duplicate_free=cKDTree(base['xyz']).query(full,workers=1)[0]>.05
        for mode in ['pairwise','region']:
            data=dict(np.load(OUT/'transport'/key/mode/'field.npz'));dec=data['certified_owner']
            np.testing.assert_array_equal(data['duplicate_free'],duplicate_free)
            again=certify(data['nearest_owner'],data['owner_distance'],data['nearest_token'],data['token_distance'],data['token_groups'][:,0])
            np.testing.assert_array_equal(again,dec)
            safe=dec>=0;counts=np.bincount(data['seeds'][:,2]);chosen=sorted(np.flatnonzero(counts),key=lambda x:(-counts[x],x))[:2]
            deletions=[]
            for frame in chosen:
                remaining=data['seeds'][data['seeds'][:,2]!=frame]
                owners,dist=nearest_groups(graph,remaining[:,0],remaining[:,1],lib)
                without=nearest_owner(owners,dist)
                np.testing.assert_array_equal(without[safe],dec[safe])
                deletions.append(dict(frame_rank=int(frame),deleted_seed_records=int(counts[frame]),certified_cells_checked=int(safe.sum()),changed_certified_cells=0))
            link=read(OUT/'transport'/key/mode/'lineage.json');part=dict(np.load(OUT/'partitions'/key/mode/'partition.npz'))
            old_seen=set();root_seen=set()
            for row in link['aligned']:
                assert row['old_id'] not in old_seen and row['root'] not in root_seen
                old_seen.add(row['old_id']);root_seen.add(row['root'])
                assert oldrecords[row['old_id']]['train_id']==atoms['train_class'][row['root']]
                expected=((part['old_point_proposed_region']==row['root'])&(base['instance']==row['old_id'])).sum()
                assert expected==row['overlap_points'] and expected>0
            for row in link['births']:
                r=row['root'];assert part['source_count'][r]>=16 and part['observed_frames'][r]>=3
                assert row['object_id'] not in oldrecords and 0<row['score']<=1
            added=np.flatnonzero(duplicate_free&(dec[sourcecell]>0));old=base['instance'].copy();cell=data['old_cell']
            active=cell>=0;active[active]&=dec[cell[active]]>=0;old[active]=dec[cell[active]]
            names=['pairwise_history'] if mode=='pairwise' else ['region_current','region_history']
            for name in names:
                dest=OUT/'maps'/key/name;g=dict(np.load(dest/'geometry.npz'));rec=read(dest/'classification.json')
                n=len(base['xyz']);np.testing.assert_array_equal(g['xyz'][:n],base['xyz'])
                np.testing.assert_array_equal(g['instance'][:n],base['instance'] if name=='region_current' else old)
                np.testing.assert_array_equal(g['added_full_source_ids'],added)
                np.testing.assert_array_equal(g['xyz'][n:],full[added]);np.testing.assert_array_equal(g['instance'][n:],dec[sourcecell[added]])
                categories={r['object_id']:r['predictions']['mean'] for r in rec['objects']}
                assert len(categories)==len(rec['objects'])
                assert set(np.unique(g['instance']))<=(set(categories)|{0})
                assert all(categories[oid]==prior for oid,prior in oldrecords.items())
                checks+=1
            rows[mode]=dict(old_coordinates_and_labels_replayed=True,all_additions_exact_measured_coordinates=True,
                unique_source_ids=True,old_classes_scores_unchanged=True,lineage_class_and_count_checked=True,
                new_identity_eligibility_checked=True,frame_deletions=deletions)
        current=np.load(OUT/'maps'/key/'region_current/geometry.npz');history=np.load(OUT/'maps'/key/'region_history/geometry.npz')
        np.testing.assert_array_equal(current['xyz'],history['xyz']);n=len(base['xyz'])
        np.testing.assert_array_equal(current['instance'][n:],history['instance'][n:])
        np.testing.assert_array_equal(current['added_full_source_ids'],history['added_full_source_ids'])
        assert sha(OUT/'maps'/key/'region_current/classification.json')==sha(OUT/'maps'/key/'region_history/classification.json')
        results[key]=dict(methods=rows,history_ablation_changes_only_old_labels=True,seconds=time.monotonic()-tick)
        print('ALL OBJECT EXPORTS VERIFIED',key,flush=True)
    dump(OUT/'delivery_audit.json',dict(complete=True,results=results,complete_map_replays=checks,
        actual_frame_removals=8,source_sha256=sha(__file__),maps_seal_sha256=sha(OUT/'maps_complete.json'),
        semantics_not_verified_by_these_checks=True))


if __name__=='__main__':main()
