"""Independent SciPy graph replay and complete export checks, no score tuning."""
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from paper_final_common import ROOT,read,dump,sha
from run_batch_stable_transport import OUT,FULL,OLD,SEM,BASE,METHODS,verify,seal


def run():
    p=verify();seal('prepared');seal('predictions_complete');results={}
    for scene in p['scenes']:
        key=scene['key'];start=time.monotonic();base=np.load(BASE/key/'geometry_graph/geometry.npz');full=np.load(FULL/key/'geometry.npz');oldmap=np.load(FULL/key/'old_to_full.npy')
        sem=np.load(SEM/'aggregate'/key/'source.npz');sourceowner=np.zeros(len(full['xyz']),np.int64);sourceowner[oldmap]=sem['owners']
        fields={d:dict(np.load(OUT/'fields'/key/d/'field.npz')) for d in ['full','camera']};exports={}
        for method in METHODS:
            folder=OUT/'maps'/key/method;g=np.load(folder/'geometry.npz')
            assert sha(folder/'classification.json')==sha(BASE/key/'geometry_graph/classification.json')
            if method=='baseline':
                assert sha(folder/'geometry.npz')==sha(BASE/key/'geometry_graph/geometry.npz');exports[method]={'exact_copy':True};continue
            added=g['added_full_source_ids'];n=len(base['xyz']);np.testing.assert_array_equal(g['xyz'][:n],base['xyz']);np.testing.assert_array_equal(g['instance'][:n],base['instance'])
            np.testing.assert_array_equal(g['xyz'][n:],full['xyz'][added]);assert (sourceowner[added]==0).all()
            np.testing.assert_array_equal(added,np.unique(added))
            domain='camera' if method=='camera_frame' else 'full';field=fields[domain];mode='nearest' if method=='full_nearest' else 'cell' if method=='full_two_cells' else 'frame'
            mapping=oldmap if domain=='camera' else np.arange(len(sourceowner));pred=field[mode+'_owner'][field['source_to_cell']]
            keep=(pred>0)&(sourceowner[mapping]==0);order=np.argsort(mapping[keep]);expected=mapping[keep][order]
            np.testing.assert_array_equal(added,expected);np.testing.assert_array_equal(g['instance'][n:],pred[keep][order]);assert set(g['instance'])<=set(base['instance'])
            exports[method]=dict(added_sources=len(added),all_coordinates_labels_replayed=True,unchanged_old_map=True,no_new_id=True)
        d=fields['full'];n=len(d['centers']);graph=csr_matrix((d['graph_weights'],d['graph_indices'],d['graph_indptr']),shape=(n,n));seed=d['seeds']
        # Replay every node's first distance using a separate upstream solver.
        distance=dijkstra(graph,indices=np.unique(seed[:,0]),min_only=True)
        np.testing.assert_allclose(distance,d['label_distance'][:,0],rtol=1e-12,atol=1e-10)
        selected=[]
        for mask in [d['frame_owner']>0,(d['nearest_owner']>0)&(d['frame_owner']<0),d['nearest_owner']==0,d['nearest_owner']<0]:
            ids=np.flatnonzero(mask)
            if len(ids):selected.extend(ids[np.linspace(0,len(ids)-1,min(2,len(ids)),dtype=int)].tolist())
        exact=[]
        for node in sorted(set(selected)):
            distances=dijkstra(graph,indices=node)[seed[:,0]]
            for mode,col in [('label',None),('frame',2),('cell',0)]:
                if col is None:groups=seed[:,1]
                else:
                    pairs,groups=np.unique(seed[:,[1,col]],axis=0,return_inverse=True)
                    np.testing.assert_array_equal(pairs,d[mode+'_groups'])
                costs=np.full(int(groups.max())+1,np.inf);np.minimum.at(costs,groups,distances)
                nearest=d[mode+'_nearest'][node];measured=d[mode+'_distance'][node]
                np.testing.assert_allclose(measured,np.sort(costs)[:2],rtol=1e-12,atol=1e-10)
                valid=nearest>=0;np.testing.assert_allclose(measured[valid],costs[nearest[valid]],rtol=1e-12,atol=1e-10)
            exact.append(int(node))
        # Actual removal of four fixed source frames; keep geometry unchanged.
        frames=np.unique(seed[:,2]);chosen=frames[np.linspace(0,len(frames)-1,min(4,len(frames)),dtype=int)];deletions=[]
        accepted=d['frame_owner']>0
        for frame in chosen:
            remaining=seed[seed[:,2]!=frame];nodes=np.unique(remaining[:,0]);lookup=np.full(n,-1,np.int64)
            minimum=np.full(n,np.iinfo(np.int64).max,np.int64);maximum=np.full(n,-1,np.int64)
            np.minimum.at(minimum,remaining[:,0],remaining[:,1]);np.maximum.at(maximum,remaining[:,0],remaining[:,1])
            unambiguous=(minimum==maximum);lookup[unambiguous]=minimum[unambiguous]
            dist,predecessor,sources=dijkstra(graph,indices=nodes,min_only=True,return_predecessors=True)
            assert np.isfinite(dist[accepted]).all();after=lookup[sources[accepted]]
            np.testing.assert_array_equal(after,d['frame_owner'][accepted])
            deletions.append(dict(source_frame=int(frame),certified_foreground_cells_checked=int(accepted.sum()),changed=0))
        results[key]=dict(exports=exports,all_first_distances_scipy_replayed=n,exhaustive_group_distance_nodes=exact,
            actual_frame_deletions=deletions,seconds=time.monotonic()-start)
        print('TRANSPORT AUDIT',key,results[key],flush=True)
    dump(OUT/'artifact_audit.json',dict(complete=True,results=results,predictions_sha256=sha(OUT/'predictions_complete.json'),
        source_sha256=sha(__file__),semantics_or_ap_guarantee=False,scope='All exports and nearest distances;8or fewer fixed nodes exhaustive pergroup;4actual acquisitionframe deletions perwindow on allcertifiedforegroundcells. Tests additionally exhaust every batch on20randomgraphs.'))


if __name__=='__main__':run()
