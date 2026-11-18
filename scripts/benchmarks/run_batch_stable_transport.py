"""One frozen full-map test of source-frame deletion-stable surface transport."""
import argparse
import shutil
import time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from paper_final_common import ROOT,read,dump,sha
from omni_semantic_map.batch_stable_transport import native_library,nearest_groups,nearest_owner,certify

OUT=ROOT/'build/benchmarks/batch_stable_transport_v1'
FULL=ROOT/'build/benchmarks/full_lidar_surface_inputs_v1'
OLD=ROOT/'build/benchmarks/discovery_measured_inputs_v1'
SEM=ROOT/'build/benchmarks/closed_vocabulary_semantics_v1'
BASE=ROOT/'build/benchmarks/signed_query_association_v1/maps'
METHODS=['baseline','full_nearest','full_two_cells','camera_frame','ours_frame']


def freeze():
    assert not OUT.exists()
    scenes=read(SEM/'protocol.json')['scenes'];inputs={}
    for s in scenes:
        key=s['key'];old=read(OLD/key/'manifest.json')
        paths=[FULL/'prepared.json',FULL/key/'geometry.npz',FULL/key/'old_to_full.npy',OLD/key/'manifest.json',old['geometry'],
            SEM/'aggregation_complete.json',SEM/'aggregate'/key/'source.npz',SEM/'protocol.json',
            BASE/key/'geometry_graph/geometry.npz',BASE/key/'geometry_graph/classification.json']
        for path in paths:inputs[str(path)]=sha(path)
    sources=[Path(__file__),ROOT/'omni_semantic_map/omni_semantic_map/batch_stable_transport.py',
        ROOT/'omni_semantic_map/omni_semantic_map/batch_stable_transport.cpp',ROOT/'omni_semantic_map/test/test_batch_stable_transport.py',
        ROOT/'scripts/benchmarks/evaluate_kitti360_development.py',ROOT/'omni_semantic_map/omni_semantic_map/kitti360_evaluation.py',
        ROOT/'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py']
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),scenes=scenes,methods=METHODS,input_hashes=inputs,source_hashes={str(f):sha(f) for f in sources},
        hypothesis='Can competing-source transport complete measured object surfaces while excluding assignments whose winning identity depends on a single acquisition-frame anchor batch? Full fixed geometry retained when removing anchors. A certificate is not semantic correctness or independent-view evidence.',
        geometry='Same60/48scans and old108RGBsemantic outputs; full5cm-source set plus exact old sources. Nodes are20cmworldcells represented by their first actual source point. Symmetric union of16NN within strict0.4m, edge cost Euclidean length. No normals, gap predicate, pseudo-depth, radius sweep or point synthesis.',
        seeds='Original predicted source owners from frozen strong baseline are foreground labels. Unowned original sources with >=1semantic observation and mean19classwinner outside car/building seed one common background label0. Others remain unlabeled. Foreground anchors override semantic disagreement; conflicting ownership(-1) provides no seed and is never added. No GT-selected seeds.',
        mechanism='On the shared graph find two closest distinct owner groups, and two closest distinct(owner,acquisition_frame) groups. Accept only if the two token owners equal the unique winning owner and second token distance is finite and strictly below nearest competing owner distance by1e-10. This bounds distance after any single frame-anchor deletion. Competitors retain all seeds, conservative. No claim that correlation/errors shared across frames are removed.',
        controls='full_nearest: unique nearest owner; full_two_cells: identical deletion certificate but batch is spatial seed cell; camera_frame: same frame certificate on old camera-source geometry only; ours_frame: fullgeometry frame certificate. Every full arm shares exact geometry/seeds/graph/classscores/export. Camera ablation intentionally varies only source domain and its deterministic graph.',
        output='Append all previously unowned actual sources in accepted foreground cells, original source order. Old fused map XYZ/IDs/classscores immutable. No new objects, historical ownership revisions, SOR/NMS or extra finalprocessing. This candidate tests geometry completion, not full object discovery. Unknown cells/background never exported. Graph transport has no path-length cutoff; record distance distribution.',
        training='No new training/model forward. Parent includes cross-window supervised intrinsic predictor and Cityscapes-supervised SegFormer. Not training-free full system, official leaderboard, or independent development.',
        budget='Two exposed windows x5complete maps, one fixed extraction and scoring. Six core tests passed before freeze. No thresholds/epochs/seeds resampling or rescue sweeps.',
        gate='OursAP50:90>=baseline eachwindow; mean>=baseline+.01 and strictly exceeds full_nearest,full_two_cells,camera_frame;>=1gained and0lostAP50matches. Ours must actually differ from both fullcontrols. Otherwise close without robot/independent expansion.',
        pending='Only a passed fixed candidate advances to same-module robot and reserved0006/0007. New labels/images remain unopened during development.'))
    print('BATCH TRANSPORT FROZEN',flush=True)


def verify():
    p=read(OUT/'protocol.json')
    for f,h in {**p['input_hashes'],**p['source_hashes']}.items():assert sha(f)==h,f
    return p


def seal(name):
    p=read(OUT/(name+'.json'));assert p['complete']
    for f,h in p['files'].items():assert sha(f)==h,f
    return p


def graph_and_seeds(xyz,old_to_domain,sem,source_frames):
    keys=np.floor(xyz/.2).astype(np.int64)
    cells,first,inverse=np.unique(keys,axis=0,return_index=True,return_inverse=True)
    centers=xyz[first];n=len(cells)
    distances,neighbors=cKDTree(centers).query(centers,k=min(17,n),distance_upper_bound=.4,workers=1)
    a=np.repeat(np.arange(n),neighbors.shape[1]);b=neighbors.ravel();d=distances.ravel()
    good=(a!=b)&(b<n)&np.isfinite(d)&(d<.4);a,b=a[good],b[good]
    pair=np.unique(np.minimum(a,b)*n+np.maximum(a,b));a,b=pair//n,pair%n
    weight=np.linalg.norm(centers[a]-centers[b],axis=1);assert (weight>0).all()
    graph=csr_matrix((np.r_[weight,weight],(np.r_[a,b],np.r_[b,a])),shape=(n,n))
    owners=sem['owners'];bg=(owners==0)&(sem['seen']>0)&~np.isin(sem['mean_response'],[2,13])
    active=(owners>0)|bg;source_ids=np.flatnonzero(active)
    seed_node=inverse[old_to_domain[source_ids]];seed_owner=owners[source_ids];seed_frame=source_frames[old_to_domain[source_ids]]
    seeds=np.unique(np.column_stack([seed_node,seed_owner,seed_frame]),axis=0)
    return graph,dict(source_to_cell=inverse,representative_source=first,centers=centers,cells=cells,seeds=seeds),dict(
        source_points=len(xyz),nodes=n,undirected_edges=len(pair),seed_records=len(seeds),foreground_source_seeds=int((owners>0).sum()),
        background_source_seeds=int(bg.sum()),seed_objects=len(np.unique(owners[owners>0])),unknown_cells=int(n-len(np.unique(seed_node))))


def prepare():
    p=verify();assert not (OUT/'execution.json').exists()
    dump(OUT/'execution.json',dict(started_epoch=time.time(),no_gt_read=True));lib=native_library(OUT/'native');files={};stats={}
    for scene in p['scenes']:
        key=scene['key'];sem=dict(np.load(SEM/'aggregate'/key/'source.npz'));old=dict(np.load(read(OLD/key/'manifest.json')['geometry']))
        full=dict(np.load(FULL/key/'geometry.npz'));mapping=np.load(FULL/key/'old_to_full.npy');np.testing.assert_array_equal(full['xyz'][mapping],old['xyz'])
        stats[key]={}
        for domain,data,mapped in [('full',full,mapping),('camera',old,np.arange(len(old['xyz'])))]:
            started=time.monotonic();graph,d,stat=graph_and_seeds(data['xyz'],mapped,sem,data['source_frame']);seeds=d['seeds']
            print('TRANSPORT GRAPH',key,domain,stat,flush=True)
            labels,dist=nearest_groups(graph,seeds[:,0],seeds[:,1],lib);d.update(label_nearest=labels,label_distance=dist,nearest_owner=nearest_owner(labels,dist))
            for grouping in (['cell','frame'] if domain=='full' else ['frame']):
                pairs,tokens=np.unique(seeds[:,[1,0 if grouping=='cell' else 2]],axis=0,return_inverse=True)
                tg,td=nearest_groups(graph,seeds[:,0],tokens,lib)
                pred=certify(labels,dist,tg,td,pairs[:,0]);d.update({grouping+'_groups':pairs,grouping+'_nearest':tg,grouping+'_distance':td,grouping+'_owner':pred})
                stat[grouping+'_accepted_foreground_cells']=int((pred>0).sum());print('TRANSPORT FIELD',key,domain,grouping,stat[grouping+'_accepted_foreground_cells'],flush=True)
            stat['seconds']=time.monotonic()-started;folder=OUT/'fields'/key/domain;folder.mkdir(parents=True,exist_ok=False)
            path=folder/'field.npz';np.savez_compressed(path,**d,graph_indptr=graph.indptr,graph_indices=graph.indices,graph_weights=graph.data)
            files[str(path)]=sha(path);stats[key][domain]=stat
            del graph,d
        del full,old,sem
    for path in (OUT/'native').iterdir():files[str(path)]=sha(path)
    dump(OUT/'prepared.json',dict(complete=True,files=files,stats=stats,no_gt_read=True,protocol_sha256=sha(OUT/'protocol.json')))


def maps():
    p=verify();seal('prepared');assert not (OUT/'predictions_complete.json').exists();files={};stats={}
    for scene in p['scenes']:
        key=scene['key'];base=np.load(BASE/key/'geometry_graph/geometry.npz');full=np.load(FULL/key/'geometry.npz');oldmapping=np.load(FULL/key/'old_to_full.npy')
        sem=np.load(SEM/'aggregate'/key/'source.npz');owner=np.zeros(len(full['xyz']),np.int64);owner[oldmapping]=sem['owners']
        old=np.load(read(OLD/key/'manifest.json')['geometry']);stats[key]={}
        fields={domain:np.load(OUT/'fields'/key/domain/'field.npz') for domain in ['full','camera']}
        for method in METHODS:
            start=time.monotonic();folder=OUT/'maps'/key/method;folder.mkdir(parents=True,exist_ok=False)
            shutil.copyfile(BASE/key/'geometry_graph/classification.json',folder/'classification.json')
            if method=='baseline':
                shutil.copyfile(BASE/key/'geometry_graph/geometry.npz',folder/'geometry.npz');added=np.empty(0,np.int64);assigned=np.empty(0,np.int64);distance=np.empty(0)
            else:
                domain='camera' if method=='camera_frame' else 'full';field=fields[domain]
                mode='nearest' if method=='full_nearest' else 'cell' if method=='full_two_cells' else 'frame'
                nodes=field['source_to_cell'];pred=field[mode+'_owner'][nodes]
                domain_to_full=oldmapping if domain=='camera' else np.arange(len(owner))
                keep=(pred>0)&(owner[domain_to_full]==0);added=domain_to_full[keep];assigned=pred[keep]
                order=np.argsort(added);added,assigned=added[order],assigned[order]
                np.testing.assert_array_equal(np.unique(added),added)
                xyz=np.concatenate([base['xyz'],full['xyz'][added]]);instance=np.r_[base['instance'],assigned]
                np.savez_compressed(folder/'geometry.npz',xyz=xyz,instance=instance,added_full_source_ids=added)
                distance=field['label_distance'][nodes[keep],0]
            stat=dict(added_sources=len(added),objects=len(np.unique(base['instance'])),new_objects=0,old_map_exact=True,
                added_original_camera_sources=int((full['old_source_id'][added]>=0).sum()),added_new_lidar_sources=int((full['old_source_id'][added]<0).sum()),
                propagation_distance_quantiles=np.quantile(distance,[0,.5,.9,.99,1]).tolist() if len(distance) else [],
                additions_by_object={str(int(o)):int((assigned==o).sum()) for o in np.unique(assigned)},seconds=time.monotonic()-start)
            dump(folder/'run.json',stat);stats[key][method]=stat
            for path in folder.iterdir():files[str(path)]=sha(path)
            print('TRANSPORT MAP',key,method,'added',len(added),flush=True)
    dump(OUT/'predictions_complete.json',dict(complete=True,files=files,stats=stats,no_gt_read=True,protocol_sha256=sha(OUT/'protocol.json')))


def score():
    from evaluate_kitti360_development import transfer,clean_json
    from omni_semantic_map.kitti360_evaluation import evaluate
    from run_discovery_query_benchmark import matched_ids
    p=verify();seal('predictions_complete');assert not (OUT/'summary.json').exists();results={};means={m:[] for m in METHODS};gained=lost=0;gate=True
    evaluator=ROOT/'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py'
    for scene in p['scenes']:
        key=scene['key'];ref=np.load(scene['reference']);rows={}
        for method in METHODS:
            folder=OUT/'maps'/key/method;g=np.load(folder/'geometry.npz');objects=read(folder/'classification.json')['objects']
            pred=transfer(g['xyz'],g['instance'],ref['xyz'],.2)
            scored=evaluate(ref['train_instance'],ref['confidence'],pred,{o['object_id']:o['predictions']['mean'] for o in objects},evaluator)
            dump(folder/'matches.json',clean_json(scored['matches']));np.savez_compressed(folder/'reference_prediction.npz',instance=pred)
            rows[method]=dict(metrics=clean_json(scored['averages']),matched_gt_ids=sorted(matched_ids(scored['matches'])),run=read(folder/'run.json'))
            means[method].append(float(scored['averages']['allAp']));print('TRANSPORT SCORE',key,method,rows[method]['metrics'],flush=True)
        expected=read(ROOT/'build/benchmarks/signed_query_association_v1/summary.json')['results'][key]['methods']['geometry_graph']['metrics']
        assert abs(rows['baseline']['metrics']['allAp']-expected['allAp'])<1e-12
        old=set(rows['baseline']['matched_gt_ids'])
        for row in rows.values():
            ids=set(row['matched_gt_ids']);row['changes_vs_baseline']=dict(gained=sorted(ids-old),lost=sorted(old-ids))
        change=rows['ours_frame']['changes_vs_baseline'];gained+=len(change['gained']);lost+=len(change['lost'])
        gate &= rows['ours_frame']['metrics']['allAp']>=rows['baseline']['metrics']['allAp'];results[key]=dict(methods=rows)
    mean={m:float(np.mean(v)) for m,v in means.items()};gate &= mean['ours_frame']>=mean['baseline']+.01 and mean['ours_frame']>max(v for m,v in mean.items() if m!='ours_frame') and gained>=1 and lost==0
    different={}
    for key in results:
        a=np.load(OUT/'maps'/key/'ours_frame/geometry.npz')
        different[key]={}
        for control in ['full_nearest','full_two_cells']:
            b=np.load(OUT/'maps'/key/control/'geometry.npz')
            different[key][control]=not(np.array_equal(a['added_full_source_ids'],b['added_full_source_ids']) and np.array_equal(a['instance'],b['instance']))
    gate &= all(any(different[key][control] for key in results) for control in ['full_nearest','full_two_cells'])
    dump(OUT/'summary.json',dict(complete=True,results=results,mean_primary_ap=mean,gained=gained,lost=lost,
        advance_gate=bool(gate),actual_map_differences=different,protocol_sha256=sha(OUT/'protocol.json'),prediction_seal_sha256=sha(OUT/'predictions_complete.json'),official_leaderboard=False,independent_test=False))
    print('TRANSPORT ADVANCE',bool(gate),mean,'gained',gained,'lost',lost,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','maps','score']);globals()[ap.parse_args().action]()
