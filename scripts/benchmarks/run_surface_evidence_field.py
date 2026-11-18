"""Finite six-arm measured-surface field experiment; references only in score."""
import argparse,copy,shutil,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from paper_final_common import ROOT,read,dump,sha
from prepare_partition_revision import PARENT,INPUT
from omni_semantic_map.surface_evidence_field import aggregate_views,solve_field,surface_graph
from omni_semantic_map.query_hypothesis_evidence import membership
from omni_semantic_map.kitti360_evaluation import evaluate
from evaluate_kitti360_development import transfer,clean_json
from run_discovery_query_benchmark import matched_ids
from run_hypothesis_admission import EVALUATOR

OUT=ROOT/'build/benchmarks/surface_evidence_field_v1'
CAPTURE=ROOT/'build/benchmarks/prethreshold_candidates_v1'
METHODS=['baseline','high_surface','weak_independent','weak_euclidean','weak_native_score','ours']


def freeze():
    assert not OUT.exists();parent=read(PARENT/'protocol.json');capture=read(CAPTURE/'predictions_complete.json');inputs={}
    for f,h in read(PARENT/'predictions_complete.json')['files'].items():assert sha(f)==h,f
    for f,h in capture['files'].items():assert sha(f)==h,f;inputs[f]=h
    for f in [CAPTURE/'predictions_complete.json',PARENT/'protocol.json',PARENT/'predictions_complete.json',PARENT/'summary.json',EVALUATOR]:inputs[str(f)]=sha(f)
    for scene in parent['scenes']:
        key=scene['key'];manifest=read(INPUT/key/'manifest.json')
        for f in [INPUT/key/'manifest.json',manifest['geometry'],INPUT/'query_cells'/key/'source_to_cell.npy',scene['reference']]:inputs[str(f)]=sha(f)
        for r in manifest['records']:inputs[r['file']]=sha(r['file'])
        for name in ['geometry.npz','classification.json']:f=PARENT/'maps'/key/'geometry_graph'/name;inputs[str(f)]=sha(f)
    sources=[Path(__file__),ROOT/'omni_semantic_map/omni_semantic_map/surface_evidence_field.py',ROOT/'omni_semantic_map/test/test_surface_evidence_field.py',ROOT/'omni_semantic_map/test/test_surface_field_export.py',ROOT/'omni_semantic_map/omni_semantic_map/query_hypothesis_evidence.py',ROOT/'omni_semantic_map/omni_semantic_map/kitti360_evaluation.py',ROOT/'scripts/benchmarks/evaluate_kitti360_development.py']
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),scenes=parent['scenes'],methods=METHODS,input_hashes=inputs,source_hashes={str(f):sha(f) for f in sources},
        scope='Existing108 frames only, exposed0000/0002. No new SAM/training/29verification inputs. Whole-window offline. No new object birth; revise geometry of existing object IDs only. Robot and independent tests require development gate.',
        control_amendment='Before freezing/inference: replace duplicate-count ablation with native-class-score weighting. One selected query per object/frame makes duplicate counting unidentifiable. Core duplicate-invariance remains tested, not an experimentally substantiated novelty claim.',
        cells='Existing20cm physical grid, original source IDs unchanged. Baseline points with absent keys append cells. Node position first actual source measurement, or first actual old map point for old-only cells. Never output cell centers. Cell visibility is not every point passing5cm depth gating.',
        retrieval='For every baseline object with>=16unique measured-visible core cells, choose same-class native positive query maximizing measured-cell IoU against its visible core; native score then query index break ties. High control restricts queries to native score>0.5. Zero intersection yields no support. Query support is fraction of visible core cells covered by selected query; native-score arm multiplies this by raw native score. No reference labels.',
        observations='Each measured-visible cell/frame is one categorical observation. Selected query supports compete across objects; null=1-max support before normalization. Missing cell/frame has reliability0, visible but unsupported yields null. Old map contributes one prior categorical observation per occupied cell, uniformly across distinct old object labels in mixed cells, not per point count. Fidelity weight is total real-view count plus that one prior. The model may erase old support if repeatedly observed without object support; report harm rather than silently preserving it.',
        graph='Symmetric8nearest physical neighbors within0.4m, existing component radius. Normals via up to16neighbors within0.4m; reliability=(lambda1-lambda0)/lambda2, zero if<3neighbors. Euclidean exp(-distance²/(2*.2²))/sqrt(degrees). Surface multiplier=(1-ri*rj)+ri*rj*dot(normal_i,normal_j)²*exp(-sum endpoint-plane-distance²/(2*.2²)). Same edges for both controls. Lambda1; no tuning.',
        solution='Convex categorical Laplacian field, CG Jacobi-preconditioned, relative tolerance1e-8,max2000iter, residual and simplex checked. Unknown disconnected components null. Weak-independent exact lambda0. Reference-free pre-score comparison of weak/high retrieval and field/independent decision differences.',
        export='Argmax existing object only if probability>0.5+1e-9; otherwise label0. Every old XYZ remains byte-for-byte as array prefix; label0 old points remain spatial null anchors. Append uniquely indexed original measurements only if nearest old geometry distance>0.05m and assigned>0. No geometry motion, duplicate source append, new class/ID, ranking change, NMS or cleanup. All conditions identical export. Baseline is exact frozen geometry_graph copy.',
        budget='Two windows x six complete maps, including two baseline copies; no model calls or fits. Source/inputs frozen before extraction, all12predictions sealed before scoring. No threshold/graph/normal/export sweep.',
        gate='Each window Ours AP50:90>=strong baseline. Mean Ours strictly exceeds every other control. Across windows at least one gained AP50 GT and zero lost; full joint mean strictly>weak_independent. Identical independent/joint outputs or no new weak support fails mechanism identifiability. Failure closes version, no expansion or parameter rescue.',
        novelty='Graph regularization and geometry retrieval are existing tools; implementation alone is not novelty. The specific conditional evidence/joint geometry hypothesis needs attributable complete-map gains.'))
    print('FROZEN SURFACE FIELD',flush=True)


def verify():
    p=read(OUT/'protocol.json')
    for f,h in {**p['input_hashes'],**p['source_hashes']}.items():assert sha(f)==h,f
    return p


def sealed(name):
    d=read(OUT/(name+'.json'));assert d['complete']
    for f,h in d['files'].items():assert sha(f)==h,f
    return d


def scene_data(scene):
    key=scene['key'];manifest=read(INPUT/key/'manifest.json');source=np.load(manifest['geometry'])['xyz'];base=dict(np.load(PARENT/'maps'/key/'geometry_graph/geometry.npz'))
    records=read(PARENT/'maps'/key/'geometry_graph/classification.json');return manifest,source,base,records


def prepare():
    p=verify();assert not (OUT/'features_complete.json').exists();capture=read(CAPTURE/'predictions_complete.json');files={};stats={}
    for scene in p['scenes']:
        start=time.monotonic();key=scene['key'];manifest,source,base,records=scene_data(scene);sourcecell=np.load(INPUT/'query_cells'/key/'source_to_cell.npy')
        cells,first=np.unique(sourcecell,return_index=True);np.testing.assert_array_equal(cells,np.arange(len(cells)))
        keys=np.floor(source[first]/.2).astype(np.int64);lookup={tuple(k):i for i,k in enumerate(keys)};positions=list(source[first]);oldcell=[]
        for xyz,k in zip(base['xyz'],np.floor(base['xyz']/.2).astype(np.int64)):
            token=tuple(k)
            if token not in lookup:lookup[token]=len(lookup);positions.append(xyz)
            oldcell.append(lookup[token])
        oldcell=np.asarray(oldcell,np.int64);positions=np.asarray(positions);n=len(positions);objects=sorted(records['objects'],key=lambda o:o['object_id']);oids=np.array([o['object_id'] for o in objects]);semantics=np.array([o['predictions']['mean']['semantic_id'] for o in objects]);K=len(oids)
        core=[np.unique(oldcell[base['instance']==oid]) for oid in oids];coremat=membership(core,n);prior=coremat.T.toarray();hasprior=prior.sum(1)>0;prior/=np.maximum(prior.sum(1,keepdims=True),1)
        prior=np.column_stack([np.zeros(n),prior]);sums={name:prior.copy() for name in ['high','weak','native']};counts=hasprior.astype(float)
        views={r['frame_id']:r for r in manifest['records']};events=[];frame_count=np.zeros(n,np.int64)
        for rec in capture['results'][key]:
            frame=rec['frame_id'];v=dict(np.load(views[frame]['file']));d=dict(np.load(rec['file']));ids=v['visible_source_ids'];xy=v['pixel_xy'];visible=np.unique(sourcecell[ids]);local=np.searchsorted(visible,sourcecell[ids])
            inside=((d['masks_packed'][:,xy[:,1],xy[:,0]//8]>>(7-xy[:,0]%8)[None,:])&1).astype(bool)
            qsets=[np.unique(local[row]) for row in inside];qmat=membership(qsets,len(visible));inter=(qmat@coremat[:,visible].T).toarray();qsize=np.diff(qmat.indptr);ownsize=np.asarray(coremat[:,visible].sum(1)).ravel();union=qsize[:,None]+ownsize[None,:]-inter
            iou=np.divide(inter,union,out=np.zeros_like(inter),where=union>0);supports={name:np.zeros((len(visible),K)) for name in ['high','weak','native']};selected=[]
            for j in range(K):
                if ownsize[j]<16:continue
                for mode,threshold in [('high',.5),('weak',0.)]:
                    candidates=np.flatnonzero((d['semantic_ids']==semantics[j])&(d['scores']>threshold)&(inter[:,j]>0))
                    if not len(candidates):continue
                    q=int(candidates[np.lexsort((candidates,-d['scores'][candidates],-iou[candidates,j]))[0]])
                    covered=float(inter[q,j]/ownsize[j]);supports[mode][qsets[q],j]=covered
                    if mode=='weak':supports['native'][qsets[q],j]=covered*float(d['scores'][q])
                    selected.append(dict(object_id=int(oids[j]),mode=mode,query=q,native_score=float(d['scores'][q]),core_coverage=covered,cell_iou=float(iou[q,j]),support_cells=len(qsets[q])))
            for mode,s in supports.items():
                pi,oi=np.nonzero(s);u,c=aggregate_views(pi.astype(np.int64),np.full(len(pi),frame,np.int64),oi.astype(np.int64)+1,s[pi,oi],np.column_stack([np.arange(len(visible)),np.full(len(visible),frame)]),len(visible),K)
                assert np.all(c==1);sums[mode][visible]+=u
            counts[visible]+=1;frame_count[visible]+=1;events.append(dict(frame_id=frame,visible_cells=len(visible),selections=selected))
        folder=OUT/'features'/key;folder.mkdir(parents=True,exist_ok=False)
        for mode,value in sums.items():
            u=np.divide(value,counts[:,None],out=np.zeros_like(value),where=counts[:,None]>0);u[counts==0,0]=1;np.save(folder/(mode+'_unary.npy'),u)
        edges,euclidean,surface,normal=surface_graph(positions)
        distance=cKDTree(base['xyz']).query(source,workers=1)[0];eligible=distance>.05
        np.savez_compressed(folder/'graph.npz',edges=edges,euclidean=euclidean,surface=surface,positions=positions,**normal)
        np.savez_compressed(folder/'index.npz',source_cell=sourcecell,old_cell=oldcell,object_ids=oids,reliability=counts,eligible_source=eligible,frame_count=frame_count)
        dump(folder/'retrieval.json',events)
        weak=[r for f in events for r in f['selections'] if r['mode']=='weak'];stat=dict(nodes=n,source_nodes=len(cells),old_only_nodes=n-len(cells),mixed_old_cells=int((coremat.T.getnnz(axis=1)>1).sum()),edges=len(edges),surface_diff_edges=int((np.abs(surface-euclidean)>1e-12).sum()),eligible_sources=int(eligible.sum()),weak_retrievals=len(weak),weak_native_below_half=sum(r['native_score']<=.5 for r in weak),never_retrieved_object_ids=sorted(set(map(int,oids))-{r['object_id'] for r in weak}),seconds=time.monotonic()-start)
        dump(folder/'run.json',stat)
        for f in folder.iterdir():files[str(f)]=sha(f)
        stats[key]=stat;print('FEATURES',key,stat,flush=True)
    dump(OUT/'features_complete.json',dict(complete=True,files=files,stats=stats,protocol_sha256=sha(OUT/'protocol.json')))


def field_labels(probability,object_ids):
    p=np.asarray(probability);ids=np.asarray(object_ids)
    if p.ndim!=2 or p.shape[1]!=len(ids)+1 or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(1),1,atol=1e-6):raise ValueError('Categorical field required')
    best=p.argmax(1);labels=np.r_[0,ids][best];labels[p[np.arange(len(p)),best]<=.5+1e-9]=0
    return labels


def export_field(base,source,index,probability):
    labels=field_labels(probability,index['object_ids']);new_labels=labels[index['source_cell']];selected=np.flatnonzero(index['eligible_source']&(new_labels>0))
    old=labels[index['old_cell']];xyz=np.concatenate([base['xyz'],source[selected]]);instances=np.r_[old,new_labels[selected]]
    np.testing.assert_array_equal(xyz[:len(base['xyz'])],base['xyz'])
    assert len(np.unique(selected))==len(selected)
    return dict(xyz=xyz,instance=instances,added_source_ids=selected),dict(old_point_ids_changed=int((old!=base['instance']).sum()),old_points_to_null=int((old==0).sum()),old_points_to_other=int(((old>0)&(old!=base['instance'])).sum()),added_points=len(selected),nonempty_objects=len(np.unique(instances[instances>0])))


def predict():
    p=verify();features=sealed('features_complete');assert not (OUT/'predictions_complete.json').exists();files={};stats={};checks={}
    for scene in p['scenes']:
        key=scene['key'];manifest,source,base,records=scene_data(scene);folder=OUT/'features'/key;graph=dict(np.load(folder/'graph.npz'));index=dict(np.load(folder/'index.npz'));independent=None;ours=None
        for method in METHODS:
            start=time.monotonic();dest=OUT/'maps'/key/method;dest.mkdir(parents=True,exist_ok=False)
            if method=='baseline':
                for name in ['geometry.npz','classification.json']:shutil.copyfile(PARENT/'maps'/key/'geometry_graph'/name,dest/name)
                stat=dict(exact_copy=True,old_point_ids_changed=0,old_points_to_null=0,old_points_to_other=0,added_points=0,nonempty_objects=len(records['objects']))
            else:
                mode='high' if method=='high_surface' else 'native' if method=='weak_native_score' else 'weak';u=np.load(folder/(mode+'_unary.npy'))
                strength=0. if method=='weak_independent' else 1.;weights=graph['euclidean'] if method=='weak_euclidean' else graph['surface']
                prob,solver=solve_field(u,index['reliability'],graph['edges'],weights,strength=strength)
                geometry,stat=export_field(base,source,index,prob);stat['solver']=solver
                np.save(dest/'cell_labels.npy',field_labels(prob,index['object_ids']));np.savez_compressed(dest/'geometry.npz',**geometry);dump(dest/'classification.json',records)
                if method=='weak_independent':independent=field_labels(prob,index['object_ids'])
                if method=='ours':ours=field_labels(prob,index['object_ids'])
            stat['seconds']=time.monotonic()-start;dump(dest/'run.json',stat);stats[key+'/'+method]=stat
            for f in dest.iterdir():files[str(f)]=sha(f)
            print('PREDICT',key,method,stat,flush=True)
        checks[key]=dict(weak_low_score_retrievals=features['stats'][key]['weak_native_below_half'],joint_vs_independent_cells_changed=int((ours!=independent).sum()),surface_vs_euclidean_cells_changed=int((ours!=np.load(OUT/'maps'/key/'weak_euclidean/cell_labels.npy')).sum()))
    identifiable=all(c['weak_low_score_retrievals']>0 and c['joint_vs_independent_cells_changed']>0 and c['surface_vs_euclidean_cells_changed']>0 for c in checks.values())
    dump(OUT/'predictions_complete.json',dict(complete=True,files=files,stats=stats,identifiability=checks,mechanism_identifiable=identifiable,protocol_sha256=sha(OUT/'protocol.json'),feature_seal_sha256=sha(OUT/'features_complete.json'),sealed_before_scoring=True))
    print('PREDICTIONS SEALED',checks,'IDENTIFIABLE',identifiable,flush=True)


def score():
    p=verify();seal=sealed('predictions_complete');assert not (OUT/'summary.json').exists()
    if not seal['mechanism_identifiable']:
        dump(OUT/'summary.json',dict(complete=True,advance_gate=False,scored=False,reason='Pre-score mechanism identifiability failed',identifiability=seal['identifiability']));return
    results={};means={m:[] for m in METHODS};gained=lost=0;gate=True
    for scene in p['scenes']:
        key=scene['key'];ref=dict(np.load(scene['reference']));rows={}
        for method in METHODS:
            folder=OUT/'maps'/key/method;g=np.load(folder/'geometry.npz');objects=read(folder/'classification.json')['objects'];projection=transfer(g['xyz'],g['instance'],ref['xyz'],.2)
            result=evaluate(ref['train_instance'],ref['confidence'],projection,{o['object_id']:o['predictions']['mean'] for o in objects},EVALUATOR)
            dump(folder/'matches.json',clean_json(result['matches']));rows[method]=dict(metrics=clean_json(result['averages']),matched_gt_ids=sorted(matched_ids(result['matches'])),run=read(folder/'run.json'));means[method].append(float(result['averages']['allAp']));print('SCORE',key,method,rows[method]['metrics'],flush=True)
        expected=read(PARENT/'summary.json')['results'][key]['methods']['geometry_graph']['metrics']
        for metric in ['allAp','allAp50%']:assert abs(rows['baseline']['metrics'][metric]-expected[metric])<1e-12
        base=set(rows['baseline']['matched_gt_ids'])
        for r in rows.values():actual=set(r['matched_gt_ids']);r['changes_vs_baseline']=dict(gained=sorted(actual-base),lost=sorted(base-actual))
        change=rows['ours']['changes_vs_baseline'];gained+=len(change['gained']);lost+=len(change['lost']);gate&=rows['ours']['metrics']['allAp']>=rows['baseline']['metrics']['allAp'];results[key]=dict(methods=rows)
    mean={m:float(np.mean(v)) for m,v in means.items()};gate&=mean['ours']>max(v for m,v in mean.items() if m!='ours') and gained>0 and lost==0
    dump(OUT/'summary.json',dict(complete=True,scored=True,results=results,mean_primary_ap=mean,advance_gate=bool(gate),gained=gained,lost=lost,development_only=True,official_leaderboard=False,protocol_sha256=sha(OUT/'protocol.json'),prediction_seal_sha256=sha(OUT/'predictions_complete.json')));print('GATE',bool(gate),mean,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','prepare','predict','score']);globals()[parser.parse_args().action]()
