"""Common full-map exporter and prospective evaluation of object partitioning.

Maps are sealed before any scoring. Old IDs are aligned by predicted overlap,
never GT. The candidate can create identities and revise historical ownership.
"""
import argparse
import copy
from pathlib import Path
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from paper_final_common import ROOT,read,dump,sha
from run_censored_partition import OUT,verify_execution
from omni_semantic_map.batch_stable_transport import native_library,nearest_groups,certify

METHODS=['baseline','pairwise_history','region_current','region_history']


def freeze():
    p=verify_execution();assert not (OUT/'map_protocol.json').exists();inputs={}
    for c in p['contexts']:
        complete=OUT/'partitions'/c['key']/'complete.json';d=read(complete);assert d['complete']
        for f,h in d['files'].items():assert sha(f)==h,f
        inputs[str(complete)]=sha(complete)
        for f in [c['reference'],c['raw_reference']]:inputs[str(f)]=sha(f)
    paths=[Path(__file__),ROOT/'omni_semantic_map/omni_semantic_map/batch_stable_transport.py',
        ROOT/'omni_semantic_map/omni_semantic_map/batch_stable_transport.cpp',
        ROOT/'scripts/benchmarks/evaluate_kitti360_development.py',
        ROOT/'omni_semantic_map/omni_semantic_map/kitti360_evaluation.py',
        ROOT/'omni_semantic_map/omni_semantic_map/kitti360_reference_identity.py']
    dump(OUT/'map_protocol.json',dict(created_epoch=time.time(),inputs=inputs,
        sources={str(f):sha(f) for f in paths},methods=METHODS,
        partition_protocol_sha256=sha(OUT/'partition_protocol.json'),
        identity='Maximum-count one-to-one Hungarian alignment between predicted regions and old objectIDs on original-map points with EXACT20cmcell correspondence, same predicted class only, positive overlaps only. No GT matching. Unmatched regions get newIDs only when>=16full measured sources in direct cells and>=3observedframes. Otherwise abstain; aligned old identities need no new-birth test.',
        seeds='Aligned/born region labels at direct measured observation cells, one unique(cell,owner,observedframe) seed. Background0 seeds only in cells whose aggregate19classsemanticwinner is notcar/building AND where current-frame outside-car/building source fraction>0.5. No seed at absent views. Frame tags are image-observation batches, not independent views or source acquisition frames.',
        transport='Use unchanged20cmcells/full16NNstrict0.4mgraph. Exact existing nearest-group solver and two-distinct-frame certificate1e-10. Full geometry never deleted for certification. This shared extrapolation is a fixed graph stability test, not true semantic confidence or novel contribution by itself.',
        export='Keep all oldXYZ as prefix. For history variants, exact-grid old points take certified>=0label (including explicitbackground0); unknown retains oldID. Current-only preserves ALL oldIDs. Append only certifiedforeground full measured sourceXYZ at distance>5cm from ANY oldXYZ, same duplicate exclusion for all methods. No synthesized geometry, postfilter, radius/prompt/score search.',
        scores='Old identities keep original classes/scores. A new identity uses its regionclass and per-observed-frame mean nativeSAM3queryscore weighted by frozen cell/query observationmass, then unweighted mean across frames. Same source model as old frontend; these are ranking heuristics, not calibrated probabilities. Separate uniform-confidence scoring of every baseline/candidate isolates ranking sensitivity.',
        controls='Exact currentstrongbaseline; fixed-initial-edge additive partition+history; same full-region partition+current-only; Ours candidate full-region partition+history. Current/history share EXACT labels, births, transport, newpoints, classification and scores; only historical point labels differ.',
        evaluation='Seal6newmaps first. Two exposed outdoor windows x4methods: original upstream-compatible AP50:90/AP50; raw11/26reference diagnostic; uniformconfidence upstream-compatible diagnostic.24complete evaluations. All eligibleGT retained. No official-server or independent-test claim.',
        gate='Ours primaryAP/AP50 >= currentstrongbaseline eachwindow; meanprimaryAP >=baseline+0.01 and strictly exceeds pairwise_history and region_current; at leastone newRAW AP50match and zero lostRAW matches acrosswindows; nonzero historical edits. Uniform-score results are diagnostic, not used to swap the main ranking protocol. Failure closes this frozen configuration without extra tuning/maps.',
        scope='Full public mechanism comparison only. Same-version robot comparison and actually unused outdoor assessment remain required. These are not fulfilled by intermediate partitions, changed point counts or a passed capacity bound.'))
    print('FULL PARTITION MAP COMPARISON FROZEN',flush=True)


def verify_maps():
    p=verify_execution();m=read(OUT/'map_protocol.json')
    for f,h in {**m['inputs'],**m['sources']}.items():assert sha(f)==h,f
    assert sha(OUT/'partition_protocol.json')==m['partition_protocol_sha256']
    return p


def maps():
    p=verify_maps();assert not (OUT/'maps_complete.json').exists();files={};results={}
    raw=read(ROOT/'build/benchmarks/prethreshold_candidates_v1/predictions_complete.json')
    for c in p['contexts']:
        key=c['key'];d=dict(np.load(OUT/'prepared'/key/'atoms.npz'));field=np.load(c['field'])
        cells=field['source_to_cell'];nc=len(field['cells']);n=len(d['cell_ids'])
        graph=csr_matrix((field['graph_weights'],field['graph_indices'],field['graph_indptr']),shape=(nc,nc))
        full=np.load(Path(c['full'])/'geometry.npz')['xyz'];base=dict(np.load(Path(c['ours'])/'geometry.npz'))
        records=read(Path(c['ours'])/'classification.json');byoid={r['object_id']:r for r in records['objects']}
        oldids=np.array(sorted(byoid),np.int64);maxold=int(oldids.max());classold=np.array([byoid[o]['predictions']['mean']['train_id'] for o in oldids])
        oldclasses={int(o):int(t) for o,t in zip(oldids,classold)}
        dtype=np.dtype([('x',np.int64),('y',np.int64),('z',np.int64)])
        fkeys=np.ascontiguousarray(field['cells'],np.int64).view(dtype).ravel()
        bkeys=np.ascontiguousarray(np.floor(base['xyz']/.2).astype(np.int64)).view(dtype).ravel()
        pos=np.searchsorted(fkeys,bkeys);valid=pos<len(fkeys);valid[valid]&=fkeys[pos[valid]]==bkeys[valid]
        oldcell=np.full(len(bkeys),-1,np.int64);oldcell[valid]=pos[valid]
        duplicate_free=cKDTree(base['xyz']).query(full,workers=1)[0]>.05
        views=read(OUT/'prepared'/key/'views.json');vocab=read(OUT/'prepared'/key/'vocabularies.json')
        rawby={v['frame_id']:v for v in raw['results'][key]};token_score=[];bgrows=[]
        for v in vocab:
            scores=np.load(rawby[v['frame_id']]['file'])['scores'];lookup=np.zeros(v['categories'])
            lookup[1:len(v['query_rows'])+1]=scores[v['query_rows']];token_score.append(lookup)
        observed_scores=np.array([token_score[v][q] for v,q in zip(d['observation_view'],d['observation_token'])])
        bg=(d['seen']>0)&~np.isin(d['semantic_class_all_cells'],[2,13])
        for rank,rec in enumerate(views):
            v=np.load(rec['file']);cc=v['cells'];sem=v['semantic']
            count=np.bincount(cc,minlength=nc);outside=np.bincount(cc,weights=~np.isin(sem,[2,13]),minlength=nc)
            selected=np.flatnonzero(bg&(outside>.5*count))
            bgrows.append(np.column_stack((selected,np.zeros(len(selected),np.int64),np.full(len(selected),rank,np.int64))))
        background=np.concatenate(bgrows);results[key]={}
        lib=native_library(OUT/'map_native')
        for mode in ['pairwise','region']:
            tick=time.monotonic();part=dict(np.load(OUT/'partitions'/key/mode/'partition.npz'))
            labels=part['root'];roots=np.unique(labels);rootpos=np.full(n,-1,np.int64);rootpos[roots]=np.arange(len(roots))
            pointroot=part['old_point_proposed_region'];available=pointroot>=0
            contingency=np.zeros((len(roots),len(oldids)),np.int64)
            np.add.at(contingency,(rootpos[pointroot[available]],np.searchsorted(oldids,base['instance'][available])),1)
            contingency[d['train_class'][roots,None]!=classold[None,:]]=0
            ii,jj=linear_sum_assignment(-contingency)
            owner=np.zeros(n,np.int64);lineage=[]
            for i,j in zip(ii,jj):
                if contingency[i,j]>0:
                    owner[roots[i]]=oldids[j];lineage.append(dict(root=int(roots[i]),old_id=int(oldids[j]),overlap_points=int(contingency[i,j])))
            nextid=maxold+1;births=[];outrecords=copy.deepcopy(records)
            for root in roots:
                if owner[root] or not part['birth_eligible'][root]:continue
                owner[root]=nextid;nextid+=1;oid=int(owner[root]);cls=int(d['train_class'][root])
                selected=labels[d['observation_atom']]==root;vv=d['observation_view'][selected]//2;ww=d['observation_mass'][selected]
                total=np.bincount(vv,weights=ww,minlength=len(views));response=np.bincount(vv,weights=ww*observed_scores[selected],minlength=len(views))
                score=float(np.mean(response[total>0]/total[total>0]))
                outrecords['objects'].append(dict(object_id=oid,observation_ids=[],selected_observation_ids=[],source_root=int(root),
                    predictions=dict(mean=dict(name='building' if cls==2 else 'car',semantic_id=11 if cls==2 else 26,train_id=cls,score=score,retained=True))))
                births.append(dict(root=int(root),object_id=oid,train_class=cls,score=score,frames=int(part['observed_frames'][root]),direct_sources=int(part['source_count'][root])))
            atomowner=owner[labels];take=atomowner[d['observation_atom']]>0
            foreground=np.column_stack((d['cell_ids'][d['observation_atom'][take]],atomowner[d['observation_atom'][take]],d['observation_view'][take]//2))
            seeds=np.unique(np.vstack((background,foreground)),axis=0)
            own,dist=nearest_groups(graph,seeds[:,0],seeds[:,1],lib)
            pairs,tokens=np.unique(seeds[:,[1,2]],axis=0,return_inverse=True)
            if len(pairs):
                token,tdist=nearest_groups(graph,seeds[:,0],tokens,lib);decision=certify(own,dist,token,tdist,pairs[:,0])
            else:token=np.full((nc,2),-1,np.int64);tdist=np.full((nc,2),np.inf);decision=np.full(nc,-1,np.int64)
            appended=np.flatnonzero(duplicate_free&(decision[cells]>0))
            revised=base['instance'].copy();eligible=oldcell>=0;eligible[eligible]&=decision[oldcell[eligible]]>=0
            revised[eligible]=decision[oldcell[eligible]]
            names=['pairwise_history'] if mode=='pairwise' else ['region_current','region_history']
            work=OUT/'transport'/key/mode;work.mkdir(parents=True,exist_ok=False)
            np.savez_compressed(work/'field.npz',seeds=seeds,nearest_owner=own,owner_distance=dist,token_groups=pairs,
                nearest_token=token,token_distance=tdist,certified_owner=decision,root_owner=owner,
                old_cell=oldcell,duplicate_free=duplicate_free)
            dump(work/'lineage.json',dict(aligned=lineage,births=births))
            for name in names:
                dest=OUT/'maps'/key/name;dest.mkdir(parents=True,exist_ok=False)
                prefix=base['instance'] if name=='region_current' else revised
                np.savez_compressed(dest/'geometry.npz',xyz=np.concatenate((base['xyz'],full[appended])),
                    instance=np.r_[prefix,decision[cells[appended]]],added_full_source_ids=appended)
                dump(dest/'classification.json',outrecords)
                stats=dict(points=len(prefix)+len(appended),added_measured_sources=len(appended),new_identities=len(births),
                    old_points_changed=int((prefix!=base['instance']).sum()),old_points_to_background=int(((prefix==0)&(base['instance']>0)).sum()),
                    old_points_to_another_foreground=int(((prefix>0)&(prefix!=base['instance'])).sum()),
                    original_coordinate_prefix_exact=True,seconds_shared_partition_export=time.monotonic()-tick)
                dump(dest/'run.json',stats);results[key][name]=stats
                print('COMPLETE OBJECT MAP',key,name,stats,flush=True)
        a=OUT/'maps'/key/'region_current/classification.json';b=OUT/'maps'/key/'region_history/classification.json'
        assert sha(a)==sha(b)
    for sub in ['maps','transport','map_native']:
        for f in (OUT/sub).rglob('*'):
            if f.is_file():files[str(f)]=sha(f)
    dump(OUT/'maps_complete.json',dict(complete=True,files=files,results=results,gt_read=False,new_maps=6,
        map_protocol_sha256=sha(OUT/'map_protocol.json'),all_predictions_sealed_before_scoring=True))


def score():
    from evaluate_kitti360_development import transfer,read_static_ply,clean_json
    from omni_semantic_map.kitti360_evaluation import evaluate
    from omni_semantic_map.kitti360_reference_identity import encode_reference
    from run_discovery_query_benchmark import matched_ids
    import runpy
    p=verify_maps();seal=read(OUT/'maps_complete.json');assert seal['complete'] and not (OUT/'map_summary.json').exists()
    for f,h in seal['files'].items():assert sha(f)==h,f
    taxonomy=runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['id2label']
    evaluator=ROOT/'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py'
    prior=read(ROOT/'build/benchmarks/reference_identity_audit_v1/summary.json');results={};gate=True;gained=lost=edits=0
    for c in p['contexts']:
        key=c['key'];ref=dict(np.load(c['reference']));raw=read_static_ply(c['raw_reference'])
        physical,_=encode_reference(raw['semantic'],raw['instance'],taxonomy,'target_only')
        rows={}
        for method in METHODS:
            source=Path(c['ours']) if method=='baseline' else OUT/'maps'/key/method
            g=np.load(source/'geometry.npz');records=read(source/'classification.json')['objects']
            predictions={r['object_id']:r['predictions']['mean'] for r in records}
            projected=transfer(g['xyz'],g['instance'],ref['xyz'],.2)
            dest=OUT/'evaluation'/key/method;dest.mkdir(parents=True,exist_ok=False)
            np.savez_compressed(dest/'reference_prediction.npz',instance=projected)
            row={}
            for mode,truth,uniform in [('primary',ref['train_instance'],False),('raw_targets',physical,False),('uniform_scores',ref['train_instance'],True)]:
                info={o:dict(v,score=1.) for o,v in predictions.items()} if uniform else predictions
                result=evaluate(truth,ref['confidence'],projected,info,evaluator)
                dump(dest/(mode+'_matches.json'),clean_json(result['matches']))
                row[mode]=dict(metrics=clean_json(result['averages']),matched_gt_ids=sorted(matched_ids(result['matches'])))
            if method=='baseline':
                for mode,oldmode in [('primary','upstream_train'),('raw_targets','target_only')]:
                    expected=prior['results'][key]['modes'][oldmode]['methods']['baseline']
                    np.testing.assert_allclose(row[mode]['metrics']['allAp'],expected['metrics']['allAp'],rtol=0,atol=1e-12)
                    assert row[mode]['matched_gt_ids']==expected['matched_gt_ids']
            rows[method]=row
            print('OBJECT MAP SCORE',key,method,{m:round(v['metrics']['allAp']*100,4) for m,v in row.items()},flush=True)
        for mode in ['primary','raw_targets','uniform_scores']:
            base=set(rows['baseline'][mode]['matched_gt_ids'])
            for row in rows.values():
                hit=set(row[mode]['matched_gt_ids']);row[mode]['changes_vs_baseline']=dict(gained=sorted(hit-base),lost=sorted(base-hit))
        ours=rows['region_history'];base=rows['baseline']
        gate &= all(ours['primary']['metrics'][m]>=base['primary']['metrics'][m] for m in ['allAp','allAp50%'])
        change=ours['raw_targets']['changes_vs_baseline'];gained+=len(change['gained']);lost+=len(change['lost'])
        edits+=seal['results'][key]['region_history']['old_points_changed'];results[key]=rows
    mean={mode:{m:float(np.mean([r[m][mode]['metrics']['allAp'] for r in results.values()])) for m in METHODS}
          for mode in ['primary','raw_targets','uniform_scores']}
    primary=mean['primary'];gate &= primary['region_history']>=primary['baseline']+.01
    gate &= primary['region_history']>max(primary[m] for m in ['pairwise_history','region_current'])
    gate &= gained>=1 and lost==0 and edits>0
    dump(OUT/'map_summary.json',dict(complete=True,results=results,mean_ap=mean,development_gate=bool(gate),
        raw_gained=gained,raw_lost=lost,historical_points_changed=edits,complete_evaluations=24,
        new_robot_runs=0,new_independent_scenes=0,goal_complete=False,
        map_protocol_sha256=sha(OUT/'map_protocol.json'),prediction_seal_sha256=sha(OUT/'maps_complete.json')))
    print('OBJECT PARTITION DEVELOPMENT GATE',bool(gate),mean,'raw gained/lost',gained,lost,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','maps','score'])
    globals()[parser.parse_args().action]()
