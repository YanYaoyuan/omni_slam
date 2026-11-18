"""Frozen whole-surface mask explanation and complete-map comparison."""
import argparse
import copy
from pathlib import Path
import time
import runpy
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from paper_final_common import ROOT,read,dump,sha
from evaluate_censored_partition import verify_maps as verify_parent,OUT as PARENT
from omni_semantic_map.mask_cover_revision import explain_surface
from omni_semantic_map.batch_stable_transport import native_library,nearest_groups,certify

OUT=ROOT/'build/benchmarks/object_mask_cover_v1'
RAW=ROOT/'build/benchmarks/prethreshold_candidates_v1/predictions_complete.json'
METHODS=['baseline','geometry_history','single_history','cover_current','cover_history']


def freeze():
    p=verify_parent();assert not OUT.exists()
    files=[PARENT/'prepared.json',PARENT/'input_protocol.json',PARENT/'map_summary.json',RAW]
    for c in p['contexts']:
        key=c['key'];files += [c['field'],c['job'],c['reference'],c['raw_reference'],
            Path(c['full'])/'geometry.npz',Path(c['ours'])/'geometry.npz',Path(c['ours'])/'classification.json',
            PARENT/'prepared'/key/'atoms.npz',PARENT/'prepared'/key/'views.json',PARENT/'prepared'/key/'vocabularies.json']
        files += [r['file'] for r in read(PARENT/'prepared'/key/'views.json')]
        files += [r['file'] for r in read(RAW)['results'][key]]
    src=[Path(__file__),ROOT/'omni_semantic_map/omni_semantic_map/mask_cover_revision.py',
        ROOT/'omni_semantic_map/test/test_mask_cover_revision.py',
        ROOT/'omni_semantic_map/omni_semantic_map/batch_stable_transport.py',
        ROOT/'omni_semantic_map/omni_semantic_map/batch_stable_transport.cpp',
        ROOT/'omni_semantic_map/omni_semantic_map/kitti360_evaluation.py',
        ROOT/'omni_semantic_map/omni_semantic_map/kitti360_reference_identity.py',
        ROOT/'scripts/benchmarks/evaluate_kitti360_development.py']
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),contexts=p['contexts'],methods=METHODS,
        inputs={str(f):sha(f) for f in files},sources={str(f):sha(f) for f in src},
        geometry='Reuse all frozen measured atoms,20cm cells and strict0.4m16NN same-class local proposal edges. Connected components, canonical minimum atom root. No nonlocal query-chain joins and no categorical-token split objective. This can merge/split legacy identities via exact overlap alignment; disconnected surfaces cannot join in this candidate.',
        visibility='Use ALL original measured-visible source pixels, including other classes/background/unknown for leakage. Each visible cell contributes one mass, divided by its visible sources. Only region membership from the geometric components defines inside; neither GT nor per-pixel semantic filtering chooses masks.',
        cover='All positive native-score same-class queries. Query cost log2(1+Q)-log2(native_score). Greedily maximize union-covered inside mass minus union-covered outside mass minus sum selected query costs. Each source counted once despite overlapping masks. Zero-mask explanation has utility0. Negative-gain queries rejected; outside/hidden geometry not observed here contributes nothing. This is a fixed coverage utility, NOT calibrated probability, exact losslessMDL, optimal set cover or proven novelty.',
        admissibility='Pool each component positive explanation utility over actual observed frames. Accept if total utility>log2(number_of_geometric_components+1) and positive explanation in at least3frames. Different frame mask IDs do not disagree merely by differing. The3frame minimum inherits the object birth evidence convention, not statistical independence. Same exact rule for singleton vs multi-mask explanation.',
        controls='Current strong baseline; all geometry components with common export; best single mask per region/frame with same costs/acceptance/export; greedy multi-mask union current-only; same multi-mask union with history. Single/multi share candidate masks and measured pixels. Native object ranking is kept common; a separate uniform-score diagnostic isolates ranking.',
        identity='One-to-one same-class maximum-count positive exact-cell overlap with old IDs. Unaligned accepted components may be born with>=16full measured sources and>=3original mask-observed frames; same inherited export rule. Old classes/scores preserved, new scores from original native token observations, not fitted to evaluations.',
        history='Keep all oldXYZ. Only an old point whose exact cell belongs to an ACCEPTED foreground component may change to its component owner, and only if the common two-frame transport certificate agrees with that owner. No rejected/hidden component is called background. Current-only changes no old IDs. This restricted direct-evidence update is COMMON to geometry/single/multi history controls; it is not the previous categorical exporter.',
        addition='Unchanged full measured graph transport and witnessed semantic background seeds; append certified foreground full measuredXYZ >5cm from all oldXYZ. No new synthesized coordinates, postfilter/NMS/SOR, GT selection or threshold search. Unknown transport remains abstention.',
        gate='Before scoring: cover_history primary AP and AP50 no worse than strong baseline in EACH window; meanAP>=baseline+.01 and strictly exceeds geometry_history,single_history,cover_current. Raw canonical-target AP50 gains>=1, losses0; some actual history changes. Otherwise close configuration without tuning. A failed control is still reported.',
        budget='Two exposed full windows, one component/cover pass each,8new complete maps,30evaluations including baseline/diagnostics.5core tests passed. CPU only,no new model inference/training. Robot and truly unused outdoor evaluation only after same-version efficacy; not claimed complete by this stage.'))
    print('OBJECT COVER PROTOCOL FROZEN',flush=True)


def verify():
    p=read(OUT/'protocol.json')
    for f,h in {**p['inputs'],**p['sources']}.items():assert sha(f)==h,f
    return p


def prepare():
    p=verify();assert not (OUT/'prepared.json').exists();results={};files={};rawseal=read(RAW)
    for c in p['contexts']:
        start=time.monotonic();key=c['key'];dest=OUT/'prepared'/key;dest.mkdir(parents=True,exist_ok=False)
        d=dict(np.load(PARENT/'prepared'/key/'atoms.npz'));n=len(d['cell_ids']);e=d['local_edges'];a,b=e//n,e%n
        graph=csr_matrix((np.ones(2*len(a),np.int8),(np.r_[a,b],np.r_[b,a])),shape=(n,n))
        nc,comp=connected_components(graph,directed=False)
        roots=np.full(nc,n,np.int64);np.minimum.at(roots,comp,np.arange(n));labels=roots[comp]
        component_class=d['train_class'][roots];assert np.all(component_class[comp]==d['train_class'])
        field=np.load(c['field']);sourcecell=field['source_to_cell'];celln=len(field['cells'])
        cellcomp=np.full(celln,-1,np.int32);cellcomp[d['cell_ids']]=comp
        counts=np.bincount(labels,weights=np.bincount(sourcecell,minlength=celln)[d['cell_ids']],minlength=n).astype(np.int64)
        nv=len(d['categories']);pairs=np.unique(labels[d['observation_atom']]*nv+d['observation_view']//2)
        observed=np.bincount(pairs//nv,minlength=n);birth=(counts>=16)&(observed>=3)
        # Exact old-coordinate mapping, no nearestGT or identity locking.
        base=np.load(Path(c['ours'])/'geometry.npz');dtype=np.dtype([('x',np.int64),('y',np.int64),('z',np.int64)])
        fkeys=np.ascontiguousarray(field['cells'],np.int64).view(dtype).ravel()
        bkeys=np.ascontiguousarray(np.floor(base['xyz']/.2),np.int64).view(dtype).ravel()
        pos=np.searchsorted(fkeys,bkeys);valid=pos<len(fkeys);valid[valid]&=fkeys[pos[valid]]==bkeys[valid]
        oldcell=np.full(len(pos),-1,np.int64);oldcell[valid]=pos[valid]
        atomlookup=np.full(celln,-1,np.int32);atomlookup[d['cell_ids']]=np.arange(n)
        known=valid.copy();known[known]&=atomlookup[oldcell[known]]>=0
        pointroot=np.full(len(pos),-1,np.int64);pointroot[known]=labels[atomlookup[oldcell[known]]]
        utility={m:np.zeros(nc) for m in ['single','cover']};support={m:np.zeros(nc,np.int32) for m in utility}
        records=[];union_checks=0;rawby={r['frame_id']:r for r in rawseal['results'][key]}
        views=read(PARENT/'prepared'/key/'views.json')
        for rank,rec in enumerate(views):
            v=dict(np.load(rec['file']));raw=dict(np.load(rawby[rec['frame_id']]['file']))
            visible_comp=cellcomp[v['cells']];denom=np.bincount(v['cells'],minlength=celln)
            weights=1./denom[v['cells']];px=v['pixel_xy'];assert abs(weights.sum()-len(np.unique(v['cells'])))<1e-8
            for cls,rid in [(2,11),(13,26)]:
                queries=np.flatnonzero((raw['semantic_ids']==rid)&(raw['scores']>0))
                if not len(queries):continue
                masks=((raw['masks_packed'][queries[:,None],px[None,:,1],px[None,:,0]//8]>>(7-px[None,:,0]%8))&1).astype(bool)
                costs=np.log2(1+len(queries))-np.log2(raw['scores'][queries].astype(float))
                inside=np.zeros((nc,len(queries)));known=visible_comp>=0
                for j in range(len(queries)):
                    take=known&masks[j]
                    inside[:,j]=np.bincount(visible_comp[take],weights=weights[take],minlength=nc)
                candidates=np.flatnonzero((component_class==cls)&np.any(inside>costs,axis=1))
                for component in candidates:
                    # Safe marginal-gain upper bound, not a confidence threshold.
                    chosen=np.flatnonzero(inside[component]>costs)
                    member=visible_comp==component
                    r=explain_surface(masks[chosen],weights,member,costs[chosen])
                    for name,mode in [('single','single'),('cover','multi')]:
                        row=r[mode];utility[name][component]+=row['utility'];support[name][component]+=row['utility']>1e-10
                        if row['queries']:
                            qq=chosen[row['queries']];union=np.any(masks[qq],axis=0)
                            # Explicit independent union evaluation over source IDs.
                            inside_mass=float(sum(weights[i] for i in np.flatnonzero(union&member)))
                            outside_mass=float(sum(weights[i] for i in np.flatnonzero(union&~member)))
                            assert abs(row['utility']-(inside_mass-outside_mass-costs[qq].sum()))<1e-7
                            union_checks+=1
                    if r['multi']['utility']>0:
                        records.append(dict(frame_id=rec['frame_id'],component=int(component),root=int(roots[component]),
                            single={**r['single'],'native_rows':queries[chosen[r['single']['queries']]].tolist()},
                            cover={**r['multi'],'native_rows':queries[chosen[r['multi']['queries']]].tolist()},visible_region_mass=float(weights[member].sum())))
            if (rank+1)%12==0:print('SURFACE MASK COVER',key,rank+1,'/',len(views),flush=True)
        accepted={name:(utility[name]>np.log2(nc+1))&(support[name]>=3) for name in utility}
        assert np.all(~accepted['single']|accepted['cover'])
        arrays=dict(root=labels,component=comp,roots=roots,component_class=component_class,source_count=counts,
            observed_frames=observed,birth_eligible=birth,old_point_proposed_region=pointroot,old_cell=oldcell)
        for name in utility:
            for fieldname,value in [('utility',utility[name]),('support_frames',support[name]),('accepted',accepted[name])]:arrays[name+'_'+fieldname]=value
        np.savez_compressed(dest/'components.npz',**arrays);dump(dest/'explanations.json',records)
        stats=dict(atoms=n,geometric_components=nc,birth_eligible_components=int(birth.sum()),
            accepted={m:int(v.sum()) for m,v in accepted.items()},
            multi_only_components=int((accepted['cover']&~accepted['single']).sum()),
            region_frames_multi_better=sum(r['cover']['utility']>r['single']['utility']+1e-8 for r in records),
            multi_mask_region_frames=sum(len(r['cover']['queries'])>1 for r in records),
            independent_union_checks=union_checks,seconds=time.monotonic()-start)
        dump(dest/'run.json',stats);results[key]=stats
        for f in dest.iterdir():files[str(f)]=sha(f)
        print('SURFACE COVER READY',key,stats,flush=True)
    dump(OUT/'prepared.json',dict(complete=True,results=results,files=files,gt_read=False,
        protocol_sha256=sha(OUT/'protocol.json')))


def maps():
    p=verify();prepared=read(OUT/'prepared.json');assert prepared['complete'] and not (OUT/'maps_complete.json').exists()
    for f,h in prepared['files'].items():assert sha(f)==h,f
    raw=read(RAW);results={};files={}
    for c in p['contexts']:
        key=c['key'];d=dict(np.load(PARENT/'prepared'/key/'atoms.npz'));part=dict(np.load(OUT/'prepared'/key/'components.npz'))
        field=np.load(c['field']);cells=field['source_to_cell'];nc=len(field['cells']);n=len(d['cell_ids'])
        graph=csr_matrix((field['graph_weights'],field['graph_indices'],field['graph_indptr']),shape=(nc,nc))
        full=np.load(Path(c['full'])/'geometry.npz')['xyz'];base=dict(np.load(Path(c['ours'])/'geometry.npz'))
        records=read(Path(c['ours'])/'classification.json');oldids=np.array(sorted(r['object_id'] for r in records['objects']))
        byoid={r['object_id']:r for r in records['objects']};classold=np.array([byoid[o]['predictions']['mean']['train_id'] for o in oldids])
        views=read(PARENT/'prepared'/key/'views.json');vocab=read(PARENT/'prepared'/key/'vocabularies.json')
        rawby={r['frame_id']:r for r in raw['results'][key]};token_score=[];bgrows=[]
        for v in vocab:
            scores=np.load(rawby[v['frame_id']]['file'])['scores'];lookup=np.zeros(v['categories'])
            lookup[1:len(v['query_rows'])+1]=scores[v['query_rows']];token_score.append(lookup)
        observed_scores=np.array([token_score[v][q] for v,q in zip(d['observation_view'],d['observation_token'])])
        bg=(d['seen']>0)&~np.isin(d['semantic_class_all_cells'],[2,13])
        for rank,rec in enumerate(views):
            v=np.load(rec['file']);cc=v['cells'];cnt=np.bincount(cc,minlength=nc)
            outside=np.bincount(cc,weights=~np.isin(v['semantic'],[2,13]),minlength=nc)
            selected=np.flatnonzero(bg&(outside>.5*cnt))
            bgrows.append(np.column_stack((selected,np.zeros(len(selected),np.int64),np.full(len(selected),rank,np.int64))))
        background=np.concatenate(bgrows);duplicate_free=cKDTree(base['xyz']).query(full,workers=1)[0]>.05
        labels=part['root'];allroots=part['roots'];oldcell=part['old_cell'];results[key]={}
        lib=native_library(OUT/'map_native')
        for mode in ['geometry','single','cover']:
            tick=time.monotonic();accepted=np.ones(len(allroots),bool) if mode=='geometry' else part[mode+'_accepted']
            roots=allroots[accepted];rootpos=np.full(n,-1,np.int64);rootpos[roots]=np.arange(len(roots))
            pointroot=part['old_point_proposed_region'];available=pointroot>=0;available[available]&=rootpos[pointroot[available]]>=0
            contingency=np.zeros((len(roots),len(oldids)),np.int64)
            np.add.at(contingency,(rootpos[pointroot[available]],np.searchsorted(oldids,base['instance'][available])),1)
            contingency[d['train_class'][roots,None]!=classold[None,:]]=0
            ii,jj=linear_sum_assignment(-contingency);owner=np.zeros(n,np.int64);lineage=[]
            for i,j in zip(ii,jj):
                if contingency[i,j]>0:
                    owner[roots[i]]=oldids[j];lineage.append(dict(root=int(roots[i]),old_id=int(oldids[j]),overlap_points=int(contingency[i,j])))
            nextid=int(oldids.max())+1;births=[];outrecords=copy.deepcopy(records)
            for root in roots:
                if owner[root] or not part['birth_eligible'][root]:continue
                owner[root]=nextid;nextid+=1;oid=int(owner[root]);cls=int(d['train_class'][root])
                selected=labels[d['observation_atom']]==root;vv=d['observation_view'][selected]//2;ww=d['observation_mass'][selected]
                total=np.bincount(vv,weights=ww,minlength=len(views));response=np.bincount(vv,weights=ww*observed_scores[selected],minlength=len(views))
                score=float(np.mean(response[total>0]/total[total>0]))
                outrecords['objects'].append(dict(object_id=oid,observation_ids=[],selected_observation_ids=[],source_root=int(root),
                    predictions=dict(mean=dict(name='building' if cls==2 else 'car',semantic_id=11 if cls==2 else 26,train_id=cls,score=score,retained=True))))
                births.append(dict(root=int(root),object_id=oid,train_class=cls,score=score))
            atomowner=owner[labels];take=atomowner[d['observation_atom']]>0
            fg=np.column_stack((d['cell_ids'][d['observation_atom'][take]],atomowner[d['observation_atom'][take]],d['observation_view'][take]//2))
            seeds=np.unique(np.vstack((background,fg)),axis=0)
            own,dist=nearest_groups(graph,seeds[:,0],seeds[:,1],lib)
            pairs,tokens=np.unique(seeds[:,[1,2]],axis=0,return_inverse=True)
            token,tdist=nearest_groups(graph,seeds[:,0],tokens,lib);decision=certify(own,dist,token,tdist,pairs[:,0])
            appended=np.flatnonzero(duplicate_free&(decision[cells]>0))
            # A direct accepted component AND agreeing certificate are required.
            cellowner=np.zeros(nc,np.int64);cellowner[d['cell_ids']]=atomowner
            old=base['instance'].copy();edit=oldcell>=0
            edit[edit]&=(cellowner[oldcell[edit]]>0)&(decision[oldcell[edit]]==cellowner[oldcell[edit]])
            old[edit]=cellowner[oldcell[edit]]
            folder=OUT/'transport'/key/mode;folder.mkdir(parents=True,exist_ok=False)
            np.savez_compressed(folder/'field.npz',seeds=seeds,nearest_owner=own,owner_distance=dist,token_groups=pairs,
                nearest_token=token,token_distance=tdist,certified_owner=decision,root_owner=owner,
                old_cell=oldcell,duplicate_free=duplicate_free,old_edit_eligible=edit,cell_direct_owner=cellowner)
            dump(folder/'lineage.json',dict(aligned=lineage,births=births))
            names=['cover_current','cover_history'] if mode=='cover' else [mode+'_history']
            for name in names:
                dest=OUT/'maps'/key/name;dest.mkdir(parents=True,exist_ok=False)
                prefix=base['instance'] if name=='cover_current' else old
                np.savez_compressed(dest/'geometry.npz',xyz=np.concatenate((base['xyz'],full[appended])),
                    instance=np.r_[prefix,decision[cells[appended]]],added_full_source_ids=appended)
                dump(dest/'classification.json',outrecords)
                stats=dict(points=len(prefix)+len(appended),added_measured_sources=len(appended),new_identities=len(births),
                    accepted_components=int(accepted.sum()),old_points_changed=int((prefix!=base['instance']).sum()),
                    direct_history_eligible=int(edit.sum()),seconds_shared_export=time.monotonic()-tick)
                dump(dest/'run.json',stats);results[key][name]=stats
                print('COMPLETE COVER MAP',key,name,stats,flush=True)
    for sub in ['maps','transport','map_native']:
        for f in (OUT/sub).rglob('*'):
            if f.is_file():files[str(f)]=sha(f)
    dump(OUT/'maps_complete.json',dict(complete=True,results=results,files=files,gt_read=False,new_maps=8,
        prepared_sha256=sha(OUT/'prepared.json'),protocol_sha256=sha(OUT/'protocol.json')))


def score():
    from evaluate_kitti360_development import transfer,read_static_ply,clean_json
    from omni_semantic_map.kitti360_evaluation import evaluate
    from omni_semantic_map.kitti360_reference_identity import encode_reference
    from run_discovery_query_benchmark import matched_ids
    p=verify();seal=read(OUT/'maps_complete.json');assert seal['complete'] and not (OUT/'summary.json').exists()
    for f,h in seal['files'].items():assert sha(f)==h,f
    taxonomy=runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['id2label']
    evaluator=ROOT/'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py'
    prior=read(PARENT/'map_summary.json');results={};gate=True;gained=lost=edits=0
    for c in p['contexts']:
        key=c['key'];ref=dict(np.load(c['reference']));raw=read_static_ply(c['raw_reference'])
        physical,_=encode_reference(raw['semantic'],raw['instance'],taxonomy,'target_only');rows={}
        for method in METHODS:
            source=Path(c['ours']) if method=='baseline' else OUT/'maps'/key/method
            g=np.load(source/'geometry.npz');predictions={r['object_id']:r['predictions']['mean'] for r in read(source/'classification.json')['objects']}
            projected=transfer(g['xyz'],g['instance'],ref['xyz'],.2);dest=OUT/'evaluation'/key/method;dest.mkdir(parents=True,exist_ok=False)
            np.savez_compressed(dest/'reference_prediction.npz',instance=projected);row={}
            for mode,truth,uniform in [('primary',ref['train_instance'],False),('raw_targets',physical,False),('uniform_scores',ref['train_instance'],True)]:
                info={o:dict(v,score=1.) for o,v in predictions.items()} if uniform else predictions
                r=evaluate(truth,ref['confidence'],projected,info,evaluator);dump(dest/(mode+'_matches.json'),clean_json(r['matches']))
                row[mode]=dict(metrics=clean_json(r['averages']),matched_gt_ids=sorted(matched_ids(r['matches'])))
                if method=='baseline':np.testing.assert_allclose(row[mode]['metrics']['allAp'],prior['results'][key]['baseline'][mode]['metrics']['allAp'],atol=1e-12,rtol=0)
            rows[method]=row
            print('COVER MAP SCORE',key,method,{m:round(v['metrics']['allAp']*100,5) for m,v in row.items()},flush=True)
        for mode in ['primary','raw_targets','uniform_scores']:
            base=set(rows['baseline'][mode]['matched_gt_ids'])
            for row in rows.values():
                hit=set(row[mode]['matched_gt_ids']);row[mode]['changes_vs_baseline']=dict(gained=sorted(hit-base),lost=sorted(base-hit))
        ours=rows['cover_history'];base=rows['baseline']
        gate &= all(ours['primary']['metrics'][m]>=base['primary']['metrics'][m] for m in ['allAp','allAp50%'])
        changes=ours['raw_targets']['changes_vs_baseline'];gained+=len(changes['gained']);lost+=len(changes['lost'])
        edits+=seal['results'][key]['cover_history']['old_points_changed'];results[key]=rows
    mean={mode:{m:float(np.mean([r[m][mode]['metrics']['allAp'] for r in results.values()])) for m in METHODS}
        for mode in ['primary','raw_targets','uniform_scores']}
    v=mean['primary'];gate &= v['cover_history']>=v['baseline']+.01
    gate &= v['cover_history']>max(v[m] for m in ['geometry_history','single_history','cover_current'])
    gate &= gained>=1 and lost==0 and edits>0
    dump(OUT/'summary.json',dict(complete=True,results=results,mean_ap=mean,development_gate=bool(gate),
        raw_gained=gained,raw_lost=lost,historical_points_changed=edits,new_maps=8,complete_evaluations=30,
        new_robot_runs=0,new_independent_scenes=0,goal_complete=False,
        protocol_sha256=sha(OUT/'protocol.json'),maps_seal_sha256=sha(OUT/'maps_complete.json')))
    print('COVER DEVELOPMENT GATE',bool(gate),mean,'raw gained/lost',gained,lost,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','prepare','maps','score'])
    globals()[p.parse_args().action]()
