"""Observation-authorized identities and writes inside geometric candidates."""
import argparse
from pathlib import Path
import itertools
import time
import runpy
import numpy as np
from scipy.sparse import csr_matrix
from paper_final_common import ROOT,read,dump,sha
from run_object_mask_cover import OUT as PARENT,PARENT as OBS,RAW,verify as verify_parent
from omni_semantic_map.revision_authorization import margin_after_view_deletion,compatible_donors
from omni_semantic_map.batch_stable_transport import native_library,nearest_groups,certify

OUT=ROOT/'build/benchmarks/revision_authorization_v1'
NEW=['identity_only','write_only','current_only','ours']
METHODS=['baseline','unconditional']+NEW


def freeze():
    p=verify_parent();assert not OUT.exists()
    files=[PARENT/'protocol.json',PARENT/'summary.json',PARENT/'maps_complete.json',PARENT/'delivery_audit.json',RAW]
    for c in p['contexts']:
        k=c['key'];files += [c['field'],c['reference'],c['raw_reference'],c['job'],Path(c['full'])/'geometry.npz',
            Path(c['ours'])/'geometry.npz',Path(c['ours'])/'classification.json',
            OBS/'prepared'/k/'atoms.npz',OBS/'prepared'/k/'views.json',
            PARENT/'prepared'/k/'components.npz',PARENT/'prepared'/k/'explanations.json',PARENT/'transport'/k/'cover/field.npz']
        for m in ['cover_current','cover_history']:files += [PARENT/'maps'/k/m/'geometry.npz',PARENT/'maps'/k/m/'classification.json']
        files += [r['file'] for r in read(RAW)['results'][k]]+[r['file'] for r in read(OBS/'prepared'/k/'views.json')]
    src=[Path(__file__),ROOT/'omni_semantic_map/omni_semantic_map/revision_authorization.py',
        ROOT/'omni_semantic_map/test/test_revision_authorization.py',
        ROOT/'omni_semantic_map/omni_semantic_map/batch_stable_transport.py',ROOT/'omni_semantic_map/omni_semantic_map/batch_stable_transport.cpp',
        ROOT/'scripts/benchmarks/evaluate_kitti360_development.py',ROOT/'omni_semantic_map/omni_semantic_map/kitti360_evaluation.py',
        ROOT/'omni_semantic_map/omni_semantic_map/kitti360_reference_identity.py']
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),contexts=p['contexts'],methods=METHODS,
        inputs={str(f):sha(f) for f in files},sources={str(f):sha(f) for f in src},
        development='Hypothesis formed after the failed complete-surface experiment and post-hoc provenance audit. Same exposed windows; not independent, no retroactive change to prior gates. Recompute evidence from raw predictions/old maps; do not read diagnostic GT-bearing feature outputs in inference.',
        candidates='Exactly the parent accepted cover components, proposed identities, oldpoint transitions and added-source pool. No new cover costs, thresholds, class exceptions or geometric radius. New identity births remain possible only in components with no same-class old footprint; no claim of splitting a wrong old identity into a fresh identity in this version.',
        footprints='Exact20cmcells of oldXYZ. Cells containing multiple distinct oldIDs are unknown footprint cells. Same-class old IDs within each accepted component are separate persistent hypotheses. ALL pair combinations within a component are compared, not only previously executed transitions.',
        identity_evidence='In each jointly footprint-visible frame, use all positive native same-class queries. Coverage fractions A(q),B(q) from per-visible-cell mass. Shared=max(score*min(A,B)); separate=sqrt(max(score*A*(1-B))*max(score*B*(1-A))). Retain continuous per-frame differences; no GT fitting or calibrated probability claim.',
        approval='A pair is compatible only if total(shared-separate)-max(0,max_frame_difference)>1e-10. This survives deleting ANY one observed frame. Missing joint views abstain. Candidate receiver remains fixed by parent overlap alignment. Donors considered by descending worst-deletion margin, stableID tie; every pair in accepted group must be compatible, preventing transitive conflicting merges. Not independence, true-identity or optimal grouping proof.',
        local_identity_field='Use unchanged same-class local atom graph, restricted by geometric components. Seed pure old-ID cells ONLY when an actual selected cover-mask union witnesses that cell in that frame. Approved donors use recipient identity, others retain theirs. A component with no same-class old points uses its parent born identity at witnessed cells. Components with old points but no usable old footprint abstain. Reuse two-distinct-observation-frame graph certificate against competing identities on this LOCAL graph.',
        write='Full method accepts only parent-added measured sources in a DIRECT accepted component, in a cell actually witnessed by that component selected mask union in >=1frame, with nonnegative local identity certificate. Label by competing local identity, not whole-component receiver. Cell witness allows other actual measurements in the same20cmcell; no requirement of repeated selection of the exact source point. Existing parent graph certificate remains a proposal constraint, not proof of final relabelled identity.',
        history='Only parent-proposed oldpoint changes may occur; donor must be approved in that point component, destination must equal fixed receiver, and local certificate must agree. Other old points retain originalIDs/XYZ. New classes and ranking scores unchanged from parent; no postfilter, retraining or GT selection.',
        controls='Strong baseline; frozen parent unconditional cover_history; identity_only uses approved history and competing local labels inside direct candidates, retains parent labels for outside-candidate additions and does not require direct cell witness (uncertified inside candidates abstain); write_only requires direct cell witness but keeps parent labels/history; current_only shares full new geometry/labels but preserves oldIDs; ours uses both.',
        gate='Ours primaryAP/AP50>=strong baseline EACH window; meanAP>=baseline+.01, strictly exceeds identity_only,write_only,current_only and unconditional. Raw-target AP50 gained>=1,lost0; actual history edits>0. All failures retained, no post-score parameter changes.',
        budget='Two exposed windows, one raw evidence pass,8new map outputs plus4reused baseline outputs,36full evaluations.6core tests passed. No GPU/model calls/training. Robot/independent effectiveness still required after this gate.'))
    print('REVISION AUTHORIZATION FROZEN',flush=True)


def verify():
    p=read(OUT/'protocol.json')
    for f,h in {**p['inputs'],**p['sources']}.items():assert sha(f)==h,f
    return p


def prepare():
    p=verify();assert not (OUT/'prepared.json').exists();rawseal=read(RAW);files={};results={}
    for c in p['contexts']:
        begin=time.monotonic();key=c['key'];dest=OUT/'prepared'/key;dest.mkdir(parents=True,exist_ok=False)
        base=np.load(Path(c['ours'])/'geometry.npz');classes={r['object_id']:r['predictions']['mean']['train_id'] for r in read(Path(c['ours'])/'classification.json')['objects']}
        d=dict(np.load(OBS/'prepared'/key/'atoms.npz'));part=dict(np.load(PARENT/'prepared'/key/'components.npz'))
        parent=dict(np.load(PARENT/'transport'/key/'cover/field.npz'));field=np.load(c['field']);nc=len(field['cells']);n=len(d['cell_ids'])
        comp=part['component'];ncomp=len(part['roots']);accepted=part['cover_accepted'];receiver=parent['root_owner'][part['roots']]
        cellatom=np.full(nc,-1,np.int32);cellatom[d['cell_ids']]=np.arange(n);cellcomp=np.full(nc,-1,np.int32);cellcomp[d['cell_ids']]=comp
        oldcell=part['old_cell'];valid=oldcell>=0
        lower=np.full(nc,np.iinfo(np.int64).max,np.int64);upper=np.full(nc,-1,np.int64)
        np.minimum.at(lower,oldcell[valid],base['instance'][valid]);np.maximum.at(upper,oldcell[valid],base['instance'][valid])
        pure=np.where((lower==upper)&(lower>0),lower,0)
        oldcomp=np.full(len(oldcell),-1,np.int32);oldcomp[valid]=cellcomp[oldcell[valid]]
        members={};all_pairs=set()
        for component in np.flatnonzero(accepted&(receiver>0)):
            ids=[int(o) for o in np.unique(base['instance'][oldcomp==component]) if classes.get(int(o))==int(part['component_class'][component])]
            members[int(component)]=ids;combined=set(ids)
            if int(receiver[component]) in classes:combined.add(int(receiver[component]))
            all_pairs.update(itertools.combinations(sorted(combined),2))
        pair_rows={pair:[] for pair in sorted(all_pairs)};witness=[];records=read(PARENT/'prepared'/key/'explanations.json')
        views=read(OBS/'prepared'/key/'views.json');rawby={r['frame_id']:r for r in rawseal['results'][key]}
        for rank,rec in enumerate(views):
            v=np.load(rec['file']);px=v['pixel_xy'];cell=v['cells'];footprint=pure[cell];weights=1./np.bincount(cell,minlength=nc)[cell]
            raw=dict(np.load(rawby[rec['frame_id']]['file']));cache={}
            def bits(q):
                if q not in cache:cache[q]=np.unpackbits(raw['masks_packed'][q],axis=1)[px[:,1],px[:,0]].astype(bool)
                return cache[q]
            for r in records:
                component=r['component']
                if r['frame_id']!=rec['frame_id'] or not accepted[component] or receiver[component]<=0:continue
                union=np.zeros(len(cell),bool)
                for q in r['cover']['native_rows']:union |= bits(q)
                nodes=np.unique(cellatom[cell[union&(cellcomp[cell]==component)]])
                if len(nodes):witness.append(np.column_stack((nodes,np.full(len(nodes),component),np.full(len(nodes),rank))))
            for (a,b),rows in pair_rows.items():
                ta=footprint==a;tb=footprint==b
                if not ta.any() or not tb.any():continue
                rid=11 if classes[a]==2 else 26;qq=np.flatnonzero((raw['semantic_ids']==rid)&(raw['scores']>0))
                if not len(qq):continue
                ma=float(weights[ta].sum());mb=float(weights[tb].sum())
                ca=np.array([weights[ta&bits(int(q))].sum()/ma for q in qq]);cb=np.array([weights[tb&bits(int(q))].sum()/mb for q in qq]);s=raw['scores'][qq]
                joint=s*np.minimum(ca,cb);ea=s*ca*(1-cb);eb=s*cb*(1-ca)
                rows.append(dict(frame_id=rec['frame_id'],shared=float(joint.max()),separate=float(np.sqrt(ea.max()*eb.max())),mass_a=ma,mass_b=mb))
            if (rank+1)%12==0:print('IDENTITY FOOTPRINT EVIDENCE',key,rank+1,'/',len(views),flush=True)
        evidence={pair:margin_after_view_deletion([r['frame_id'] for r in rows],[r['shared'] for r in rows],[r['separate'] for r in rows]) for pair,rows in pair_rows.items()}
        groups={};aliases={}
        for component,ids in members.items():
            group=compatible_donors(int(receiver[component]),ids,evidence);groups[component]=group
            aliases[component]={o:int(receiver[component]) if o in group['accepted'] else o for o in ids}
        observations=np.unique(np.concatenate(witness),axis=0) if witness else np.empty((0,3),np.int64)
        seeds=[];witness_node=np.zeros(n,bool)
        for node,component,frame in observations:
            node,component,frame=map(int,[node,component,frame]);witness_node[node]=True;oid=int(pure[d['cell_ids'][node]])
            if oid in aliases[component]:label=aliases[component][oid]
            elif not members[component]:label=int(receiver[component])
            else:continue
            seeds.append((node,label,frame))
        seeds=np.unique(np.asarray(seeds,np.int64).reshape(-1,3),axis=0)
        edges=d['local_edges'];left,right=edges//n,edges%n
        centers=field['centers'][d['cell_ids']];length=np.linalg.norm(centers[left]-centers[right],axis=1)
        graph=csr_matrix((np.r_[length,length],(np.r_[left,right],np.r_[right,left])),shape=(n,n))
        lib=native_library(OUT/'native');owners,dist=nearest_groups(graph,seeds[:,0],seeds[:,1],lib)
        token_pairs,tokens=np.unique(seeds[:,[1,2]],axis=0,return_inverse=True)
        if len(token_pairs):
            token,tdist=nearest_groups(graph,seeds[:,0],tokens,lib);decision=certify(owners,dist,token,tdist,token_pairs[:,0])
        else:token=np.full((n,2),-1,np.int64);tdist=np.full((n,2),np.inf);decision=np.full(n,-1,np.int64)
        np.savez_compressed(dest/'field.npz',cell_atom=cellatom,old_component=oldcomp,pure_old_cell_id=pure,witnesses=observations,
            witness_node=witness_node,seeds=seeds,nearest_owner=owners,owner_distance=dist,token_groups=token_pairs,
            nearest_token=token,token_distance=tdist,certified_owner=decision,graph_indptr=graph.indptr,graph_indices=graph.indices,graph_weights=graph.data)
        dump(dest/'groups.json',dict(members=members,groups=groups,aliases=aliases))
        dump(dest/'evidence.json',{f'{a}_{b}':dict(**evidence[a,b],observations=pair_rows[a,b]) for a,b in evidence})
        stats=dict(candidate_components=len(members),candidate_pairs=len(evidence),jointly_observed_pairs=sum(bool(r) for r in pair_rows.values()),
            pair_approvals=sum(r['approved'] for r in evidence.values()),accepted_donor_relations=sum(len(g['accepted']) for g in groups.values()),
            witness_cells=int(witness_node.sum()),witness_cell_frames=len(observations),label_frame_seeds=len(seeds),certified_cells=int((decision>=0).sum()),
            seconds=time.monotonic()-begin)
        dump(dest/'run.json',stats);results[key]=stats
        for f in dest.iterdir():files[str(f)]=sha(f)
        print('AUTHORIZED IDENTITY FIELD',key,stats,flush=True)
    dump(OUT/'prepared.json',dict(complete=True,results=results,files=files,gt_decoded=False,protocol_sha256=sha(OUT/'protocol.json')))


def maps():
    p=verify();prep=read(OUT/'prepared.json');assert prep['complete'] and not (OUT/'maps_complete.json').exists()
    for f,h in prep['files'].items():assert sha(f)==h,f
    results={};files={}
    for c in p['contexts']:
        key=c['key'];base=dict(np.load(Path(c['ours'])/'geometry.npz'));parent=dict(np.load(PARENT/'maps'/key/'cover_history/geometry.npz'));nold=len(base['xyz'])
        field=np.load(c['field']);part=np.load(PARENT/'prepared'/key/'components.npz');pf=np.load(PARENT/'transport'/key/'cover/field.npz')
        data=dict(np.load(OUT/'prepared'/key/'field.npz'));groups=read(OUT/'prepared'/key/'groups.json')['groups'];full=np.load(Path(c['full'])/'geometry.npz')['xyz']
        source=parent['added_full_source_ids'];cells=field['source_to_cell'][source];nodes=data['cell_atom'][cells]
        direct=nodes>=0;direct[direct]&=part['cover_accepted'][part['component'][nodes[direct]]]
        direct[direct]&=pf['root_owner'][part['root'][nodes[direct]]]>0
        positive=np.zeros(len(source),bool);positive[direct]=data['witness_node'][nodes[direct]]
        local=np.full(len(source),-1,np.int64);local[direct]=data['certified_owner'][nodes[direct]]
        historical=base['instance'].copy();eligible=np.zeros(nold,bool)
        for component,group in groups.items():
            component=int(component);recipient=group['recipient'];donors=group['accepted']
            selected=(data['old_component']==component)&np.isin(base['instance'],donors)&(parent['instance'][:nold]==recipient)
            ii=np.flatnonzero(selected);nn=data['cell_atom'][part['old_cell'][ii]]
            selected[ii]&=data['certified_owner'][nn]==recipient
            historical[selected]=recipient;eligible|=selected
        records=read(PARENT/'maps'/key/'cover_history/classification.json');results[key]={}
        for method in NEW:
            if method=='identity_only':keep=~direct|(local>=0);labels=parent['instance'][nold:].copy();labels[direct]=local[direct];prefix=historical
            elif method=='write_only':keep=direct&positive;labels=parent['instance'][nold:];prefix=parent['instance'][:nold]
            else:keep=direct&positive&(local>=0);labels=local;prefix=base['instance'] if method=='current_only' else historical
            added=source[keep];dest=OUT/'maps'/key/method;dest.mkdir(parents=True,exist_ok=False)
            np.savez_compressed(dest/'geometry.npz',xyz=np.concatenate((base['xyz'],full[added])),instance=np.r_[prefix,labels[keep]],added_full_source_ids=added)
            dump(dest/'classification.json',records)
            stats=dict(points=nold+len(added),added_sources=len(added),parent_added_sources=len(source),
                changed_added_labels=int((labels[keep]!=parent['instance'][nold:][keep]).sum()),old_points_changed=int((prefix!=base['instance']).sum()))
            dump(dest/'run.json',stats);results[key][method]=stats
            print('AUTHORIZED COMPLETE MAP',key,method,stats,flush=True)
        np.savez_compressed(OUT/'prepared'/key/'write_decisions.npz',direct=direct,witness=positive,local_label=local,authorized_history=historical,history_eligible=eligible)
    for f in (OUT/'maps').rglob('*'):
        if f.is_file():files[str(f)]=sha(f)
    dump(OUT/'maps_complete.json',dict(complete=True,results=results,files=files,new_maps=8,gt_decoded=False,
        protocol_sha256=sha(OUT/'protocol.json'),prepared_sha256=sha(OUT/'prepared.json')))


def score():
    from evaluate_kitti360_development import transfer,read_static_ply,clean_json
    from omni_semantic_map.kitti360_evaluation import evaluate
    from omni_semantic_map.kitti360_reference_identity import encode_reference
    from run_discovery_query_benchmark import matched_ids
    p=verify();seal=read(OUT/'maps_complete.json');assert seal['complete'] and not (OUT/'summary.json').exists()
    for f,h in seal['files'].items():assert sha(f)==h,f
    taxonomy=runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['id2label']
    evaluator=ROOT/'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py';prior=read(PARENT/'summary.json')
    results={};gate=True;gained=lost=edits=0
    for c in p['contexts']:
        key=c['key'];ref=dict(np.load(c['reference']));raw=read_static_ply(c['raw_reference']);physical,_=encode_reference(raw['semantic'],raw['instance'],taxonomy,'target_only');rows={}
        for method in METHODS:
            path=Path(c['ours']) if method=='baseline' else PARENT/'maps'/key/'cover_history' if method=='unconditional' else OUT/'maps'/key/method
            g=np.load(path/'geometry.npz');pred={r['object_id']:r['predictions']['mean'] for r in read(path/'classification.json')['objects']}
            labels=transfer(g['xyz'],g['instance'],ref['xyz'],.2);dest=OUT/'evaluation'/key/method;dest.mkdir(parents=True,exist_ok=False)
            np.savez_compressed(dest/'reference_prediction.npz',instance=labels);row={}
            for mode,truth,uniform in [('primary',ref['train_instance'],False),('raw_targets',physical,False),('uniform_scores',ref['train_instance'],True)]:
                info={o:dict(v,score=1.) for o,v in pred.items()} if uniform else pred
                r=evaluate(truth,ref['confidence'],labels,info,evaluator);dump(dest/(mode+'_matches.json'),clean_json(r['matches']))
                row[mode]=dict(metrics=clean_json(r['averages']),matched_gt_ids=sorted(matched_ids(r['matches'])))
                if method in ['baseline','unconditional']:
                    old='baseline' if method=='baseline' else 'cover_history'
                    np.testing.assert_allclose(row[mode]['metrics']['allAp'],prior['results'][key][old][mode]['metrics']['allAp'],atol=1e-12,rtol=0)
            rows[method]=row;print('AUTHORIZED MAP SCORE',key,method,{m:round(v['metrics']['allAp']*100,5) for m,v in row.items()},flush=True)
        for mode in ['primary','raw_targets','uniform_scores']:
            before=set(rows['baseline'][mode]['matched_gt_ids'])
            for row in rows.values():
                after=set(row[mode]['matched_gt_ids']);row[mode]['changes_vs_baseline']=dict(gained=sorted(after-before),lost=sorted(before-after))
        gate &= all(rows['ours']['primary']['metrics'][m]>=rows['baseline']['primary']['metrics'][m] for m in ['allAp','allAp50%'])
        changes=rows['ours']['raw_targets']['changes_vs_baseline'];gained+=len(changes['gained']);lost+=len(changes['lost'])
        edits+=seal['results'][key]['ours']['old_points_changed'];results[key]=rows
    mean={mode:{m:float(np.mean([r[m][mode]['metrics']['allAp'] for r in results.values()])) for m in METHODS}
          for mode in ['primary','raw_targets','uniform_scores']}
    v=mean['primary'];gate &= v['ours']>=v['baseline']+.01 and v['ours']>max(v[m] for m in ['identity_only','write_only','current_only','unconditional'])
    gate &= gained>=1 and lost==0 and edits>0
    dump(OUT/'summary.json',dict(complete=True,results=results,mean_ap=mean,development_gate=bool(gate),raw_gained=gained,raw_lost=lost,
        old_points_changed=edits,new_maps=8,complete_evaluations=36,new_robot_runs=0,new_independent_scenes=0,goal_complete=False,
        protocol_sha256=sha(OUT/'protocol.json'),maps_seal_sha256=sha(OUT/'maps_complete.json')))
    print('REVISION AUTHORIZATION GATE',bool(gate),mean,'raw gain/loss',gained,lost,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','prepare','maps','score'])
    globals()[p.parse_args().action]()
