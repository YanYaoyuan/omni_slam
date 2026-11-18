"""Five-arm complete-map test of reference-free historical geometry revision.

Shared external semantic evidence; GT only enters score() after all ten maps
are sealed. Geometry regions change the readout of measured query surfaces.
"""
import argparse,copy,shutil,time
from pathlib import Path
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from capture_closed_vocabulary_semantics import OUT as SEM,BASE,INPUT
from omni_semantic_map.counterfactual_geometry_revision import CounterfactualRevision,connected_cells,nearest_indices
from audit_semantic_counterevidence import winner
from evaluate_kitti360_development import transfer,clean_json
from omni_semantic_map.kitti360_evaluation import evaluate
from run_discovery_query_benchmark import matched_ids

OUT=ROOT/'build/benchmarks/counterfactual_geometry_revision_v1'
METHODS=['baseline','semantic_point','semantic_region','local_counterfactual','ours_counterfactual']
EVALUATOR=ROOT/'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py'


def freeze():
    assert not OUT.exists();parent=read(SEM/'protocol.json');capture=read(SEM/'predictions_complete.json');aggregation=read(SEM/'aggregation_complete.json');assert capture['complete'] and aggregation['complete'];inputs={}
    for d in [capture,aggregation]:
        for f,h in d['files'].items():assert sha(f)==h,f;inputs[f]=h
    for f in [SEM/'protocol.json',SEM/'predictions_complete.json',SEM/'aggregation_protocol.json',SEM/'aggregation_complete.json',EVALUATOR,ROOT/'build/benchmarks/signed_query_association_v1/summary.json']:
        inputs[str(f)]=sha(f)
    for scene in parent['scenes']:
        key=scene['key'];m=read(INPUT/key/'manifest.json')
        for f in [m['geometry'],INPUT/key/'manifest.json',BASE/key/'geometry_graph/geometry.npz',BASE/key/'geometry_graph/classification.json',scene['reference'],parent['references'][key]['raw']]:inputs[str(f)]=sha(f)
    modules=['counterfactual_geometry_revision','spatial_acceptance_readout','kitti360_evaluation'];sources=[Path(__file__),ROOT/'scripts/benchmarks/audit_semantic_counterevidence.py',ROOT/'scripts/benchmarks/evaluate_kitti360_development.py',ROOT/'scripts/benchmarks/run_discovery_query_benchmark.py',ROOT/'omni_semantic_map/test/test_counterfactual_geometry_revision.py']+[ROOT/'omni_semantic_map/omni_semantic_map'/f'{m}.py' for m in modules]
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),input_hashes=inputs,source_hashes={str(f):sha(f) for f in sources},scenes=parent['scenes'],methods=METHODS,
        hypothesis='Before committing removal of a historical geometry region, evaluate the actual semantic readout change at all affected measured surfaces, including neighbouring cells and alternative surviving objects. This can avoid harms hidden by point-local semantic agreement.',
        evidence='Exact sealed externalCityscapesSegFormer mean probabilities on actual source measurements. No new model/GT training. All observed source points,one equal total weight per20cm physical cell;no class-balanced GT weights. First/mean/majority source diagnostics were seen during development;this is not a blind test.',
        probability='Map class set is the frozen existing object train classes. Model posterior of each mapped class plus one no-tracked-class probability equal to sum of remaining19classes. Missing semantic observation excluded,not mapped to background. The no-object event is represented by absence of nearby map geometry,never by inserting a null point.',
        proposals='For each old object,20cm cells containing its original fused points. Cell response is mean19classposterior of all observed original sources in that cell. Unique positive winner different from declared objectclass proposes that cell.26-neighbour components perobject;remove every old point of that object in componentcells. Sort by descending removedpointcount,objectID,lexicographiccellkey;atmost500regions/window. Same regions/order for all region arms;no GT,score reranking or newcoordinates.',
        controls='baseline exact geometry_graph. semantic_point deletes oldpoints whose directly projected mean posterior uniquely predicts anotherclass,unseen/ambiguousretain. semantic_region deletes every proposedregion. local_counterfactual tests actual before/after mapreadout but only at measured queries in proposedcells. Ours tests all queries whose current nearestpoint is removed,including neighbouringcells. Shared semantic model and data;baseline has no extra semantic model cost.',
        objective='Weighted negative log probability of nearest surviving map point semanticclass within strict20cm;fallback is no-tracked-class probability. Stable original-index ties. For each proposal in fixed order, recompute exact alternative nearest points only for currently affectedqueries. Accept region atomically iff own acceptance-domain risk decreases by>1e-9. Ours objective monotonically decreases;this is model self-consistency,not GT accuracy or a calibrated performance guarantee. No independent-view acceptance claim.',
        export='Subset of original mapXYZ and original pointobjectIDs only. Save retainedbaseindices for complete provenance;all classscores unchanged,drop metadata only for objects with no remaining points. No newobjects,IDmerge/split,coordinatechanges,syntheticpoints,additional cleanup or postfilter. Method scope is historical objectgeometry revision,not newSLAM/localization or identity-association algorithm.',
        budget='2exposedwindows x5complete maps,each regionarm onepass <=500regions/window. CPUonly after shared108externalmodel forwards. No thresholds/cellresolution/order/model/iteration search after results. All maps sealed beforeGT scores.',
        primary='Same unchanged adaptedKITTI360 confidenceweighted building/car AP50:90,thresholds.50:.05:.90;AP50 and AP25 secondary. Full existing releasedreferencewindows,not officialleaderboard. Baseline must numerically reproduce previousgeometry_graph metrics.',
        gate='Ours primaryAP no lower than baseline in eitherwindow;mean >= baseline+0.01 AND strictly exceeds every othercontrol mean; >=1new AP50matchedGT and zero lost vsbaseline acrosswindows. Otherwise close thisfixedversion without parameter rescue or robot/independent expansion.',
        novelty_scope='A proposed spatial counterfactual selection mechanism using shared external semantic evidence. OrdinarySegFormer,connectedcomponents,nearestprojection and greedydeletion are not standalone innovations. No novelty/success claim without map-level evidence and subsequent independent/robot validation.'))
    print('FROZEN COUNTERFACTUAL MAP REVISION',flush=True)


def verify():
    p=read(OUT/'protocol.json')
    for f,h in {**p['input_hashes'],**p['source_hashes']}.items():assert sha(f)==h,f
    return p


def sealed(name):
    d=read(OUT/(name+'.json'));assert d['complete']
    for f,h in d['files'].items():assert sha(f)==h,f
    return d


def prepare():
    p=verify();assert not (OUT/'prepared.json').exists();files={};stats={}
    for scene in p['scenes']:
        key=scene['key'];source=np.load(read(INPUT/key/'manifest.json')['geometry']);base=np.load(BASE/key/'geometry_graph/geometry.npz');sem=np.load(SEM/'aggregate'/key/'source.npz');objects=read(BASE/key/'geometry_graph/classification.json')['objects'];classes=sorted({o['predictions']['mean']['train_id'] for o in objects});class_index={c:i for i,c in enumerate(classes)};point_class=np.full(len(base['xyz']),-1,np.int16)
        for obj in objects:point_class[base['instance']==obj['object_id']]=class_index[obj['predictions']['mean']['train_id']]
        assert (point_class>=0).all();qids=np.flatnonzero(sem['seen']>0);prob=sem['mean_probabilities'][qids];qxyz=source['xyz'][qids];qkeys=np.floor(qxyz/.2).astype(np.int64);mkeys=np.floor(base['xyz']/.2).astype(np.int64);keys,inverse=np.unique(np.concatenate([qkeys,mkeys]),axis=0,return_inverse=True);qcell=inverse[:len(qids)];mcell=inverse[len(qids):];counts=np.bincount(qcell,minlength=len(keys));mu=np.column_stack([np.bincount(qcell,weights=prob[:,i],minlength=len(keys)) for i in range(19)]);np.divide(mu,counts[:,None],out=mu,where=counts[:,None]>0);pred=winner(mu,np.arange(19));proposals=[]
        for obj in sorted(objects,key=lambda x:x['object_id']):
            oid=obj['object_id'];old=np.flatnonzero(base['instance']==oid);cells=np.unique(mcell[old]);cells=cells[(pred[cells]>=0)&(pred[cells]!=obj['predictions']['mean']['train_id'])]
            for part in connected_cells(keys[cells]):
                region=cells[part];remove=old[np.isin(mcell[old],region)];proposals.append(dict(object_id=oid,cell_ids=region.tolist(),map_ids=remove.tolist(),points=len(remove),minimum_cell=keys[region].min(axis=0).tolist()))
        proposals.sort(key=lambda r:(-r['points'],r['object_id'],r['minimum_cell']));total=len(proposals);proposals=proposals[:500]
        for j,r in enumerate(proposals):r['region_id']=j
        covered=np.concatenate([np.asarray(r['map_ids'],np.int64) for r in proposals]) if proposals else np.empty(0,np.int64);assert len(np.unique(covered))==len(covered)
        remaining=[c for c in range(19) if c not in classes];posterior=np.column_stack([prob[:,classes],prob[:,remaining].sum(1)]);np.testing.assert_allclose(posterior.sum(1),1.,rtol=0,atol=1e-6);weights=1./counts[qcell]
        folder=OUT/'prepared'/key;folder.mkdir(parents=True,exist_ok=False);f=folder/'context.npz';np.savez_compressed(f,query_source_ids=qids,query_cell=qcell,map_cell=mcell,cell_keys=keys,probabilities=posterior,weights=weights,point_classes=point_class,train_classes=np.array(classes));files[str(f)]=sha(f);f=folder/'regions.json';dump(f,proposals);files[str(f)]=sha(f)
        stats[key]=dict(query_sources=len(qids),observed_cells=int((counts>0).sum()),proposed_regions=total,selected_regions=len(proposals),candidate_removed_points=len(covered),classes=classes);print('PREPARED REVISION',key,stats[key],flush=True)
    dump(OUT/'prepared.json',dict(complete=True,protocol_sha256=sha(OUT/'protocol.json'),files=files,stats=stats,gt_read=False))


def maps():
    p=verify();sealed('prepared');assert not (OUT/'predictions_complete.json').exists();files={};stats={}
    for scene in p['scenes']:
        key=scene['key'];base=np.load(BASE/key/'geometry_graph/geometry.npz');classification=read(BASE/key/'geometry_graph/classification.json');source=np.load(read(INPUT/key/'manifest.json')['geometry']);ctx=np.load(OUT/'prepared'/key/'context.npz');regions=read(OUT/'prepared'/key/'regions.json');xyz=source['xyz'][ctx['query_source_ids']];sem=np.load(SEM/'aggregate'/key/'map.npz');rows={}
        for method in METHODS:
            start=time.monotonic();keep=np.ones(len(base['xyz']),bool);events=[];audit={}
            if method=='semantic_point':keep &= (sem['mean_response']<0)|(sem['mean_response']==sem['declared_train_id'])
            elif method=='semantic_region':
                for r in regions:keep[r['map_ids']]=False
            elif method in ['local_counterfactual','ours_counterfactual']:
                state=CounterfactualRevision(base['xyz'],ctx['point_classes'],xyz,ctx['probabilities'],ctx['weights'])
                for i,r in enumerate(regions):
                    local=np.isin(ctx['query_cell'],r['cell_ids']) if method=='local_counterfactual' else None
                    event=state.trial(r['map_ids'],local);event.update(region_id=r['region_id'],object_id=r['object_id']);events.append(event)
                    if (i+1)%50==0:print('REVISING',key,method,i+1,len(regions),flush=True)
                keep=state.active.copy();audit=state.audit()
                if method=='ours_counterfactual':assert audit['final_loss']<=audit['initial_loss']+1e-7 and all(e['full_gain']>0 for e in events if e['accepted'])
            folder=OUT/'maps'/key/method;folder.mkdir(parents=True,exist_ok=False);retained=np.flatnonzero(keep)
            if method=='baseline':
                for name in ['geometry.npz','classification.json']:shutil.copyfile(BASE/key/'geometry_graph'/name,folder/name)
            else:
                np.savez_compressed(folder/'geometry.npz',xyz=base['xyz'][retained],instance=base['instance'][retained],retained_base_indices=retained)
                export=copy.deepcopy(classification);present=set(base['instance'][retained]);export['objects']=[o for o in export['objects'] if o['object_id'] in present];dump(folder/'classification.json',export)
            dump(folder/'events.json',events);run=dict(removed_points=int((~keep).sum()),remaining_points=len(retained),remaining_objects=len(np.unique(base['instance'][retained])),proposed_regions=len(regions),accepted_regions=sum(e['accepted'] for e in events),accepted_global_harm=sum(e['accepted'] and e['full_gain']<0 for e in events),seconds=time.monotonic()-start,audit=audit,retained_coordinates_and_ids_exact=True,new_objects=0)
            dump(folder/'run.json',run)
            for f in folder.iterdir():files[str(f)]=sha(f)
            rows[method]=run;print('MAP COMPLETE',key,method,run,flush=True)
        stats[key]=rows
    dump(OUT/'predictions_complete.json',dict(complete=True,protocol_sha256=sha(OUT/'protocol.json'),prepared_sha256=sha(OUT/'prepared.json'),files=files,stats=stats,sealed_before_gt_scoring=True))


def score():
    p=verify();seal=sealed('predictions_complete');assert not (OUT/'summary.json').exists();results={};means={m:[] for m in METHODS};gate=True;gained=lost=0
    for scene in p['scenes']:
        key=scene['key'];ref=np.load(scene['reference']);rows={}
        for method in METHODS:
            folder=OUT/'maps'/key/method;g=np.load(folder/'geometry.npz');objects=read(folder/'classification.json')['objects'];pred=transfer(g['xyz'],g['instance'],ref['xyz'],.2);result=evaluate(ref['train_instance'],ref['confidence'],pred,{o['object_id']:o['predictions']['mean'] for o in objects},EVALUATOR);dump(folder/'matches.json',clean_json(result['matches']));np.savez_compressed(folder/'reference_prediction.npz',instance=pred)
            rows[method]=dict(metrics=clean_json(result['averages']),matched_gt_ids=sorted(matched_ids(result['matches'])),run=read(folder/'run.json'));means[method].append(float(result['averages']['allAp']));print('FULL MAP SCORE',key,method,rows[method]['metrics'],flush=True)
        expected=read(ROOT/'build/benchmarks/signed_query_association_v1/summary.json')['results'][key]['methods']['geometry_graph']['metrics']
        for metric in ['allAp','allAp50%']:assert abs(rows['baseline']['metrics'][metric]-expected[metric])<1e-12
        base=set(rows['baseline']['matched_gt_ids'])
        for row in rows.values():ids=set(row['matched_gt_ids']);row['changes_vs_baseline']=dict(gained=sorted(ids-base),lost=sorted(base-ids))
        change=rows['ours_counterfactual']['changes_vs_baseline'];gained+=len(change['gained']);lost+=len(change['lost']);gate &= rows['ours_counterfactual']['metrics']['allAp']>=rows['baseline']['metrics']['allAp'];results[key]=dict(methods=rows)
    mean={m:float(np.mean(v)) for m,v in means.items()};gate &= mean['ours_counterfactual']>=mean['baseline']+.01 and mean['ours_counterfactual']>max(v for m,v in mean.items() if m!='ours_counterfactual') and gained>=1 and lost==0
    dump(OUT/'summary.json',dict(complete=True,results=results,mean_primary_ap=mean,advance_gate=bool(gate),gained=gained,lost=lost,protocol_sha256=sha(OUT/'protocol.json'),prediction_seal_sha256=sha(OUT/'predictions_complete.json'),official_leaderboard=False,independent_test=False));print('ADVANCE GATE',bool(gate),mean,flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','prepare','maps','score']);globals()[a.parse_args().action]()
