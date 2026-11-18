"""Bounded post-review interventions on exposed windows; immutable original maps.

No inference, fitting, threshold search, or new-scene claim. Freeze -> export all
maps -> score. Fixed baseline ID-to-GT correspondence is diagnostic only.
"""
import argparse,time,resource,platform
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from paper_final_common import ROOT,read,dump,sha
from omni_semantic_map.batch_stable_transport import nearest_owner,nearest_groups,native_library,certify
from omni_semantic_map.revision_authorization import margin_after_view_deletion,compatible_donors

SRC=ROOT/'build/benchmarks/revision_assessment_v1'
OUT=ROOT/'build/benchmarks/reviewer_causal_revision_20261117'
SCENES=['kitti360_0009','kitti360_0010']
NEW=['witness_history','witness_no_history','nearest_witness_history','ours_no_history','ours_unconditional_history']
OLD=['strong_baseline','unconditional','history_only','full_authorization']

def paths(k):return SRC/'comparison_maps'/k,SRC/'authorization/prepared'/k

def freeze():
    assert not OUT.exists()
    sources=[Path(__file__),ROOT/'scripts/benchmarks/evaluate_kitti360_development.py',ROOT/'scripts/benchmarks/run_discovery_query_benchmark.py',ROOT/'omni_semantic_map/omni_semantic_map/kitti360_evaluation.py',ROOT/'omni_semantic_map/omni_semantic_map/batch_stable_transport.py',ROOT/'omni_semantic_map/omni_semantic_map/batch_stable_transport.cpp',ROOT/'omni_semantic_map/omni_semantic_map/revision_authorization.py']
    inputs=[SRC/'summary.json',SRC/'per_object.json',SRC/'comparison_maps_complete.json',ROOT/'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py']
    for k in SCENES:
        maps,prep=paths(k)
        inputs += list(prep.glob('*'))+[SRC/'fields'/k/'field.npz',SRC/'evaluation'/k/'reference.npz']
        for m in OLD:inputs += [maps/m/'geometry.npz',maps/m/'classification.json',SRC/'evaluation'/k/m/'reference_prediction.npz']
    dump(OUT/'protocol.json',dict(date='2026-11-17',created_epoch=time.time(),scenes=SCENES,new_methods=NEW,reused_methods=OLD,
      sources={str(f):sha(f) for f in sources},inputs={str(f):sha(f) for f in inputs if f.is_file()},
      scope='Post-hoc interventions on both already-exposed 0009/0010 windows. Not an independent test or a new acceptance gate. Freeze all arms before exporting/scoring; preserve all outcomes.',
      budget='Exactly10newmaps,30newAPevaluations; original4methods reused per window. No model calls/training/threshold search.',
      interventions=dict(witness_history='Ours old-ID prefix; direct+witness additions with parent labels, no local competition/certificate.',witness_no_history='Identical witness additions/labels; baseline old-ID prefix.',nearest_witness_history='Ours old-ID prefix; direct+witness additions with unique nearest local seed identity, without two-frame-group stability certificate.',ours_no_history='Exact Ours new coordinates/labels; baseline old-ID prefix.',ours_unconditional_history='Exact Ours new coordinates/labels; unconditional parent old-ID prefix.'),
      fixed_correspondence='On canonical reference domain only, assign each baseline predictedID to same-class GT with largest confidence-weighted intersection; tie lowestGT; absent overlap unmapped. Many-to-one allowed to respect fragments. Freeze this map BEFORE diagnostics and reuse for every intervention. This is post-hoc reference-area evidence, NOT source-point accuracy, AP matching, or GT used by inference.',
      history_domain='Nearest baseline source ownership fixed at strict20cm. Restrict to references whose fixed source changed ID and whose GT and both endpoint correspondences are known. Correct-to-correct counts as same-object consolidation; wrong-to-correct correction; correct-to-wrong damage; wrong-to-wrong unresolved. Unmapped and void tracked separately, never silently counted correct.',
      identification='Full-vs-no-history holds appended sources/labels fixed; witness_history-vs-witness_no_history tests history at another support policy. Full-vs-nearest holds history fixed but changes added-point certification/retention. Full-vs-witness_history changes both local assignment and certificate. No independent benefit inferred solely from point counts.',
      profiling='Replay cached identity-pair decisions and graph certificate; report computation time separately from IO/export. CPU peakRSS is whole driver high-water mark, not isolated module memory or online systemFPS.'))
    print('FROZEN POST-REVIEW CAUSAL PROTOCOL',flush=True)

def verify():
    p=read(OUT/'protocol.json')
    for f,h in {**p['sources'],**p['inputs']}.items():assert sha(f)==h,f
    return p

def maps():
    p=verify();assert not (OUT/'maps_complete.json').exists();stats={};files={}
    lib=native_library(OUT/'native')
    for k in SCENES:
        wall=time.monotonic();mp,pp=paths(k);base=dict(np.load(mp/'strong_baseline/geometry.npz'));parent=dict(np.load(mp/'unconditional/geometry.npz'));ours=dict(np.load(mp/'full_authorization/geometry.npz'));n=len(base['xyz']);d=dict(np.load(pp/'write_decisions.npz'));f=dict(np.load(pp/'field.npz'));cell=np.load(SRC/'fields'/k/'field.npz')['source_to_cell'];source=parent['added_full_source_ids'];nodes=f['cell_atom'][cell[source]]
        np.testing.assert_array_equal(ours['xyz'][:n],base['xyz']);np.testing.assert_array_equal(ours['instance'][:n],d['authorized_history'])
        t=time.perf_counter();owner=nearest_owner(f['nearest_owner'],f['owner_distance']);near=np.full(len(source),-1,np.int64);near[d['direct']]=owner[nodes[d['direct']]];nearest_s=time.perf_counter()-t
        graph=csr_matrix((f['graph_weights'],f['graph_indices'],f['graph_indptr']),shape=(len(owner),len(owner)))
        t=time.perf_counter();oa,od=nearest_groups(graph,f['seeds'][:,0],f['seeds'][:,1],lib);tok,td=nearest_groups(graph,f['seeds'][:,0],np.unique(f['seeds'][:,[1,2]],axis=0,return_inverse=True)[1],lib);cert=certify(oa,od,tok,td,f['token_groups'][:,0]);graph_s=time.perf_counter()-t
        np.testing.assert_array_equal(oa,f['nearest_owner']);np.testing.assert_allclose(od,f['owner_distance'],atol=0,rtol=0);np.testing.assert_array_equal(cert,f['certified_owner'])
        ev=read(pp/'evidence.json');t=time.perf_counter()
        for v in ev.values():
            z=v['observations'];got=margin_after_view_deletion([x['frame_id'] for x in z],[x['shared'] for x in z],[x['separate'] for x in z]);assert got['approved']==v['approved']
        pair_s=time.perf_counter()-t;stats[k]={'profile':dict(nearest_cached_decision_s=nearest_s,graph_certificate_compute_s=graph_s,pair_evidence_compute_s=pair_s,cached_original_raw_evidence_prepare_s=read(pp/'run.json')['seconds'],graph_nodes=graph.shape[0],graph_directed_edges=graph.nnz,seed_rows=len(f['seeds']),identity_pairs=len(ev)),'methods':{}}
        for m in NEW:
            t=time.perf_counter()
            if m.startswith('ours_'):
                g={z:v.copy() for z,v in ours.items()};g['instance'][:n]=base['instance'] if m=='ours_no_history' else parent['instance'][:n]
            else:
                keep=d['direct']&d['witness'];labels=parent['instance'][n:]
                if m=='nearest_witness_history':keep&=near>=0;labels=near
                prefix=base['instance'] if m=='witness_no_history' else d['authorized_history']
                g=dict(xyz=np.r_[base['xyz'],parent['xyz'][n:][keep]],instance=np.r_[prefix,labels[keep]],added_full_source_ids=source[keep])
            compute=time.perf_counter()-t;dest=OUT/'maps'/k/m;dest.mkdir(parents=True,exist_ok=False);t=time.perf_counter();np.savez_compressed(dest/'geometry.npz',**g);dump(dest/'classification.json',read(mp/'full_authorization/classification.json'));export=time.perf_counter()-t
            assert np.array_equal(g['xyz'][:n],base['xyz']);assert np.isfinite(g['xyz']).all();assert np.all(g['instance']>=0)
            if m.startswith('ours_'):
                np.testing.assert_array_equal(g['xyz'],ours['xyz']);np.testing.assert_array_equal(g['instance'][n:],ours['instance'][n:])
            row=dict(points=len(g['xyz']),added_points=len(g['added_full_source_ids']),old_id_edits=int((g['instance'][:n]!=base['instance']).sum()),assembly_compute_s=compute,export_s=export)
            dump(dest/'run.json',row);stats[k]['methods'][m]=row
            for path in dest.iterdir():files[str(path)]=sha(path)
            print('EXPORTED',k,m,row,flush=True)
        stats[k]['driver_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;stats[k]['driver_wall_s']=time.monotonic()-wall
    dump(OUT/'maps_complete.json',dict(complete=True,new_maps=10,reference_labels_read_by_map_stage=False,protocol_sha256=sha(OUT/'protocol.json'),files=files,results=stats,host=dict(platform=platform.platform(),processor=platform.processor())))

def score():
    from evaluate_kitti360_development import transfer,clean_json
    from omni_semantic_map.kitti360_evaluation import evaluate
    from run_discovery_query_benchmark import matched_ids
    verify();seal=read(OUT/'maps_complete.json');assert seal['complete'];assert not (OUT/'summary.json').exists()
    for f,h in seal['files'].items():assert sha(f)==h
    old=read(SRC/'summary.json');results={};object_rows={};history={}
    for k in SCENES:
        mp,pp=paths(k);ref=dict(np.load(SRC/'evaluation'/k/'reference.npz'));gt=ref['raw_target_instance'];weight=ref['confidence'];rows=dict(old['results'][k]);preds={m:np.load(SRC/'evaluation'/k/m/'reference_prediction.npz')['instance'] for m in OLD};infos={}
        for m in OLD+NEW:
            folder=mp/m if m in OLD else OUT/'maps'/k/m;infos[m]={v['object_id']:v['predictions']['mean'] for v in read(folder/'classification.json')['objects']}
            if m in OLD:continue
            g=np.load(folder/'geometry.npz');t=time.monotonic();preds[m]=transfer(g['xyz'],g['instance'],ref['xyz'],.2);row={};dest=OUT/'evaluation'/k/m;dest.mkdir(parents=True,exist_ok=False);np.savez_compressed(dest/'reference_prediction.npz',instance=preds[m])
            for mode,truth,uniform in [('primary',ref['train_instance'],False),('raw_targets',gt,False),('uniform_scores',ref['train_instance'],True)]:
                info={o:dict(v,score=1.) for o,v in infos[m].items()} if uniform else infos[m];z=evaluate(truth,weight,preds[m],info,ROOT/'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py');matched=set(matched_ids(z['matches']));baseline=set(old['results'][k]['strong_baseline'][mode]['matched_gt_ids']);row[mode]=dict(metrics=clean_json(z['averages']),matched_gt_ids=sorted(matched),changes_vs_baseline=dict(gained=sorted(matched-baseline),lost=sorted(baseline-matched)))
            rows[m]=row;print('SCORED',k,m,100*row['primary']['metrics']['allAp'],'seconds',time.monotonic()-t,flush=True)
        # Full per-object same-class IoU, including uncovered eligible objects.
        mass=np.bincount(np.maximum(gt,0),weights=weight);ids=np.flatnonzero(np.isin(np.arange(len(mass))//1000,[2,13])&(mass>=100));objects=[]
        for gid in ids:
            obj=dict(gt_id=int(gid),methods={})
            for m in OLD+NEW:
                ppred=preds[m];tot=np.bincount(ppred,weights=weight,minlength=max(infos[m],default=0)+1);inter=np.bincount(ppred[gt==gid],weights=weight[gt==gid],minlength=len(tot));cand=np.array([o for o,v in infos[m].items() if v['train_id']==gid//1000],np.int64);ious=inter[cand]/np.maximum(1e-12,mass[gid]+tot[cand]-inter[cand]);j=int(ious.argmax()) if len(cand) else None;obj['methods'][m]=dict(object_id=int(cand[j]) if j is not None else None,best_iou=float(ious[j]) if j is not None else 0.)
            objects.append(obj)
        object_rows[k]=objects
        # Freeze baseline correspondence once. Never rematch per method or assign new IDs using outcomes.
        base=np.load(mp/'strong_baseline/geometry.npz');n=len(base['xyz']);nearest=transfer(base['xyz'],np.arange(n,dtype=np.int64)+1,ref['xyz'],.2)-1;valid=nearest>=0;np.testing.assert_array_equal(np.where(valid,base['instance'][np.maximum(nearest,0)],0),preds['strong_baseline'])
        mapping={};correspondence=[]
        for oid,meta in infos['strong_baseline'].items():
            target=(preds['strong_baseline']==oid)&(gt//1000==meta['train_id'])&(gt>0);hist=np.bincount(gt[target],weights=weight[target]);gid=int(hist.argmax()) if len(hist) and hist.max()>0 else 0;mapping[oid]=gid;correspondence.append(dict(predicted_id=int(oid),fixed_gt=gid,overlap_mass=float(hist[gid]) if gid else 0.))
        dump(OUT/'evaluation'/k/'fixed_correspondence.json',dict(rule='Baseline same-class max weighted intersection; many-to-one; no threshold',records=correspondence))
        hist_rows={};oldlabels=np.where(valid,base['instance'][np.maximum(nearest,0)],0);oldgt=np.array([mapping.get(int(o),0) for o in oldlabels]);oldcorrect=oldgt==gt
        for m in ['unconditional','full_authorization','ours_no_history','ours_unconditional_history','witness_history','witness_no_history']:
            path=mp/m if m in OLD else OUT/'maps'/k/m;g=np.load(path/'geometry.npz');prefix=g['instance'][:n];newlabel=np.where(valid,prefix[np.maximum(nearest,0)],0);newgt=np.array([mapping.get(int(o),0) for o in newlabel]);changed=valid&(newlabel!=oldlabels);known=changed&(gt>0)&(oldgt>0)&(newgt>0);newcorrect=newgt==gt
            masks={'correction':known&~oldcorrect&newcorrect,'damage':known&oldcorrect&~newcorrect,'same_object_consolidation':known&oldcorrect&newcorrect,'still_wrong':known&~oldcorrect&~newcorrect,'unknown_correspondence':changed&(gt>0)&((oldgt==0)|(newgt==0)),'void_reference':changed&(gt<=0)}
            assert np.array_equal(np.sum(np.stack(list(masks.values())),axis=0),changed.astype(int))
            cats={z:dict(reference_points=int(mask.sum()),confidence_mass=float(weight[mask].sum())) for z,mask in masks.items()};pairs=[]
            changed_sources=np.flatnonzero(prefix!=base['instance'])
            for a,b in np.unique(np.column_stack([base['instance'][changed_sources],prefix[changed_sources]]),axis=0):
                take=(base['instance'][changed_sources]==a)&(prefix[changed_sources]==b);ga,gb=mapping.get(int(a),0),mapping.get(int(b),0);pairs.append(dict(old_id=int(a),new_id=int(b),source_points=int(take.sum()),fixed_old_gt=ga,fixed_new_gt=gb,relation='unmapped' if not ga or not gb else 'same_GT_consolidation' if ga==gb else 'cross_GT_transition'))
            hist_rows[m]=dict(changed_source_points=len(changed_sources),changed_reference_points=int(changed.sum()),changed_confidence_mass=float(weight[changed].sum()),categories=cats,transitions=pairs)
        history[k]=hist_rows;results[k]=rows
    mean={mode:{m:float(np.mean([rows[m][mode]['metrics']['allAp'] for rows in results.values()])) for m in OLD+NEW} for mode in ['primary','raw_targets','uniform_scores']}
    contrasts={k:{'historical_edits_at_full_support_pp':100*(v['full_authorization']['primary']['metrics']['allAp']-v['ours_no_history']['primary']['metrics']['allAp']),'history_protection_at_full_support_pp':100*(v['full_authorization']['primary']['metrics']['allAp']-v['ours_unconditional_history']['primary']['metrics']['allAp']),'historical_edits_at_witness_support_pp':100*(v['witness_history']['primary']['metrics']['allAp']-v['witness_no_history']['primary']['metrics']['allAp']),'certificate_vs_nearest_pp':100*(v['full_authorization']['primary']['metrics']['allAp']-v['nearest_witness_history']['primary']['metrics']['allAp']),'local_policy_vs_witness_pp':100*(v['full_authorization']['primary']['metrics']['allAp']-v['witness_history']['primary']['metrics']['allAp'])} for k,v in results.items()}
    dump(OUT/'per_object.json',dict(complete=True,results=object_rows));dump(OUT/'history_diagnostics.json',dict(complete=True,results=history,scope=p['history_domain'] if (p:=read(OUT/'protocol.json')) else ''))
    dump(OUT/'summary.json',dict(complete=True,results=results,mean_ap=mean,contrasts=contrasts,new_maps=10,new_scores=30,post_hoc=True,independent_test=False,new_models=0,new_training=0,protocol_sha256=sha(OUT/'protocol.json'),maps_sha256=sha(OUT/'maps_complete.json')))
    print('CAUSAL AUDIT COMPLETE',mean['primary'],contrasts,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','maps','score']);globals()[p.parse_args().action]()
