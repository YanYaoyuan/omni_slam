"""Exact full-window readout replay followed by fixed-output spatial credit.

No learned fit, new mapper, parameter search, new AP or independent test. Query
coordinates come from the existing evaluation domain; the soft operator is
training/diagnostic infrastructure, not a test-time GT-dependent predictor.
"""
import argparse,time
from pathlib import Path
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from evaluate_kitti360_development import transfer
from omni_semantic_map.spatial_acceptance_readout import build_readout,hard_readout,first_two_readouts,expected_labels

OUT=ROOT/'build/benchmarks/spatial_readout_audit_v1'
PARENT=ROOT/'build/benchmarks/object_context_acceptance_v1'
INPUT=ROOT/'build/benchmarks/discovery_measured_inputs_v1'
METHODS=['baseline','expanded_geometry','simple_context','category_context','ours_object_context']


def freeze():
    assert not OUT.exists();p=read(PARENT/'protocol.json');seal=read(PARENT/'predictions_complete.json');inputs={}
    for f,h in seal['files'].items():assert sha(f)==h,f;inputs[f]=h
    for f in [PARENT/'protocol.json',PARENT/'predictions_complete.json',PARENT/'artifact_audit.json',PARENT/'object_diagnostics.json']:
        inputs[str(f)]=sha(f)
    for scene in p['scenes']:
        key=scene['key'];manifest=read(INPUT/key/'manifest.json')
        for f in [scene['reference'],manifest['geometry'],INPUT/key/'manifest.json',PARENT/'features'/key/'candidates.npz',PARENT/'training_labels'/key/'TRAINING_ONLY.npz']:
            inputs[str(f)]=sha(f)
    sources=[Path(__file__),ROOT/'omni_semantic_map/omni_semantic_map/spatial_acceptance_readout.py',
             ROOT/'omni_semantic_map/test/test_spatial_acceptance_readout.py',ROOT/'scripts/benchmarks/evaluate_kitti360_development.py']
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),scenes=p['scenes'],input_hashes=inputs,source_hashes={str(f):sha(f) for f in sources},
         operator='Same fixed20cm nearest-map projection. Old XYZ always present. Per query sort all candidate sources strictly closer than nearest old point by distance then candidate append index. Old point wins equal distance; radius strictly less than0.2 as original cKDTree query. First accepted source wins, otherwise original map label or0if unobserved.',
         soft='Each source is independent categorical among object IDs and rejection. Expected label contribution=product of rejection probabilities of closer sources times current object probability; remaining probability falls back to old map. Exact marginal labels under this model, not exact expectation of nonlinear IoU. No new coordinates, radius, score threshold or geometry modification.',
         phases='Read only existing reference XYZ to build adjacency and reproduce all10frozen map projections. Then check0/1categorical probabilities against the entire Ours projection and half-acceptance probability normalization. Only after both scene replays are sealed, read reference labels for attribution.',
         attribution='Fixed Ours output. For every accepted source, compute label changes if ONLY that source is removed, using next accepted candidate or old map as backup. Keep original baseline best same-class eligibleGT per object; no IoU>0.5filter. Report exact per-object IoU retention effect, source supervision sign agreement, and whole-coalition vs sum of single-deletion effects. Unmatched objects have zero IoU target and are counted; their false-positive AP cost is NOT captured by this diagnostic.',
         verification='Full-map readout for all five existing conditions in both windows. Direct cKDTree projection after deleting first/last accepted source and the source changing most reference locations, at most3per window. These are diagnostic equality checks, not scored new method maps or a GT-selected repair.',
         budget='Existing two exposed windows only; no new inference/training, maps, scores, robot or independent data. Freeze before computation. No parameter sweep.',
         gate='Do not design map-level training until exact full-reference hard replay, deterministic endpoints, finite normalized soft outputs, and direct single-deletion checks pass. This gate establishes operator correctness only, not a successful method. Prior candidate failure gates stay closed.',
         limitation='Reference-domain spatial credit is benchmark-adapter dependent. A future training method would use labeled training scenes only, require equal-supervision controls and truly unused tests, and must not be sold as official full-input benchmark performance. Per-object IoU attribution is not AP attribution.'))
    print('FROZEN SPATIAL READOUT AUDIT',flush=True)


def verify():
    p=read(OUT/'protocol.json')
    for f,h in {**p['input_hashes'],**p['source_hashes']}.items():assert sha(f)==h,f
    return p


def replay():
    import torch
    p=verify();assert not (OUT/'replay_complete.json').exists();files={};stats={};torch.set_num_threads(1)
    for scene in p['scenes']:
        tick=time.monotonic();key=scene['key'];reference=np.load(scene['reference'])['xyz'];base=dict(np.load(PARENT/'maps'/key/'baseline/geometry.npz'));candidate=np.load(PARENT/'features'/key/'candidates.npz')['source_ids'];xyz=np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz']
        adjacency=build_readout(base['xyz'],base['instance'],xyz[candidate],reference,.2);folder=OUT/key;folder.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(folder/'adjacency.npz',**adjacency);replayed={}
        for method in METHODS:
            owner=np.zeros(len(candidate),np.int64) if method=='baseline' else np.load(PARENT/'maps'/key/method/'decisions.npz')['owners']
            g=dict(np.load(PARENT/'maps'/key/method/'geometry.npz'));result=hard_readout(adjacency,owner);actual=transfer(g['xyz'],g['instance'],reference,.2)
            np.testing.assert_array_equal(result,actual,err_msg=key+'/'+method+' projection mismatch')
            np.savez_compressed(folder/(method+'_projection.npz'),labels=result);replayed[method]=dict(reference_positions=len(reference),mismatches=0)
        owner=np.load(PARENT/'maps'/key/'ours_object_context/decisions.npz')['owners'];ids=owner[:,None];prob=torch.tensor((ids>0).astype(float));label_count=max(int(base['instance'].max()),int(owner.max()))+1
        max_error=0.;soft_min=1.;soft_max=0.;y=np.load(folder/'ours_object_context_projection.npz')['labels']
        for start in range(0,len(reference),20000):
            end=min(start+20000,len(reference));offset=adjacency['offsets'];lo,hi=offset[start],offset[end]
            sub=dict(offsets=offset[start:end+1]-lo,sources=adjacency['sources'][lo:hi],fallback=adjacency['fallback'][start:end],candidate_count=adjacency['candidate_count'])
            hard=expected_labels(sub,prob,ids,label_count).numpy()
            np.testing.assert_array_equal(hard.argmax(1),y[start:end]);np.testing.assert_array_equal(hard.max(1),1.)
            soft=expected_labels(sub,prob*.5,ids,label_count).numpy();error=float(np.max(np.abs(soft.sum(1)-1)))
            assert np.isfinite(soft).all() and soft.min()>=-1e-12 and soft.max()<=1+1e-12
            max_error=max(max_error,error);soft_min=min(soft_min,float(soft.min()));soft_max=max(soft_max,float(soft.max()))
        assert max_error<1e-12
        counts=np.diff(adjacency['offsets']);hist=np.bincount(adjacency['sources'],minlength=len(candidate))
        stat=dict(queries=len(reference),candidates=len(candidate),edges=len(adjacency['sources']),active_queries=int((counts>0).sum()),
                  max_candidates_per_query=int(counts.max()),candidate_sources_with_no_spatial_effect=int((hist==0).sum()),
                  candidate_reference_exposures_quantiles=np.quantile(hist,[0,.25,.5,.75,.9,.99,1]).tolist(),
                  replay=replayed,probability_endpoint_exact=True,half_acceptance_max_sum_error=max_error,soft_min=soft_min,soft_max=soft_max,seconds=time.monotonic()-tick)
        dump(folder/'replay.json',stat)
        for f in folder.iterdir():files[str(f)]=sha(f)
        stats[key]=stat;print('SPATIAL REPLAY',key,stat,flush=True)
    dump(OUT/'replay_complete.json',dict(complete=True,files=files,stats=stats,protocol_sha256=sha(OUT/'protocol.json'),reference_labels_read=False,new_map=False))


def iou_counts(labels,truth,weight,targets):
    n=len(targets);mass=np.bincount(labels,weights=weight,minlength=n)
    inter=np.bincount(labels,weights=weight*(truth==targets[labels]),minlength=n)
    return mass,inter


def attribute():
    p=verify();seal=read(OUT/'replay_complete.json');assert seal['complete'];assert not (OUT/'attribution_complete.json').exists()
    for f,h in seal['files'].items():assert sha(f)==h,f
    diagnostics=read(PARENT/'object_diagnostics.json')['results'];results={};files={}
    for scene in p['scenes']:
        tick=time.monotonic();key=scene['key'];folder=OUT/key;a=dict(np.load(folder/'adjacency.npz'));features=dict(np.load(PARENT/'features'/key/'candidates.npz'));decisions=dict(np.load(PARENT/'maps'/key/'ours_object_context/decisions.npz'));owner=decisions['owners'];count=len(owner)
        ref=dict(np.load(scene['reference']));truth=ref['train_instance'];w=ref['confidence'].astype(float);baseline=np.load(folder/'baseline_projection.npz')['labels'];current=np.load(folder/'ours_object_context_projection.npz')['labels']
        got,winner,backup=first_two_readouts(a,owner);np.testing.assert_array_equal(got,current)
        objects=diagnostics[key]['baseline'];oids=np.array([o['object_id'] for o in objects]);n=int(oids.max())+1;targets=np.full(n,-1,np.int64);mass_gt=np.bincount(truth,weights=w);gtmass=np.zeros(n)
        for o in objects:
            if o['overlaps']:targets[o['object_id']]=o['overlaps'][0]['gt_id'];gtmass[o['object_id']]=mass_gt[o['overlaps'][0]['gt_id']]
        pmass,intersection=iou_counts(current,truth,w,targets);bm,bi=iou_counts(baseline,truth,w,targets)
        iou=intersection/np.maximum(1e-12,gtmass+pmass-intersection);base_iou=bi/np.maximum(1e-12,gtmass+bm-bi)
        for method,calculated in [('baseline',base_iou),('ours_object_context',iou)]:
            for o in diagnostics[key][method]:
                oid=o['object_id'];expected=next((q['iou'] for q in o['overlaps'] if q['gt_id']==targets[oid]),0.)
                np.testing.assert_allclose(calculated[oid],expected,rtol=1e-10,atol=1e-12)
        change=(winner>=0)&(current!=backup);sid=winner[change];old=current[change];new=backup[change];ww=w[change];tt=truth[change]
        delta_mass=np.bincount(sid*n+old,weights=-ww,minlength=count*n)+np.bincount(sid*n+new,weights=ww,minlength=count*n)
        delta_inter=np.bincount(sid*n+old,weights=-ww*(tt==targets[old]),minlength=count*n)+np.bincount(sid*n+new,weights=ww*(tt==targets[new]),minlength=count*n)
        delta_mass=delta_mass.reshape(count,n);delta_inter=delta_inter.reshape(count,n)
        after_i=intersection+delta_inter;after_m=pmass+delta_mass;assert after_i[:,oids].min()>=-1e-7 and after_m[:,oids].min()>=-1e-7
        after_iou=after_i/np.maximum(1e-12,gtmass+after_m-after_i);effect=iou[None,:]-after_iou
        mean_effect=effect[:,oids].mean(1);changed_locations=np.bincount(sid,minlength=count);changed_mass=np.bincount(sid,weights=ww,minlength=count)
        won=winner>=0;winner_mass=np.bincount(winner[won],weights=w[won],minlength=count)
        y=dict(np.load(PARENT/'training_labels'/key/'TRAINING_ONLY.npz'));correct=((features['owners']==owner[:,None])&features['valid']&y['positive']).any(1);accepted=owner>0
        groups=dict(known_correct=accepted&y['eligible']&correct,known_wrong=accepted&y['eligible']&~correct,unknown=accepted&~y['eligible']);group_stats={}
        for name,mask in groups.items():
            values=mean_effect[mask];group_stats[name]=dict(sources=int(mask.sum()),helps=int((values>1e-12).sum()),harms=int((values<-1e-12).sum()),zero=int((np.abs(values)<=1e-12).sum()),
                 sources_changing_any_label=int((changed_locations[mask]>0).sum()),sum_individual_mean_iou_effect=float(values.sum()),changed_reference_weight=float(changed_mass[mask].sum()))
        assert sum(v['sources'] for v in group_stats.values())==accepted.sum()
        # Direct recomputation is an equality check, not a scored repair method.
        chosen=np.flatnonzero(accepted);chosen=sorted(set([int(chosen[0]),int(chosen[-1]),int(changed_locations.argmax())])) if len(chosen) else []
        oldmap=dict(np.load(PARENT/'maps'/key/'baseline/geometry.npz'));full=dict(np.load(PARENT/'maps'/key/'ours_object_context/geometry.npz'));accepted_indices=np.flatnonzero(accepted);checks=[]
        for i in chosen:
            if not accepted[i]:continue
            pos=len(oldmap['xyz'])+int(np.searchsorted(accepted_indices,i));keep=np.arange(len(full['xyz']))!=pos
            actual=transfer(full['xyz'][keep],full['instance'][keep],ref['xyz'],.2);want=current.copy();want[winner==i]=backup[winner==i]
            np.testing.assert_array_equal(actual,want)
            m,inter=iou_counts(actual,truth,w,targets);after=inter/np.maximum(1e-12,gtmass+m-inter)
            np.testing.assert_allclose(after[oids],after_iou[i,oids],rtol=1e-10,atol=1e-12)
            checks.append(dict(candidate_index=i,source_id=int(features['source_ids'][i]),changed_locations=int((actual!=current).sum()),exact=True))
        np.savez_compressed(folder/'source_effects.npz',source_ids=features['source_ids'],accepted_owner=owner,mean_retention_iou_effect=mean_effect,
                            per_object_retention_iou_effect=effect[:,oids],object_ids=oids,changed_locations=changed_locations,changed_reference_weight=changed_mass,winner_reference_weight=winner_mass)
        object_rows=[dict(object_id=int(o),target_gt=int(targets[o]),baseline_iou=float(base_iou[o]),current_iou=float(iou[o]),
                         coalition_iou_change=float(iou[o]-base_iou[o]),sum_single_retention_effect=float(effect[accepted,o].sum())) for o in oids]
        result=dict(groups=group_stats,objects=object_rows,unmatched_object_targets=int((targets[oids]<0).sum()),direct_deletion_checks=checks,
                    accepted_sources=int(accepted.sum()),accepted_without_any_label_effect=int((accepted&(changed_locations==0)).sum()),
                    whole_coalition_mean_iou_change=float((iou-base_iou)[oids].mean()),sum_single_retention_mean_iou_effect=float(mean_effect[accepted].sum()),seconds=time.monotonic()-tick)
        dump(folder/'attribution.json',result)
        for name in ['source_effects.npz','attribution.json']:f=folder/name;files[str(f)]=sha(f)
        results[key]=result;print('SPATIAL ATTRIBUTION',key,{k:v for k,v in result.items() if k!='objects'},flush=True)
    dump(OUT/'attribution_complete.json',dict(complete=True,results=results,files=files,protocol_sha256=sha(OUT/'protocol.json'),replay_seal_sha256=sha(OUT/'replay_complete.json'),
         gt_assisted_diagnostic=True,new_training=False,new_method_map=False,new_ap=False,
         scope='Single-source deletion effects conditioned on the complete frozen Ours map. Effects interact and cannot be summed as a global decomposition or used as a label-free predictor. Targets fixed to baseline best same-class eligibleGT, including weak matches. Unmatched object false-positive AP cost is not represented.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','replay','attribute']);globals()[parser.parse_args().action]()
