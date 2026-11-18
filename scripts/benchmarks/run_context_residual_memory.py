"""Frozen development test of context separation, including robot replay.

Fixed existing object memberships/geometry; no learned fitting or GT in readout.
Historical replay is conditional on final associations, not causal SLAM.
"""
import argparse
import time
from pathlib import Path
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from omni_semantic_map.context_residual_memory import ContextObservation,ContextResidualMemory,unit
from omni_semantic_map.full_ovo_evaluation import BACKGROUND,voxel_metrics
from omni_semantic_map.mesh_instance_evaluation import transfer_labels
from omni_semantic_map.kitti360_evaluation import evaluate,kernel
from evaluate_kitti360_development import transfer,clean_json

OUT=ROOT/'build/benchmarks/context_residual_memory_v1'
METHODS=['learned','masked','bbox','fixed_context','object_context','shared_context']
META=ROOT/'build/datasets/kitti360_prepare'


def freeze():
    assert not OUT.exists()
    scenes=[]
    for suffix in ['0000','0002','0003']:
        key=f'2013_05_28_drive_{suffix}_sync'
        if suffix=='0000':
            r=ROOT/'build/benchmarks/kitti360_connected_development_v1'
            cache=r/'cache';maps=r/'maps/cg/history2';ev=r/'evaluation/reference.npz'
            front=ROOT/'build/benchmarks/kitti360_development_v1/frontend'/key
        else:
            r=ROOT/'build/benchmarks/kitti360_reserved_v1'/key
            cache=r/'connected/cache';maps=r/'base/maps/cg/history2';ev=r/'evaluation/reference.npz';front=r/'frontend'/key
        scenes.append(dict(key=key,kind='kitti360',cache=str(cache),fixed_map=str(maps),triplets=str(front/'triplets'),reference=str(ev)))
    old=read(ROOT/'build/benchmarks/vggt_fullmap_v1/protocol.json')
    s=next(s for s in old['scenes'] if s['key']=='scenenn_014')
    scenes.append(dict(key=s['key'],kind='scenenn',cache=s['cache'],triplets=str(Path(s['cache']).parent/'triplets'),
        fixed_map=str(ROOT/'build/benchmarks/vggt_fullmap_v1/maps/scenenn_014/vggt_single/history2'),reference=s['reference']))
    scenes.append(dict(key='rosbag2245',kind='robot',cache=str(ROOT/'build/benchmarks/evidence_retention_20261111/livo_full_strong_v2/cache_supportcc'),
        triplets=str(ROOT/'build/benchmarks/evidence_retention_20261111/livo_full_strong_v2/triplets'),
        fixed_map=str(ROOT/'build/experiments/garage_current_20261112/current'),reference=None))
    hashes={};refs={}
    for s in scenes:
        cache=Path(s['cache']);m=read(cache/'manifest.json');assert m['complete']
        paths=[cache/'manifest.json',cache/'text_features.npy',cache/'vocabulary.json']
        paths += [Path(s['fixed_map'])/f for f in ['geometry.npz','classification.json']]
        for rec in m['records']:
            f=cache/'frames'/rec['file'];assert sha(f)==rec['sha256']
            paths += [f,Path(s['triplets'])/rec['file']]
        for path in paths:hashes[str(path)]=sha(path)
        if s['reference']:refs[s['reference']]=sha(s['reference'])
    sources=[Path(__file__),ROOT/'omni_semantic_map/omni_semantic_map/context_residual_memory.py',
        ROOT/'omni_semantic_map/test/test_context_residual_memory.py',
        ROOT/'omni_semantic_map/omni_semantic_map/kitti360_evaluation.py',
        ROOT/'scripts/benchmarks/evaluate_kitti360_development.py',
        ROOT/'omni_semantic_map/omni_semantic_map/full_ovo_evaluation.py',
        ROOT/'omni_semantic_map/omni_semantic_map/mesh_instance_evaluation.py']
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),scenes=scenes,methods=METHODS,
        input_hashes=hashes,reference_hashes=refs,source_hashes={str(f):sha(f) for f in sources},
        mechanism='Within-object centered regression of learned descriptor on global descriptor, scalar beta pooled equally over objects with >=2 distinct frames; clamp beta to[0,1]. Object residual mean=mean learned-beta*mean global. Zero context variation yields beta0; degenerate residual falls back to original mean. Raw cached vectors retained, no GT.',
        methods_detail=dict(learned='Exact original unique-frame mean',masked='Existing masked crop mean',bbox='Existing bbox crop mean',
            fixed_context='learned mean - 0.5 global mean, fixed coefficient control',object_context='Fit beta separately per object',
            shared_context='Candidate: shared beta across objects; later evidence can revise previous object descriptors'),
        scope='Four already-exposed development scenes plus one330-frame robot cache. Fixed final memberships and geometry. Prefix replay uses final membership oracle only in the sense of future predicted associations, NOT GT; explicitly noncausal map-readout audit, not online association/SLAM or geometry revision.',
        budget='Six fixed readouts per scene, five scenes, no new neural inference/training/mapping; no parameter sweep. All outputs and history logs sealed before scoring this version.',
        gate='Advance only if shared_context beats the best of learned/masked/bbox/fixed_context by >=1 percentage point in pooled outdoor AP50 or indoor PQ, with no >0.5 point loss to the strongest simple control in the other domain. Also report object_context ablation, no selecting per-scene winners for output.',
        robot='Report historical class revisions, descriptor changes, foreground counts and replay cost only; no independent reference, no accuracy claim.',
        stop='If gate fails, freeze branch and report. No automatic new threshold or independent-data run. New mechanism/robot accuracy/generalization goal remains unmet.'))
    print('FROZEN',[(s['key'],s['kind']) for s in scenes],flush=True)


def verify(inputs=False):
    p=read(OUT/'protocol.json')
    for f,h in p['source_hashes'].items():assert sha(f)==h,f
    if inputs:
        for f,h in p['input_hashes'].items():assert sha(f)==h,f
    return p


def classify(descriptor,text,vocab,kind):
    scores=text@descriptor;order=np.argsort(-scores,kind='stable');best=int(order[0])
    name=vocab['names'][best];row=dict(name=name,score=float(scores[best]),margin=float(scores[best]-scores[int(order[1])]))
    if kind=='kitti360':
        semantic=int(vocab['semantic_ids'][best]);row.update(semantic_id=semantic,retained=semantic in vocab['target_semantic_ids'])
    else:
        semantic=int(vocab['replica_class_ids'][best]);row.update(replica_class_id=semantic,retained=semantic not in BACKGROUND)
    return row


def predict():
    p=verify(True);files={}
    assert not (OUT/'predictions_complete.json').exists()
    for s in p['scenes']:
        dest=OUT/s['key'];dest.mkdir(exist_ok=False);cache=Path(s['cache']);m=read(cache/'manifest.json')
        rows=read(Path(s['fixed_map'])/'classification.json')['objects']
        owners={int(i):r['object_id'] for r in rows for i in r['observation_ids']}
        assert len(owners)==sum(len(r['observation_ids']) for r in rows)
        text=np.load(cache/'text_features.npy').astype(np.float64);text/=np.linalg.norm(text,axis=1,keepdims=True)
        vocab=read(cache/'vocabulary.json');memory=ContextResidualMemory();locals_={};history=[];previous={};started=time.monotonic()
        for rec in m['records']:
            f=np.load(cache/'frames'/rec['file']);t=np.load(Path(s['triplets'])/rec['file'])['triplets']
            assert t.shape==(rec['masks'],3,f['features'].shape[-1])
            added=set()
            for j in range(rec['masks']):
                oid=rec['observation_offset']+j
                if oid not in owners:continue
                owner=owners[oid];added.add(owner)
                memory.add(ContextObservation(oid,rec['frame_id'],owner,float(f['areas'][j]),f['features'][j],t[j,0]))
                locals_[oid]=t[j,1:].astype(np.float64)
            result=memory.readout();current={};changed=[];descriptor_changes=0
            for oid,r in result['objects'].items():
                c=classify(r['descriptor'],text,vocab,s['kind']);current[oid]=dict(name=c['name'],descriptor=r['descriptor'])
                if oid in previous:
                    if not np.allclose(r['descriptor'],previous[oid]['descriptor'],rtol=0,atol=1e-12) and oid not in added:descriptor_changes+=1
                    if previous[oid]['name']!=c['name']:
                        changed.append(dict(object_id=oid,before=previous[oid]['name'],after=c['name'],received_new_observation=oid in added))
            history.append(dict(frame_id=rec['frame_id'],fit=result['fit'],class_changes=changed,
                historical_descriptor_changes_without_new_observation=descriptor_changes));previous=current
        outcomes=[];descriptors={method:[] for method in METHODS}
        result={method:memory.readout(mode) for method,mode in [('learned','none'),('fixed_context','fixed'),('object_context','local'),('shared_context','shared')]}
        for row in rows:
            oid=row['object_id'];selected=result['learned']['objects'][oid]['selected_observation_ids']
            assert selected==row['selected_observation_ids']
            r=dict(object_id=oid,observation_ids=row['observation_ids'],selected_observation_ids=selected,methods={})
            for method in METHODS:
                if method in result:descriptor=result[method]['objects'][oid]['descriptor']
                else:descriptor=unit(np.stack([locals_[i][0 if method=='masked' else 1] for i in selected]).mean(0))
                c=classify(descriptor,text,vocab,s['kind'])
                if method=='learned':
                    old=row['predictions']['mean'];assert c['name']==old['name'] and c['retained']==old['retained'];assert abs(c['score']-old['score'])<1e-6
                    c['score']=old['score']
                r['methods'][method]=c;descriptors[method].append(descriptor)
            r['shared_fit']=result['shared_context']['objects'][oid]['local_fit']
            r['shared_residual_norm']=result['shared_context']['objects'][oid]['residual_norm'];outcomes.append(r)
        np.savez_compressed(dest/'descriptors.npz',**{k:np.stack(v) for k,v in descriptors.items()})
        dump(dest/'readouts.json',outcomes);dump(dest/'history.json',history)
        dump(dest/'run.json',dict(complete=True,frames=len(m['records']),objects=len(rows),
            final_fit=result['shared_context']['fit'],wall_seconds=time.monotonic()-started,
            geometry_sha256=sha(Path(s['fixed_map'])/'geometry.npz'),noncausal_fixed_membership_replay=True,
            historical_class_revisions_without_new_observation=sum(not c['received_new_observation'] for h in history for c in h['class_changes'])))
        for f in dest.iterdir():files[str(f)]=sha(f)
        print('PREDICTED',s['key'],result['shared_context']['fit'],flush=True)
    dump(OUT/'predictions_complete.json',dict(complete=True,files=files,protocol_sha256=sha(OUT/'protocol.json')))


def rename(value,scene):
    if isinstance(value,list):return [rename(v,scene) for v in value]
    if isinstance(value,dict):return {k:(scene+'/'+v if k=='filename' else rename(v,scene)) for k,v in value.items()}
    return value


def score():
    p=verify();seal=read(OUT/'predictions_complete.json')
    for f,h in seal['files'].items():assert sha(f)==h,f
    assert not (OUT/'summary.json').exists()
    results={};pooled={method:{} for method in METHODS}
    for s in p['scenes']:
        dest=OUT/s['key'];rows=read(dest/'readouts.json');geometry=np.load(Path(s['fixed_map'])/'geometry.npz');methods={}
        if s['reference']:
            assert sha(s['reference'])==p['reference_hashes'][s['reference']];ref=np.load(s['reference'])
            if s['kind']=='kitti360':all_ids=transfer(geometry['xyz'],geometry['instance'],ref['xyz'],.2)
            else:
                valid=ref['valid'] if 'valid' in ref else np.ones(len(ref['instance']),bool)
                all_ids,_=transfer_labels(geometry['xyz'],geometry['instance'],ref['xyz'][valid],mode='nearest_5cm')
        for method in METHODS:
            pred={r['object_id']:r['methods'][method] for r in rows if r['methods'][method]['retained']}
            entry=dict(foreground_objects=len(pred),foreground_points=int(np.isin(geometry['instance'],list(pred)).sum()),
                class_changes=[dict(object_id=r['object_id'],before=r['methods']['learned']['name'],after=r['methods'][method]['name']) for r in rows if r['methods'][method]['name']!=r['methods']['learned']['name']])
            if s['reference']:
                projected=np.where(np.isin(all_ids,list(pred)),all_ids,0)
                if s['kind']=='kitti360':
                    result=evaluate(ref['train_instance'],ref['confidence'],projected,pred,META/'evalInstanceLevelSemanticLabeling.py')
                    entry['metrics']=clean_json(result['averages']);dump(dest/(method+'_matches.json'),clean_json(result['matches']))
                    pooled[method][s['key']]=rename(clean_json(result['matches'])['window'],s['key'])
                else:entry['metrics']=voxel_metrics(ref['instance'][valid],projected)
            methods[method]=entry
        results[s['key']]=dict(kind=s['kind'],methods=methods,run=read(dest/'run.json'))
        print('SCORED',s['key'],{k:((v.get('metrics') or {}).get('allAp50%',(v.get('metrics') or {}).get('observed_voxel_instance_pq'))) for k,v in methods.items()},flush=True)
    args,ns=kernel(META/'evalInstanceLevelSemanticLabeling.py');outdoor={}
    for method,matches in pooled.items():
        filenames=[r['filename'] for entry in matches.values() for rows in entry['prediction'].values() for r in rows];assert len(filenames)==len(set(filenames))
        outdoor[method]=clean_json(ns['computeAverages'](ns['evaluateMatches'](matches,args),args));dump(OUT/'pooled_matches'/(method+'.json'),matches)
    simple=['learned','masked','bbox','fixed_context'];indoor=results['scenenn_014']['methods']
    outdelta=outdoor['shared_context']['allAp50%']-max(outdoor[m]['allAp50%'] for m in simple)
    indelta=indoor['shared_context']['metrics']['observed_voxel_instance_pq']-max(indoor[m]['metrics']['observed_voxel_instance_pq'] for m in simple)
    gate=(outdelta>=.01 and indelta>=-.005) or (indelta>=.01 and outdelta>=-.005)
    dump(OUT/'summary.json',clean_json(dict(complete=True,results=results,outdoor_pooled=outdoor,
        advance_gate=bool(gate),candidate_delta_best_simple=dict(outdoor_ap50=outdelta,indoor_pq=indelta),
        development_only=True,robot_accuracy_available=False,protocol_sha256=sha(OUT/'protocol.json'),prediction_seal_sha256=sha(OUT/'predictions_complete.json'))))
    print('GATE',gate,'deltas',outdelta,indelta,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','predict','score']);a=ap.parse_args();globals()[a.action]()
