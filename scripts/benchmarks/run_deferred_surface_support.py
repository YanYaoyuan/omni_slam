"""Frozen support-recovery control, using sealed SAM3/CG inputs unchanged."""
import argparse,time
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from run_discovery_query_benchmark import OUT as PARENT,INPUT,verify as verify_parent,matched_ids

OUT=ROOT/'build/benchmarks/deferred_surface_support_v1'
METHODS=['largest','all','pointwise','component']


def freeze():
    assert not OUT.exists();p=verify_parent(True)
    seals=[PARENT/(name+'_complete.json') for name in ['masks','features','supports','predictions']]
    for path in seals:
        for f,h in read(path)['files'].items():assert sha(f)==h,f
    sources=[__file__,ROOT/'omni_semantic_map/omni_semantic_map/deferred_surface_support.py',ROOT/'scripts/benchmarks/discovery_cg_backend.py']
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),scenes=p['scenes'],methods=METHODS,
        parent_protocol_sha256=sha(PARENT/'protocol.json'),input_seals={str(f):sha(f) for f in seals},source_hashes={str(f):sha(f) for f in sources},
        cell_size_m=.2,component_radius_m=.4,descriptor_cosine=.7,component_fraction=.5,
        scope='Previously exposed development windows0000/0002 and unlabelled robot2245. New reserved0004/0005 data remains unused. No new model inference or training.',
        inputs='All frames from frozen SAM3 car/building frontend. Same exclusive masks, SigLIP learned-merger features, measured geometry, native CG/history2 backend and semantic readout.',
        mechanism='Retain largest-component baseline supports. For every raw mask component, confirm a20cm cell only with a different-frame, same-class mask with descriptor cosine>=0.7. Recover the whole measured component when at least ceil(0.5*distinct cells) are confirmed. Earliest confirmation time is logged; repeated points within a cell do not increase evidence.',
        controls='Largest: exact previous all-frame connected baseline. All: every actual exclusive raw-mask support. Pointwise: largest plus only cells with compatible cross-view witnesses. Component: largest plus whole components confirmed by majority cells. Native16-point and3-observation persistence unchanged in every arm.',
        timing='Delayed window-end support revision followed by complete native CG replay. Future witnesses cannot be presented as available at the source frame. Not an online incremental implementation or real-time claim.',
        unknown='No mask support is treated as unknown, not as negative evidence. No synthetic depth/points, GT prompts, alternate image selection, post-hoc threshold search or semantic relabeling.',
        budget='12 maps:4 arms times3 scenes, identical60/48/330 image inputs. No parameter sweep.',
        primary='Native adapted KITTI allAp: AP over0.50..0.90 thresholds. Geometry recovery is assessed across IoUs; also report AP50, class breakdown and all per-target matched identities.',
        gate='Mean primary AP must improve by>=0.02 over BOTH largest and all. Primary AP and AP50 must not decrease on either outdoor window versus either control; no lost AP50-matched GT versus either. At least one gained unique AP50 GT versus largest across windows. Whole-component contribution additionally needs mean primary AP strictly greater than pointwise. Otherwise seal branch without threshold tuning.',
        novelty='Candidate support-recovery mechanism, not established novelty or advantage. All-component retention is an engineering control; multi-view confirmation alone is not claimed as new.'))
    print('FROZEN',flush=True)


def verify(data=False):
    p=read(OUT/'protocol.json');verify_parent(data);assert sha(PARENT/'protocol.json')==p['parent_protocol_sha256']
    for f,h in {**p['input_seals'],**p['source_hashes']}.items():assert sha(f)==h,f
    if data:
        for path in p['input_seals']:
            for f,h in read(path)['files'].items():assert sha(f)==h,f
    return p


def scene_arrays(key):
    records=read(PARENT/'masks_complete.json')['scenes'][key]['records'];manifest=read(INPUT/key/'manifest.json')
    features=[];frames=[];areas=[];classes=[];scores=[]
    for r in records:
        z=np.load(r['file']);features.append(np.load(PARENT/'frontend'/key/'features'/f'{r["frame_id"]:06d}.npy'))
        frames.extend([r['frame_id']]*r['masks']);areas.extend(z['areas']);classes.extend(z['semantic_ids']);scores.extend(z['scores'])
    return records,manifest,np.concatenate(features),np.asarray(frames),np.asarray(areas),np.asarray(classes),np.asarray(scores)


def prepare():
    from omni_semantic_map.deferred_surface_support import recover
    p=verify(True);assert not (OUT/'supports_complete.json').exists();files={};stats={}
    for s in p['scenes']:
        key=s['key'];records,manifest,f,frames,areas,classes,scores=scene_arrays(key);byframe={r['frame_id']:r for r in manifest['records']}
        xyz=np.load(manifest['geometry'])['xyz'];cells=np.load(INPUT/'query_cells'/key/'source_to_cell.npy');observations=[];start=time.monotonic()
        for r in records:
            measured=np.load(byframe[r['frame_id']]['file']);ids=measured['visible_source_ids'];px=measured['pixel_xy'];seg=np.load(r['file'])['segmentation'];labels=seg[px[:,1],px[:,0]]
            base=np.load(PARENT/'frontend'/key/'supports'/f'{r["frame_id"]:06d}.npz')
            for j in range(r['masks']):
                oid=r['observation_offset']+j;raw=ids[labels==j];baseline=base['point_ids'][base['offsets'][j]:base['offsets'][j+1]]
                assert np.isin(baseline,raw).all()
                observations.append(dict(id=oid,frame=r['frame_id'],semantic=int(classes[oid]),raw=raw,baseline=baseline))
        result,events=recover(observations,xyz,cells,f,radius=p['component_radius_m'],similarity=p['descriptor_cosine'],fraction=p['component_fraction'])
        recovery_seconds=time.monotonic()-start
        # Recompute two real prefixes: future observations must not accelerate
        # promotion of historical measured supports.
        prefix_checks=[]
        for rank in sorted(set([len(records)//3,2*len(records)//3])):
            cutoff=records[rank]['frame_id'];part=[o for o in observations if o['frame']<=cutoff]
            partial,_=recover(part,xyz,cells,f,radius=p['component_radius_m'],similarity=p['descriptor_cosine'],fraction=p['component_fraction'])
            for o in part:
                oid=o['id']
                for method in ['pointwise','component']:
                    full=result[oid][method][result[oid][method+'_times']<=cutoff]
                    assert np.array_equal(full,partial[oid][method]),(key,cutoff,oid,method)
            prefix_checks.append(dict(cutoff=cutoff,observations=len(part),equal=True))
        counts={m:0 for m in METHODS};promotions={m:0 for m in ['pointwise','component']};delay=[]
        for method in METHODS:
            folder=OUT/'supports'/key/method;folder.mkdir(parents=True,exist_ok=False)
            for r in records:
                ids=[r['observation_offset']+j for j in range(r['masks'])];pieces=[result[i][method] for i in ids]
                points=np.concatenate(pieces) if pieces else np.empty(0,np.int64);offsets=np.r_[0,np.cumsum([len(v) for v in pieces])]
                assert len(np.unique(points))==len(points)
                path=folder/f'{r["frame_id"]:06d}.npz';np.savez_compressed(path,point_ids=points,offsets=offsets);files[str(path)]=sha(path);counts[method]+=len(points)
                if method in promotions:
                    promotions[method]+=sum(len(result[i][method])-len(result[i]['largest']) for i in ids)
                    for i in ids:
                        assert np.isin(result[i]['largest'],result[i][method]).all() and np.isin(result[i][method],result[i]['all']).all()
            if method=='largest':
                for r in records:
                    a=np.load(folder/f'{r["frame_id"]:06d}.npz');b=np.load(PARENT/'frontend'/key/'supports'/f'{r["frame_id"]:06d}.npz')
                    assert np.array_equal(a['point_ids'],b['point_ids']) and np.array_equal(a['offsets'],b['offsets'])
        for e in events:assert e['promotion_frame']>=e['source_frame'];delay.append(e['promotion_frame']-e['source_frame'])
        path=OUT/'supports'/key/'events.json';dump(path,events);files[str(path)]=sha(path)
        stats[key]=dict(raw_observations=len(observations),point_observation_counts=counts,added_point_observations=promotions,
            component_promotions=len(events),promotion_delay_frame_ids=dict(max=max(delay,default=0),median=float(np.median(delay)) if delay else 0),
            recovery_seconds=recovery_seconds,prefix_checks=prefix_checks,baseline_support_exact=True)
        print('PREPARED',key,stats[key],flush=True)
    dump(OUT/'supports_complete.json',dict(complete=True,files=files,stats=stats,protocol_sha256=sha(OUT/'protocol.json')))


def predict():
    from discovery_cg_backend import build_map
    p=verify(True);seal=read(OUT/'supports_complete.json');assert not (OUT/'predictions_complete.json').exists();files={};replays={}
    for f,h in seal['files'].items():assert sha(f)==h
    tax=ROOT/'build/benchmarks/kitti360_development_v1/taxonomy';text=np.load(tax/'text_features.npy');vocab=read(tax/'vocabulary.json')
    for s in p['scenes']:
        key=s['key'];records,manifest,f,frames,areas,classes,scores=scene_arrays(key);geometry=np.load(manifest['geometry'])
        for method in METHODS:
            folder=OUT/'maps'/key/method
            def loader(r):return np.load(OUT/'supports'/key/method/f'{r["frame_id"]:06d}.npz')
            run=build_map(folder,geometry,records,loader,f,frames,areas,classes,scores,text,vocab)
            if method=='largest':
                old=PARENT/'maps'/key/'all_frames';a=np.load(folder/'geometry.npz');b=np.load(old/'geometry.npz')
                for name in a.files:assert np.array_equal(a[name],b[name]),(key,name)
                for name in ['classification.json','nodes.json','proposals.json','graph.json','native_events.json']:assert read(folder/name)==read(old/name),(key,name)
                replays[key]=dict(geometry_bitwise_equal=True,classification_and_graph_equal=True)
            for path in folder.iterdir():files[str(path)]=sha(path)
            print('MAPPED',key,method,run['objects'],flush=True)
    dump(OUT/'predictions_complete.json',dict(complete=True,files=files,baseline_replays=replays,
        protocol_sha256=sha(OUT/'protocol.json'),supports_seal_sha256=sha(OUT/'supports_complete.json')))


def score():
    from omni_semantic_map.kitti360_evaluation import evaluate
    from evaluate_kitti360_development import transfer,clean_json
    p=verify();parent=read(PARENT/'protocol.json');seal=read(OUT/'predictions_complete.json');assert not (OUT/'summary.json').exists();results={};sums={m:[] for m in METHODS};no_harm=True;gained=0
    for f,h in seal['files'].items():assert sha(f)==h
    for s in p['scenes']:
        key=s['key'];methods={};matches={}
        if s['reference']:assert sha(s['reference'])==parent['reference_hashes'][s['reference']];ref=np.load(s['reference'])
        for method in METHODS:
            folder=OUT/'maps'/key/method;run=read(folder/'run.json');methods[method]=dict(run=run)
            if not s['reference']:continue
            g=np.load(folder/'geometry.npz');projected=transfer(g['xyz'],g['instance'],ref['xyz'],.2);rows=read(folder/'classification.json')['objects']
            prediction={r['object_id']:r['predictions']['mean'] for r in rows}
            evaluated=evaluate(ref['train_instance'],ref['confidence'],projected,prediction,ROOT/'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py')
            dump(folder/'matches.json',clean_json(evaluated['matches']));matches[method]=matched_ids(evaluated['matches'])
            methods[method]['metrics']=clean_json(evaluated['averages']);methods[method]['matched_gt_ids']=sorted(matches[method]);sums[method].append(evaluated['averages']['allAp'])
            print('SCORED',key,method,'AP',evaluated['averages']['allAp'],'AP50',evaluated['averages']['allAp50%'],flush=True)
        changes={}
        if matches:
            for control in ['largest','all','pointwise']:
                changes[control]=dict(gained=sorted(matches['component']-matches[control]),lost=sorted(matches[control]-matches['component']))
                if control in ['largest','all']:
                    no_harm &= not changes[control]['lost'] and all(methods['component']['metrics'][metric]+1e-12>=methods[control]['metrics'][metric] for metric in ['allAp','allAp50%'])
            gained+=len(changes['largest']['gained'])
        results[key]=dict(methods=methods,changes_vs_controls=changes)
    means={m:float(np.mean(sums[m])) for m in METHODS};gate=bool(no_harm and gained>0 and all(means['component']-means[c]>=.02 for c in ['largest','all']) and means['component']>means['pointwise'])
    dump(OUT/'summary.json',clean_json(dict(complete=True,results=results,mean_primary_ap=means,advance_gate=gate,no_harm=bool(no_harm),gained_vs_largest=gained,
        prediction_seal_sha256=sha(OUT/'predictions_complete.json'),protocol_sha256=sha(OUT/'protocol.json'),development_only=True,robot_accuracy_available=False)))
    print('ADVANCE',gate,means,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','prepare','predict','score']);globals()[parser.parse_args().action]()
