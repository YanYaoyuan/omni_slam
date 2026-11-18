"""Freeze, predict, and evaluate category evidence without changing map identity."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from audit_evidence_retention import ROOT,OUT,dump
from public_dino_protocol import DEST as DINO,PARENT,read,verify
from omni_semantic_map.flat_benchmark import sha256
from omni_semantic_map.category_evidence import METHODS,readout
from omni_semantic_map.shared_observations import select_views

DEST=OUT/'public_category_evidence_v1'


def freeze():
    assert not DEST.exists()
    parent=read(DINO/'protocol.json');verify(parent)
    scenes=parent['scenes']
    for e in scenes:
        key=e['key'];e['fixed_map']=str(PARENT/'runs'/key/'native_bridge_2')
        e['parent_evaluation']=str(DINO/'evaluation'/key)
        deps=[Path(e['fixed_map'])/f for f in ['geometry.npz','classification.json']]
        deps+=[Path(e['parent_evaluation'])/f for f in ['semantic_reference.json','metrics.json','native_bridge_2.npz']]
        deps+=[DINO/'features'/key/'manifest.json']
        e['readout_dependencies']={str(p):sha256(p) for p in deps}
    sources=[Path(__file__).resolve(),ROOT/'omni_semantic_map/omni_semantic_map/category_evidence.py',
             ROOT/'omni_semantic_map/test/test_category_evidence.py',
             ROOT/'omni_semantic_map/omni_semantic_map/semantic_instance_diagnostic.py']
    dump(DEST/'protocol.json',dict(scenes=scenes,methods=METHODS,
        source_sha256={**parent['source_sha256'],**{str(p):sha256(p) for p in sources}},
        dino_protocol_sha256=sha256(DINO/'protocol.json'),nyu_prediction_mapping=parent['nyu_prediction_mapping'],
        input='Exact final native_bridge_2 memberships, geometry, vocabulary, SigLIP features; one largest-area mask per frame with original ID ties.',
        quality='sqrt(raw mask area) * exclusive_area/raw_area. Exclusivity is same-frame proposal competition, NOT calibrated occlusion or visibility. All-zero quality falls back to uniform.',
        consensus='Median pairwise cosine excluding self. Retain centrality >= median centrality (all ties retained), quality-weight those views. Fewer than3 views or any missing feature retains quality weights. No thresholds fitted.',
        controls={'mean':'Exact original unique-frame descriptor mean','area':'Original area weights','exclusive_quality':'Quality only','quality_siglip':'Quality + SigLIP consensus','quality_dino':'Quality + frozen DINO consensus','score_median':'Coordinatewise median per-class score; not a descriptor-weighted mean'},
        primary='Fixed original foreground point assignments AND fixed geometric matches. All methods classify these same objects; predicted background/unknown category counts wrong, never removes an object.',
        secondary='Same geometry/membership, but update foreground filtering from new category, then transfer assigned points with original nearest<=5cm protocol. Report changes separately.',
        comparison='Compare quality_dino against BOTH quality_siglip and exclusive_quality; report all methods and all scenes, do not select per scene.',
        diagnostic='After all predictions: fixed-matched-object correct/wrong transitions, unexpressible GT classes, per-view GT opportunities, sufficient common-competitor convex-weighting impossibility certificate.',
        no_training=True,no_sweep=True,development_only=True,
        limits='All11 scenes already viewed, provided poses. No independent test claim or official macro semantic PQ. Existing area/density experiments preserved; SAM3 and training plans unchanged.'))
    for p in sources:
        target=DEST/'sources'/p.name;target.parent.mkdir(exist_ok=True);target.write_bytes(p.read_bytes())


def predict(key,spec):
    from omni_semantic_map.full_ovo_evaluation import BACKGROUND
    e=next(e for e in spec['scenes'] if e['key']==key);cache=Path(e['cache']);manifest=read(cache/'manifest.json')
    assert sha256(cache/'manifest.json')==e['hashes'][str(cache/'manifest.json')]
    for p,h in e['readout_dependencies'].items():assert sha256(p)==h
    for p,h in manifest['files_sha256'].items():assert sha256(cache/p)==h
    arrays={k:[] for k in ['features','areas','exclusive_areas']};frames=[]
    for r in manifest['records']:
        assert sha256(cache/'frames'/r['file'])==r['sha256']
        f=np.load(cache/'frames'/r['file']);assert len(frames)==r['observation_offset']
        for k in arrays:arrays[k].append(f[k])
        frames.extend([r['frame_id']]*r['masks'])
    a={k:np.concatenate(v) for k,v in arrays.items()};frames=np.array(frames)
    d=np.zeros((len(frames),384),np.float32);df=DINO/'features'/key;dm=read(df/'manifest.json')
    assert dm['complete'] and dm['protocol_sha256']==spec['dino_protocol_sha256']
    for r in dm['records']:
        assert sha256(df/r['file'])==r['sha256']
        values=np.load(df/r['file'])['mean'];lo=r['observation_offset'];d[lo:lo+len(values)]=values
    text=np.load(cache/'text_features.npy');text=text/np.linalg.norm(text,axis=1,keepdims=True)
    vocab=read(cache/'vocabulary.json');rows=read(Path(e['fixed_map'])/'classification.json')['objects'];predictions=[]
    for row in rows:
        ids=np.array(select_views(row['observation_ids'],frames,a['areas'],top_k=0))
        assert ids.tolist()==row['selected_observation_ids']
        result=dict(object_id=row['object_id'],selected_observation_ids=ids.tolist(),base_retained=row['predictions']['mean']['retained'],methods={})
        for method in METHODS:
            out=readout(a['features'][ids],a['areas'][ids],a['exclusive_areas'][ids],text,method=method,dino=d[ids])
            order=np.argsort(-out['scores'],kind='stable');best=int(order[0]);cls=int(vocab['replica_class_ids'][best])
            result['methods'][method]=dict(name=vocab['names'][best],replica_class_id=cls,retained=cls not in BACKGROUND,
                score=float(out['scores'][best]),margin=float(out['scores'][best]-out['scores'][order[1]]),
                weights=out['weights'].tolist(),fallback=out['fallback'],active_views=int((out['weights']>0).sum()),
                effective_sample_size=out['effective_sample_size'])
            if method=='mean':
                assert cls==row['predictions']['mean']['replica_class_id']
                np.testing.assert_allclose(out['scores'][best],row['predictions']['mean']['score'],rtol=0,atol=1e-12)
        predictions.append(result)
    out=DEST/'runs'/key;out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'view_scores.npz',scores=a['features'].astype(np.float64)@text.astype(np.float64).T,
                        frames=frames,areas=a['areas'],exclusive_areas=a['exclusive_areas'])
    dump(out/'predictions.json',dict(complete=True,no_gt=True,objects=predictions,protocol_sha256=sha256(DEST/'protocol.json')))
    dump(out/'run.json',dict(complete=True,no_gt=True,mean_classification_exact=True,
        fixed_geometry_sha256=sha256(Path(e['fixed_map'])/'geometry.npz'),
        files_sha256={p.name:sha256(p) for p in out.iterdir()}))
    print('PREDICTED',key,len(predictions),flush=True)


def evaluate(key,spec):
    from omni_semantic_map.semantic_instance_diagnostic import semantic_metrics
    from omni_semantic_map.full_ovo_evaluation import voxel_metrics
    from omni_semantic_map.mesh_instance_evaluation import transfer_labels
    e=next(e for e in spec['scenes'] if e['key']==key);run=DEST/'runs'/key;state=read(run/'run.json')
    for f,h in state['files_sha256'].items():assert sha256(run/f)==h
    for p,h in e['readout_dependencies'].items():assert sha256(p)==h
    rows=read(run/'predictions.json')['objects'];old=read(Path(e['parent_evaluation'])/'metrics.json')['methods']['native_bridge_2']
    ref=np.load(e['reference']);assert sha256(e['reference'])==e['hashes'][e['reference']]
    truth=ref['instance'];valid=ref['valid'] if 'valid' in ref.files else np.ones(len(truth),bool)
    reference=read(Path(e['parent_evaluation'])/'semantic_reference.json');gt={int(i):c for i,c in reference['gt_classes'].items()}
    sem_valid=valid&((truth==0)|np.isin(truth,list(gt)))
    fixed=np.load(Path(e['parent_evaluation'])/'native_bridge_2.npz')['instance']
    geometry=np.load(Path(e['fixed_map'])/'geometry.npz');methods={};changes=[]
    base_matches={m['gt_id']:m for m in old['semantic']['matches']}
    for name in METHODS:
        classes={r['object_id']:(r['methods'][name]['replica_class_id'] if e['dataset']=='Replica' else spec['nyu_prediction_mapping'].get(r['methods'][name]['name'],-1)) for r in rows}
        primary=semantic_metrics(truth[sem_valid],fixed[sem_valid],gt,classes)
        assert [(m['gt_id'],m['predicted_id'],m['iou']) for m in primary['matches']]==[(m['gt_id'],m['predicted_id'],m['iou']) for m in old['semantic']['matches']]
        if name=='mean':assert primary==old['semantic']
        for m in primary['matches']:
            b=base_matches[m['gt_id']]
            if m['class_correct']!=b['class_correct']:changes.append(dict(method=name,gt_id=m['gt_id'],object_id=m['predicted_id'],before=b,after=m))
        keep=[r['object_id'] for r in rows if r['methods'][name]['retained']]
        original_keep=[r['object_id'] for r in rows if r['base_retained']]
        if keep==original_keep:projected=fixed
        else:
            labels=np.where(np.isin(geometry['instance'],keep),geometry['instance'],0)
            projected,_=transfer_labels(geometry['xyz'],labels,ref['xyz'],mode='nearest_5cm')
        instance=voxel_metrics(truth[valid],projected[valid]);semantic=semantic_metrics(truth[sem_valid],projected[sem_valid],gt,classes)
        if name=='mean':assert instance['observed_voxel_instance_pq']==old['instance']['observed_voxel_instance_pq']
        methods[name]=dict(fixed_foreground=primary,refiltered=dict(instance=instance,semantic=semantic),
            removed_foreground=sorted(set(original_keep)-set(keep)),added_foreground=sorted(set(keep)-set(original_keep)),
            fallbacks=dict(Counter(r['methods'][name]['fallback'] for r in rows)))
    # Post-hoc sufficient certificate: a single competing class beats every
    # GT-equivalent text vector in every view. No nonnegative view weighting can fix it.
    scores=np.load(run/'view_scores.npz')['scores'];vocab=read(Path(e['cache'])/'vocabulary.json')
    category=np.array(vocab['replica_class_ids'] if e['dataset']=='Replica' else [spec['nyu_prediction_mapping'].get(n,-1) for n in vocab['names']])
    byid={r['object_id']:r for r in rows};diagnostics=[]
    for match in old['semantic']['matches']:
        row=byid[match['predicted_id']];vs=scores[row['selected_observation_ids']];gtcols=category==match['gt_class']
        item=dict(gt_id=match['gt_id'],object_id=match['predicted_id'],gt_class=match['gt_class'],baseline_correct=match['class_correct'],
                  baseline_name=row['methods']['mean']['name'],views=len(vs),expressible=bool(gtcols.any()))
        if gtcols.any():
            target=vs[:,gtcols].max(1);dominators=np.flatnonzero((vs>target[:,None]+1e-12).all(0)&~gtcols)
            item.update(any_single_view_correct=bool((category[vs.argmax(1)]==match['gt_class']).any()),
                        correct_single_views=int((category[vs.argmax(1)]==match['gt_class']).sum()),
                        common_dominators=[vocab['names'][i] for i in dominators],convex_reweighting_blocked=bool(len(dominators)))
        diagnostics.append(item)
    out=DEST/'evaluation'/key;out.mkdir(parents=True,exist_ok=False)
    dump(out/'metrics.json',dict(complete=True,key=key,dataset=e['dataset'],methods=methods,changes=changes,
        diagnostic=diagnostics,baseline_reproduced=True,fixed_matches_verified=True,
        run_sha256=sha256(run/'run.json'),protocol_sha256=sha256(DEST/'protocol.json')))
    print('EVALUATED',key,{m:(v['fixed_foreground']['correct_class_matches'],round(v['fixed_foreground']['class_aware_pq']*100,3)) for m,v in methods.items()},flush=True)


def summarize(spec):
    scenes=[read(DEST/'evaluation'/e['key']/'metrics.json') for e in spec['scenes']];datasets={}
    for ds in ['Replica','SceneNN']:
        subset=[s for s in scenes if s['dataset']==ds];methods={}
        for name in METHODS:
            values=[s['methods'][name] for s in subset]
            delta=np.array([100*(s['methods'][name]['fixed_foreground']['class_aware_pq']-s['methods']['mean']['fixed_foreground']['class_aware_pq']) for s in subset])
            methods[name]=dict(fixed_class_pq100=float(np.mean([v['fixed_foreground']['class_aware_pq'] for v in values])*100),
                delta_fixed_class_pq100=float(delta.mean()),wins=int((delta>1e-8).sum()),ties=int((abs(delta)<=1e-8).sum()),losses=int((delta< -1e-8).sum()),
                fixed_correct=sum(v['fixed_foreground']['correct_class_matches'] for v in values),
                fixed_matches=sum(v['fixed_foreground']['geometry_matches'] for v in values),
                class_gains=sum(c['after']['class_correct'] for s in subset for c in s['changes'] if c['method']==name),
                class_losses=sum(c['before']['class_correct'] for s in subset for c in s['changes'] if c['method']==name),
                refiltered_instance_pq100=float(np.mean([v['refiltered']['instance']['observed_voxel_instance_pq'] for v in values])*100),
                refiltered_class_pq100=float(np.mean([v['refiltered']['semantic']['class_aware_pq'] for v in values])*100),
                refiltered_correct=sum(v['refiltered']['semantic']['correct_class_matches'] for v in values),
                refiltered_matches=sum(v['refiltered']['semantic']['geometry_matches'] for v in values))
        wrong=[d for s in subset for d in s['diagnostic'] if not d['baseline_correct']]
        datasets[ds]=dict(methods=methods,baseline_wrong=len(wrong),unexpressible=sum(not d['expressible'] for d in wrong),
            convex_blocked=sum(d.get('convex_reweighting_blocked',False) for d in wrong),
            single_view_opportunity=sum(d.get('any_single_view_correct',False) for d in wrong))
    dump(DEST/'summary.json',dict(complete=True,datasets=datasets,scenes=scenes,protocol_sha256=sha256(DEST/'protocol.json')))
    print(datasets,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--predict',action='store_true');p.add_argument('--evaluate',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    spec=read(DEST/'protocol.json');verify(spec)
    if a.predict:
        for e in spec['scenes']:predict(e['key'],spec)
        dump(DEST/'predictions_complete.json',dict(complete=True,files={e['key']:sha256(DEST/'runs'/e['key']/'run.json') for e in spec['scenes']}))
    if a.evaluate:
        gate=read(DEST/'predictions_complete.json');assert gate['complete']
        for key,h in gate['files'].items():assert sha256(DEST/'runs'/key/'run.json')==h
        for e in spec['scenes']:evaluate(e['key'],spec)
        summarize(spec)
