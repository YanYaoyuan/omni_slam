"""Bounded factorial diagnostic: adaptive versus frozen identity paths."""
import argparse,time
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from run_deferred_surface_support import OUT as PARENT,METHODS,verify as verify_parent,scene_arrays
from run_discovery_query_benchmark import OUT as FRONT,matched_ids

OUT=ROOT/'build/benchmarks/fixed_identity_recovery_control_v1'


def freeze():
    assert not OUT.exists();p=verify_parent(True)
    for name in ['supports','predictions']:
        for f,h in read(PARENT/(name+'_complete.json'))['files'].items():assert sha(f)==h,f
    sources=[__file__,ROOT/'scripts/benchmarks/fixed_identity_geometry_replay.py']
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),scenes=p['scenes'],methods=METHODS,
        sources={str(f):sha(f) for f in sources},parent_protocol_sha256=sha(PARENT/'protocol.json'),
        inputs={str(PARENT/(name+'.json')):sha(PARENT/(name+'.json')) for name in ['supports_complete','predictions_complete','summary']},
        hypothesis='Does the support-recovery regression remain when the entire baseline identity path is fixed?',
        control='Same exact support inputs for largest/all/pointwise/component, original native geometry kernels. Trace baseline ingestion, accepted observations, merge order, periodic/final denoise; replay with changed geometry but frozen ownership, rejection, retained objects, classes and scores. Ignore newly accepted/unowned observations as required by this intervention.',
        validation='Replay largest must match every recorded native merge/denoise cloud digest and final geometry bitwise in all3 scenes before any target score. Baseline trace capture also must reproduce old arrays and classification/graph.',
        scope='Causal diagnostic of the full identity path, not a pure one-line association intervention. No new proposal/threshold/model, no claim of online operation or object birth. A successful diagnostic is not a new method validation.',
        budget='3 baseline trace captures plus12 geometry replays;2 exposed outdoor windows and unlabelled robot. No new held-out data access.',
        decision='Report every arm. If frozen-identity component recovery still worsens original mean AP50:90 or AP50, identity freezing alone is insufficient; do not call the regression solved. Any later new mechanism requires separate design, controls and validation.'))
    print('FROZEN diagnostic',flush=True)


def verify(data=False):
    p=read(OUT/'protocol.json');verify_parent(data);assert sha(PARENT/'protocol.json')==p['parent_protocol_sha256']
    for f,h in {**p['inputs'],**p['sources']}.items():assert sha(f)==h,f
    if data:
        for name in ['supports','predictions']:
            for f,h in read(PARENT/(name+'_complete.json'))['files'].items():assert sha(f)==h,f
    return p


def inputs(key):
    r,m,f,frames,areas,classes,scores=scene_arrays(key);geometry=np.load(m['geometry']);tax=ROOT/'build/benchmarks/kitti360_development_v1/taxonomy'
    return r,geometry,f,frames,areas,classes,scores,np.load(tax/'text_features.npy'),read(tax/'vocabulary.json')


def trace():
    from fixed_identity_geometry_replay import capture
    p=verify(True);assert not (OUT/'trace_complete.json').exists();files={}
    for s in p['scenes']:
        key=s['key'];r,g,f,frames,areas,classes,scores,text,vocab=inputs(key)
        def loader(rec):return np.load(PARENT/'supports'/key/'largest'/f'{rec["frame_id"]:06d}.npz')
        folder=OUT/'traces'/key;events=capture(folder,g,r,loader,f,frames,areas,classes,scores,text,vocab)
        old=PARENT/'maps'/key/'largest';a=np.load(folder/'geometry.npz');b=np.load(old/'geometry.npz')
        for name in a.files:assert np.array_equal(a[name],b[name]),(key,name)
        for name in ['classification.json','nodes.json','proposals.json','graph.json','native_events.json']:assert read(folder/name)==read(old/name)
        for path in folder.iterdir():files[str(path)]=sha(path)
        print('TRACED',key,len(events),'operations; original reproduced',flush=True)
    dump(OUT/'trace_complete.json',dict(complete=True,files=files,protocol_sha256=sha(OUT/'protocol.json')))


def predict():
    from fixed_identity_geometry_replay import replay
    p=verify(True);seal=read(OUT/'trace_complete.json');assert not (OUT/'predictions_complete.json').exists();files={}
    for f,h in seal['files'].items():assert sha(f)==h
    for s in p['scenes']:
        key=s['key'];r,g,f,frames,areas,classes,scores,text,vocab=inputs(key)
        for method in METHODS:
            def loader(rec):return np.load(PARENT/'supports'/key/method/f'{rec["frame_id"]:06d}.npz')
            folder=OUT/'maps'/key/method
            replay(folder,g,r,loader,f,classes,text,vocab,OUT/'traces'/key,assert_native=method=='largest')
            for path in folder.iterdir():files[str(path)]=sha(path)
            print('REPLAYED',key,method,read(folder/'run.json'),flush=True)
    dump(OUT/'predictions_complete.json',dict(complete=True,files=files,protocol_sha256=sha(OUT/'protocol.json'),trace_seal_sha256=sha(OUT/'trace_complete.json')))


def score():
    from omni_semantic_map.kitti360_evaluation import evaluate
    from evaluate_kitti360_development import transfer,clean_json
    p=verify();seal=read(OUT/'predictions_complete.json');assert not (OUT/'summary.json').exists();results={};parent=read(PARENT/'summary.json');front=read(FRONT/'protocol.json')
    for f,h in seal['files'].items():assert sha(f)==h
    for s in p['scenes']:
        key=s['key'];methods={}
        if s['reference']:assert sha(s['reference'])==front['reference_hashes'][s['reference']];ref=np.load(s['reference'])
        for method in METHODS:
            folder=OUT/'maps'/key/method;run=read(folder/'run.json');row=dict(run=run)
            if s['reference']:
                g=np.load(folder/'geometry.npz');projected=transfer(g['xyz'],g['instance'],ref['xyz'],.2);cls=read(folder/'classification.json')['objects'];prediction={r['object_id']:r['predictions']['mean'] for r in cls}
                score=evaluate(ref['train_instance'],ref['confidence'],projected,prediction,ROOT/'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py')
                row['metrics']=clean_json(score['averages']);row['matched_gt_ids']=sorted(matched_ids(score['matches']));dump(folder/'matches.json',clean_json(score['matches']))
                if method=='largest':assert row['metrics']==parent['results'][key]['methods']['largest']['metrics']
                old=set(parent['results'][key]['methods']['largest']['matched_gt_ids']);new=set(row['matched_gt_ids']);row['changes_vs_original']=dict(gained=sorted(new-old),lost=sorted(old-new))
                row['adaptive_metrics']=parent['results'][key]['methods'][method]['metrics']
                print('SCORED',key,method,'fixed AP',row['metrics']['allAp'],'AP50',row['metrics']['allAp50%'],'adaptive AP',row['adaptive_metrics']['allAp'],flush=True)
            methods[method]=row
        results[key]=dict(methods=methods)
    dump(OUT/'summary.json',clean_json(dict(complete=True,results=results,diagnostic_only=True,
        prediction_seal_sha256=sha(OUT/'predictions_complete.json'),protocol_sha256=sha(OUT/'protocol.json'))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','trace','predict','score']);globals()[parser.parse_args().action]()
