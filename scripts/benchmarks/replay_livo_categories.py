"""Replay real atomic LIVO cached evidence through an isolated category stream."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from audit_evidence_retention import ROOT,OUT,dump
from public_dino_protocol import read
from omni_semantic_map.flat_benchmark import sha256
from omni_semantic_map.category_stream import CategoryStream
from omni_semantic_map.consensus_category import classify
from omni_semantic_map.shared_observations import observation_table,select_views
from omni_semantic_map.full_ovo_evaluation import BACKGROUND

DEST=OUT/'livo_category_stream_v1'
CACHE=OUT/'livo_full_strong_v2/cache_supportcc'


def main():
    import run_livo_scale_ablation as scale
    from omni_semantic_map.replay_fusion import ConceptGraphsBackend
    DEST.mkdir(exist_ok=False)
    features,frames,areas=observation_table(CACHE);text=np.load(CACHE/'text_features.npy');vocab=read(CACHE/'vocabulary.json')
    job=read(OUT/'jobs/livo_full_v2.json');capture=read(job['capture']);assert capture['complete'];byframe={r['frame_id']:r for r in capture['frames']}
    cm=read(CACHE/'manifest.json');assert cm['complete']
    model=ROOT/'build/benchmarks/ovo_full_20261110/model_artifacts.json'
    space=hashlib.sha256((sha256(model)+sha256(CACHE/'text_features.npy')+sha256(CACHE/'vocabulary.json')).encode()).hexdigest()
    stream=CategoryStream(text,vocab,space);text=stream.text
    source_paths=[Path(__file__).resolve(),ROOT/'omni_semantic_map/omni_semantic_map/category_stream.py',ROOT/'omni_semantic_map/omni_semantic_map/category_node.py',
        ROOT/'omni_semantic_map/omni_semantic_map/consensus_category.py',ROOT/'omni_semantic_map/omni_semantic_map/category_evidence.py']
    dump(DEST/'protocol.json',dict(scope=__doc__,feature_space=space,vocabulary_directory=str(CACHE),
        sources={str(p):sha256(p) for p in source_paths},capture_sha256=sha256(job['capture']),cache_sha256=sha256(CACHE/'manifest.json'),
        model_manifest_sha256=sha256(model),causal='Native sparse scale-adapted CG integration replay; categories see only received observations. Final offline history map is a separate last revision.',
        acceptance='No geometry/membership/retention changes caused by category consumer; exact replay of previous scale both map; prefix readout agrees with batch on that prefix.',
        limitations='Cached SAM2/SigLIP evidence from real atomic LIVO packets; not online inference or an independent semantic accuracy benchmark.'))
    trial=DEST/'mapper';trial.mkdir();(trial/'protocol.json').write_bytes((OUT/'scale_ablation_v1/protocol.json').read_bytes());scale.ROOT_OUT=trial
    original=ConceptGraphsBackend.integrate;revision=0;timings=[];comparisons=0
    events=(DEST/'events.jsonl').open('x');outputs=(DEST/'expected.jsonl').open('x')
    def dispatch(frame,new,objects,kind):
        nonlocal revision,comparisons
        event=dict(schema_version=1,session_id='rosbag2245_strong_category_v1',feature_space=space,revision=revision,frame_id=int(frame),observations=new,objects=objects,kind=kind)
        began=time.monotonic();out=stream.apply(event);timings.append(time.monotonic()-began)
        for src,dst in zip(objects,out['objects']):
            assert (src['id'],src['retained'],src['geometry_revision'])==(dst['id'],dst['retained'],dst['geometry_revision'])
            batch=classify(src['observation_ids'],features,frames,areas,text,vocab)
            assert dst['category']==batch
            assert all(frames[i]<=frame for i in dst['category']['selected_observation_ids'])
            comparisons+=1
        events.write(json.dumps(event,allow_nan=False)+'\n');outputs.write(json.dumps(out,allow_nan=False)+'\n');revision+=1
    def integrate(engine,observations,pose):
        result=original(engine,observations,pose)
        frame=observations[0].frame_id;packet=byframe[frame]
        incoming=[dict(id=int(o.observation_id),frame_id=int(frame),area=int(areas[o.observation_id]),feature=o.image_feature.tolist(),
            evidence=dict(rgb=packet['rgb'],packet=packet['packet'],keyframe_id=int(frame),image_stamp_ns=packet['image_stamp_ns'])) for o in observations]
        objects=[]
        for o in engine._objects:
            ids=list(map(int,o['observation_ids']));selected=select_views(ids,frames,areas,top_k=0)
            desc=features[selected].astype(np.float64).mean(0);desc/=np.linalg.norm(desc);best=int((text@desc).argmax())
            xyz=np.asarray(o['pcd'].points)
            objects.append(dict(id='object_'+str(min(ids)+1),observation_ids=ids,retained=int(vocab['replica_class_ids'][best]) not in BACKGROUND,
                geometry_revision=hashlib.sha256(xyz.tobytes()).hexdigest()))
        dispatch(frame,incoming,objects,'causal_keyframe');return result
    ConceptGraphsBackend.integrate=integrate
    try:scale.execute('cache_supportcc','both')
    finally:ConceptGraphsBackend.integrate=original
    # Assert that attaching the observer did not change the original mapper.
    expected=OUT/'scale_ablation_v1/cache_supportcc/both';actual=trial/'cache_supportcc/both'
    assert read(expected/'classification.json')==read(actual/'classification.json')
    a,b=np.load(actual/'geometry.npz'),np.load(expected/'geometry.npz')
    for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    final=OUT/'history_graph_v1/native_bridge_2';rows=read(final/'classification.json')['objects'];objects=[]
    for r in rows:
        ids=r['observation_ids'];objects.append(dict(id='object_'+str(min(ids)+1),observation_ids=ids,retained=r['predictions']['mean']['retained'],
            geometry_revision='offline_history_2:'+sha256(final/'geometry.npz')))
    dispatch(int(frames.max()),[],objects,'offline_history_finalization');events.close();outputs.close()
    changes=[]
    finalout=json.loads((DEST/'expected.jsonl').read_text().splitlines()[-1]);byid={o['id']:o for o in finalout['objects']}
    for r in rows:
        old=r['predictions']['mean'];new=byid['object_'+str(min(r['observation_ids'])+1)]['category']
        if old['name']!=new['name']:changes.append(dict(alias=min(r['observation_ids'])+1,before=old['name'],after=new['name'],retained=old['retained']))
    dump(DEST/'summary.json',dict(complete=True,no_gt=True,causal_keyframes=len(cm['records']),revisions=revision,observations=len(features),
        prefix_batch_comparisons=comparisons,mapper_geometry_membership_and_mean_exact=True,final_objects=len(rows),retained_objects=sum(o['retained'] for o in objects),
        category_changes=changes,category_processing_median_s=float(np.median(timings)),category_processing_p95_s=float(np.quantile(timings,.95)),
        timing_scope='CPU category processing only; excludes SAM2/SigLIP extraction, transport, mapper and independent verification.',
        events_sha256=sha256(DEST/'events.jsonl'),expected_sha256=sha256(DEST/'expected.jsonl')))
    print('LIVO CATEGORY REPLAY',revision,comparisons,changes,flush=True)


if __name__=='__main__':main()
