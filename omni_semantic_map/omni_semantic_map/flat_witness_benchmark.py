"""Fixed-input, scene-development experiment for co-visibility evidence policies."""
import argparse,json,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .flat_benchmark import frame_index,read_pose,sha256,evaluate
from .flat_dino_benchmark import load_cache,write_frame
from .witness_graph import WitnessConfig,WitnessGraph


def run(sequence,cache,output,policy='witness',appearance=True):
    ids,_=frame_index(sequence);meta,groups,hashes=load_cache(cache,sequence);source=Path(meta['source_run'])
    poses=np.array([read_pose(sequence/(f+'_pose.txt')) for f in ids])
    config=WitnessConfig(appearance_weight=.6 if appearance else 0.)
    tracker=WitnessGraph(config,policy)
    output.mkdir(parents=True,exist_ok=False);(output/'frames').mkdir()
    protocol=dict(complete=False,sequence=str(sequence.resolve()),method='witness_graph_'+policy,mode='predicted',
        config=asdict(config),conflict_policy=policy,appearance_enabled=appearance,depth_stride=4,frames_expected=len(ids),
        gt_segmentation_used_by_mapper=False,pose_source='provided poses',scope='causal instance output; no filtering',
        development_only=True,feature_cache_sha256=sha256(cache/'cache.json'),
        source_sha256={p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name('witness_graph.py'),Path(__file__).with_name('evidence_association.py'),Path(__file__).with_name('flat_dino_benchmark.py')]})
    (output/'run.json').write_text(json.dumps(protocol,indent=2));started=time.monotonic();emitted=set()
    for k,(frame,group) in enumerate(zip(ids,groups)):
        for obs in group:
            with np.load(source/obs['support_file']) as d:obs['support_xyz']=d['local_points'].copy()
        labels=tracker.add_frame(group,poses);emitted.update(labels)
        write_frame(output,frame,group,labels,source)
        if k%50==0:print(sequence.name,policy,appearance,k,len(np.unique(tracker.labels())),flush=True)
    labels=tracker.labels()
    (output/'assignments.json').write_text(json.dumps(labels.tolist()))
    (output/'revisions.json').write_text(json.dumps(tracker.events))
    (output/'input_sha256.json').write_text(json.dumps(hashes,indent=2))
    protocol.update(complete=True,frames_processed=len(ids),object_records=len(emitted),final_map_objects=len(np.unique(labels)),wall_s=time.monotonic()-started)
    (output/'run.json').write_text(json.dumps(protocol,indent=2))
    evaluate(sequence,output,output/'metrics.json')
    final=output/'final';(final/'frames').mkdir(parents=True,exist_ok=False)
    for frame,group in zip(ids,groups):write_frame(final,frame,group,[labels[o['observation_id']] for o in group],source)
    p=dict(protocol,method=protocol['method']+'_final',object_records=len(np.unique(labels)),scope='retrospective final assignments, not causal output')
    (final/'run.json').write_text(json.dumps(p,indent=2));(final/'input_sha256.json').write_text(json.dumps(hashes,indent=2))
    evaluate(sequence,final,final/'metrics.json')
    return protocol


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ['sequence','cache','output']:p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--policy',choices=['hard','off','witness'],default='witness');p.add_argument('--no-appearance',action='store_true')
    a=p.parse_args();print(json.dumps(run(a.sequence,a.cache,a.output,a.policy,not a.no_appearance),indent=2))

if __name__=='__main__':main()
