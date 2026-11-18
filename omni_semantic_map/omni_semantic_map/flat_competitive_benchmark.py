"""Frozen visibility-ownership method on Flat; GT only used after mapping."""
import argparse,json,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from PIL import Image
from .flat_benchmark import frame_index,read_pose,sha256,camera_points,evaluate
from .flat_dino_benchmark import load_cache,write_frame
from .competitive_graph import CompetitiveGraph,CompetitiveConfig


def run(sequence,cache,output):
    ids,_=frame_index(sequence);meta,groups,hashes=load_cache(cache,sequence)
    source=Path(meta['source_run']);poses=np.array([read_pose(sequence/(f+'_pose.txt')) for f in ids])
    views=[]
    for frame,group in zip(ids,groups):
        depth=np.asarray(Image.open(sequence/(frame+'_depth.tiff')))
        _,valid,_,_=camera_points(depth,4)
        owner=np.full(valid.shape,-1,np.int32)
        for o in group:
            with np.load(source/o['support_file']) as f:
                o['support_xyz']=f['local_points'].copy();uv=f['pixels']
                if np.any(owner[uv[:,1]//4,uv[:,0]//4]>=0):raise ValueError('non-disjoint source masks')
                owner[uv[:,1]//4,uv[:,0]//4]=o['observation_id']
        views.append(dict(owner=owner,depth=depth[::4,::4],valid=valid,stride=4,
                          intrinsic=np.array([[320.,0,320.],[0,320.,240.],[0,0,1.]])))
    output.mkdir(parents=True,exist_ok=False);(output/'frames').mkdir()
    cfg=CompetitiveConfig();t=CompetitiveGraph(views,cfg)
    protocol=dict(complete=False,sequence=str(sequence.resolve()),method='competitive',mode='predicted',
        config=asdict(cfg),depth_stride=4,frames_expected=len(ids),gt_segmentation_used_by_mapper=False,
        pose_source='provided poses',scope='causal output',development_only=True,
        feature_cache_sha256=sha256(cache/'cache.json'),source_sha256={p.name:sha256(p) for p in
            [Path(__file__),Path(__file__).with_name('competitive_graph.py'),Path(__file__).with_name('ownership_graph.py')]})
    (output/'run.json').write_text(json.dumps(protocol,indent=2));start=time.monotonic();emitted=set()
    for k,(frame,group) in enumerate(zip(ids,groups)):
        labels=t.add_frame(group,poses);emitted.update(labels);write_frame(output,frame,group,labels,source)
        if k%50==0:print(sequence.name,k,len(np.unique(t.labels())),flush=True)
    labels=t.labels();(output/'assignments.json').write_text(json.dumps(labels.tolist()))
    (output/'revisions.json').write_text(json.dumps(t.events));(output/'input_sha256.json').write_text(json.dumps(hashes,indent=2))
    protocol.update(complete=True,frames_processed=len(ids),object_records=len(emitted),final_map_objects=len(np.unique(labels)),wall_s=time.monotonic()-start)
    (output/'run.json').write_text(json.dumps(protocol,indent=2));evaluate(sequence,output,output/'metrics.json')
    final=output/'final';(final/'frames').mkdir(parents=True)
    for frame,group in zip(ids,groups):write_frame(final,frame,group,[labels[o['observation_id']] for o in group],source)
    (final/'run.json').write_text(json.dumps(dict(protocol,method='competitive_final',object_records=len(np.unique(labels)),scope='retrospective final assignments'),indent=2))
    (final/'input_sha256.json').write_text(json.dumps(hashes,indent=2));evaluate(sequence,final,final/'metrics.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['sequence','cache','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.sequence,a.cache,a.output)

if __name__=='__main__':main()
