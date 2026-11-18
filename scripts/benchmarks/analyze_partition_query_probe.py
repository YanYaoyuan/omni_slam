"""GT-free probe readout; no acceptance decision, training or map alteration."""
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from omni_semantic_map.query_hypothesis_evidence import membership,retrieve

OUT=ROOT/'build/benchmarks/partition_query_probe_v1'


def unpack(d, selected=None):
    packed=d['masks_packed'] if selected is None else d['masks_packed'][selected:selected+1]
    return np.unpackbits(packed,axis=2)[:,:,:int(d['original_size'][0])].astype(bool)


def main():
    p=read(OUT/'protocol.json');done=read(OUT/'predictions_complete.json');inputs=read(OUT/'verification_inputs.json')
    assert done['protocol_sha256']==sha(OUT/'protocol.json') and done['execution_sha256']==sha(OUT/'execution.json')
    for f,h in {**done['files'],**inputs['files']}.items():assert sha(f)==h,f
    acquired={(r['key'],tuple(r['object_ids'])):r for r in done['acquisition']}
    verification={(r['key'],r['frame_id'],r['semantic_id']):r for r in done['verification']}
    views={(r['key'],r['frame_id']):r for r in inputs['records']}
    rows=[]
    for pair in p['pairs']:
        if pair['status']!='ready':continue
        key=pair['key'];row=dict(key=key,object_ids=pair['object_ids'],acquisition_frame=pair['acquisition_frame'],prompts={},verification=[])
        cell=np.load(ROOT/'build/benchmarks/discovery_measured_inputs_v1/query_cells'/key/'source_to_cell.npy');count=int(cell.max())+1
        src=dict(np.load(pair['source_sets']));sets=[np.unique(cell[src[k]]) for k in ['a','b']]
        view=dict(np.load(pair['acquisition_view']));xy=view['pixel_xy'];sourceids=view['visible_source_ids']
        for query in acquired[(key,tuple(pair['object_ids']))]['queries']:
            d=dict(np.load(query['file']));mask=unpack(d,query['selected'])[0]
            source=sourceids[mask[xy[:,1],xy[:,0]]];sets.append(np.unique(cell[source]))
            points=d['point_coords'].astype(int);actual=mask[points[:,1],points[:,0]]
            row['prompts'][query['name']]=dict(selected_quality=float(d['scores'][query['selected']]),measured_sources=len(source),cells=len(sets[-1]),
                prompt_labels_satisfied=bool(np.array_equal(actual,d['point_labels'].astype(bool))),
                original_a_coverage=float(np.isin(src['a'],source).sum()/max(1,np.isin(src['a'],sourceids).sum())),
                original_b_coverage=float(np.isin(src['b'],source).sum()/max(1,np.isin(src['b'],sourceids).sum())))
        names=['original_a','original_b','split_a','split_b','joint_1','joint_2'];h=membership(sets,count)
        for frame in pair['verification_frames']:
            rec=views[(key,frame)];view=dict(np.load(rec['file']));data=dict(np.load(verification[(key,frame,pair['semantic_id'])]['file']))
            masks=unpack(data);xy=view['pixel_xy'];visible=cell[view['visible_source_ids']]
            supports=[np.unique(visible[m[xy[:,1],xy[:,0]]]) for m in masks];q=membership(supports,count)
            answer=retrieve(h,np.full(len(sets),pair['semantic_id']),q,np.full(len(supports),pair['semantic_id']),data['scores'],np.unique(visible))
            result=dict(frame_id=frame,readout={name:dict(query=int(answer['query'][j]),agreement=float(answer['agreement'][j]),native_score=float(answer['score'][j]),visible_cells=int(answer['visible_cells'][j])) for j,name in enumerate(names)})
            for title,i,j in [('original_pair',0,1),('prompted_split',2,3)]:
                qa,qb=int(answer['query'][i]),int(answer['query'][j]);available=min(answer['visible_cells'][[i,j]])>=16 and min(qa,qb)>=0
                intersection=len(np.intersect1d(supports[qa],supports[qb])) if min(qa,qb)>=0 else 0
                union=len(np.union1d(supports[qa],supports[qb])) if min(qa,qb)>=0 else 0
                result[title]=dict(measured16_each=bool(available),same_query=bool(qa==qb) if min(qa,qb)>=0 else None,
                    query_mask_cell_iou=float(intersection/max(1,union)) if min(qa,qb)>=0 else None)
            row['verification'].append(result)
        rows.append(row)
        print('READOUT',key,pair['object_ids'],'prompts',row['prompts'],flush=True)
        for r in row['verification']:print('VIEW',r['frame_id'],r['original_pair'],r['readout'],flush=True)
    dump(OUT/'readout.json',dict(complete=True,post_prediction_analysis=True,gt_read=False,maps_changed=False,
        protocol_sha256=sha(OUT/'protocol.json'),predictions_sha256=sha(OUT/'predictions_complete.json'),source_sha256=sha(__file__),results=rows,
        limitations='Diagnostic readout only. No new acceptance thresholds, partition selection or AP claim. Query IDs may duplicate an object, so report overlap as well; differing query IDs alone do not establish two objects. Nearby unused frames remain temporally correlated.'))


if __name__=='__main__':main()
