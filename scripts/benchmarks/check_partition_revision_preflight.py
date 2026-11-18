"""Check frozen candidate coverage and replay revision selection on old probe.

This is mechanism preflight, not a new map experiment or independent test.
"""
import hashlib
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from prepare_partition_revision import OUT
from analyze_partition_query_probe import OUT as PROBE,unpack
from omni_semantic_map.partition_revision import select_partition


def main():
    seal=read(OUT/'candidates_complete.json');protocol=read(OUT/'candidate_protocol.json')
    assert seal['complete'] and seal['protocol_sha256']==sha(OUT/'candidate_protocol.json')
    for f,h in {**seal['files'],**protocol['sources']}.items():assert sha(f)==h,f
    coverage=[]
    for key in seal['stats']:
        selected=read(OUT/key/'selected.json');groups=read(OUT/key/'groups.json')
        allnodes=[j for g in groups for j in g['node_ids']];assert len(allnodes)==len(set(allnodes)) and sorted(allnodes)==list(range(len(allnodes)))
        chosen=[g for row in selected for g in row['groups']];assert len(chosen)==len(set(chosen))
        for row in selected:
            ours=max(row['actions'],key=lambda a:(a['utility'],-a['frame_id']))
            visible=max(row['actions'],key=lambda a:(a['min_visible'],-a['frame_id']))
            digest=hashlib.sha256(f'{key}:{row["groups"][0]}:{row["groups"][1]}:0'.encode()).digest()
            random=row['actions'][int.from_bytes(digest[:8],'little')%len(row['actions'])]
            assert row['policies']==dict(ours=ours,max_visible=visible,random=random)
        coverage.append(dict(key=key,selected=len(selected),single_eligible_view=sum(len(r['actions'])==1 for r in selected),
            ours_differs_from_maxvisible=sum(r['policies']['ours']['frame_id']!=r['policies']['max_visible']['frame_id'] for r in selected),
            random_differs_from_maxvisible=sum(r['policies']['random']['frame_id']!=r['policies']['max_visible']['frame_id'] for r in selected)))
    dump(OUT/'control_identifiability.json',dict(complete=True,scenes=coverage,selection_sha256=sha(OUT/'candidates_complete.json'),
        decision='Drop active-view-choice claim and redundant view-policy map arms before GPU/scoring. Use common maximum-visibility acquisition views; compare ordinary/symmetric prompts, verification, and historical revision. Do not change frozen candidate pool or search for views that favor Ours.'))
    p=read(PROBE/'protocol.json');done=read(PROBE/'predictions_complete.json');inputs=read(PROBE/'verification_inputs.json');analysis=read(PROBE/'readout.json')
    for f,h in done['files'].items():assert sha(f)==h,f
    acquisitions={(r['key'],tuple(r['object_ids'])):r for r in done['acquisition']}
    verification={(r['key'],r['frame_id'],r['semantic_id']):r for r in done['verification']}
    view_index={(r['key'],r['frame_id']):r for r in inputs['records']}
    old_analysis={(r['key'],tuple(r['object_ids'])):r for r in analysis['results']}
    result=[]
    for row in p['pairs']:
        if row['status']!='ready':continue
        key=row['key'];pair=tuple(row['object_ids']);cell=np.load(ROOT/'build/benchmarks/discovery_measured_inputs_v1/query_cells'/key/'source_to_cell.npy')
        original=dict(np.load(row['source_sets']));parts=[np.unique(cell[original[name]]) for name in ['a','b']]
        view=dict(np.load(row['acquisition_view']));ids=view['visible_source_ids'];xy=view['pixel_xy'];new=[]
        for query in acquisitions[(key,pair)]['queries']:
            data=dict(np.load(query['file']));mask=unpack(data,query['selected'])[0]
            new.append(np.unique(cell[ids[mask[xy[:,1],xy[:,0]]]]))
        candidates=dict(keep=parts,split=new[:2],merge=[np.union1d(new[2],new[3])])
        checks=old_analysis[(key,pair)]['prompts']
        valid=dict(keep=True,split=all(checks[k]['prompt_labels_satisfied'] for k in ['split_a','split_b']),
                   merge=all(checks[k]['prompt_labels_satisfied'] for k in ['joint_1','joint_2']))
        validation=[]
        for frame in row['verification_frames']:
            v=dict(np.load(view_index[(key,frame)]['file']));xy=v['pixel_xy'];visible=cell[v['visible_source_ids']]
            data=dict(np.load(verification[(key,frame,row['semantic_id'])]['file']));masks=unpack(data)
            queries=[np.unique(visible[m[xy[:,1],xy[:,0]]]) for m in masks]
            validation.append(dict(frame_id=frame,visible_cells=visible,queries=queries,scores=data['scores'],used_for_prompt=False))
        decision=select_partition(parts,candidates,validation,valid)
        result.append(dict(key=key,objects=list(pair),valid=valid,decision=decision))
        print('CORE REPLAY',key,pair,decision['selected'],decision['scores'],flush=True)
    dump(OUT/'probe_core_replay.json',dict(complete=True,source_sha256=sha(__file__),
        module_sha256=sha(ROOT/'omni_semantic_map/omni_semantic_map/partition_revision.py'),
        probe_predictions_sha256=sha(PROBE/'predictions_complete.json'),results=result,
        limitation='Existing exposed capability cases, no GT or map updates. Checks decision execution, not true object identity or benchmark gain. Full-map candidates must retain unseen historical support; this diagnostic scores acquisition mask support only.'))
    print('PREFLIGHT COMPLETE',coverage,flush=True)


if __name__=='__main__':main()
