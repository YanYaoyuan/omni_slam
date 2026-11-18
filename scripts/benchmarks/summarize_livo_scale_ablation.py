"""Summarize sparse-scale controls without treating object count as accuracy."""
import json
import numpy as np
from audit_evidence_retention import OUT, REPORT, dump
from omni_semantic_map.flat_benchmark import sha256


def main():
    root=OUT/'scale_ablation_v1';p=root/'protocol.json';spec=json.loads(p.read_text())
    rows=[]
    for cache in spec['inputs']:
        for variant in spec['variants']:
            path=root/cache/variant;run=json.loads((path/'run.json').read_text())
            assert run['complete'] and run['intervention_protocol_sha256']==sha256(p)
            for f,h in run['files_sha256'].items():assert sha256(path/f)==h
            d=json.loads((path/'diagnostics.json').read_text());online=d['online_objects'];final=d['final_objects']
            foreground=[o for o in final if o['retained']]
            classifications=json.loads((path/'classification.json').read_text())['objects']
            target_rows=[]
            for alias in spec['fixed_visual_targets']:
                hit=[o for o in final if alias in o['fixed_target_observations']]
                target_rows.append(dict(original_alias=alias,output=hit[0] if hit else None))
            # Count conflicts in final membership, without mistaking them for GT errors.
            members=[i for o in classifications for i in o['observation_ids']]
            assert len(members)==len(set(members)), 'cross-object or within-object observation duplication'
            first=next(t['output'] for t in target_rows if t['original_alias']==6)
            returned=next(t['output'] for t in target_rows if t['original_alias']==3075)
            rows.append(dict(cache=cache,variant=variant,online_objects=len(online),
                single_view_objects=sum(o['distinct_frames']==1 for o in online),
                multiview_objects=sum(o['distinct_frames']>=3 for o in online),
                final_regions=len(final),retained_predictions=len(foreground),
                retained_below_16_points=sum(o['points']<16 for o in foreground),
                objects_with_same_frame_extras=sum(o['same_frame_extra_members']>0 for o in final),
                same_frame_extras=sum(o['same_frame_extra_members'] for o in final),
                targets=target_rows,start_return_same_output=bool(first and returned and first['alias']==returned['alias']),
                backend_wall_s=run['wall_s'],run_sha256=sha256(path/'run.json'),
                native_exact_reproduction=d['native_exact_reproduction']))
    result=dict(complete=True,protocol_sha256=sha256(p),rows=rows,
        scope='Eight fixed development controls on existing LIVO packets. Counts, conflicts and geometry are diagnostics, not semantic recall/PQ or identity accuracy.',
        same_frame_scope='Extra masks from the same frame can be overlapping SAM detections or distinct physical objects; requires evidence review.',
        selection_scope='Fixed four observation aliases chosen in protocol before these runs; not a complete scene recall test.')
    dest=REPORT/'scale_ablation_v1';dump(dest/'summary.json',result)
    dump(dest/'protocol.json',spec)
    print(json.dumps([{k:v for k,v in r.items() if k not in ['targets','run_sha256']} for r in rows],indent=2))


if __name__=='__main__':main()
