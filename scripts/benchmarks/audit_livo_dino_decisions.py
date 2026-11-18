"""Exhaustive changed-decision evidence; counts do not imply identity accuracy."""
import json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from audit_evidence_retention import OUT,REPORT,dump
from omni_semantic_map.flat_benchmark import sha256


def main():
    root=OUT/'dino_association_v1';dest=REPORT/'dino_association_v1';dest.mkdir(exist_ok=False)
    cache=OUT/'livo_full_strong_v2/cache_supportcc';m=json.loads((cache/'manifest.json').read_text())
    lookup={i:(r,i-r['observation_offset']) for r in m['records'] for i in range(r['observation_offset'],r['observation_offset']+r['masks'])}
    job=json.loads((OUT/'jobs/livo_full_v2.json').read_text());frames={f['frame_id']:f for f in job['frames']}
    baseline=json.loads((root/'baseline/cache_supportcc/both/classification.json').read_text())['objects'];baseline_members={i:r for r in baseline for i in r['observation_ids']}
    def label(row):return dict(alias=min(row['observation_ids'])+1,name=row['predictions']['mean']['name'],retained=row['predictions']['mean']['retained']) if row else None
    def picture(oid,title):
        record,k=lookup[oid];fid=record['frame_id'];f=np.load(cache/'frames'/record['file']);rgb=cv2.imread(frames[fid]['rgb'])
        mask=f['segmentation']==k;rgb[mask]=(rgb[mask]*.6+np.array([0,210,255])*.4).astype(np.uint8)
        cv2.rectangle(rgb,(0,0),(rgb.shape[1],28),(20,20,20),-1)
        cv2.putText(rgb,f'{title} | frame {fid} | obs {oid+1}',(7,19),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
        return rgb
    summaries=[];cases=[]
    base_groups={tuple(sorted(o['observation_ids'])) for o in baseline if o['predictions']['mean']['retained']}
    for variant in ['baseline','dino_pooled','dino_local']:
        path=root/variant;d=json.loads((path/'association.json').read_text());assert d['complete']
        rows=json.loads((Path(d['output'])/'classification.json').read_text())['objects'];index={i:r for r in rows for i in r['observation_ids']}
        fg=[o for o in rows if o['predictions']['mean']['retained']]
        groups={tuple(sorted(o['observation_ids'])) for o in fg}
        members=[i for o in rows for i in o['observation_ids']];assert len(members)==len(set(members))
        baseline_fg_events=[e for e in d['events'] if baseline_members.get(e['observation_id'],{}).get('predictions',{}).get('mean',{}).get('retained',False)]
        summaries.append(dict(variant=variant,counts=d['counts'],regions=len(rows),foreground=len(fg),
            foreground_membership_groups_unchanged=len(groups&base_groups),foreground_membership_groups_new=len(groups-base_groups),
            ambiguous_baseline_retained_foreground_queries=len(baseline_fg_events),baseline_foreground_decisions_changed=sum(e['changed'] for e in baseline_fg_events),
            foreground_reasons=dict(Counter(e['reason'] for e in baseline_fg_events)),
            first_return=next(h for h in d['causal_history'] if h['return_alias'] is not None),
            association_sha256=sha256(path/'association.json'),run_sha256=d['run_sha256']))
        for e in d['events']:
            if not e['changed']:continue
            a=next(c for c in e['candidates'] if c['alias']==e['native_alias']);b=next(c for c in e['candidates'] if c['alias']==e['chosen_alias'])
            native_obs=a['best_evidence_observation'];chosen_obs=b['best_evidence_observation'];assert native_obs is not None and chosen_obs is not None
            img=np.concatenate([picture(e['observation_id'],'Current'),picture(native_obs,'Native candidate'),picture(chosen_obs,'DINO candidate')],axis=1)
            image=dest/f'{variant}_{e["observation_id"]+1}.jpg';assert cv2.imwrite(str(image),img)
            cases.append(dict(variant=variant,**e,native_evidence_observation=native_obs,chosen_evidence_observation=chosen_obs,
                baseline_final=label(baseline_members.get(e['observation_id'])),variant_final=label(index.get(e['observation_id'])),
                evidence_image=str(image),image_sha256=sha256(image),visual_review='pending'))
    dump(dest/'summary.json',dict(complete=True,summaries=summaries,cases=cases,
        scope='All changed causal choices, no score-based selection. Evidence frame for each candidate is its highest DINO-scoring prior member, as used by the actual decision.',
        visual_review_complete=False))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
