"""Full candidate coverage and bounded, common queries before map experiments.

This stage reads neither reference labels nor scored object diagnostics.
It freezes selection only; full-map execution requires its own final protocol.
"""
import hashlib,time
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from run_signed_query_association import OUT as PARENT,INPUT,pieces,sealed
from omni_semantic_map.signed_query_association import positive_components
from omni_semantic_map.query_hypothesis_evidence import membership
from omni_semantic_map.kitti360_input import load_poses,load_times
from prepare_partition_query_probe import anchor,META

OUT=ROOT/'build/benchmarks/partition_revision_v1'


def full_partitions(key,n,pairs):
    geometric=positive_components(n,pairs['left'],pairs['right'],pairs['features'][:,0]>.5)
    owner=np.zeros(n,np.int64)
    for i,g in enumerate(geometric):owner[g]=i
    signed={j:{j} for j in range(n)};assignment=np.arange(n)
    for event in read(PARENT/'maps'/key/'ours_signed/groups.json')['merge_trace']:
        a,b=int(assignment[event['left'][0]]),int(assignment[event['right'][0]])
        assert signed[a]==set(event['left']) and signed[b]==set(event['right'])
        merged=signed.pop(a)|signed.pop(b);signed[min(a,b)]=merged;assignment[list(merged)]=min(a,b)
    disagreements=set()
    for g in signed.values():
        ids=sorted(set(owner[list(g)]));disagreements.update((a,b) for j,a in enumerate(ids) for b in ids[j+1:])
    return geometric,[(int(a),int(b)) for a,b in sorted(disagreements)]


def main():
    assert not OUT.exists()
    for stage in ['nodes_complete','pairs_complete','predictions_complete']:sealed(stage)
    parent=read(PARENT/'protocol.json')
    prior=read(ROOT/'build/benchmarks/partition_query_probe_v1/verification_inputs.json')
    sources={str(ROOT/'scripts/benchmarks/prepare_partition_revision.py'):sha(__file__),
             str(ROOT/'scripts/benchmarks/prepare_partition_query_probe.py'):sha(ROOT/'scripts/benchmarks/prepare_partition_query_probe.py')}
    dump(OUT/'candidate_protocol.json',dict(created_epoch=time.time(),sources=sources,
         parent_seals={s:sha(PARENT/(s+'.json')) for s in ['nodes_complete','pairs_complete','predictions_complete']},
         earlier_probe_inputs_sha256=sha(ROOT/'build/benchmarks/partition_query_probe_v1/verification_inputs.json'),
         candidate_policy='All geometric connected components, INCLUDING non-exported provisional observations; all unordered geometric-component pairs co-grouped in a frozen full signed partition reconstructed from its trace. Failed learned graph is a dispute proposal source only, never the committed new map.',
         action_policy='Same >=16 exclusive measured-source visible requirement and fixed separation/visibility utility as capability probe. Candidate acquisition views must also have two exact-pose unused validation frame IDs at fixed +1/+6 neighborhoods. Validation RGB/geometry/queries are not inspected by selection.',
         selection='Rank pairs by min(exclusive source-set sizes) times maximum action utility, stable group IDs. Greedily select up to4 per semantic class, without reusing a geometric component. Fixed car/building quotas; no reallocation or GT. Same pairs for all controls.',
         view_controls='For each common pair: maximum dispute utility; maximum minimum visible source count; uniformly random eligible acquisition index using SHA256(key,a,b,seed0) modulo eligible count. Stable earliest ties. All policies use4 decodes per selected pair on1008x1008 images; report cache reuse and actual encode costs separately.',
         outputs='Candidate metadata, source supports, every eligible action, three selected action policies and unused validation frame IDs. No training, predictions, acceptance decisions, maps or metrics.',
         limit='Up to8 pairs per window across2 exposed development windows. All previous108 images and5 probe verification images excluded from the new validation pool. These are correlated unused observations, not independent benchmark scenes.'))
    stats={};files={};downloads={}
    for scene in parent['scenes']:
        key=scene['key'];nodes=dict(np.load(PARENT/'nodes'/key/'nodes.npz'));meta=read(PARENT/'nodes'/key/'nodes.json');pairs=dict(np.load(PARENT/'pairs'/key/'features.npz'));n=len(meta)
        groups,disagreements=full_partitions(key,n,pairs)
        raw_sets=pieces(nodes['source_ids'],nodes['source_offsets'])
        sets=[np.unique(np.concatenate([raw_sets[j] for j in g])) for g in groups]
        classes=np.array([nodes['semantic_ids'][g[0]] for g in groups]);assert all(np.all(nodes['semantic_ids'][g]==classes[i]) for i,g in enumerate(groups))
        manifest=read(INPUT/key/'manifest.json');source_count=manifest['source_points'];h=membership(sets,source_count)
        overlaps=(h@h.T).toarray();sizes=np.diff(h.indptr)
        job=read(scene['job']);sequence=job['scene'];poses=load_poses(META/sequence/'poses.txt')
        images=load_times(META/sequence/'image_00/timestamps.txt');scans=load_times(META/sequence/'velodyne_points/timestamps.txt')
        excluded={r['frame_id'] for r in manifest['records']}|{r['frame_id'] for r in prior['records'] if r['key']==key}
        oldframes=[r['frame_id'] for r in manifest['records']]
        unused=set(f for f in poses if min(oldframes)<=f<=max(oldframes) and f not in excluded and f<min(len(images),len(scans)))
        candidates={}
        for a,b in disagreements:
            exclusive=[int(sizes[a]-overlaps[a,b]),int(sizes[b]-overlaps[a,b])]
            candidates[(a,b)]=dict(groups=[a,b],semantic_id=int(classes[a]),exclusive_sources=exclusive,actions=[])
            assert classes[a]==classes[b]
        for rec in manifest['records']:
            heldout=[]
            for offset in [1,6]:
                values=[f for f in range(rec['frame_id']+offset,rec['frame_id']+offset+5) if f in unused]
                if values:heldout.append(values[0])
            if len(heldout)!=2:continue
            view=dict(np.load(rec['file']));ids=view['visible_source_ids'];xy=view['pixel_xy'];visible=h[:,ids].toarray().astype(bool)
            counts=visible.sum(1)
            for (a,b),candidate in candidates.items():
                if min(candidate['exclusive_sources'])<16 or min(counts[[a,b]])<16:continue
                va=visible[a]&~visible[b];vb=visible[b]&~visible[a]
                if min(va.sum(),vb.sum())<16:continue
                pa,pb=xy[va],xy[vb];ca,cb=np.median(pa,axis=0),np.median(pb,axis=0)
                distance=np.linalg.norm(ca-cb);spread=np.sqrt(np.mean(np.sum((pa-ca)**2,axis=1)))+np.sqrt(np.mean(np.sum((pb-cb)**2,axis=1)))+1
                fa,fb=va.sum()/candidate['exclusive_sources'][0],vb.sum()/candidate['exclusive_sources'][1]
                utility=float(2*fa*fb/(fa+fb)*distance/spread)
                anchor_a,source_a=anchor(pa,ids[va]);anchor_b,source_b=anchor(pb,ids[vb])
                if anchor_a[0]==anchor_b[0]:continue
                candidate['actions'].append(dict(frame_id=rec['frame_id'],rgb=rec['rgb'],measured_view=rec['file'],
                    utility=utility,min_visible=int(min(va.sum(),vb.sum())),point_a=anchor_a,point_b=anchor_b,
                    anchor_sources=[source_a,source_b],verification_frames=heldout))
        selected=[];used=set()
        for semantic in [11,26]:
            valid=[c for c in candidates.values() if c['semantic_id']==semantic and c['actions']]
            for c in valid:c['priority']=min(c['exclusive_sources'])*max(a['utility'] for a in c['actions'])
            ranked=sorted(valid,key=lambda c:(-c['priority'],c['groups']))
            kept=0
            for c in ranked:
                if any(g in used for g in c['groups']):continue
                used.update(c['groups']);chosen=dict(c);chosen['policies']={}
                choices=dict(ours=max(c['actions'],key=lambda a:(a['utility'],-a['frame_id'])),
                             max_visible=max(c['actions'],key=lambda a:(a['min_visible'],-a['frame_id'])))
                digest=hashlib.sha256(f'{key}:{c["groups"][0]}:{c["groups"][1]}:0'.encode()).digest()
                choices['random']=c['actions'][int.from_bytes(digest[:8],'little')%len(c['actions'])]
                for policy,action in choices.items():
                    chosen['policies'][policy]=action
                    downloads.setdefault(sequence,set()).update(action['verification_frames'])
                selected.append(chosen);kept+=1
                if kept==4:break
        folder=OUT/key;folder.mkdir(parents=True)
        offsets=np.r_[0,np.cumsum([len(v) for v in sets])]
        np.savez_compressed(folder/'groups.npz',source_ids=np.concatenate(sets),source_offsets=offsets,semantic_ids=classes)
        dump(folder/'groups.json',[dict(group=i,node_ids=g,old_ids=sorted(meta[j]['old_object_id'] for j in g if meta[j]['old_object_id']>=0),
              seed_frames=sorted(set(meta[j]['seed_frame'] for j in g if meta[j]['seed_frame']>=0))) for i,g in enumerate(groups)])
        dump(folder/'candidates.json',list(candidates.values()));dump(folder/'selected.json',selected)
        stats[key]=dict(groups=len(groups),disputes=len(disagreements),with_eligible_action=sum(bool(c['actions']) for c in candidates.values()),
                        selected_pairs=len(selected),selected_classes={str(s):sum(c['semantic_id']==s for c in selected) for s in [11,26]},
                        old_groups_in_selected=sum(bool(any(meta[j]['old_object_id']>=0 for j in groups[g])) for c in selected for g in c['groups']))
        for f in folder.iterdir():files[str(f)]=sha(f)
        print('COVERAGE',key,stats[key],flush=True)
        for c in selected:print('SELECTED',c['groups'],c['semantic_id'],{k:v['frame_id'] for k,v in c['policies'].items()},flush=True)
    dump(OUT/'candidates_complete.json',dict(complete=True,protocol_sha256=sha(OUT/'candidate_protocol.json'),files=files,stats=stats,
         downloads={k:sorted(v) for k,v in downloads.items()},no_gt_read=True,no_new_model_calls=True,no_maps_changed=True))


if __name__=='__main__':main()
