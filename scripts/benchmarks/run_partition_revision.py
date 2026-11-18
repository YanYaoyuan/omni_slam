"""Frozen six-arm complete-map experiment for partition-conditioned revision."""
import argparse,copy,shutil,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from prepare_partition_revision import OUT,PARENT,INPUT,META
from run_signed_query_association import pieces,ragged,sealed as parent_sealed
from run_hypothesis_admission import BASE,EVALUATOR
from omni_semantic_map.partition_revision import partition_fit,select_partition,revise_history,propose_measured_sources
from omni_semantic_map.query_hypothesis_evidence import membership
from omni_semantic_map.sparse_lidar_support import project_measured_visibility
from omni_semantic_map.kitti360_input import load_left_calibration,project_lidar
from omni_semantic_map.kitti360_evaluation import evaluate
from evaluate_kitti360_development import transfer,clean_json
from run_discovery_query_benchmark import matched_ids

METHODS=['baseline','ordinary','self_validation','current_only','retrieval_only','ours']


def freeze():
    assert not (OUT/'protocol.json').exists()
    parent_sealed('predictions_complete');parent_sealed('nodes_complete')
    parent=read(PARENT/'protocol.json');probe=read(ROOT/'build/benchmarks/partition_query_probe_v1/protocol.json')
    source_files=[Path(__file__),ROOT/'scripts/benchmarks/predict_partition_revision.py',ROOT/'omni_semantic_map/omni_semantic_map/partition_revision.py',ROOT/'omni_semantic_map/test/test_partition_revision.py',ROOT/'omni_semantic_map/omni_semantic_map/sparse_lidar_support.py',ROOT/'omni_semantic_map/omni_semantic_map/kitti360_input.py',ROOT/'omni_semantic_map/omni_semantic_map/query_hypothesis_evidence.py',ROOT/'omni_semantic_map/omni_semantic_map/kitti360_evaluation.py',ROOT/'scripts/benchmarks/evaluate_kitti360_development.py']
    source_files += [ROOT/'omni_semantic_map/test/test_partition_revision_export.py',ROOT/'scripts/benchmarks/run_signed_query_association.py',ROOT/'scripts/benchmarks/run_discovery_query_benchmark.py']
    sources={str(f):sha(f) for f in source_files};inputs={}
    for name in ['candidates_complete','validation_cells_complete','verification_inputs']:
        f=OUT/(name+'.json');d=read(f);assert d['complete'];inputs[str(f)]=sha(f)
        for path,h in d['files'].items():assert sha(path)==h,path;inputs[path]=h
    for f in [OUT/'common_view_execution_scope.json',OUT/'control_identifiability.json',PARENT/'predictions_complete.json',PARENT/'summary.json',EVALUATOR]:inputs[str(f)]=sha(f)
    for scene in parent['scenes']:
        key=scene['key'];manifest=read(INPUT/key/'manifest.json');job=read(scene['job'])
        for f in [INPUT/key/'manifest.json',Path(manifest['geometry']),INPUT/'query_cells'/key/'source_to_cell.npy',Path(scene['job']),PARENT/'nodes'/key/'nodes.npz',PARENT/'nodes'/key/'nodes.json',BASE/'maps'/key/'intrinsic_geometry/geometry.npz']:
            inputs[str(f)]=sha(f)
        for name in ['geometry.npz','classification.json','groups.json']:f=PARENT/'maps'/key/'geometry_graph'/name;inputs[str(f)]=sha(f)
        selected=read(OUT/key/'selected.json');frames={r['policies']['max_visible']['frame_id'] for r in selected}
        for frame in job['frames']:
            if frame['frame_id'] in frames:
                for field in ['rgb','raw_scan','measured']:inputs[frame[field]]=sha(frame[field])
        manifest_frames={r['frame_id']:r for r in manifest['records']}
        for frame in frames:f=manifest_frames[frame]['file'];inputs[f]=sha(f)
    for f in (META/'calibration').glob('*.txt'):inputs[str(f)]=sha(f)
    for f in [Path('/home/user/vbot/sam3/sam3/model_builder.py'),Path('/home/user/vbot/sam3/sam3/model/sam3_image.py'),Path('/home/user/vbot/sam3/sam3/model/sam3_image_processor.py')]:sources[str(f)]=sha(f)
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),scenes=parent['scenes'],methods=METHODS,source_hashes=sources,input_hashes=inputs,
        checkpoint=probe['checkpoint'],checkpoint_sha256=probe['checkpoint_sha256'],
        baseline='Exact frozen geometry_graph map, stronger than original intrinsic baseline. Failed signed graph only proposed disputes; never the starting map.',
        actions='Frozen16 pairs,8/window; common maximum-measured-visibility acquisition actions. No active-view-selection claim. Ordinary4positive prompts and symmetric4positive/negative/joint prompts; native highest-IoU mask only. Keep always valid; split/merge valid only when both associated selected masks satisfy all supplied point labels.',
        candidates='Source-support geometry, not GT: keep original pair supports; split preserves historical support while changing exclusive current-mask assignments; merge unions historical pair with both joint-mask supports. Current-only uses identical Ours decision but appends only current observed supports and never changes old point IDs. Retrieval-only compares keep vs original-support union, no new masks.',
        validation='Native class-only queries on unused actual-LiDAR20cm cell observations; same views for all candidate partitions. Both original parts need16visible cells; fit=score-weighted measured IoU, injective two-query matching discounted by query overlap, equal-view averaging; strict gain>1e-9, ties/unknown keep. Self-validation intentionally substitutes acquisition class-only observation; it is never called independent validation. Acquisition and new-view class observations both use current actual returns in the same physical grid.',
        history='Old map XYZ prefix preserved exactly. Split changes only affected object IDs at points passing current acquisition5cm depth gate and exactly one selected mask; unseen/overlap retains label. Merge unifies both historical IDs to the smallest existing/synthetic ID. No old point deletion or movement. Current-only skips all historical relabeling.',
        birth='Common census for ALL arms, including self-validation: original member seed frames plus acquisition/new verification frames where candidate has>=16visible cells and a same-class native query has score>0.1 and measured IoU>0.5. A new object needs>=3 distinct census frames, original native group score>0.1, and>=16 actual output points. Shared census consumes new views even in self-validation; ONLY its partition selection is self-validated. This is explicitly not an ablation removing all new-view information.',
        scores='Existing-object scores unchanged; merged existing score=max constituent existing scores. New-object score=mean of per-frame maximum native ORIGINAL member seed score; verification and interactive-quality scores never rerank objects. Same rule in all arms.',
        export='Original native core sources, intrinsic additions and geometry_graph additions globally locked against duplicate append. No new validation-scan points may enter output geometry. Proposals allocate remaining original measured sources in descending native score/stable pair order; no source claimed twice. Atomic pair transaction: if a required new part lacks birth eligibility or16outputpoints, keep whole pair. Disjoint selected groups prohibit unrelated-old-object edits. No additional NMS,DBSCAN or denoising.',
        budget='16old acquisition image encodes+29new validation encodes,128instance decodes (64ordinary+64symmetric),45class decodes; cache shared across six arms. Twelve map outputs (two baseline copies,ten revised). No training or parameter search. Report actual counts if image/class overlap alters cache reuse.',
        gate='Each exposed development window: Ours primary AP50:90 strictly exceeds every five controls (including cheaper retrieval and geometry baseline), and net gained AP50 GTs vs baseline>0. Ours mean must exceed current-only. Otherwise close version, no post-score parameter sweep or expansion. This is the original object-hypothesis mechanism gate, not a retroactive change to previous failed gates.',
        scope='Window-end offline, development0000/0002 only. New nearby frames are not independent benchmark scenes. No robot/new-scene assessment until development gate. Full12-map prediction seal precedes reference scoring.'))
    print('FROZEN PARTITION REVISION',flush=True)


def verify():
    p=read(OUT/'protocol.json')
    for f,h in {**p['source_hashes'],**p['input_hashes']}.items():assert sha(f)==h,f
    return p


def unpack(data):
    return np.unpackbits(data['masks_packed'],axis=2)[:,:,:int(data['original_size'][0])].astype(bool)


def make_view(frame,queries,pixels,cells,used_for_prompt):
    masks=unpack(queries);supports=[np.unique(cells[m[pixels[:,1],pixels[:,0]]]) for m in masks]
    width=int(max(cells.max(initial=0),0))+1;q=membership(supports,width)
    intersection=(q@q.T).toarray().astype(np.float64);size=np.diff(q.indptr);union=size[:,None]+size[None,:]-intersection
    overlap=np.divide(intersection,union,out=np.zeros_like(intersection),where=union>0)
    return dict(frame_id=frame,queries=supports,scores=queries['scores'],visible_cells=np.unique(cells),query_overlap=overlap,used_for_prompt=used_for_prompt)


def self_select(original,candidates,view,valid):
    # Explicit ablation: do not weaken select_partition's anti-self-validation guard.
    counts=[len(np.intersect1d(s,view['visible_cells'])) for s in original]
    if min(counts)<16:return dict(selected='keep',reason='self_view_insufficient',scores={},views=[])
    scores={k:partition_fit(parts,view['queries'],view['scores'],view['visible_cells'],view['query_overlap']) for k,parts in candidates.items() if valid.get(k,False)}
    best=max(scores,key=lambda k:(scores[k],k=='keep'))
    if scores[best]<=scores['keep']+1e-9 or sum(abs(v-scores[best])<=1e-9 for v in scores.values())>1:best='keep'
    return dict(selected=best,reason='self_validation_ablation',scores=scores,views=[dict(frame_id=view['frame_id'],original_visible=counts,scores=scores)])


def census_frames(cells,seed_frames,views):
    frames=set(seed_frames)
    for view in views:
        visible=np.intersect1d(cells,view['visible_cells'])
        if len(visible)<16:continue
        for query,score in zip(view['queries'],view['scores']):
            if score<=.1:continue
            common=len(np.intersect1d(visible,query))
            if common/max(1,len(visible)+len(query)-common)>.5:frames.add(view['frame_id']);break
    return sorted(frames)


def context(scene,outputs):
    key=scene['key'];job=read(scene['job']);manifest=read(INPUT/key/'manifest.json');source=np.load(manifest['geometry'])['xyz']
    cell=np.load(INPUT/'query_cells'/key/'source_to_cell.npy');data=dict(np.load(OUT/key/'groups.npz'));sets=pieces(data['source_ids'],data['source_offsets']);groups=read(OUT/key/'groups.json');nodes=read(PARENT/'nodes'/key/'nodes.json')
    folder=PARENT/'maps'/key/'geometry_graph';base=dict(np.load(folder/'geometry.npz'));classification=read(folder/'classification.json');records={o['object_id']:o for o in classification['objects']}
    node_group={j:g['group'] for g in groups for j in g['node_ids']};group_oid={}
    for g in read(folder/'groups.json')['groups']:
        gid=node_group[g['node_ids'][0]];assert all(node_group[j]==gid for j in g['node_ids']);group_oid[gid]=g['object_id']
    parent_nodes=dict(np.load(PARENT/'nodes'/key/'nodes.npz'));parent_sets=pieces(parent_nodes['source_ids'],parent_nodes['source_offsets'])
    intrinsic=np.load(BASE/'maps'/key/'intrinsic_geometry/geometry.npz')
    protected=np.unique(np.concatenate([base['added_source_ids'],intrinsic['added_source_ids']]+[parent_sets[i] for i in np.flatnonzero(parent_nodes['old'])]))
    selected=read(OUT/key/'selected.json');jobframes={r['frame_id']:r for r in job['frames']};oldviews={r['frame_id']:r for r in manifest['records']}
    verification=read(OUT/'validation_cells_complete.json')['results'][key];vframes={r['frame_id']:r for r in verification['records']}
    cellkeys=np.load(verification['cell_keys']);lookup={tuple(k):i for i,k in enumerate(cellkeys)};calib=load_left_calibration(META/'calibration')
    queries={(r['role'],r['frame_id'],r['semantic_id']):r['file'] for r in outputs['queries'] if r['key']==key}
    instance={(tuple(r['groups']),r['policy']):r['files'] for r in outputs['instances'] if r['key']==key}
    viewcache={};cases=[]
    for row in selected:
        ga,gb=row['groups'];action=row['policies']['max_visible'];frame=action['frame_id'];semantic=row['semantic_id'];acquisition=jobframes[frame]
        raw=np.fromfile(acquisition['raw_scan'],dtype='<f4').reshape(-1,4)[:,:3];projected=project_lidar(raw,calib);ids=projected['selected_indices'];pose=np.asarray(acquisition['pose']);world=projected['camera_xyz'][ids]@pose[:3,:3].T+pose[:3,3]
        assigned=[]
        for k in np.floor(world/.2).astype(np.int64):
            token=tuple(k)
            if token not in lookup:lookup[token]=len(lookup)
            assigned.append(lookup[token])
        yy,xx=np.nonzero(projected['source_point_index']>=0);np.testing.assert_array_equal(projected['source_point_index'][yy,xx],ids)
        acquisition_view=make_view(frame,dict(np.load(queries[('acquisition',frame,semantic)])),np.column_stack([xx,yy]),np.asarray(assigned),True)
        validation=[]
        for vf in action['verification_frames']:
            cachekey=(vf,semantic)
            if cachekey not in viewcache:
                v=dict(np.load(vframes[vf]['file']));viewcache[cachekey]=make_view(vf,dict(np.load(queries[('validation',vf,semantic)])),v['pixel_xy'],v['cell_ids'],False)
            validation.append(viewcache[cachekey])
        measured=dict(np.load(oldviews[frame]['file']));pix=measured['pixel_xy'];sourceids=measured['visible_source_ids'];policies={}
        depth=np.load(acquisition['measured'])['depth'];x,y,_,visible=project_measured_visibility(base['xyz'],pose,job['config']['cam'],depth,SimpleNamespace(depth_absolute_m=.05,depth_relative=0.))
        for policy in ['ordinary','symmetric']:
            supports=[];history_masks=[];compliance=[]
            for f in instance[((ga,gb),policy)]:
                d=dict(np.load(f));mask=unpack(d)[int(d['selected'])];points=d['point_coords'].astype(int)
                compliance.append(bool(np.array_equal(mask[points[:,1],points[:,0]],d['point_labels'].astype(bool))))
                supports.append(sourceids[mask[pix[:,1],pix[:,0]]]);history_masks.append(visible&mask[y,x])
            split=propose_measured_sources(sets[ga],sets[gb],supports[0],supports[1],'split')
            merge=propose_measured_sources(sets[ga],sets[gb],supports[2],supports[3],'merge')
            policies[policy]=dict(supports=supports,history_masks=history_masks,source_candidates=dict(keep=[sets[ga],sets[gb]],split=split,merge=merge),
                                  valid=dict(keep=True,split=all(compliance[:2]),merge=all(compliance[2:])),prompt_compliance=compliance)
        cases.append(dict(row=row,policies=policies,acquisition=acquisition_view,validation=validation))
    return dict(key=key,source=source,cell=cell,sets=sets,groups=groups,nodes=nodes,base=base,classification=classification,records=records,group_oid=group_oid,protected=protected,cases=cases)


def native_score(ctx,gids):
    existing=[ctx['group_oid'][g] for g in gids if g in ctx['group_oid']]
    if existing:return max(ctx['records'][o]['predictions']['mean']['score'] for o in existing)
    frames={}
    for g in gids:
        for j in ctx['groups'][g]['node_ids']:
            node=ctx['nodes'][j]
            if node['seed_frame']>=0:frames[node['seed_frame']]=max(frames.get(node['seed_frame'],0.),node['score'])
    return float(np.mean(list(frames.values()))) if frames else 0.


def export_method(ctx,method,folder):
    base=ctx['base'];labels=base['instance'].copy();records=copy.deepcopy(ctx['records']);maxid=max(records);decisions=[];transactions=[];changed_sources=[]
    for case in ctx['cases']:
        row=case['row'];ga,gb=row['groups'];policy='ordinary' if method=='ordinary' else 'symmetric';bundle=case['policies'][policy]
        if method=='retrieval_only':
            source_candidates=dict(keep=[ctx['sets'][ga],ctx['sets'][gb]],merge=[np.union1d(ctx['sets'][ga],ctx['sets'][gb])]);valid=dict(keep=True,merge=True)
        else:source_candidates=bundle['source_candidates'];valid=bundle['valid']
        candidate_cells={k:[np.unique(ctx['cell'][s]) for s in parts] for k,parts in source_candidates.items()};original=candidate_cells['keep']
        if method=='self_validation':decision=self_select(original,candidate_cells,case['acquisition'],valid)
        else:decision=select_partition(original,candidate_cells,case['validation'],valid)
        event=dict(groups=[ga,gb],acquisition_frame=row['policies']['max_visible']['frame_id'],decision=decision,valid=valid,prompt_compliance=bundle['prompt_compliance'] if method!='retrieval_only' else None,applied=False)
        decisions.append(event);operation=decision['selected']
        if operation=='keep':continue
        oa=ctx['group_oid'].get(ga,maxid+1+ga);ob=ctx['group_oid'].get(gb,maxid+1+gb)
        gids_parts=[[ga,gb]] if operation=='merge' else [[ga],[gb]];oids=[min(oa,ob)] if operation=='merge' else [oa,ob]
        proposed=source_candidates[operation]
        if method=='current_only':
            masks=bundle['supports'][:2] if operation=='split' else bundle['supports'][2:]
            proposed=propose_measured_sources([],[],*masks,operation,include_history=False)
        prospective=labels.copy()
        if method!='current_only':
            if operation=='merge':prospective=revise_history(labels,oa,ob,'merge')
            else:prospective=revise_history(labels,oa,ob,'split',*bundle['history_masks'][:2])
        parts=[];eligible=True
        for index,(gids,oid,sources) in enumerate(zip(gids_parts,oids,proposed)):
            seeds=sorted(set(f for g in gids for f in ctx['groups'][g]['seed_frames']))
            census=census_frames(candidate_cells[operation][index],seeds,[case['acquisition']]+case['validation'])
            score=native_score(ctx,gids);existing=oid in ctx['records']
            if method=='current_only' and existing:score=ctx['records'][oid]['predictions']['mean']['score']
            birth_ok=existing or (len(census)>=3 and score>.1)
            eligible&=birth_ok;parts.append(dict(groups=gids,object_id=oid,sources=sources,score=score,existing=existing,census=census))
        event['birth']=[{k:v for k,v in part.items() if k!='sources'} for part in parts]
        if not eligible:event['skip_reason']='shared_birth_rule';continue
        transactions.append(dict(parts=parts,oa=oa,ob=ob,operation=operation,case=case,event=event,priority=max(p['score'] for p in parts)))
    used=np.zeros(len(ctx['source']),bool);used[ctx['protected']]=True;added=[];added_labels=[]
    for transaction in sorted(transactions,key=lambda t:(-t['priority'],t['event']['groups'])):
        oa,ob=transaction['oa'],transaction['ob'];operation=transaction['operation'];case=transaction['case'];bundle=case['policies']['ordinary' if method=='ordinary' else 'symmetric'];event=transaction['event']
        prospective=labels.copy()
        if method!='current_only':prospective=revise_history(labels,oa,ob,operation,*(bundle['history_masks'][:2] if operation=='split' else []))
        local=used.copy();free_parts=[];ok=True
        for part in transaction['parts']:
            free=part['sources'][~local[part['sources']]];local[free]=True;free_parts.append(free)
            if not part['existing'] and len(free)+int((prospective==part['object_id']).sum())<16:ok=False
        if not ok:event['skip_reason']='insufficient_exclusive_output_points';continue
        event['old_point_ids_changed']=int((labels!=prospective).sum());labels=prospective;used=local;event['applied']=True;event['added_points']=0
        if operation=='merge' and method!='current_only':
            target=transaction['parts'][0]['object_id']
            for oid in [oa,ob]:
                if oid!=target:records.pop(oid,None)
        for part,free in zip(transaction['parts'],free_parts):
            oid=part['object_id'];gids=part['groups'];sem=case['row']['semantic_id']
            existing=[ctx['group_oid'][g] for g in gids if g in ctx['group_oid']]
            record=copy.deepcopy(ctx['records'][oid]) if oid in ctx['records'] else dict(object_id=oid,observation_ids=[],selected_observation_ids=[],predictions=dict(mean=dict(name='car' if sem==26 else 'building',semantic_id=sem,train_id=13 if sem==26 else 2,retained=True,score=part['score'])))
            record['predictions']['mean']['score']=part['score'];record['revision_groups']=gids;record['revision_census_frames']=part['census'];record['revision_operation']=operation
            if len(existing)>1 and method!='current_only':
                for field in ['observation_ids','selected_observation_ids']:
                    record[field]=sorted(set(v for old in existing for v in ctx['records'][old].get(field,[])))
            records[oid]=record;added.append(free);added_labels.append(np.full(len(free),oid,np.int64));event['added_points']+=len(free)
    source_ids=np.concatenate(added) if added else np.empty(0,np.int64);newlabels=np.concatenate(added_labels) if added_labels else np.empty(0,np.int64)
    assert len(np.unique(source_ids))==len(source_ids) and not np.isin(source_ids,ctx['protected']).any()
    xyz=np.concatenate([base['xyz'],ctx['source'][source_ids]]);instances=np.r_[labels,newlabels];np.testing.assert_array_equal(xyz[:len(base['xyz'])],base['xyz'])
    present=set(map(int,np.unique(instances)));assert present<=set(records)
    export=copy.deepcopy(ctx['classification']);export['objects']=[records[i] for i in sorted(present)]
    np.savez_compressed(folder/'geometry.npz',xyz=xyz,instance=instances,added_source_ids=source_ids);dump(folder/'classification.json',export);dump(folder/'decisions.json',decisions)
    return dict(objects=len(present),new_objects=len(present-set(ctx['records'])),removed_object_ids=sorted(set(ctx['records'])-present),added_points=len(source_ids),old_point_ids_changed=int((labels!=base['instance']).sum()),applied_pairs=sum(d['applied'] for d in decisions),selected_revisions=sum(d['decision']['selected']!='keep' for d in decisions),birth_rejections=sum(d.get('skip_reason')=='shared_birth_rule' for d in decisions),old_coordinates_exact=True,new_validation_scan_points_added=False)


def maps():
    p=verify();outputs=read(OUT/'model_outputs_complete.json');assert outputs['protocol_sha256']==sha(OUT/'protocol.json')
    for f,h in outputs['files'].items():assert sha(f)==h,f
    assert not (OUT/'predictions_complete.json').exists();files={};stats={}
    for scene in p['scenes']:
        started=time.monotonic();ctx=context(scene,outputs);key=scene['key'];print('CONTEXT',key,time.monotonic()-started,flush=True)
        for method in METHODS:
            folder=OUT/'maps'/key/method;folder.mkdir(parents=True,exist_ok=False);started=time.monotonic()
            if method=='baseline':
                for name in ['geometry.npz','classification.json']:shutil.copyfile(PARENT/'maps'/key/'geometry_graph'/name,folder/name)
                stat=dict(objects=len(ctx['records']),new_objects=0,added_points=0,old_point_ids_changed=0,applied_pairs=0,exact_copy=True)
            else:stat=export_method(ctx,method,folder)
            stat['seconds']=time.monotonic()-started;dump(folder/'run.json',stat)
            for f in folder.iterdir():files[str(f)]=sha(f)
            stats[key+'/'+method]=stat;print('MAP',key,method,stat,flush=True)
    dump(OUT/'predictions_complete.json',dict(complete=True,files=files,stats=stats,protocol_sha256=sha(OUT/'protocol.json'),model_seal_sha256=sha(OUT/'model_outputs_complete.json'),sealed_before_scoring=True))


def score():
    p=verify();seal=read(OUT/'predictions_complete.json');assert seal['protocol_sha256']==sha(OUT/'protocol.json')
    for f,h in seal['files'].items():assert sha(f)==h,f
    assert not (OUT/'summary.json').exists();results={};means={m:[] for m in METHODS};gate=True
    for scene in p['scenes']:
        key=scene['key'];ref=dict(np.load(scene['reference']));rows={}
        for method in METHODS:
            folder=OUT/'maps'/key/method;g=np.load(folder/'geometry.npz');objects=read(folder/'classification.json')['objects'];projected=transfer(g['xyz'],g['instance'],ref['xyz'],.2)
            r=evaluate(ref['train_instance'],ref['confidence'],projected,{o['object_id']:o['predictions']['mean'] for o in objects},EVALUATOR);dump(folder/'matches.json',clean_json(r['matches']));rows[method]=dict(metrics=clean_json(r['averages']),matched_gt_ids=sorted(matched_ids(r['matches'])),run=read(folder/'run.json'));means[method].append(float(r['averages']['allAp']))
            print('SCORE',key,method,rows[method]['metrics'],flush=True)
        expected=read(PARENT/'summary.json')['results'][key]['methods']['geometry_graph']['metrics']
        for metric in ['allAp','allAp50%']:assert abs(rows['baseline']['metrics'][metric]-expected[metric])<1e-12
        base=set(rows['baseline']['matched_gt_ids'])
        for row in rows.values():actual=set(row['matched_gt_ids']);row['changes_vs_baseline']=dict(gained=sorted(actual-base),lost=sorted(base-actual))
        ours=rows['ours'];changes=ours['changes_vs_baseline'];gate&=ours['metrics']['allAp']>max(r['metrics']['allAp'] for m,r in rows.items() if m!='ours') and len(changes['gained'])>len(changes['lost']);results[key]=dict(methods=rows)
    mean={m:float(np.mean(v)) for m,v in means.items()};gate&=mean['ours']>mean['current_only']
    dump(OUT/'summary.json',dict(complete=True,results=results,mean_primary_ap=mean,advance_gate=bool(gate),development_only=True,official_leaderboard=False,protocol_sha256=sha(OUT/'protocol.json'),prediction_seal_sha256=sha(OUT/'predictions_complete.json')));print('GATE',bool(gate),mean,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','maps','score']);globals()[parser.parse_args().action]()
