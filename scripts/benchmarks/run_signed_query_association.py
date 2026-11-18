"""Joint historical identity merging and multi-view measured geometry formation."""
import argparse,copy,runpy,shutil,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import logit
from paper_final_common import ROOT,read,dump,sha
from run_hypothesis_admission import OUT as PREVIOUS,BASE,INPUT,EVIDENCE,OWNERSHIP,EVALUATOR,verify as verify_previous
from extract_query_hypothesis_evidence import sealed as evidence_sealed
from omni_semantic_map.query_hypothesis_evidence import membership,retrieve
from omni_semantic_map.hypothesis_admission import fit_quality,predict_quality
from omni_semantic_map.signed_query_association import FEATURES,GEOMETRY_FEATURES,pair_features,positive_components,signed_groups
from evaluate_kitti360_development import read_static_ply,transfer,clean_json
from omni_semantic_map.kitti360_evaluation import evaluate
from run_discovery_query_benchmark import matched_ids

OUT=ROOT/'build/benchmarks/signed_query_association_v1'
MC=ROOT/'build/benchmarks/visibility_independent_20261112/adapters/MaskClusteringCPU'
METHODS=['baseline','geometry_graph','mc_core','positive_graph','geometry_signed','ours_signed']
CLASSES={11:2,26:13}


def freeze():
    assert not OUT.exists();p=verify_previous();assert not read(PREVIOUS/'summary.json')['advance_gate'];evidence_sealed('index_complete');evidence_sealed('evidence_complete')
    sources=[Path(__file__),ROOT/'omni_semantic_map/omni_semantic_map/signed_query_association.py',ROOT/'omni_semantic_map/test/test_signed_query_association.py',ROOT/'omni_semantic_map/omni_semantic_map/hypothesis_admission.py',ROOT/'omni_semantic_map/omni_semantic_map/query_hypothesis_evidence.py']
    sources += [MC/f for f in ['graph/node.py','graph/construction.py','graph/iterative_clustering.py']]
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),scenes=p['scenes'],methods=METHODS,source_hashes={str(f.resolve()):sha(f) for f in sources},parent_protocol_sha256=sha(PREVIOUS/'protocol.json'),parent_summary_sha256=sha(PREVIOUS/'summary.json'),evidence_index_sha256=sha(EVIDENCE/'index_complete.json'),evidence_readout_sha256=sha(EVIDENCE/'evidence_complete.json'),
        decision='Previous fixed-component admission failed and remains closed. This separate experiment changes the object representation: groups can merge historical IDs and union measured components across views. It is not another acceptance-score/threshold search.',
        nodes='Unlearned native original object source provenance nodes, plus all original hypothesis seeds with native score>0.1, plus a single fixed expansion to weak seed components already selected by an anchor at another frame with>=16 measured-visible cells and cellIoU>0.5. No recursive node acquisition or GT selection. Original component sets, no baseline occupancy in features or training labels. Old nodes use native source provenance, excluding later learned geometry additions.',
        readout='Reuse sealed raw-seed readouts; compute same reader for original old-object supports. One query per node/frame, actual measured visibility. Pair excludes both seed frames. Query-mask overlap uses current measured-cell IoU of the two selected raw queries, so differing native query IDs are not automatically a contradiction.',
        features=list(FEATURES),
        labels='Training-only nearest known reference within0.2m. Node needs>=16 known source points,>=0.5 known fraction and>0.5 confidence-weighted dominant-GT purity. Same predicted-class dominant identity is positive anchor; confidently wrong-class dominant identity is valid negative anchor. Mixed/unknown nodes excluded from pair supervision. A pair is positive only if both valid nodes share a nonnegative same-class GT anchor. Training edges also require common measured opportunity excluding both seeds.',
        fits='Four fixed logistic fits: geometry11features and full22features, two scene-crossfit folds. Same L2=0.001 including intercept, L-BFGS-B300/gtol1e-6/ftol1e-10 as prior optimizer; train-only standardization, no oversampling, no threshold or checkpoint search. Exposed0000/0002 development only.',
        methods_detail='Geometry graph: cellIoU>0.5 connected components. MC-core: unchanged upstream CPU iterative-clustering core and observer-quantile schedule,connect0.9, on shared selected-query observer matrices. This is a readout/core adaptation, NOT native full MaskClustering: original backprojection/containment/undersegmentation/postprocessing are replaced by common raw-query readout/export. Positive graph: learned full pair probability>0.5 connected components. Geometry signed: signed agglomeration of geometry-only logits. Ours signed: signed agglomeration of full pair logits. Absent measured opportunity gets zero learned edge cost. All groups same-class. Standard clustering/linear heads not claimed novel by themselves.',
        signed_objective='Greedy positive total-gain merging, sum of all known cross-group log odds. Missing evidence zero; negative evidence can defeat positive bridge. Clip probabilities only to[1e-6,1-1e-6] for finite logits; no additional merge threshold.',
        export='Same union of group measured source sets in all arms. Any group with old objects preserves their original point coordinates and merges IDs to smallest old ID, score=max old native score. No splitting/deleting old points. New group needs>=3 distinct seed frames,>=1 native anchor>0.1 and>=16 unoccupied measured sources; score=mean of per-frame max native score. Classes fixed by same-class graph. Lock all original provenance plus frozen intrinsic additions; allocate new sources once by descending group score/stable output ID. Same native intrinsic baseline, no extra DBSCAN/NMS/cleanup.',
        budget='Four fits,ten new maps plus two exact baseline copies;zero new model inferences. No new windows or robot in this development stage. Seal all maps before final scoring.',
        gate='Ours mean AP50:90>=best of all five controls+0.02, no per-window AP50:90/AP50 regression vs intrinsic baseline,>=1 gained AP50 GT and0lost. Otherwise close version without tuning or expansion. Success permits separately frozen robot and genuinely unused-window assessment.',
        limitations='Window-end offline only; no independent view validation claim because old native objects already contain observations. Native upstream core retained with explicitly changed evidence adapter. No official leaderboard, first-method or established-novelty claim. Real robot labels still unavailable.'))
    print('FROZEN',flush=True)


def verify():
    p=read(OUT/'protocol.json');verify_previous();assert sha(PREVIOUS/'protocol.json')==p['parent_protocol_sha256'];assert sha(PREVIOUS/'summary.json')==p['parent_summary_sha256']
    assert sha(EVIDENCE/'index_complete.json')==p['evidence_index_sha256'];assert sha(EVIDENCE/'evidence_complete.json')==p['evidence_readout_sha256']
    for f,h in p['source_hashes'].items():assert sha(f)==h,f
    return p


def sealed(name):
    d=read(OUT/(name+'.json'))
    for f,h in d['files'].items():assert sha(f)==h,f
    return d


def ragged(sets):return np.concatenate(sets),np.r_[0,np.cumsum([len(v) for v in sets])]
def pieces(values,offsets):return [values[a:b] for a,b in zip(offsets[:-1],offsets[1:])]


def nodes():
    p=verify();index=evidence_sealed('index_complete');evidence_sealed('evidence_complete');assert not (OUT/'nodes_complete.json').exists();files={};stats={}
    for s in p['scenes']:
        key=s['key'];start=time.monotonic();meta=read(EVIDENCE/'index'/key/'hypotheses.json');raw=dict(np.load(EVIDENCE/'index'/key/'hypotheses.npz'));history=dict(np.load(EVIDENCE/'evidence'/key/'readout.npz'));anchors=np.array([j for j,m in enumerate(meta) if m['score']>.1]);lookup={(m['frame_id'],q):j for j,m in enumerate(meta) for q in m['aliases']};selected=set(map(int,anchors))
        for k,f in enumerate(history['witness_frames']):
            good=anchors[(history['visible_cells'][anchors,k]>=16)&(history['agreement'][anchors,k]>.5)&(history['query_ids'][anchors,k]>=0)]
            selected.update(lookup[(int(f),int(q))] for q in np.unique(history['query_ids'][good,k]) if (int(f),int(q)) in lookup)
        selected=np.array(sorted(selected),np.int64);provenance=read(OWNERSHIP/'features'/key/'core_sources.json');objects=read(BASE/'maps'/key/'intrinsic_geometry/classification.json')['objects'];manifest=read(INPUT/key/'manifest.json');xyz=np.load(manifest['geometry'])['xyz'];cell_ids=np.load(INPUT/'query_cells'/key/'source_to_cell.npy');source_sets=[];metadata=[]
        for o in sorted(objects,key=lambda o:o['object_id']):
            ids=np.array(provenance[str(o['object_id'])],np.int64);source_sets.append(ids);metadata.append(dict(old_object_id=o['object_id'],hypothesis_index=-1,seed_frame=-1,semantic_id=o['predictions']['mean']['semantic_id'],score=o['predictions']['mean']['score']))
        old_count=len(metadata)
        for j in selected:
            a,b=raw['source_offsets'][j:j+2];source_sets.append(raw['source_ids'][a:b]);m=meta[j];metadata.append(dict(old_object_id=-1,hypothesis_index=int(j),seed_frame=m['frame_id'],semantic_id=m['semantic_id'],score=m['score']))
        cells=[np.unique(cell_ids[v]) for v in source_sets];n=len(cells);t=len(history['witness_frames']);classes=np.asarray([m['semantic_id'] for m in metadata]);h=membership(cells,index['results'][key]['cell_count']);centers=[];rms=[]
        for ids in source_sets:
            points=xyz[ids];center=points.mean(0) if len(ids) else np.zeros(3);centers.append(center);rms.append(float(np.sqrt(np.mean(np.sum((points-center)**2,axis=1)))) if len(ids) else 0.)
        result={f:np.zeros((n,t),history[f].dtype) for f in ['query_ids','agreement','scores','visible_cells']};result['query_ids'][:]=-1
        for f in result:result[f][old_count:]=history[f][selected]
        overlaps=np.zeros((t,400,400),np.float32)
        for k,r in enumerate(index['results'][key]['frames']):
            data=dict(np.load(r['file']));q=membership(pieces(data['cell_ids'],data['offsets']),h.shape[1]);z=retrieve(h[:old_count],classes[:old_count],q,data['semantic_ids'],data['scores'],data['visible_cells']);good=z['query']>=0;result['visible_cells'][:old_count,k]=z['visible_cells'];result['query_ids'][np.flatnonzero(good),k]=data['query_ids'][z['query'][good]];result['agreement'][:old_count,k]=z['agreement'];result['scores'][:old_count,k]=z['score']
            intersections=(q@q.T).toarray();sizes=np.diff(q.indptr);union=sizes[:,None]+sizes[None,:]-intersections;iou=np.divide(intersections,union,out=np.zeros_like(intersections),where=union>0);assert data['query_ids'].max(initial=0)<400;overlaps[k][np.ix_(data['query_ids'],data['query_ids'])]=iou
        folder=OUT/'nodes'/key;folder.mkdir(parents=True,exist_ok=False);sv,so=ragged(source_sets);cv,co=ragged(cells);np.savez_compressed(folder/'nodes.npz',source_ids=sv,source_offsets=so,cell_ids=cv,cell_offsets=co,centers=np.asarray(centers),rms=np.asarray(rms),semantic_ids=classes,scores=np.asarray([m['score'] for m in metadata]),seed_frames=np.asarray([m['seed_frame'] for m in metadata]),old=np.arange(n)<old_count,original_hypotheses=selected);np.savez_compressed(folder/'readout.npz',**result,witness_frames=history['witness_frames']);np.savez_compressed(folder/'query_overlap.npz',iou=overlaps);dump(folder/'nodes.json',metadata)
        for f in folder.iterdir():files[str(f)]=sha(f)
        stats[key]=dict(nodes=n,old_nodes=old_count,anchors=len(anchors),weak_nodes=len(selected)-len(anchors),seconds=time.monotonic()-start);print('NODES',key,stats[key],flush=True)
    dump(OUT/'nodes_complete.json',dict(complete=True,files=files,stats=stats))


def pairs():
    p=verify();sealed('nodes_complete');assert not (OUT/'pairs_complete.json').exists();files={};stats={}
    for s in p['scenes']:
        key=s['key'];start=time.monotonic();folder=OUT/'nodes'/key;d=dict(np.load(folder/'nodes.npz'));r=dict(np.load(folder/'readout.npz'));overlaps=np.load(folder/'query_overlap.npz')['iou'];sets=pieces(d['cell_ids'],d['cell_offsets']);h=membership(sets,int(d['cell_ids'].max())+1);intersection=(h@h.T).toarray();a,b=np.triu_indices(len(sets),1);same=d['semantic_ids'][a]==d['semantic_ids'][b];a=a[same];b=b[same];x=np.zeros((len(a),len(FEATURES)));opportunity=np.zeros(len(a),bool);geometry={f:d[f] for f in ['centers','rms','semantic_ids','scores','seed_frames']};geometry['old']=d['old'].astype(np.int64);geometry['cells']=np.diff(d['cell_offsets'])
        for lo in range(0,len(a),2048):
            hi=min(lo+2048,len(a));left=a[lo:hi];right=b[lo:hi];qa=np.maximum(r['query_ids'][left],0);qb=np.maximum(r['query_ids'][right],0);qov=overlaps[np.arange(len(r['witness_frames']))[None,:],qa,qb];geometry['intersections']=intersection[left,right];x[lo:hi],opportunity[lo:hi]=pair_features(left,right,geometry,r,qov)
        path=OUT/'pairs'/key/'features.npz';path.parent.mkdir(parents=True,exist_ok=False);np.savez_compressed(path,left=a,right=b,features=x,opportunity=opportunity);files[str(path)]=sha(path);stats[key]=dict(pairs=len(a),common_measured_pairs=int(opportunity.sum()),seconds=time.monotonic()-start);print('PAIRS',key,stats[key],flush=True)
    dump(OUT/'pairs_complete.json',dict(complete=True,files=files,stats=stats))


def labels():
    p=verify();sealed('pairs_complete');assert not (OUT/'labels_complete.json').exists();files={};stats={};references=read(OWNERSHIP/'protocol.json')['references'];taxonomy=runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['id2label']
    for s in p['scenes']:
        key=s['key'];d=dict(np.load(OUT/'nodes'/key/'nodes.npz'));pairs=dict(np.load(OUT/'pairs'/key/'features.npz'));xyz=np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz'];ref=dict(np.load(s['reference']));info=references[key];assert sha(info['raw'])==info['sha256'];raw=read_static_ply(info['raw']);train=np.array([taxonomy[int(v)].trainId for v in raw['semantic']]);nearest=transfer(ref['xyz'],np.arange(1,len(raw)+1),xyz,.2);ix=np.maximum(nearest-1,0);known=(nearest>0)&(train[ix]!=255)&(ref['confidence'][ix]>0);truth=ref['train_instance'][ix];weight=ref['confidence'][ix];sets=pieces(d['source_ids'],d['source_offsets']);anchor=np.full(len(sets),-1,np.int64);valid=np.zeros(len(sets),bool)
        for j,ids in enumerate(sets):
            selected=ids[known[ids]]
            if len(selected)<16 or len(selected)<.5*len(ids):continue
            mass=np.bincount(truth[selected],weights=weight[selected]);gid=int(mass.argmax());purity=mass[gid]/mass.sum()
            if purity<=.5:continue
            valid[j]=True
            if gid//1000==CLASSES[int(d['semantic_ids'][j])]:anchor[j]=gid
        a,b=pairs['left'],pairs['right'];eligible=valid[a]&valid[b]&pairs['opportunity'];positive=(anchor[a]>=0)&(anchor[a]==anchor[b]);folder=OUT/'training_labels'/key;folder.mkdir(parents=True,exist_ok=False);path=folder/'TRAINING_ONLY.npz';np.savez_compressed(path,node_anchor=anchor,node_valid=valid,eligible=eligible,positive=positive);files[str(path)]=sha(path);stats[key]=dict(training_pairs=int(eligible.sum()),positive_pairs=int((eligible&positive).sum()));print('LABELS',key,stats[key],flush=True)
    dump(OUT/'labels_complete.json',dict(complete=True,files=files,stats=stats))


def train():
    p=verify();sealed('pairs_complete');sealed('labels_complete');assert not (OUT/'trained.json').exists();files={};stats={};keys=[s['key'] for s in p['scenes']]
    for target in keys:
        training=next(k for k in keys if k!=target);a=dict(np.load(OUT/'pairs'/training/'features.npz'));b=dict(np.load(OUT/'pairs'/target/'features.npz'));labels=dict(np.load(OUT/'training_labels'/training/'TRAINING_ONLY.npz'));scores={}
        for name,dim in [('geometry',len(GEOMETRY_FEATURES)),('evidence',len(FEATURES))]:
            model=fit_quality(a['features'][labels['eligible'],:dim],labels['positive'][labels['eligible']]);scores[name]=predict_quality(model,b['features'][:,:dim]);model.update(training=training,target=target,dimensions=dim);folder=OUT/'models'/target;folder.mkdir(parents=True,exist_ok=True);path=folder/(name+'.json');dump(path,clean_json(model));files[str(path)]=sha(path);stats[target+'/'+name]={k:model[k] for k in ['training','training_rows','positive_rows','loss','iterations']};print('TRAINED',target,name,stats[target+'/'+name],flush=True)
        path=OUT/'models'/target/'probabilities.npz';np.savez_compressed(path,**scores);files[str(path)]=sha(path)
    dump(OUT/'trained.json',dict(complete=True,files=files,stats=stats))


def mc_groups(data,readout):
    import torch
    torch.set_num_threads(1);sys.path.insert(0,str(MC))
    from graph.node import Node
    from graph.construction import get_observer_num_thresholds
    from graph.iterative_clustering import iterative_clustering
    query=readout['query_ids'];visible=(readout['visible_cells']>=16)&(readout['witness_frames'][None,:]!=data['seed_frames'][:,None]);selected=visible&(query>=0);tokens=query+400*np.arange(query.shape[1])[None,:];dictionary=np.unique(tokens[selected]);contained=np.zeros((len(query),len(dictionary)),np.float32);rows,columns=np.nonzero(selected);contained[rows,np.searchsorted(dictionary,tokens[rows,columns])]=1
    vf=torch.from_numpy(visible.astype(np.float32));cm=torch.from_numpy(contained);nodes=[Node([(j,1)],vf[j],cm[j],{j},(0,j),None) for j in range(len(query))]
    if not visible.any():return [[j] for j in range(len(query))]
    thresholds=get_observer_num_thresholds(vf);clustered=iterative_clustering(nodes,thresholds,.9,False)
    return sorted([sorted(n.point_ids) for n in clustered],key=lambda g:g[0])


def maps():
    p=verify();sealed('trained');sealed('pairs_complete');sealed('nodes_complete');assert not (OUT/'predictions_complete.json').exists();files={};stats={}
    for s in p['scenes']:
        key=s['key'];d=dict(np.load(OUT/'nodes'/key/'nodes.npz'));readout=dict(np.load(OUT/'nodes'/key/'readout.npz'));pairs=dict(np.load(OUT/'pairs'/key/'features.npz'));prob=dict(np.load(OUT/'models'/key/'probabilities.npz'));meta=read(OUT/'nodes'/key/'nodes.json');sets=pieces(d['source_ids'],d['source_offsets']);xyz=np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz'];basefolder=BASE/'maps'/key/'intrinsic_geometry';base=dict(np.load(basefolder/'geometry.npz'));classification=read(basefolder/'classification.json');by_id={o['object_id']:o for o in classification['objects']};old_ids=[m['old_object_id'] for m in meta if m['old_object_id']>=0];locked=np.unique(np.r_[np.concatenate([sets[j] for j in np.flatnonzero(d['old'])]),base['added_source_ids']])
        for method in METHODS:
            start=time.monotonic();folder=OUT/'maps'/key/method;folder.mkdir(parents=True,exist_ok=False);merges=[]
            if method=='baseline':
                for name in ['geometry.npz','classification.json']:shutil.copyfile(basefolder/name,folder/name)
                stat=dict(objects=len(by_id),new_objects=0,old_ids_merged=0,added_points=0,exact_copy=True,seconds=time.monotonic()-start)
            else:
                if method=='geometry_graph':groups=positive_components(len(meta),pairs['left'],pairs['right'],pairs['features'][:,0]>.5)
                elif method=='mc_core':groups=mc_groups(d,readout)
                elif method=='positive_graph':groups=positive_components(len(meta),pairs['left'],pairs['right'],(prob['evidence']>.5)&pairs['opportunity'])
                else:
                    name='geometry' if method=='geometry_signed' else 'evidence';weights=np.where(pairs['opportunity'],logit(np.clip(prob[name],1e-6,1-1e-6)),0.);groups,merges=signed_groups(len(meta),pairs['left'],pairs['right'],weights)
                assignments={};proposals=[];next_id=max(old_ids)+1
                for members in groups:
                    cls=np.unique(d['semantic_ids'][members]);assert len(cls)==1;sem=int(cls[0]);old=sorted(meta[j]['old_object_id'] for j in members if meta[j]['old_object_id']>=0);raw=[j for j in members if meta[j]['hypothesis_index']>=0];frames=sorted(set(int(d['seed_frames'][j]) for j in raw))
                    if old:
                        oid=old[0];score=max(by_id[i]['predictions']['mean']['score'] for i in old)
                        for i in old:assignments[i]=oid
                    else:
                        if len(frames)<3 or max((d['scores'][j] for j in raw),default=0)<=.1:continue
                        oid=next_id;next_id+=1;score=float(np.mean([max(d['scores'][j] for j in raw if d['seed_frames'][j]==f) for f in frames]))
                    ids=np.unique(np.concatenate([sets[j] for j in members]));proposals.append(dict(object_id=oid,semantic_id=sem,score=score,old_ids=old,node_ids=members,frames=frames,source_ids=ids))
                assert set(assignments)==set(old_ids);used=np.zeros(len(xyz),bool);used[locked]=True;new_xyz=[];new_instance=[];new_sources=[];records=[];group_records=[]
                for proposal in sorted(proposals,key=lambda q:(-q['score'],q['object_id'])):
                    ids=proposal.pop('source_ids');free=ids[~used[ids]]
                    if not proposal['old_ids'] and len(free)<16:continue
                    used[free]=True;oid=proposal['object_id'];new_xyz.append(xyz[free]);new_instance.append(np.full(len(free),oid,np.int64));new_sources.append(free);proposal['added_points']=len(free);group_records.append(proposal)
                    if len(proposal['old_ids'])==1:record=copy.deepcopy(by_id[proposal['old_ids'][0]])
                    else:
                        sem=proposal['semantic_id'];observations=sorted(set(v for i in proposal['old_ids'] for v in by_id[i].get('observation_ids',[])));record=dict(object_id=oid,observation_ids=observations,selected_observation_ids=observations,predictions=dict(mean=dict(name='car' if sem==26 else 'building',semantic_id=sem,train_id=CLASSES[sem],score=proposal['score'],retained=True)))
                    record['source_node_ids']=proposal['node_ids'];record['merged_old_ids']=proposal['old_ids'];records.append(record)
                old_instance=np.array([assignments[int(v)] for v in base['instance']],np.int64);added=np.concatenate(new_sources) if new_sources else np.empty(0,np.int64);points=np.concatenate([base['xyz']]+new_xyz);instance=np.concatenate([old_instance]+new_instance);assert len(np.unique(added))==len(added) and not np.isin(added,locked).any();np.testing.assert_array_equal(points[:len(base['xyz'])],base['xyz']);np.testing.assert_array_equal(points[len(base['xyz']):],xyz[added])
                np.savez_compressed(folder/'geometry.npz',xyz=points,instance=instance,added_source_ids=added);export=copy.deepcopy(classification);export['objects']=sorted(records,key=lambda q:q['object_id']);dump(folder/'classification.json',export);dump(folder/'groups.json',dict(groups=group_records,old_id_assignment=assignments,merge_trace=merges));stat=dict(objects=len(records),new_objects=sum(not r['old_ids'] for r in group_records),old_ids_merged=sum(len(r['old_ids'])-1 for r in group_records if r['old_ids']),old_point_ids_changed=int((old_instance!=base['instance']).sum()),added_points=len(added),old_point_coordinates_exact=True,new_sources_unique_and_measured=True,seconds=time.monotonic()-start)
            dump(folder/'run.json',stat)
            for f in folder.iterdir():files[str(f)]=sha(f)
            stats[key+'/'+method]=stat;print('MAP',key,method,stat,flush=True)
    dump(OUT/'predictions_complete.json',dict(complete=True,files=files,stats=stats,protocol_sha256=sha(OUT/'protocol.json'),sealed_before_final_map_scoring=True))


def score():
    p=verify();sealed('predictions_complete');assert not (OUT/'summary.json').exists();results={};means={m:[] for m in METHODS};gained=lost=0;no_harm=True
    for s in p['scenes']:
        key=s['key'];ref=dict(np.load(s['reference']));rows={}
        for method in METHODS:
            folder=OUT/'maps'/key/method;g=np.load(folder/'geometry.npz');objects=read(folder/'classification.json')['objects'];projected=transfer(g['xyz'],g['instance'],ref['xyz'],.2);r=evaluate(ref['train_instance'],ref['confidence'],projected,{o['object_id']:o['predictions']['mean'] for o in objects},EVALUATOR);dump(folder/'matches.json',clean_json(r['matches']));row=dict(metrics=clean_json(r['averages']),matched_gt_ids=sorted(matched_ids(r['matches'])),run=read(folder/'run.json'));rows[method]=row;means[method].append(r['averages']['allAp']);print('SCORE',key,method,row['metrics'],flush=True)
        expected=read(BASE/'summary.json')['results'][key]['methods']['intrinsic_geometry']['metrics']
        for metric in ['allAp','allAp50%']:assert abs(rows['baseline']['metrics'][metric]-expected[metric])<1e-12
        base=set(rows['baseline']['matched_gt_ids'])
        for row in rows.values():actual=set(row['matched_gt_ids']);row['changes_vs_baseline']=dict(gained=sorted(actual-base),lost=sorted(base-actual))
        ours=rows['ours_signed'];gained+=len(ours['changes_vs_baseline']['gained']);lost+=len(ours['changes_vs_baseline']['lost']);no_harm&=all(ours['metrics'][m]>=rows['baseline']['metrics'][m]-1e-12 for m in ['allAp','allAp50%']);results[key]=dict(methods=rows)
    mean={m:float(np.mean(v)) for m,v in means.items()};gate=bool(mean['ours_signed']>=max(v for m,v in mean.items() if m!='ours_signed')+.02 and no_harm and gained>=1 and lost==0);dump(OUT/'summary.json',dict(complete=True,results=results,mean_primary_ap=mean,advance_gate=gate,no_metric_regression=bool(no_harm),gained_gt=gained,lost_gt=lost,development_crossfit=True,official_leaderboard=False,protocol_sha256=sha(OUT/'protocol.json'),prediction_seal_sha256=sha(OUT/'predictions_complete.json')));print('GATE',gate,mean,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','nodes','pairs','labels','train','maps','score']);globals()[parser.parse_args().action]()
