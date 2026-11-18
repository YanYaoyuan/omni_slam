"""Bounded crossfit comparison of measured object-hypothesis admission."""
import argparse
import copy
import runpy
import shutil
import time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from paper_final_common import ROOT,read,dump,sha
from extract_query_hypothesis_evidence import OUT as EVIDENCE,verify as verify_reader,sealed as reader_sealed
from capture_prethreshold_candidates import INPUT
from run_intrinsic_surface_ownership import OUT as BASE,verify as verify_base
from run_surface_ownership import OUT as OWNERSHIP
from evaluate_kitti360_development import read_static_ply,transfer,clean_json
from omni_semantic_map.hypothesis_admission import FEATURES,SEED_FEATURES,feature_row,fit_quality,predict_quality,select_hypotheses
from omni_semantic_map.kitti360_evaluation import evaluate,EVALUATOR_SHA256
from run_discovery_query_benchmark import matched_ids

OUT=ROOT/'build/benchmarks/hypothesis_admission_v1'
METHODS=['baseline','native01','three_frame','seed_quality','ours_evidence','shuffled_evidence']
TRAINED=['seed_quality','ours_evidence','shuffled_evidence']
CLASSES={11:2,26:13}
EVALUATOR=ROOT/'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py'


def freeze():
    assert not OUT.exists();p=verify_reader();reader_sealed('index_complete');reader_sealed('evidence_complete');verify_base(False)
    assert sha(EVALUATOR)==EVALUATOR_SHA256
    sources=[Path(__file__),ROOT/'omni_semantic_map/omni_semantic_map/hypothesis_admission.py',ROOT/'omni_semantic_map/test/test_hypothesis_admission.py',ROOT/'omni_semantic_map/omni_semantic_map/kitti360_evaluation.py',ROOT/'scripts/benchmarks/evaluate_kitti360_development.py']
    inputs={str(EVIDENCE/'index_complete.json'):sha(EVIDENCE/'index_complete.json'),str(EVIDENCE/'evidence_complete.json'):sha(EVIDENCE/'evidence_complete.json'),str(EVIDENCE/'technical_amendment_npz_io.json'):sha(EVIDENCE/'technical_amendment_npz_io.json')}
    for f in [BASE/'summary.json',OWNERSHIP/'protocol.json',ROOT/'build/datasets/kitti360_prepare/labels.py']:
        inputs[str(f)]=sha(f)
    references=read(OWNERSHIP/'protocol.json')['references']
    for s in p['scenes']:
        key=s['key'];manifest=INPUT/key/'manifest.json';m=read(manifest)
        for f in [manifest,Path(m['geometry']),INPUT/'query_cells'/key/'source_to_cell.npy',OWNERSHIP/'features'/key/'core_sources.json',BASE/'maps'/key/'intrinsic_geometry/geometry.npz',BASE/'maps'/key/'intrinsic_geometry/classification.json',Path(s['reference']),Path(references[key]['raw'])]:
            inputs[str(f)]=sha(f)
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),scenes=p['scenes'],methods=METHODS,source_hashes={str(f.resolve()):sha(f) for f in sources},input_hashes=inputs,
        candidate='All already-sealed raw-query largest components. Features, training labels and class-wise shuffle use full original components without any learned-baseline occupancy input. Only final map eligibility/allocation locks source IDs from original observation provenance plus frozen intrinsic-geometry additions. Common final eligibility: at least16 free sources and at least half original source count remains. Common NMS uses original component cells, then removes locked/already allocated sources. This prevents target-trained opposite-fold baseline occupancy leaking into quality-head training features/labels.',
        baseline='Frozen intrinsic_geometry crossfit maps, mean development AP50:90=22.639. Preserve original map point prefix, object IDs and scores. New objects only; no old-point deletion or reassignment. Candidate component geometry is fixed; this experiment does not claim cross-view shape completion.',
        mechanism='Conditional object-quality admission: use actual measured visibility and previously sealed raw-query correspondence statistics to predict whether an uncommitted geometric proposal can become a complete same-class object. Linear quality head separates evidence-value test from a larger architecture claim. Standard NMS is shared, not claimed novel.',
        features=list(FEATURES),
        training='Two crossfit folds, target0000 trained only on0002 and vice versa. Logistic BCE with L2=0.001 including intercept, L-BFGS-B max300,gtol1e-6,ftol1e-10; no oversampling/tuning/checkpoint selection. Standardization from training rows only. Seed-only9features; evidence23features; shuffled23features with history blocks permuted within predicted class and scene across full original pool, seed0. Three fits per fold, six total. Training-feature/label domain is independent of both frozen learned baseline maps; their source occupancy is used only for final map allocation.',
        supervision='Training-only standalone proposal quality proxy: union of full-reference points within0.2m of original measured component sources, before final baseline-source exclusion. Confidence-weighted best same-class full-GT IoU>0.5 is positive; GT mass>=100. Training rows need predicted mass>=100 and at least half predicted confidence mass from raw nonignored trainId!=255 labels. Remaining known proposals negative. Unknown-dominated proposals excluded from training, not removed from predictions. Radius-union proxy is not final map AP; final projection uses unchanged nearest-map-point kernel with competition.',
        controls='Exact baseline; native score>0.1; native score>0 and>=3 other frames with>=16visible cells and cellIoU>0.5; seed-quality posterior>0.5; Ours evidence posterior>0.5; shuffled-evidence posterior>0.5. Same greedy descending-score same-class cellIoU>0.5 suppression, stable original-index ties, then source-exclusive allocation and>=16remaining points. No final extra cleaning or GT-based selection.',
        timing='Window-end offline experiment with all evidence available after window. No per-frame/real-time or zero-delay claim.',
        prediction_policy='Seal GT-free candidates/features before label generation. Each scene prediction uses only model trained on other scene. Seal all12 map outputs before any final map scoring. Exposed development data; no claim of independent generalization. No0004/0005 access or robot execution in this development stage.',
        gate='Ours meanAP50:90>=strongest of all five controls+0.02, neither scene AP50:90/AP50 below frozen intrinsic baseline, at least one gained AP50 GT and zero lost. All outputs reported; failure stops this candidate without threshold/model sweeps. Passing permits separately frozen robot/independent assessment, not an automatic generalization claim.',
        budget='0new foundation-model inferences,6CPU fits,10new maps+2exact baseline copies,2exposed windows. No architecture/hyperparameter sweeps. Robot and new independent tests conditional on this development result.',
        scope='A candidate mechanism experiment, not established novelty/SOTA. Different raw camera/LiDAR input task from official accumulated XYZRGB leaderboard. Reference/GPU/cache limitations retained.'))
    print('FROZEN',flush=True)


def verify():
    p=read(OUT/'protocol.json');verify_reader();verify_base(False)
    for f,h in {**p['source_hashes'],**p['input_hashes']}.items(): assert sha(f)==h,f
    return p


def sealed(name):
    d=read(OUT/(name+'.json'))
    for f,h in d['files'].items(): assert sha(f)==h,f
    return d


def split_values(values,offsets):
    return [values[a:b] for a,b in zip(offsets[:-1],offsets[1:])]


def ragged(sets):
    return (np.concatenate(sets) if sets else np.empty(0,np.int64)),np.r_[0,np.cumsum([len(s) for s in sets])]


def features():
    p=verify();reader_sealed('index_complete');reader_sealed('evidence_complete');assert not (OUT/'features_complete.json').exists();files={};stats={}
    for s in p['scenes']:
        key=s['key'];start=time.monotonic();manifest=read(INPUT/key/'manifest.json');xyz=np.load(manifest['geometry'])['xyz'];cells=np.load(INPUT/'query_cells'/key/'source_to_cell.npy')
        base=np.load(BASE/'maps'/key/'intrinsic_geometry/geometry.npz');provenance=read(OWNERSHIP/'features'/key/'core_sources.json')
        added=base['added_source_ids'];np.testing.assert_array_equal(base['xyz'][-len(added):],xyz[added])
        locked=np.unique(np.r_[np.concatenate([np.asarray(v,np.int64) for v in provenance.values()]),added]);occupied=np.zeros(len(xyz),bool);occupied[locked]=True
        meta=read(EVIDENCE/'index'/key/'hypotheses.json');d=np.load(EVIDENCE/'index'/key/'hypotheses.npz');raw=d['source_ids'];offsets=d['source_offsets'];hist=dict(np.load(EVIDENCE/'evidence'/key/'readout.npz'))
        n=len(meta);x=np.zeros((n,len(FEATURES)),float);eligible=np.zeros(n,bool);sources=[];cell_sets=[]
        for j,m in enumerate(meta):
            a,b=offsets[j:j+2];ids=raw[a:b];free=ids[~occupied[ids]];hc=np.unique(cells[ids]);sources.append(ids);cell_sets.append(hc)
            eligible[j]=len(free)>=16 and len(free)>=.5*len(ids)
            history={f:hist[f][j] for f in ['visible_cells','query_ids','agreement','scores','class_logits']}
            x[j]=feature_row(xyz[ids],len(hc),m,history,hist['witness_frames'])
        shuffled=x.copy();permutation=np.arange(n);rng=np.random.RandomState(0);sem=np.asarray([m['semantic_id'] for m in meta])
        for cls in sorted(CLASSES):
            rows=np.flatnonzero(sem==cls);permutation[rows]=rng.permutation(rows)
        shuffled[:,len(SEED_FEATURES):]=x[permutation,len(SEED_FEATURES):]
        counts=((hist['agreement']>.5)&(hist['visible_cells']>=16)&(hist['seed_frames'][:,None]!=hist['witness_frames'][None,:])).sum(1)
        vals,off=ragged(sources);cv,co=ragged(cell_sets);folder=OUT/'features'/key;folder.mkdir(parents=True,exist_ok=False);path=folder/'candidates.npz'
        np.savez_compressed(path,features=x,shuffled_features=shuffled,permutation=permutation,eligible=eligible,source_ids=vals,source_offsets=off,cell_ids=cv,cell_offsets=co,locked_sources=locked,semantic_ids=sem,native_scores=np.asarray([m['score'] for m in meta]),confirmation_count=counts)
        files[str(path)]=sha(path);stats[key]=dict(all_hypotheses=n,eligible=int(eligible.sum()),locked_sources=len(locked),source_count=len(xyz),seconds=time.monotonic()-start)
        print('FEATURES',key,stats[key],flush=True)
    dump(OUT/'features_complete.json',dict(complete=True,files=files,stats=stats,protocol_sha256=sha(OUT/'protocol.json')))


def labels():
    p=verify();sealed('features_complete');assert not (OUT/'training_labels_complete.json').exists();files={};stats={};parent=read(OWNERSHIP/'protocol.json');taxonomy=runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['id2label']
    for s in p['scenes']:
        key=s['key'];start=time.monotonic();data=dict(np.load(OUT/'features'/key/'candidates.npz'));xyz=np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz'];ref=dict(np.load(s['reference']));info=parent['references'][key];assert sha(info['raw'])==info['sha256'];raw=read_static_ply(info['raw']);known=np.asarray([taxonomy[int(v)].trainId!=255 for v in raw['semantic']]);truth=ref['train_instance'];weight=ref['confidence'].astype(float)
        assert len(known)==len(truth);tree=cKDTree(ref['xyz']);mass=np.bincount(truth,weights=weight);n=len(data['eligible']);best_iou=np.zeros(n);best_gt=np.full(n,-1,np.int64);trainable=np.zeros(n,bool);predicted_mass=np.zeros(n);known_fraction=np.zeros(n)
        sources=split_values(data['source_ids'],data['source_offsets']);rows=np.arange(n)
        for rank,j in enumerate(rows):
            neighborhoods=tree.query_ball_point(xyz[sources[j]],.2,workers=1)
            covered=np.unique(np.concatenate(neighborhoods)).astype(np.int64) if len(neighborhoods) else np.empty(0,np.int64)
            if not len(covered): continue
            pm=float(weight[covered].sum());predicted_mass[j]=pm;known_fraction[j]=float(weight[covered][known[covered]].sum()/max(pm,1e-12));trainable[j]=pm>=100 and known_fraction[j]>=.5
            intersection=np.bincount(truth[covered],weights=weight[covered],minlength=len(mass));targets=np.flatnonzero((np.arange(len(mass))//1000==CLASSES[int(data['semantic_ids'][j])])&(mass>=100))
            if len(targets):
                quality=intersection[targets]/np.maximum(pm+mass[targets]-intersection[targets],1e-12);winner=int(np.argmax(quality));best_iou[j]=quality[winner];best_gt[j]=targets[winner]
            if (rank+1)%1000==0: print('LABELS',key,rank+1,len(rows),flush=True)
        folder=OUT/'training_labels'/key;folder.mkdir(parents=True,exist_ok=False);path=folder/'TRAINING_ONLY.npz';np.savez_compressed(path,eligible=trainable,positive=best_iou>.5,best_iou=best_iou,best_gt=best_gt,predicted_mass=predicted_mass,known_fraction=known_fraction);files[str(path)]=sha(path);stats[key]=dict(training_rows=int(trainable.sum()),positives=int((trainable&(best_iou>.5)).sum()),seconds=time.monotonic()-start)
        print('LABELS COMPLETE',key,stats[key],flush=True)
    dump(OUT/'training_labels_complete.json',dict(complete=True,files=files,stats=stats))


def train():
    p=verify();sealed('features_complete');sealed('training_labels_complete');assert not (OUT/'trained.json').exists();files={};stats={};keys=[s['key'] for s in p['scenes']]
    for target in keys:
        training=next(k for k in keys if k!=target);a=dict(np.load(OUT/'features'/training/'candidates.npz'));b=dict(np.load(OUT/'features'/target/'candidates.npz'));labels=dict(np.load(OUT/'training_labels'/training/'TRAINING_ONLY.npz'));use=labels['eligible'];scores={}
        for method in TRAINED:
            name='shuffled_features' if method=='shuffled_evidence' else 'features';dim=len(SEED_FEATURES) if method=='seed_quality' else len(FEATURES);model=fit_quality(a[name][use,:dim],labels['positive'][use]);scores[method]=predict_quality(model,b[name][:,:dim]);model.update(training=training,target=target,method=method,dimensions=dim)
            folder=OUT/'models'/target;folder.mkdir(parents=True,exist_ok=True);path=folder/(method+'.json');dump(path,clean_json(model));files[str(path)]=sha(path);stats[target+'/'+method]={k:model[k] for k in ['training','training_rows','positive_rows','loss','iterations']};print('TRAINED',target,method,stats[target+'/'+method],flush=True)
        path=OUT/'models'/target/'probabilities.npz';np.savez_compressed(path,**scores);files[str(path)]=sha(path)
    dump(OUT/'trained.json',dict(complete=True,files=files,stats=stats))


def maps():
    p=verify();sealed('features_complete');sealed('trained');assert not (OUT/'predictions_complete.json').exists();files={};stats={}
    for s in p['scenes']:
        key=s['key'];data=dict(np.load(OUT/'features'/key/'candidates.npz'));prob=dict(np.load(OUT/'models'/key/'probabilities.npz'));xyz=np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz'];basefolder=BASE/'maps'/key/'intrinsic_geometry';base=dict(np.load(basefolder/'geometry.npz'));classification=read(basefolder/'classification.json');meta=read(EVIDENCE/'index'/key/'hypotheses.json')
        sources=split_values(data['source_ids'],data['source_offsets']);cells=split_values(data['cell_ids'],data['cell_offsets'])
        for method in METHODS:
            start=time.monotonic();folder=OUT/'maps'/key/method;folder.mkdir(parents=True,exist_ok=False)
            if method=='baseline':
                for name in ['geometry.npz','classification.json']: shutil.copyfile(basefolder/name,folder/name)
                stat=dict(objects=len(classification['objects']),new_objects=0,added_points=0,baseline_exact=True,seconds=time.monotonic()-start)
            else:
                scores=prob[method] if method in TRAINED else data['native_scores'];eligible=data['eligible'].copy();threshold=.5 if method in TRAINED else (.1 if method=='native01' else 0.)
                if method=='three_frame': eligible &= data['confirmation_count']>=3
                selected,claimed=select_hypotheses(sources,cells,data['semantic_ids'],scores,eligible,len(xyz),data['locked_sources'],threshold)
                records=copy.deepcopy(classification);next_id=max(o['object_id'] for o in records['objects'])+1;point_ids=[];owners=[]
                for j,ids in zip(selected,claimed):
                    sem=int(data['semantic_ids'][j]);record=dict(object_id=next_id,observation_ids=[],selected_observation_ids=[],source_hypothesis_index=j,seed_frame=meta[j]['frame_id'],seed_query_id=meta[j]['query_id'],predictions=dict(mean=dict(name='car' if sem==26 else 'building',semantic_id=sem,train_id=CLASSES[sem],score=float(scores[j]),retained=True)))
                    records['objects'].append(record);point_ids.append(ids);owners.append(np.full(len(ids),next_id,np.int64));next_id+=1
                added=np.concatenate(point_ids) if point_ids else np.empty(0,np.int64);oid=np.concatenate(owners) if owners else np.empty(0,np.int64)
                assert len(np.unique(added))==len(added) and not np.isin(added,data['locked_sources']).any()
                new_xyz=np.concatenate((base['xyz'],xyz[added]));new_instance=np.r_[base['instance'],oid];np.testing.assert_array_equal(new_xyz[:len(base['xyz'])],base['xyz']);np.testing.assert_array_equal(new_instance[:len(base['instance'])],base['instance']);assert records['objects'][:len(classification['objects'])]==classification['objects']
                np.savez_compressed(folder/'geometry.npz',xyz=new_xyz,instance=new_instance,added_source_ids=added);dump(folder/'classification.json',records);dump(folder/'decisions.json',dict(selected_hypotheses=selected,new_object_ids=list(range(next_id-len(selected),next_id)),threshold=threshold))
                stat=dict(objects=len(records['objects']),new_objects=len(selected),added_points=len(added),old_prefix_exact=True,old_object_records_exact=True,unique_free_measured_sources=True,seconds=time.monotonic()-start)
            dump(folder/'run.json',stat)
            for f in folder.iterdir():files[str(f)]=sha(f)
            stats[key+'/'+method]=stat;print('MAP',key,method,stat,flush=True)
    dump(OUT/'predictions_complete.json',dict(complete=True,files=files,stats=stats,protocol_sha256=sha(OUT/'protocol.json'),all_predictions_sealed_before_map_scoring=True))


def score():
    p=verify();sealed('predictions_complete');assert not (OUT/'summary.json').exists();results={};means={m:[] for m in METHODS};gained=lost=0;no_harm=True
    for s in p['scenes']:
        key=s['key'];ref=dict(np.load(s['reference']));rows={}
        for method in METHODS:
            folder=OUT/'maps'/key/method;g=np.load(folder/'geometry.npz');objects=read(folder/'classification.json')['objects'];pred=transfer(g['xyz'],g['instance'],ref['xyz'],.2);r=evaluate(ref['train_instance'],ref['confidence'],pred,{o['object_id']:o['predictions']['mean'] for o in objects},EVALUATOR);dump(folder/'matches.json',clean_json(r['matches']));row=dict(metrics=clean_json(r['averages']),matched_gt_ids=sorted(matched_ids(r['matches'])),run=read(folder/'run.json'));rows[method]=row;means[method].append(r['averages']['allAp']);print('SCORE',key,method,row['metrics'],flush=True)
        old=read(BASE/'summary.json')['results'][key]['methods']['intrinsic_geometry']['metrics']
        for metric in ['allAp','allAp50%']: assert abs(rows['baseline']['metrics'][metric]-old[metric])<1e-12
        base=set(rows['baseline']['matched_gt_ids'])
        for row in rows.values():
            actual=set(row['matched_gt_ids']);row['changes_vs_baseline']=dict(gained=sorted(actual-base),lost=sorted(base-actual))
        ours=rows['ours_evidence'];gained+=len(ours['changes_vs_baseline']['gained']);lost+=len(ours['changes_vs_baseline']['lost']);no_harm &= all(ours['metrics'][m]>=rows['baseline']['metrics'][m]-1e-12 for m in ['allAp','allAp50%']);results[key]=dict(methods=rows)
    mean={m:float(np.mean(v)) for m,v in means.items()};strongest=max(mean[m] for m in METHODS if m!='ours_evidence');gate=bool(mean['ours_evidence']>=strongest+.02 and no_harm and gained>=1 and lost==0)
    dump(OUT/'summary.json',dict(complete=True,results=results,mean_primary_ap=mean,advance_gate=gate,no_metric_regression=bool(no_harm),gained_gt=gained,lost_gt=lost,development_crossfit=True,official_leaderboard=False,protocol_sha256=sha(OUT/'protocol.json'),prediction_seal_sha256=sha(OUT/'predictions_complete.json')));print('GATE',gate,mean,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','features','labels','train','maps','score']);globals()[parser.parse_args().action]()
