"""All-source audit of existing19-class SAM3 counterevidence, not a map method.

Predict using sealed native scores first; only score() reads reference labels.
No threshold sweep or inference calls. Unknown/ties never become background.
"""
import argparse,time,runpy
from pathlib import Path
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from capture_semantic_context import OUT as CONTEXT,INPUT
from audit_mask_observation_code import source_anchors,PARENT
from evaluate_kitti360_development import transfer,read_static_ply

OUT=ROOT/'build/benchmarks/semantic_counterevidence_audit_v1'
METHODS=['first_observation','mean_response','frame_majority']


def winner(x,classes):
    row=np.arange(len(x));j=x.argmax(1);best=x[row,j]
    unique=(np.sum(x==best[:,None],axis=1)==1)&(best>0)
    return np.where(unique,np.asarray(classes)[j],-1).astype(np.int16)


def freeze():
    assert not OUT.exists();cp=read(CONTEXT/'protocol.json');capture=read(CONTEXT/'predictions_complete.json');assert capture['complete'];parent=read(ROOT/'build/benchmarks/object_context_acceptance_v1/protocol.json');pins={}
    for f,h in capture['files'].items():assert sha(f)==h,f;pins[f]=h
    for f in [CONTEXT/'protocol.json',CONTEXT/'predictions_complete.json',ROOT/'build/benchmarks/object_context_acceptance_v1/protocol.json']:pins[str(f)]=sha(f)
    for f,h in parent['input_hashes'].items():assert sha(f)==h,f;pins[f]=h
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),source_sha256=sha(__file__),input_hashes=pins,scenes=parent['scenes'],references=parent['references'],vocabulary=cp['vocabulary'],methods=METHODS,
         purpose='All83existing objects,all108original frames; audit whether native19-class evidence can identify other-class contamination in original source provenance. No8case restriction,GT-drivenselection,newmodel,map,AP,training or independent-test claim.',
         prediction='All measured sources,original exact visible pixels. First available native class winner; mean native class responses over measured frames; majority of unique positive per-frame class winners. Exact ties/allzero stayunknown=-1. No comparison across missing frames. Native maxima are not calibrated class posteriors.',
         labels='Existing object semantic classes from frozen geometry_graph. Original core/intrinsic/graph source provenance;conflictsignored. GTsemantic nearestreference20cm,rawtrain255/confidence0/missingignored;roadtrain0valid. Semantic accuracy/error detection only,not instance accuracy or full fused-map quality.',
         evaluation='After all predictions sealed,measure precision/recall of counterevidence against old-class contamination on known source anchors. No rejectthresholds,selectedsubsets,newmapedits or postscore parameter changes. All assignedobject counts and unknowns reported. Shared existing supervised/exposed parent limitations remain.'))
    print('FROZEN SEMANTIC COUNTEREVIDENCE AUDIT',flush=True)


def verify():
    p=read(OUT/'protocol.json');assert sha(__file__)==p['source_sha256']
    for f,h in p['input_hashes'].items():assert sha(f)==h,f
    return p


def predict():
    p=verify();assert not (OUT/'predictions_complete.json').exists();capture=read(CONTEXT/'predictions_complete.json');classes=np.array([v['train_id'] for v in p['vocabulary']]);files={};stats={}
    for scene in p['scenes']:
        key=scene['key'];m=read(INPUT/key/'manifest.json');xyz=np.load(m['geometry'])['xyz'];n=len(xyz);owners=source_anchors(key,n);objects=read(PARENT/'maps'/key/'geometry_graph/classification.json')['objects'];declared=np.full(n,-1,np.int16)
        for obj in objects:declared[owners==obj['object_id']]=obj['predictions']['mean']['train_id']
        total=np.zeros((n,19),np.float64);seen=np.zeros(n,np.int16);votes=np.zeros((n,19),np.int16);first=np.full(n,-1,np.int16);frame_rows=[]
        for rec in capture['results'][key]:
            d=np.load(rec['file']);sid=d['source_ids'];x=d['point_scores'];w=winner(x,classes);first[sid[seen[sid]==0]]=w[seen[sid]==0];total[sid]+=x;seen[sid]+=1
            valid=w>=0;np.add.at(votes,(sid[valid],w[valid]),1)
            frame_rows.append(dict(frame=rec['frame_id'],measured=len(sid),unknown=int((~valid).sum())))
        mean=np.divide(total,seen[:,None],out=np.zeros_like(total),where=seen[:,None]>0);preds=dict(first_observation=first,mean_response=winner(mean,classes),frame_majority=winner(votes,np.arange(19)))
        folder=OUT/key;folder.mkdir(exist_ok=False);f=folder/'predictions.npz';np.savez_compressed(f,owners=owners,declared_train_id=declared,seen=seen,**preds);files[str(f)]=sha(f)
        stats[key]=dict(source_points=n,assigned_sources=int((owners>0).sum()),conflicting_sources=int((owners<0).sum()),objects=len(objects),observed_assigned_sources=int(((owners>0)&(seen>0)).sum()),revisited_assigned_sources=int(((owners>0)&(seen>=2)).sum()),frames=frame_rows,
             mean_vs_first_changed=int((preds['mean_response']!=first).sum()),majority_vs_mean_changed=int((preds['frame_majority']!=preds['mean_response']).sum()))
        print('SEMANTIC READOUT',key,stats[key]['objects'],stats[key]['assigned_sources'],'revisited',stats[key]['revisited_assigned_sources'],flush=True)
    dump(OUT/'predictions_complete.json',dict(complete=True,protocol_sha256=sha(OUT/'protocol.json'),files=files,results=stats,gt_read=False,new_maps=0))


def score():
    p=verify();seal=read(OUT/'predictions_complete.json');assert seal['complete'] and not (OUT/'summary.json').exists()
    for f,h in seal['files'].items():assert sha(f)==h,f
    taxonomy=runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['id2label'];results={}
    for scene in p['scenes']:
        key=scene['key'];d=np.load(OUT/key/'predictions.npz');xyz=np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz'];ref=np.load(scene['reference']);raw=read_static_ply(p['references'][key]['raw']);train=np.array([taxonomy[int(v)].trainId for v in raw['semantic']]);expected=train*1000+raw['instance'].astype(np.int64)%1000;expected[train==255]=0;np.testing.assert_array_equal(expected,ref['train_instance']);nn=transfer(ref['xyz'],np.arange(1,len(train)+1),xyz,.2);ix=np.maximum(nn-1,0);truth=train[ix];known=(nn>0)&(truth!=255)&(ref['confidence'][ix]>0);w=ref['confidence'][ix];assigned=d['owners']>0;take=assigned&known;bad=truth!=d['declared_train_id'];rows={}
        for method in METHODS:
            pred=d[method];evidence=pred>=0;reject=evidence&(pred!=d['declared_train_id']);tp=int((take&reject&bad).sum());fp=int((take&reject&~bad).sum());fn=int((take&~reject&bad).sum());confusion=np.zeros((19,20),np.int64);np.add.at(confusion,(truth[take],np.where(pred[take]>=0,pred[take],19)),1)
            rows[method]=dict(known_anchors=int(take.sum()),known_contaminated=int((take&bad).sum()),correctly_flagged=tp,incorrectly_flagged=fp,missed_contamination=fn,unknown_reference_flagged=int((assigned&~known&reject).sum()),unknown_prediction=int((take&~evidence).sum()),
                 error_detection_precision=tp/max(1,tp+fp),error_detection_recall=tp/max(1,tp+fn),semantic_accuracy=float((pred[take]==truth[take]).mean()),confidence_weighted_accuracy=float(w[take&(pred==truth)].sum()/w[take].sum()),confusion=confusion.tolist())
            print('COUNTEREVIDENCE',key,method,{k:v for k,v in rows[method].items() if k!='confusion'},flush=True)
        objects=[]
        for oid in sorted(np.unique(d['owners'][assigned])):
            use=take&(d['owners']==oid);objects.append(dict(object_id=int(oid),known=int(use.sum()),contaminated=int((use&bad).sum()),methods={m:dict(correctly_flagged=int((use&(d[m]>=0)&(d[m]!=d['declared_train_id'])&bad).sum()),incorrectly_flagged=int((use&(d[m]>=0)&(d[m]!=d['declared_train_id'])&~bad).sum())) for m in METHODS}))
        results[key]=dict(methods=rows,objects=objects)
    dump(OUT/'summary.json',dict(complete=True,prediction_seal_sha256=sha(OUT/'predictions_complete.json'),results=results,new_maps=0,new_ap=0,scope='All assigned source anchors; semantic contamination detection,not identity/full-map accuracy. Native scores not calibrated; no success gate or parameter selection.'))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','predict','score']);globals()[a.parse_args().action]()
