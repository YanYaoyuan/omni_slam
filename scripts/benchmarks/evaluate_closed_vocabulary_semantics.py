"""All-source and actual-map semantic diagnostics for the external control.

Aggregation reads no GT and is sealed before score(). No instance map changes.
"""
import argparse
import runpy
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from capture_closed_vocabulary_semantics import OUT,BASE,INPUT,verify
from audit_semantic_counterevidence import OUT as SAM_AUDIT,winner,METHODS
from evaluate_kitti360_development import transfer,read_static_ply


def aggregate():
    p=verify();seal=read(OUT/'predictions_complete.json');assert seal['complete'] and not (OUT/'aggregation_complete.json').exists();files={};results={}
    for f,h in seal['files'].items():assert sha(f)==h,f
    dump(OUT/'aggregation_protocol.json',dict(capture_seal_sha256=sha(OUT/'predictions_complete.json'),source_sha256=sha(__file__),
         rule='First observed class,mean19classprobabilitiesoverobservedframes,majorityofframewinners. Unique positive maximum only;ties and never observed=-1. Exact originalsource/map sampling fromsealedcapture,notnearestsemanticpoint interpolation. NoGTorconfidencefiltering.',
         scope='Shared external semantic evidence,not newmethod or map. Diagnostic reads same raw taxonomy after aggregate predictions sealed. Source comparison to priorSAM3;map has no identical denseSAM3samplecache,so no domain-mismatched comparison.'))
    for scene in p['scenes']:
        key=scene['key'];manifest=read(INPUT/key/'manifest.json');base=np.load(BASE/key/'geometry_graph/geometry.npz');anchors=np.load(SAM_AUDIT/key/'predictions.npz');records=read(BASE/key/'geometry_graph/classification.json')['objects'];map_class=np.full(len(base['xyz']),-1,np.int16)
        for obj in records:map_class[base['instance']==obj['object_id']]=obj['predictions']['mean']['train_id']
        assert (map_class>=0).all();sizes={'source':len(anchors['owners']),'map':len(base['xyz'])};state={}
        for domain,n in sizes.items():state[domain]=dict(total=np.zeros((n,19),np.float64),seen=np.zeros(n,np.int16),first=np.full(n,-1,np.int16),votes=np.zeros((n,19),np.int16))
        for rec in seal['results'][key]:
            d=np.load(rec['file'])
            for domain in sizes:
                sid=d[domain+'_ids'];x=d[domain+'_probabilities'];np.testing.assert_allclose(x.sum(1),1.,rtol=0,atol=1e-6);s=state[domain];pred=winner(x,np.arange(19));first=s['seen'][sid]==0;s['first'][sid[first]]=pred[first];s['total'][sid]+=x;s['seen'][sid]+=1;valid=pred>=0;np.add.at(s['votes'],(sid[valid],pred[valid]),1)
        folder=OUT/'aggregate'/key;folder.mkdir(parents=True,exist_ok=False);stats={}
        for domain,s in state.items():
            mean=np.divide(s['total'],s['seen'][:,None],out=np.zeros_like(s['total']),where=s['seen'][:,None]>0).astype(np.float32);preds=dict(first_observation=s['first'],mean_response=winner(mean,np.arange(19)),frame_majority=winner(s['votes'],np.arange(19)))
            assert all(np.all(v[s['seen']==0]==-1) for v in preds.values())
            owner=anchors['owners'] if domain=='source' else base['instance'];declared=anchors['declared_train_id'] if domain=='source' else map_class
            path=folder/(domain+'.npz');np.savez_compressed(path,owners=owner,declared_train_id=declared,seen=s['seen'],mean_probabilities=mean,**preds);files[str(path)]=sha(path);take=owner>0;stats[domain]=dict(points=int(take.sum()),observed=int((take&(s['seen']>0)).sum()),revisited=int((take&(s['seen']>=2)).sum()))
        results[key]=stats;print('AGGREGATED',key,stats,flush=True)
    dump(OUT/'aggregation_complete.json',dict(complete=True,protocol_sha256=sha(OUT/'aggregation_protocol.json'),files=files,results=results,gt_read=False,new_maps=0))


def metrics(pred,declared,take,bad,truth,known,assigned):
    evidence=pred>=0;reject=evidence&(pred!=declared);tp=int((take&reject&bad).sum());fp=int((take&reject&~bad).sum());fn=int((take&~reject&bad).sum());confusion=np.zeros((19,20),np.int64);np.add.at(confusion,(truth[take],np.where(pred[take]>=0,pred[take],19)),1)
    return dict(known_points=int(take.sum()),known_contaminated=int((take&bad).sum()),correctly_flagged=tp,incorrectly_flagged=fp,missed_contamination=fn,unknown_reference_flagged=int((assigned&~known&reject).sum()),unknown_prediction=int((take&~evidence).sum()),error_detection_precision=tp/max(1,tp+fp),error_detection_recall=tp/max(1,tp+fn),semantic_accuracy=float((pred[take]==truth[take]).mean()),confusion=confusion.tolist())


def score():
    p=verify();seal=read(OUT/'aggregation_complete.json');assert seal['complete'] and not (OUT/'summary.json').exists();ap=read(OUT/'aggregation_protocol.json');assert sha(__file__)==ap['source_sha256']
    for f,h in seal['files'].items():assert sha(f)==h,f
    taxonomy=runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['id2label'];results={}
    for scene in p['scenes']:
        key=scene['key'];source=np.load(read(INPUT/key/'manifest.json')['geometry']);base=np.load(BASE/key/'geometry_graph/geometry.npz');ref=np.load(scene['reference']);raw=read_static_ply(p['references'][key]['raw']);train=np.array([taxonomy[int(v)].trainId for v in raw['semantic']]);expected=train*1000+raw['instance'].astype(np.int64)%1000;expected[train==255]=0;np.testing.assert_array_equal(expected,ref['train_instance']);domains={}
        for domain,xyz in [('source',source['xyz']),('map',base['xyz'])]:
            d=np.load(OUT/'aggregate'/key/(domain+'.npz'));nn=transfer(ref['xyz'],np.arange(1,len(train)+1),xyz,.2);ix=np.maximum(nn-1,0);truth=train[ix];known=(nn>0)&(truth!=255)&(ref['confidence'][ix]>0);assigned=d['owners']>0;take=assigned&known;bad=truth!=d['declared_train_id'];rows={}
            for method in METHODS:
                rows[method]=metrics(d[method],d['declared_train_id'],take,bad,truth,known,assigned);print('EXTERNAL SEMANTIC',key,domain,method,{k:v for k,v in rows[method].items() if k!='confusion'},flush=True)
            objects=[]
            for oid in sorted(np.unique(d['owners'][assigned])):
                use=take&(d['owners']==oid);objects.append(dict(object_id=int(oid),known=int(use.sum()),contaminated=int((use&bad).sum()),methods={m:dict(correctly_flagged=int((use&(d[m]>=0)&(d[m]!=d['declared_train_id'])&bad).sum()),incorrectly_flagged=int((use&(d[m]>=0)&(d[m]!=d['declared_train_id'])&~bad).sum())) for m in METHODS}))
            domains[domain]=dict(methods=rows,objects=objects)
        results[key]=domains
    dump(OUT/'summary.json',dict(complete=True,aggregation_seal_sha256=sha(OUT/'aggregation_complete.json'),results=results,new_maps=0,new_ap=0,scope='External Cityscapes supervised model,all83objects,108frames. Semantic contamination detection at source/fused-map points;not instance matching,officialbenchmark score,independenttest or Ours innovation.'))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['aggregate','score']);globals()[a.parse_args().action]()
