"""Replay frozen inference and inspect every complete-map object."""
import numpy as np
import torch
from paper_final_common import read,dump,sha
from run_map_responsibility_supervision import OUT,PARENT,INPUT,METHODS,verify,seal
from run_object_context_acceptance import logits
from omni_semantic_map.surface_ownership import allocate
from evaluate_kitti360_development import transfer


def main():
    p=verify();seal('trained');seal('prepared');pred=seal('predictions_complete');summary=read(OUT/'summary.json')
    assert summary['prediction_seal_sha256']==sha(OUT/'predictions_complete.json')
    checks={};details={};torch.set_num_threads(1)
    for scene in p['scenes']:
        key=scene['key'];d=dict(np.load(PARENT/'features'/key/'candidates.npz'));xyz=np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz'];base=dict(np.load(PARENT/'maps'/key/'baseline/geometry.npz'));cls=read(PARENT/'maps'/key/'baseline/classification.json');n=len(base['xyz']);details[key]={};means=[];stds=[]
        ref=dict(np.load(scene['reference']));truth=ref['train_instance'];weight=ref['confidence'].astype(float);gtmass=np.bincount(truth,weights=weight)
        for method in METHODS:
            f=OUT/'maps'/key/method;g=dict(np.load(f/'geometry.npz'))
            np.testing.assert_array_equal(g['xyz'][:n],base['xyz']);np.testing.assert_array_equal(g['instance'][:n],base['instance']);assert read(f/'classification.json')==cls
            if method=='baseline':assert sha(f/'geometry.npz')==sha(PARENT/'maps'/key/'baseline/geometry.npz')
            else:
                m=torch.load(OUT/'models'/key/method/'model.pt',map_location='cpu',weights_only=False);trainkey=next(s['key'] for s in p['scenes'] if s['key']!=key);assert m['training']==[trainkey];means.append(m['mean']);stds.append(m['std'])
                td=dict(np.load(PARENT/'features'/trainkey/'candidates.npz'));xx=td['object_features'][td['valid']]
                np.testing.assert_array_equal(m['mean'],xx.mean(0));np.testing.assert_array_equal(m['std'],np.maximum(xx.std(0),1e-6))
                owner,confidence=allocate(logits(m,d['object_features']),d['valid'],d['owners']);saved=dict(np.load(f/'decisions.npz'))
                np.testing.assert_array_equal(owner,saved['owners']);np.testing.assert_array_equal(confidence,saved['confidence']);np.testing.assert_array_equal(saved['source_ids'],d['source_ids'])
                ids=d['source_ids'][owner>0];np.testing.assert_array_equal(g['added_source_ids'],ids);np.testing.assert_array_equal(g['xyz'][n:],xyz[ids]);np.testing.assert_array_equal(g['instance'][n:],owner[owner>0]);assert len(np.unique(ids))==len(ids)
            projected=transfer(g['xyz'],g['instance'],ref['xyz'],.2);rows=[]
            for obj in cls['objects']:
                oid=obj['object_id'];take=projected==oid;mass=float(weight[take].sum());inter=np.bincount(truth[take],weights=weight[take],minlength=len(gtmass));eligible=np.flatnonzero((np.arange(len(gtmass))//1000==obj['predictions']['mean']['train_id'])&(gtmass>=100)&(inter>0))
                overlaps=[dict(gt_id=int(gt),iou=float(inter[gt]/max(1e-12,gtmass[gt]+mass-inter[gt]))) for gt in eligible];overlaps.sort(key=lambda q:(-q['iou'],q['gt_id']))
                rows.append(dict(object_id=oid,reference_mass=mass,overlaps=overlaps))
            details[key][method]=rows;checks[key+'/'+method]=dict(old_geometry_ids_classes_exact=True,source_append_exact=True,inference_replayed=method!='baseline')
        for mean,std in zip(means,stds):np.testing.assert_array_equal(mean,means[0]);np.testing.assert_array_equal(std,stds[0])
        print('AUDIT',key,'all four maps and training normalization exact',flush=True)
    dump(OUT/'object_diagnostics.json',dict(complete=True,results=details,scope='All original object IDs; post-score confidence-weighted IoU on full reference domain. Not training assignments or new predictions.'))
    dump(OUT/'artifact_audit.json',dict(complete=True,checks=checks,source_sha256=sha(__file__),prediction_seal_sha256=sha(OUT/'predictions_complete.json')))


if __name__=='__main__':main()
