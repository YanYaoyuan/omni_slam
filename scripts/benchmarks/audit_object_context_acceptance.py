"""Verify fixed exports/inference and explain all objects without new variants."""
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from run_object_context_acceptance import OUT,PARENT,BASE,INPUT,METHODS,TRAINED,verify,sealed,logits,simple_veto
from omni_semantic_map.surface_ownership import allocate
from evaluate_kitti360_development import transfer


def main():
    import torch
    p=verify();seal=sealed('predictions_complete');sealed('trained');sealed('features_complete');summary=read(OUT/'summary.json')
    assert summary['prediction_seal_sha256']==sha(OUT/'predictions_complete.json')
    torch.set_num_threads(1);checks={};details={};source_quality={};differences={}
    columns={v['semantic_id']:i for i,v in enumerate(p['vocabulary'])}
    for scene in p['scenes']:
        key=scene['key'];source=np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz'];d=dict(np.load(OUT/'features'/key/'candidates.npz'))
        truth=dict(np.load(OUT/'training_labels'/key/'TRAINING_ONLY.npz'));base=dict(np.load(PARENT/'maps'/key/'geometry_graph/geometry.npz'));classification=read(PARENT/'maps'/key/'geometry_graph/classification.json');objects=classification['objects'];sem={o['object_id']:o['predictions']['mean']['semantic_id'] for o in objects}
        core=read(OUT/'features'/key/'core_sources.json');locked=np.unique(np.concatenate([np.asarray(v,np.int64) for v in core.values()]))
        assert not np.isin(d['source_ids'],locked).any();n=len(base['xyz']);ref=dict(np.load(scene['reference']));w=ref['confidence'].astype(float);gt=ref['train_instance'];gmass=np.bincount(gt,weights=w)
        geometry_model=torch.load(BASE/'models'/key/'intrinsic_geometry/model.pt',map_location='cpu',weights_only=False)
        zgeo=logits(geometry_model,d['object_features']);outputs={};permethod={};source_quality[key]={};details[key]={}
        for method in METHODS:
            folder=OUT/'maps'/key/method;g=dict(np.load(folder/'geometry.npz'));outputs[method]=g
            np.testing.assert_array_equal(g['xyz'][:n],base['xyz']);np.testing.assert_array_equal(g['instance'][:n],base['instance']);assert read(folder/'classification.json')==classification
            if method=='baseline':assert sha(folder/'geometry.npz')==sha(PARENT/'maps'/key/'geometry_graph/geometry.npz')
            else:
                decisions=dict(np.load(folder/'decisions.npz'));np.testing.assert_array_equal(decisions['source_ids'],d['source_ids']);v=d['valid'];z=zgeo
                if method=='simple_context':v=simple_veto(d['object_features'],d['owners'],v,sem,columns)
                elif method in TRAINED:
                    model=torch.load(OUT/'models'/key/method/'model.pt',map_location='cpu',weights_only=False);assert model['training']==[s['key'] for s in p['scenes'] if s['key']!=key]
                    z=logits(model,d['object_features' if method=='ours_object_context' else 'category_features'])
                assigned,confidence=allocate(z,v,d['owners'],independent=True)
                np.testing.assert_array_equal(assigned,decisions['owners']);np.testing.assert_array_equal(confidence,decisions['confidence'])
                selected=assigned>0;np.testing.assert_array_equal(g['added_source_ids'],d['source_ids'][selected]);np.testing.assert_array_equal(g['xyz'][n:],source[d['source_ids'][selected]]);np.testing.assert_array_equal(g['instance'][n:],assigned[selected])
                assert len(np.unique(g['added_source_ids']))==len(g['added_source_ids'])
                eligible=truth['eligible'];positive=((d['owners']==assigned[:,None])&truth['positive']&d['valid']).any(1);known=selected&eligible
                ww=truth['weight'];mass=float(ww[known].sum());correct=float(ww[known&positive].sum())
                source_quality[key][method]=dict(added_sources=int(selected.sum()),known_added=int(known.sum()),unknown_added=int((selected&~eligible).sum()),correct_known=int((known&positive).sum()),weighted_precision=correct/mass if mass else None,
                     candidate_positive_recall=int((known&positive).sum())/max(1,int((eligible&truth['positive'].any(1)).sum())))
                permethod[method]=assigned
            projected=transfer(g['xyz'],g['instance'],ref['xyz'],.2);rows=[]
            for obj in objects:
                oid=obj['object_id'];take=projected==oid;pmass=float(w[take].sum());inter=np.bincount(gt[take],weights=w[take],minlength=len(gmass));eligible_gt=np.flatnonzero((np.arange(len(gmass))//1000==obj['predictions']['mean']['train_id'])&(gmass>=100)&(inter>0));overlaps=[]
                for gid in eligible_gt:
                    overlaps.append(dict(gt_id=int(gid),iou=float(inter[gid]/max(1e-12,gmass[gid]+pmass-inter[gid])),intersection=float(inter[gid]),gt_mass=float(gmass[gid])))
                overlaps.sort(key=lambda x:(-x['iou'],x['gt_id']))
                rows.append(dict(object_id=oid,semantic_id=obj['predictions']['mean']['semantic_id'],reference_mass=pmass,overlaps=overlaps))
            details[key][method]=rows
            checks[key+'/'+method]=dict(old_xyz_and_ids_exact=True,classification_unchanged=True,source_append_exact=True,inference_replayed=method!='baseline')
        differences[key]=dict(
            geometry_vs_simple_source_decisions=int((permethod['expanded_geometry']!=permethod['simple_context']).sum()),
            category_vs_object_source_decisions=int((permethod['category_context']!=permethod['ours_object_context']).sum()),
            geometry_vs_simple_geometry_hash_equal=sha(OUT/'maps'/key/'expanded_geometry/geometry.npz')==sha(OUT/'maps'/key/'simple_context/geometry.npz'),
            category_vs_object_geometry_hash_equal=sha(OUT/'maps'/key/'category_context/geometry.npz')==sha(OUT/'maps'/key/'ours_object_context/geometry.npz'))
        print('AUDIT',key,source_quality[key],differences[key],flush=True)
    dump(OUT/'object_diagnostics.json',dict(complete=True,results=details,scope='Post-score all object IDs, confidence-weighted full reference IoU; no candidate selection or fitting.'))
    dump(OUT/'artifact_audit.json',dict(complete=True,source_sha256=sha(__file__),prediction_seal_sha256=sha(OUT/'predictions_complete.json'),checks=checks,differences=differences,source_quality=source_quality,
         source_quality_scope='Post-score nearest-reference20cm labels for candidate sources with known source/owner supervision. Unknown/ignored source groups excluded and counted. Not official reference-area AP and not precision of all added points.'))
    print('AUDIT COMPLETE',flush=True)


if __name__=='__main__':main()
