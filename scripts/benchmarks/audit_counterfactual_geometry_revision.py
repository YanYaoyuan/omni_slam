"""Audit retained geometry, accepted transactions, and full-reference objects."""
import numpy as np
from paper_final_common import read,dump,sha
from run_counterfactual_geometry_revision import OUT,BASE,SEM,METHODS,verify,sealed


def overlaps(projected,objects,truth,weight):
    gtmass=np.bincount(truth,weights=weight);rows=[]
    for obj in objects:
        oid=obj['object_id'];take=projected==oid;mass=float(weight[take].sum());inter=np.bincount(truth[take],weights=weight[take],minlength=len(gtmass));eligible=np.flatnonzero((np.arange(len(gtmass))//1000==obj['predictions']['mean']['train_id'])&(gtmass>=100)&(inter>0))
        values=[dict(gt_id=int(gt),iou=float(inter[gt]/max(1e-12,gtmass[gt]+mass-inter[gt]))) for gt in eligible];values.sort(key=lambda x:(-x['iou'],x['gt_id']));rows.append(dict(object_id=oid,reference_mass=mass,overlaps=values))
    return rows


def main():
    p=verify();seal=sealed('predictions_complete');summary=read(OUT/'summary.json');assert summary['prediction_seal_sha256']==sha(OUT/'predictions_complete.json');checks={};details={};identifiability={}
    for scene in p['scenes']:
        key=scene['key'];base=np.load(BASE/key/'geometry_graph/geometry.npz');cls=read(BASE/key/'geometry_graph/classification.json');regions=read(OUT/'prepared'/key/'regions.json');n=len(base['xyz']);ref=np.load(scene['reference']);truth=ref['train_instance'];weight=ref['confidence'].astype(float);keep_by={};pred_by={};details[key]={}
        for method in METHODS:
            folder=OUT/'maps'/key/method;g=np.load(folder/'geometry.npz');ids=np.arange(n) if method=='baseline' else g['retained_base_indices'];np.testing.assert_array_equal(g['xyz'],base['xyz'][ids]);np.testing.assert_array_equal(g['instance'],base['instance'][ids]);assert np.all(np.diff(ids)>0);present=set(g['instance']);expected=dict(cls,objects=[o for o in cls['objects'] if o['object_id'] in present]);assert read(folder/'classification.json')==expected
            keep=np.zeros(n,bool);keep[ids]=True;wanted=np.ones(n,bool)
            if method=='baseline':assert sha(folder/'geometry.npz')==sha(BASE/key/'geometry_graph/geometry.npz')
            elif method=='semantic_point':
                d=np.load(SEM/'aggregate'/key/'map.npz');wanted=(d['mean_response']<0)|(d['mean_response']==d['declared_train_id'])
            else:
                selected=range(len(regions)) if method=='semantic_region' else [e['region_id'] for e in read(folder/'events.json') if e['accepted']]
                for j in selected:wanted[regions[j]['map_ids']]=False
            np.testing.assert_array_equal(keep,wanted);run=read(folder/'run.json');assert run['removed_points']==int((~keep).sum())
            if method in ['local_counterfactual','ours_counterfactual']:
                events=read(folder/'events.json');assert len(events)==len(regions)
                for e,r in zip(events,regions):assert e['object_id']==r['object_id'] and e['region_id']==r['region_id'] and e['accepted']==(e['gain']>1e-9)
                audit=run['audit'];np.testing.assert_allclose(audit['initial_loss']-audit['final_loss'],sum(e['full_gain'] for e in events if e['accepted']),rtol=1e-10,atol=1e-7)
                if method=='ours_counterfactual':assert all(e['full_gain']>0 for e in events if e['accepted'])
            projected=np.load(folder/'reference_prediction.npz')['instance'];assert len(projected)==len(truth);pred_by[method]=projected;keep_by[method]=keep;details[key][method]=overlaps(projected,cls['objects'],truth,weight);checks[key+'/'+method]=dict(original_coordinates_ids_scores_exact=True,retention_matches_declared_policy=True,removed_points=int((~keep).sum()),geometry_sha256=sha(folder/'geometry.npz'),projection_sha256=sha(folder/'reference_prediction.npz'))
        table={};ours={(r['object_id'],v['gt_id']):v['iou'] for r in details[key]['ours_counterfactual'] for v in r['overlaps']}
        for other in ['baseline','semantic_point','semantic_region','local_counterfactual']:
            old={(r['object_id'],v['gt_id']):v['iou'] for r in details[key][other] for v in r['overlaps']};pairs=set(old)|set(ours);changed=sum(abs(old.get(pair,0)-ours.get(pair,0))>1e-12 for pair in pairs);flips=sum((old.get(pair,0)>t)!=(ours.get(pair,0)>t) for pair in pairs for t in np.arange(.5,.95,.05))
            table['ours_vs_'+other]=dict(retained_point_decisions_changed=int((keep_by[other]!=keep_by['ours_counterfactual']).sum()),reference_labels_changed=int((pred_by[other]!=pred_by['ours_counterfactual']).sum()),object_gt_iou_pairs_changed=int(changed),threshold_crossings_50_to_90=int(flips))
        identifiability[key]=table;print('IDENTIFIABILITY',key,table,flush=True)
    dump(OUT/'object_diagnostics.json',dict(complete=True,results=details,scope='All83originalobjects,fullreferenceconfidenceweightedoverlaps;postscoreonly,fixedpredictions.'))
    dump(OUT/'artifact_audit.json',dict(complete=True,source_sha256=sha(__file__),prediction_seal_sha256=sha(OUT/'predictions_complete.json'),checks=checks,identifiability=identifiability,all_maps_distinct={key:len({checks[key+'/'+m]['geometry_sha256'] for m in METHODS})==len(METHODS) for key in details}))


if __name__=='__main__':main()
