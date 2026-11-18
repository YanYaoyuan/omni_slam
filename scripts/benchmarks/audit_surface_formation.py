"""Independent source export and signed-objective audit; optional GT diagnosis."""
import argparse
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from run_surface_formation import OUT,BASE,INPUT,SEM,METHODS,verify,seal


def exports():
    p=verify();seal('prepared');seal('predictions_complete');checks={};canonical={}
    for scene in p['scenes']:
        key=scene['key'];d=np.load(OUT/'prepared'/key/'graph.npz')
        atomkeys=d['atomkeys'];anchor=atomkeys[:,4];sid=d['source_ids'];sourceatom=d['source_atoms'];n=len(anchor)
        left,right=d['left'],d['right'];base=np.load(BASE/key/'geometry_graph/geometry.npz');old=read(BASE/key/'geometry_graph/classification.json');oldrows={r['object_id']:r for r in old['objects']}
        m=read(INPUT/key/'manifest.json');xyz=np.load(m['geometry'])['xyz'];owners=np.load(SEM/'aggregate'/key/'source.npz')['owners'];canonical[key]={}
        for method in METHODS:
            folder=OUT/'maps'/key/method;g=np.load(folder/'geometry.npz');rows=read(folder/'classification.json')['objects'];byid={r['object_id']:r for r in rows}
            for oid,row in oldrows.items():assert byid[oid]==row
            size=len(base['xyz']);np.testing.assert_array_equal(g['xyz'][:size],base['xyz']);np.testing.assert_array_equal(g['instance'][:size],base['instance'])
            out=np.zeros(len(xyz),np.int64)
            if method=='baseline':
                assert sha(folder/'geometry.npz')==sha(BASE/key/'geometry_graph/geometry.npz')
                checks[key+'/'+method]=dict(exact_baseline=True)
            else:
                added=g['added_source_ids'];assert np.all(np.diff(added)>0) and np.all(owners[added]==0)
                np.testing.assert_array_equal(g['xyz'][size:],xyz[added]);assert len(g['instance'])==size+len(added)
                groups=np.load(folder/'partition.npz')['root'];assert groups.shape==(n,)
                cost={'geometry':np.ones(len(left)),'positive_only':np.maximum(d['mean_cost'],0),'first_view':d['first_cost'],'ours_surface':d['mean_cost']}[method]
                trace=read(folder/'merge_trace.json');same_initial=(anchor[left]>0)&(anchor[left]==anchor[right])
                change=float(cost[groups[left]==groups[right]].sum()-cost[same_initial].sum())
                np.testing.assert_allclose(change,trace['objective_gain'],rtol=1e-9,atol=1e-7)
                # Independently replay initial anchor contraction and root trace.
                parent=np.arange(n);sizes=np.ones(n,np.int64);tag=anchor.copy();first={}
                for j,a in enumerate(anchor):
                    if not a:continue
                    if int(a) in first:
                        root=first[int(a)];parent[j]=root;sizes[root]+=1;sizes[j]=0
                    else:first[int(a)]=j
                for a,b,gain,na,nb in trace['merges']:
                    assert a<b and parent[a]==a and parent[b]==b and gain>0
                    assert sizes[a]==na and sizes[b]==nb
                    assert not(tag[a] and tag[b] and tag[a]!=tag[b])
                    parent[b]=a;sizes[a]+=sizes[b];sizes[b]=0;tag[a]=max(tag[a],tag[b])
                for j in range(n):
                    root=j
                    while parent[root]!=root:root=int(parent[root])
                    assert groups[j]==root
                records=read(folder/'groups.json');assign=np.zeros(n,np.int64)
                for oid,r in records.items():assign[r['root']]=int(oid)
                free=anchor[sourceatom]==0;pred=assign[groups[sourceatom[free]]];valid=pred>0
                np.testing.assert_array_equal(sid[free][valid],added);np.testing.assert_array_equal(pred[valid],g['instance'][size:])
                bits=np.zeros(n,np.uint64);np.bitwise_or.at(bits,groups,d['framebits'])
                free_count=np.bincount(groups[sourceatom[free]],minlength=n)
                old_by_group={}
                for a in np.flatnonzero(anchor>0):
                    root=int(groups[a]);assert old_by_group.get(root,int(anchor[a]))==int(anchor[a]);old_by_group[root]=int(anchor[a])
                expected=set()
                for root in np.unique(groups[sourceatom[free]]):
                    if root in old_by_group or (free_count[root]>=16 and int(bits[root]).bit_count()>=3):expected.add(int(root))
                assert expected=={r['root'] for r in records.values()}
                for oid,r in records.items():
                    oid=int(oid);root=r['root'];assert r['sources']==free_count[root]
                    assert r['observed_frames']==int(bits[root]).bit_count()
                    select=(groups[sourceatom]==root)&free
                    if oid in oldrows:assert old_by_group[root]==oid
                    else:
                        assert root not in old_by_group
                        assert byid[oid]['predictions']['mean']['train_id']==atomkeys[root,0]
                        np.testing.assert_allclose(byid[oid]['predictions']['mean']['score'],d['confidence'][select].mean(),rtol=0,atol=0)
                newowner=g['instance'][size:];out[added]=newowner
                # Canonicalize births by their minimum actual source ID, avoiding
                # apparent differences from arbitrary new-object numbering.
                for oid in np.unique(newowner):
                    if oid not in oldrows:
                        take=newowner==oid;out[added[take]]=-(int(added[take].min())+1)
                checks[key+'/'+method]=dict(old_geometry_ids_scores_exact=True,unique_new_measured_sources=True,
                    no_old_sources_reused=True,export_matches_partition=True,anchor_constraints_replayed=True,
                    signed_objective_replayed=True,birth_rules_scores_verified=True,added_sources=len(added),new_objects=len(rows)-len(oldrows))
            canonical[key][method]=out
    differences={key:{method:dict(changed_source_ownership=int((v!=values['ours_surface']).sum()),
                        changed_admission=int(((v!=0)!=(values['ours_surface']!=0)).sum()))
                     for method,v in values.items() if method!='ours_surface'} for key,values in canonical.items()}
    dump(OUT/'export_audit.json',dict(complete=True,checks=checks,canonical_source_differences=differences,
        source_sha256=sha(__file__),prediction_seal_sha256=sha(OUT/'predictions_complete.json'),gt_read=False))
    print('EXPORT AUDIT',differences,flush=True)


def objects():
    from audit_counterfactual_geometry_revision import overlaps
    from audit_object_formation_limits import best_union
    p=verify();seal('predictions_complete');summary=read(OUT/'summary.json');details={};formation={}
    for scene in p['scenes']:
        key=scene['key'];ref=np.load(scene['reference']);details[key]={}
        formation[key]={};gtmass=np.bincount(ref['train_instance'],weights=ref['confidence'].astype(float))
        for method in METHODS:
            folder=OUT/'maps'/key/method;pred=np.load(folder/'reference_prediction.npz')['instance'];rows=read(folder/'classification.json')['objects']
            details[key][method]=overlaps(pred,rows,ref['train_instance'],ref['confidence'].astype(float))
            classes={r['object_id']:r['predictions']['mean']['train_id'] for r in rows}
            info={r['object_id']:r for r in details[key][method]};out=[]
            for gt in np.flatnonzero(np.isin(np.arange(len(gtmass))//1000,[2,13])&(gtmass>=100)):
                ids=[oid for oid,tid in classes.items() if tid==gt//1000]
                mass=np.array([info[oid]['reference_mass'] for oid in ids]);iou=np.array([next((v['iou'] for v in info[oid]['overlaps'] if v['gt_id']==gt),0.) for oid in ids])
                inter=iou*(gtmass[gt]+mass)/(1+iou)
                union,selected=best_union(inter,np.maximum(mass-inter,0),gtmass[gt])
                out.append(dict(gt_id=int(gt),actual_best_iou=float(np.where(mass>=100,iou,0).max(initial=0)),
                    independent_oracle_union_iou=union,oracle_selected_object_ids=[ids[j] for j in selected],
                    assigned_same_class_coverage=float(inter.sum()/gtmass[gt])))
            formation[key][method]=out
        print('OBJECT AUDIT',key,{m:len(v) for m,v in details[key].items()},flush=True)
    dump(OUT/'object_diagnostics.json',dict(complete=True,results=details,source_sha256=sha(__file__),posthoc_gt=True,summary_sha256=sha(OUT/'summary.json')))
    counts={key:{m:{name:dict(actual_over_half=sum(r['actual_best_iou']>.5 for r in rows if r['gt_id']//1000==tid),
                   oracle_union_over_half=sum(r['independent_oracle_union_iou']>.5 for r in rows if r['gt_id']//1000==tid))
                   for name,tid in [('building',2),('car',13)]} for m,rows in methods.items()} for key,methods in formation.items()}
    dump(OUT/'formation_diagnostics.json',dict(complete=True,diagnostic_only=True,results=formation,counts=counts,
        scope='Each GT independently selects best union of frozen prediction regions; incompatible oracle choices may occur. No map exported or AP scored. Used only to distinguish recovered surface support from whole-object fragmentation.'))
    print('FORMATION DIAGNOSIS',counts,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['exports','objects']);globals()[ap.parse_args().action]()
