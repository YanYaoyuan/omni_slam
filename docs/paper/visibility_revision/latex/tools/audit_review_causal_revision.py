"""Independent export/readout checks; no new maps or parameter changes."""
import numpy as np
from review_causal_revision import verify,OUT,SRC,SCENES,NEW,read,dump,sha
verify();seal=read(OUT/'maps_complete.json')
for f,h in seal['files'].items():assert sha(f)==h,f
rows={}
for k in SCENES:
    base=np.load(SRC/'comparison_maps'/k/'strong_baseline/geometry.npz');ours=np.load(SRC/'comparison_maps'/k/'full_authorization/geometry.npz');ref=np.load(SRC/'evaluation'/k/'reference.npz');oldpred=np.load(SRC/'evaluation'/k/'full_authorization/reference_prediction.npz')['instance'];n=len(base['xyz']);out={}
    for m in NEW:
        g=np.load(OUT/'maps'/k/m/'geometry.npz');pred=np.load(OUT/'evaluation'/k/m/'reference_prediction.npz')['instance'];np.testing.assert_array_equal(g['xyz'][:n],base['xyz']);idx=np.linspace(0,len(pred)-1,16,dtype=int)
        for i in idx:
            d=np.linalg.norm(g['xyz']-ref['xyz'][i],axis=1);j=int(d.argmin());expected=g['instance'][j] if d[j]<.2 else 0
            assert pred[i]==expected,(k,m,i)
        out[m]=dict(bruteforce_readout_checks=16,changed_reference_ids_vs_ours=int(np.count_nonzero(pred!=oldpred)))
    near=np.load(OUT/'maps'/k/'nearest_witness_history/geometry.npz');extra=np.setdiff1d(near['added_full_source_ids'],ours['added_full_source_ids']);out['certificate_difference']=dict(nearest_additional_source_ids=len(extra),reference_id_differences=out['nearest_witness_history']['changed_reference_ids_vs_ours'],scores_equal=read(OUT/'summary.json')['contrasts'][k]['certificate_vs_nearest_pp']==0)
    rows[k]=out
    print(k,out['certificate_difference'],flush=True)
dump(OUT/'validation.json',dict(complete=True,original_inputs_and_sources_unchanged=True,all_export_seals_verified=True,bruteforce_readout_checks=160,results=rows,script_sha256=sha(__file__)))
