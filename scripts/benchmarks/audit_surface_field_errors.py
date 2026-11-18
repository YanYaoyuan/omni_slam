"""Explain frozen map errors by nearest-point provenance, without new variants."""
import numpy as np
from paper_final_common import read,dump,sha
from run_surface_evidence_field import OUT,PARENT,verify,sealed
from evaluate_kitti360_development import transfer


def main():
    protocol=verify();seal=sealed('predictions_complete');summary=read(OUT/'summary.json');assert summary['scored'];details=read(PARENT/'object_diagnostics.json')['results'];results={}
    for scene in protocol['scenes']:
        key=scene['key'];ref=np.load(scene['reference']);truth=ref['train_instance'];w=ref['confidence'];base=np.load(OUT/'maps'/key/'baseline/geometry.npz');ours=np.load(OUT/'maps'/key/'ours/geometry.npz');n=len(base['xyz'])
        oldprojection=transfer(base['xyz'],base['instance'],ref['xyz'],.2)
        nearest=transfer(ours['xyz'],np.arange(1,len(ours['xyz'])+1,dtype=np.int64),ref['xyz'],.2);valid=nearest>0;ix=np.maximum(nearest-1,0);assigned=np.where(valid,ours['instance'][ix],0)
        isnew=valid&(ix>=n);oldvalid=valid&(ix<n);changed=oldvalid&(ours['instance'][ix]!=base['instance'][np.minimum(ix,n-1)])
        pieces=dict(old_unchanged=oldvalid&~changed,old_relabelled=changed,new_measurement=isnew)
        anchors={o['object_id']:o['overlaps'][0] for o in details[key]['geometry_graph'] if o['overlaps'] and o['overlaps'][0]['iou']>.5};rows=[]
        totals=dict(old_unchanged=dict(correct=0.,outside_target=0.),old_relabelled=dict(correct=0.,outside_target=0.),new_measurement=dict(correct=0.,outside_target=0.))
        lost_mass=dict(no_spatial_transfer=0.,old_null=0.,old_other=0.,new_other=0.)
        for oid,anchor in anchors.items():
            gid=anchor['gt_id'];target=truth==gid;pred=assigned==oid;intersection=float(w[target&pred].sum());pred_mass=float(w[pred].sum());gt_mass=float(w[target].sum());parts={}
            for name,mask in pieces.items():
                parts[name]=dict(correct=float(w[mask&pred&target].sum()),outside_target=float(w[mask&pred&~target].sum()))
                for field,value in parts[name].items():totals[name][field]+=value
            np.testing.assert_allclose(sum(r['correct'] for r in parts.values()),intersection,atol=.01)
            np.testing.assert_allclose(sum(r['outside_target'] for r in parts.values()),pred_mass-intersection,atol=.01)
            loss=target&(oldprojection==oid)&(assigned!=oid);causes=dict(no_spatial_transfer=float(w[loss&~valid].sum()),old_null=float(w[loss&oldvalid&(assigned==0)].sum()),old_other=float(w[loss&oldvalid&(assigned!=0)].sum()),new_other=float(w[loss&isnew].sum()))
            np.testing.assert_allclose(sum(causes.values()),float(w[loss].sum()),atol=.01)
            for field,value in causes.items():lost_mass[field]+=value
            rows.append(dict(object_id=oid,gt_id=gid,baseline_iou=anchor['iou'],ours_iou=intersection/max(gt_mass+pred_mass-intersection,1e-12),parts=parts,baseline_correct_surface_lost=causes))
        result=dict(objects=rows,totals=totals,baseline_correct_surface_lost=lost_mass)
        results[key]=result;print('ERROR PROVENANCE',key,totals,lost_mass,flush=True)
    dump(OUT/'error_provenance.json',dict(complete=True,source_sha256=sha(__file__),prediction_seal_sha256=sha(OUT/'predictions_complete.json'),results=results,
        scope='Post-score audit of baseline AP50-correct object IDs only. Official confidence-weighted reference masses partitioned by nearest frozen Ours map point provenance. Outside-target includes other and void labels, not necessarily reliable semantic negatives. Uses GT only after all predictions/scoring. Not a counterfactual causal intervention and not new method scores. No map, threshold, inference or trained parameter changed.'))


if __name__=='__main__':main()
