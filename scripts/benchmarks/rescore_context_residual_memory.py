"""Correct the indoor projection adapter; predictions and frozen code unchanged.

KITTI projects the complete map then filters labels. Existing indoor protocol
filters predicted background source points BEFORE its nearest-neighbor query.
The initial generic adapter incorrectly used the KITTI order for SceneNN.
"""
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from run_context_residual_memory import OUT,METHODS,verify
from omni_semantic_map.mesh_instance_evaluation import transfer_labels
from omni_semantic_map.full_ovo_evaluation import voxel_metrics
from omni_semantic_map.instance_error_audit import audit_instances,filter_transition


def main():
    p=verify(True);seal=read(OUT/'predictions_complete.json')
    for f,h in seal['files'].items():assert sha(f)==h
    assert not (OUT/'corrected_summary.json').exists()
    dump(OUT/'evaluation_correction.json',dict(source_sha256=sha(__file__),original_summary_sha256=sha(OUT/'summary.json'),
        reason=__doc__,prediction_seal_sha256=sha(OUT/'predictions_complete.json'),
        no_new_predictions=True,no_parameter_changes=True,authoritative_summary='corrected_summary.json'))
    result=read(OUT/'summary.json');s=next(s for s in p['scenes'] if s['kind']=='scenenn');key=s['key']
    ref=np.load(s['reference']);assert sha(s['reference'])==p['reference_hashes'][s['reference']]
    valid=ref['valid'] if 'valid' in ref else np.ones(len(ref['instance']),bool)
    geometry=np.load(s['fixed_map']+'/geometry.npz');rows=read(OUT/key/'readouts.json');audits={};predictions={}
    for method in METHODS:
        retained=[r['object_id'] for r in rows if r['methods'][method]['retained']]
        labels=np.where(np.isin(geometry['instance'],retained),geometry['instance'],0)
        projected,_=transfer_labels(geometry['xyz'],labels,ref['xyz'],mode='nearest_5cm')
        score=voxel_metrics(ref['instance'][valid],projected[valid])
        result['results'][key]['methods'][method]['metrics']=score
        audits[method]=audit_instances(ref['instance'][valid],projected[valid]);predictions[method]=projected
        dump(OUT/key/(method+'_corrected_object_audit.json'),audits[method])
        print(method,'PQ',score['observed_voxel_instance_pq']*100,flush=True)
    old=read(ROOT/'build/benchmarks/vggt_fullmap_v1/evaluation.json')
    expected=next(r for r in old['results'] if r['scene']==key)['methods']['vggt_single']['metrics']
    assert result['results'][key]['methods']['learned']['metrics']==expected
    np.savez_compressed(OUT/key/'corrected_reference_predictions.npz',**predictions)
    result['indoor_transitions_vs_masked']={m:filter_transition(audits['masked'],audits[m]) for m in METHODS}
    simple=['learned','masked','bbox','fixed_context'];indoor=result['results'][key]['methods'];outdoor=result['outdoor_pooled']
    outdelta=outdoor['shared_context']['allAp50%']-max(outdoor[m]['allAp50%'] for m in simple)
    indelta=indoor['shared_context']['metrics']['observed_voxel_instance_pq']-max(indoor[m]['metrics']['observed_voxel_instance_pq'] for m in simple)
    result.update(advance_gate=bool((outdelta>=.01 and indelta>=-.005) or (indelta>=.01 and outdelta>=-.005)),
        candidate_delta_best_simple=dict(outdoor_ap50=outdelta,indoor_pq=indelta),
        indoor_projection_corrected=True,original_baseline_reproduced=True,
        correction_sha256=sha(OUT/'evaluation_correction.json'))
    dump(OUT/'corrected_summary.json',result);print('GATE',result['advance_gate'],result['candidate_delta_best_simple'],flush=True)


if __name__=='__main__':main()
