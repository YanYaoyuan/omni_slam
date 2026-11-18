"""Post-completion integrity and geometry audit; no inference or tuning."""
from collections import Counter
from pathlib import Path
import numpy as np
from audit_evidence_retention import ROOT,OUT,dump
from public_dino_protocol import read,verify
from omni_semantic_map.flat_benchmark import sha256
from omni_semantic_map.full_ovo_evaluation import voxel_metrics
from omni_semantic_map.mesh_instance_evaluation import transfer_labels
from omni_semantic_map.semantic_instance_diagnostic import semantic_metrics
from omni_semantic_map.instance_error_audit import audit_instances,filter_transition

DEST=OUT/'category_holdout_v1'


def main():
    spec=read(DEST/'selection.json');verify(spec);summary=read(DEST/'summary.json');assert summary['complete']
    amendment=read(DEST/'annotation_amendment.json');assert sha256(amendment['source'])==amendment['sha256']
    assert summary['selection_sha256']==sha256(DEST/'selection.json')==amendment['original_selection_sha256']
    assert read(DEST/'queue_status.json')['complete']
    gate=read(DEST/'predictions_complete.json');assert gate['complete'] and set(gate['files'])==set(spec['scenes'])
    for scene,h in gate['files'].items():assert sha256(DEST/'predictions'/(scene+'.json'))==h
    for f in read(DEST/'download.json')['files']:assert sha256(f['path'])==f['sha256']
    out=DEST/'completion_audit';out.mkdir(exist_ok=False);integrity=[];scores=[]
    for scene in spec['scenes']:
        job_path=DEST/'jobs'/(scene+'.json');job=read(job_path)
        for p,h in job['input_sha256'].items():assert sha256(p)==h
        for p,h in job['source_sha256'].items():assert sha256(ROOT/p)==h
        for phase in ['masks','triplets','features','geometry']:
            state=read(DEST/'frontend'/scene/(phase+'_run.json'));assert state['complete'] and state['job_sha256']==sha256(job_path)
            for f,h in state['files_sha256'].items():assert sha256(DEST/'frontend'/scene/f)==h
        cache=DEST/'frontend'/scene/'cache';cm=read(cache/'manifest.json')
        for f,h in cm['files_sha256'].items():assert sha256(cache/f)==h
        for r in cm['records']:assert sha256(cache/'frames'/r['file'])==r['sha256']
        mapping=DEST/'history/runs'/('scenenn_'+scene);state=read(mapping/'run.json')
        assert state['complete'] and state['no_gt'] and state['baseline_exact_geometry_memberships_classes']
        for f,h in state['artifacts'].items():assert sha256(mapping/f)==h
        for v in state['variants']:
            for f,h in v['hashes'].items():assert sha256(mapping/v['variant']/f)==h
        pred=read(DEST/'predictions'/(scene+'.json'));assert pred['complete'] and pred['no_gt']
        assert sha256(pred['geometry'])==pred['geometry_sha256']
        ids=[r['object_id'] for r in pred['objects'] if r['base_retained']]
        integrity.append(dict(scene=scene,geometry_frames=cm['frames'],semantic_frames=len(cm['records']),
                              regions=len(pred['objects']),retained=len(ids),scorable=scene in amendment['scored_scenes']))
        if scene not in amendment['scored_scenes']:continue
        ref=np.load(DEST/'evaluation'/scene/'reference.npz');truth=ref['instance'];valid=ref['valid']
        gt={int(k):v for k,v in read(DEST/'evaluation'/scene/'category_reference.json')['classes'].items()}
        semvalid=valid&((truth==0)|np.isin(truth,list(gt)));old=read(DEST/'evaluation'/scene/'metrics.json')
        assert old['prediction_sha256']==gate['files'][scene]
        per_method={};baseline_audit=None
        for name in ['baseline','native_bridge_1','native_bridge_2']:
            geo=np.load(mapping/name/'geometry.npz');rows=read(mapping/name/'classification.json')['objects']
            kept=[r['object_id'] for r in rows if r['predictions']['mean']['retained']]
            projected,_=transfer_labels(geo['xyz'],np.where(np.isin(geo['instance'],kept),geo['instance'],0),ref['xyz'],mode='nearest_5cm')
            instance=voxel_metrics(truth[valid],projected[valid]);audit=audit_instances(truth[valid],projected[valid])
            classes={r['object_id']:spec['nyu_prediction_mapping'].get(r['predictions']['mean']['name'],-1) for r in rows}
            semantic=semantic_metrics(truth[semvalid],projected[semvalid],gt,classes)
            if baseline_audit is None:baseline_audit=audit
            if name=='native_bridge_2':
                np.testing.assert_array_equal(projected,np.load(DEST/'evaluation'/scene/'fixed_prediction.npz')['instance'])
                assert instance==old['fixed_instance'] and semantic==old['methods']['mean']['fixed']
                for method in ['area','consensus_sqrt_area']:
                    classes={r['object_id']:spec['nyu_prediction_mapping'].get(r['methods'][method]['name'],-1) for r in pred['objects']}
                    assert semantic_metrics(truth[semvalid],projected[semvalid],gt,classes)==old['methods'][method]['fixed']
            per_method[name]=dict(instance=instance,semantic=semantic,predicted_error_counts=audit['predicted_counts'],gt_error_counts=audit['gt_counts'],matched_transition=filter_transition(baseline_audit,audit))
            dump(out/(scene+'_'+name+'_instances.json'),audit)
        scores.append(dict(scene=scene,methods=per_method))
    macro={}
    for name in ['baseline','native_bridge_1','native_bridge_2']:
        values=[r['methods'][name] for r in scores]
        macro[name]=dict(instance_pq100=float(np.mean([v['instance']['observed_voxel_instance_pq'] for v in values])*100),
            category_pq100=float(np.mean([v['semantic']['class_aware_pq'] for v in values])*100),
            totals={k:sum(v['instance'][k] for v in values) for k in ['tp','fp','fn']},
            predicted_errors=dict(sum((Counter(v['predicted_error_counts']) for v in values),Counter())),
            gt_errors=dict(sum((Counter(v['gt_error_counts']) for v in values),Counter())))
    dump(out/'audit.json',dict(complete=True,scope='Post-hoc integrity and fixed-reference geometry audit after unseen category results; no parameter changes.',
        integrity=integrity,scene_geometry=scores,macro_geometry=macro,all_fixed_category_scores_reproduced=True,
        selection_sha256=sha256(DEST/'selection.json'),summary_sha256=sha256(DEST/'summary.json'),source_sha256=sha256(__file__)))
    print('AUDIT COMPLETE',macro,flush=True)


if __name__=='__main__':main()
