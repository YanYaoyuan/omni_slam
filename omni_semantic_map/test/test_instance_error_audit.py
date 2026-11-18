import numpy as np
from omni_semantic_map.instance_error_audit import audit_instances, filter_transition


def test_equal_pure_fragments_are_not_a_missing_detection():
    truth=np.ones(20,dtype=int);pred=np.repeat([4,5],10)
    r=audit_instances(truth,pred)
    assert r['predicted_counts']=={'fragmentation_consistent':2}
    assert r['gt_counts']=={'fragmentation_consistent':1}


def test_mixed_objects_and_background_have_distinct_signatures():
    truth=np.repeat([1,2,0],10);pred=np.repeat([4,4,5],10)
    r=audit_instances(truth,pred)
    assert r['predicted_counts']=={'mixed_foreground':1,'background_dominated':1}
    assert r['gt_counts']=={'mixed_prediction':2}


def test_partial_support_and_low_coverage_are_separate():
    truth=np.repeat([1,2],20);pred=np.zeros(40,int);pred[:6]=3
    r=audit_instances(truth,pred)
    assert r['predicted_counts']=={'partial_coverage':1}
    assert r['gt_counts']=={'insufficient_iou':1,'very_low_prediction_coverage':1}


def test_filter_transition_counts_losses_and_gains_despite_equal_tp():
    truth=np.repeat([1,2],10)
    a=audit_instances(truth,np.repeat([4,0],10));b=audit_instances(truth,np.repeat([0,8],10))
    assert filter_transition(a,b)==dict(lost_matched_gt=[1],gained_matched_gt=[2],retained_matched_gt=[])
