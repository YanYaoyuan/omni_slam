import numpy as np
import pytest
from omni_semantic_map.dino_history_evidence import local_similarity,relative_vote
from omni_semantic_map.semantic_instance_diagnostic import semantic_metrics


def test_local_mutual_matching_and_missing_support():
    assert local_similarity(np.eye(3)*2,np.eye(3))==1
    assert np.isnan(local_similarity(np.eye(2),np.eye(2)))
    assert np.isnan(local_similarity(np.empty((0,3)),np.eye(3)))


def test_relative_rule_requires_both_branches_and_all_competitors():
    assert relative_vote([.9],[.8],[.7])['pass_vote']
    assert not relative_vote([.9],[.6],[.7])['pass_vote']
    assert not relative_vote([.9],[.7],[.7])['pass_vote']
    assert relative_vote([.9],[.8],[.7,np.nan])['reason']=='unknown_native_fallback'
    assert relative_vote([],[],[])['pass_vote']


def test_perfect_geometry_wrong_class_is_not_semantic_success():
    result=semantic_metrics(np.array([1,1,0]),np.array([5,5,0]),{1:4},{5:18})
    assert result['instance_pq']==1 and result['class_aware_pq']==0
    assert (result['class_aware_tp'],result['class_aware_fp'],result['class_aware_fn'])==(0,1,1)


def test_unknown_prediction_counts_wrong_and_missing_gt_is_rejected():
    result=semantic_metrics(np.array([1,1,2,2]),np.array([5,5,6,6]),{1:4,2:18},{5:4})
    assert result['class_aware_pq']==.5 and result['matched_class_accuracy']==.5
    assert result['unknown_predicted_classes']==1
    with pytest.raises(ValueError):semantic_metrics(np.array([1]),np.array([1]),{},{1:4})
