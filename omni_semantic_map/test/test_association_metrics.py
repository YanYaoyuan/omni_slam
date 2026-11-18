from omni_semantic_map.association_metrics import pair_score_with_misses
from omni_semantic_map.replica_benchmark import pair_score


def test_unassigned_detections_never_become_a_positive_pair():
    s=pair_score_with_misses([1,1,2,2],[0,0,3,3])
    assert s['precision']==1. and s['recall']==.5
    assert s['unassigned_observations']==2
    missing=pair_score_with_misses([1,1,2,2],[0,0,0,0])
    assert missing['precision'] is None and missing['recall']==0.


def test_all_assigned_matches_frozen_metric():
    for gt,pred in [([1,1,2,2],[8,8,9,9]),([1,1,2,2],[8,8,8,8]),([1,1,1,1],[8,8,9,9])]:
        old=pair_score(gt,pred);new=pair_score_with_misses(gt,pred)
        assert all(old[k]==new[k] for k in old)
