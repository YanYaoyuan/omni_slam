from pathlib import Path
import numpy as np
from omni_semantic_map.kitti360_evaluation import evaluate

SOURCE = Path(__file__).resolve().parents[2]/'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py'


def test_perfect_two_class_predictions():
    truth = np.r_[np.full(300, 2001), np.full(300, 13001)]
    pred = np.r_[np.ones(300, int), np.full(300, 2)]
    out = evaluate(truth, np.ones(600), pred, {1:dict(semantic_id=11, score=.7), 2:dict(semantic_id=26, score=.6)}, SOURCE)
    assert out['averages']['allAp50%'] == 1
    assert out['averages']['allAp'] == 1


def test_half_overlap_is_not_an_ap50_match():
    truth = np.full(300, 2001)
    pred = np.r_[np.ones(150, int), np.zeros(150, int)]
    out = evaluate(truth, np.ones(300), pred, {1:dict(semantic_id=11, score=.7)}, SOURCE)
    assert out['averages']['classes']['building']['ap50%'] == 0
    assert out['averages']['classes']['building']['ap25%'] == 1


def test_confidence_weighted_region_threshold_not_raw_point_count():
    truth = np.full(300, 2001)
    out = evaluate(truth, np.full(300, .25), np.ones(300, int), {1:dict(semantic_id=11, score=.7)}, SOURCE)
    assert out['matches']['window']['groundTruth']['building'][0]['vert_count'] == 75
    assert not out['matches']['window']['prediction']['building']
    assert np.isnan(out['averages']['classes']['building']['ap50%'])


def test_wrong_semantic_class_cannot_match_correct_geometry():
    truth = np.full(300, 2001)
    out = evaluate(truth, np.ones(300), np.ones(300, int), {1:dict(semantic_id=26, score=.7)}, SOURCE)
    assert out['averages']['classes']['building']['ap50%'] == 0
