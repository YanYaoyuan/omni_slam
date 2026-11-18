import copy
import pytest
from omni_semantic_map.evaluation import annotation_template, evaluate


def data(groups):
    return {"objects": [dict(id=str(i), label="car", observations=[
        dict(keyframe_id=k, stamp=float(k), evidence="frame.jpg") for k in group])
        for i, group in enumerate(groups)]}


def label(document, identities):
    template = annotation_template(document)
    for row, identity in zip(template["observations"], identities):
        row.update(valid_detection=True, gt_instance_id=identity)
    return template


def test_unannotated_is_not_perfect_or_zero_accuracy():
    d = data([[0, 1], [2]])
    result = evaluate(d, annotation_template(d))
    assert result["status"] == "no_identity_labels"
    assert result["pair_precision"] is None and result["pair_recall"] is None
    assert result["pending_observations"] == 3


def test_split_object_loses_recall_but_not_precision():
    d = data([[0, 1], [2, 3]])
    result = evaluate(d, label(d, ["A"] * 4))
    assert result["pair_precision"] == 1
    assert result["pair_recall"] == pytest.approx(1 / 3)
    assert result["split_gt_instances"] == 1
    assert result["merged_predicted_objects"] == 0


def test_false_merge_loses_precision_but_not_recall():
    d = data([[0, 1, 2, 3]])
    result = evaluate(d, label(d, ["A", "A", "B", "B"]))
    assert result["pair_precision"] == pytest.approx(1 / 3)
    assert result["pair_recall"] == 1
    assert result["merged_predicted_objects"] == 1


def test_missing_and_false_detections_are_disclosed_and_not_identity_negatives():
    d = data([[0, 1, 2, 3]])
    a = label(d, ["A", "A"])
    a["observations"][2]["valid_detection"] = False
    a["observations"].pop()
    r = evaluate(d, a)
    assert r["pair_f1"] == 1
    assert r["status"] == "partial_annotation"
    assert r["rejected_detection_observations"] == r["pending_observations"] == 1


def test_reject_stale_map_duplicate_and_unknown_labels():
    d = data([[0, 1]])
    a = annotation_template(d)
    changed = copy.deepcopy(d)
    changed["objects"][0]["id"] = "another"
    with pytest.raises(ValueError, match="source mismatch"):
        evaluate(changed, a)
    a["observations"].append(a["observations"][0])
    with pytest.raises(ValueError, match="duplicate or unknown"):
        evaluate(d, a)


def test_unique_singletons_have_undefined_pair_metrics():
    d = data([[0], [1]])
    r = evaluate(d, label(d, ["A", "B"]))
    assert r["pair_precision"] is None and r["pair_recall"] is None
