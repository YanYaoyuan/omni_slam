"""Regression for many-to-one raw-label mappings at the evaluator boundary."""
import numpy as np
from run_joint_semantic_revision import class_records


def test_building_keeps_canonical_raw_id_and_unchanged_fields():
    original = dict(objects=[dict(object_id=1, predictions=dict(mean=dict(
        train_id=2, semantic_id=11, name='building', score=.7, retained=True)))])
    q = np.eye(19)[[2]]
    result, changes = class_records(original, dict(object_ids=[1], object_probability=q))
    assert result['objects'][0]['predictions'] == original['objects'][0]['predictions']
    assert changes == []


def test_new_building_uses_evaluated_id_instead_of_garage_alias():
    original = dict(objects=[dict(object_id=1, predictions=dict(mean=dict(
        train_id=13, semantic_id=26, name='car', score=.7)))])
    result, changes = class_records(original, dict(object_ids=[1], object_probability=np.eye(19)[[2]]))
    assert result['objects'][0]['predictions']['mean'] == dict(train_id=2, semantic_id=11, name='building', score=.7)
    assert changes == [dict(object_id=1, before=13, after=2)]
