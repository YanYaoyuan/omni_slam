import numpy as np
import pytest

from omni_semantic_map.full_ovo_evaluation import ovo_labels, voxel_metrics, cg_geometry
from omni_semantic_map.mesh_instance_evaluation import transfer_labels


def classification():
    return {'objects': [dict(object_id=0, replica_class_id=20),
                        dict(object_id=2, replica_class_id=93),
                        dict(object_id=7, replica_class_id=-1),
                        dict(object_id=8, replica_class_id=37)]}


def test_zero_is_an_object_unknown_and_door_retained():
    ids = np.array([-1, 0, 2, 7, 8])
    np.testing.assert_array_equal(ovo_labels(ids, classification(), filter_background=True), [0, 1, 0, 8, 9])
    np.testing.assert_array_equal(ovo_labels(ids, classification(), filter_background=False), [0, 1, 3, 8, 9])
    np.testing.assert_array_equal(ids, [-1, 0, 2, 7, 8])


def test_missing_or_duplicate_classification_fails():
    with pytest.raises(ValueError, match='missing'):
        ovo_labels(np.array([9]), classification(), filter_background=True)
    c = classification(); c['objects'].append(c['objects'][0])
    with pytest.raises(ValueError, match='duplicate'):
        ovo_labels(np.array([0]), c, filter_background=False)


def test_native_object_zero_reaches_geometry_score():
    points = np.array([[0., 0., 0.], [1., 0., 0.]])
    labels = ovo_labels(np.array([0, 2]), classification(), filter_background=True)
    predicted, _ = transfer_labels(points, labels, points, mode='nearest_5cm')
    m = voxel_metrics(np.array([42, 0]), predicted)
    assert m['tp'] == 1 and m['observed_voxel_instance_pq'] == 1


def test_background_predictions_count_as_false_positives():
    m = voxel_metrics(np.array([42, 42, 0, 0]), np.array([1, 1, 3, 3]))
    assert m['tp'] == 1 and m['fp'] == 1
    assert m['observed_voxel_instance_pq'] == pytest.approx(2/3)


def test_unobserved_geometry_cannot_fill_a_far_reference_point():
    q, _ = transfer_labels(np.array([[0., 0., 0.]]), np.array([1]),
                           np.array([[0.2, 0., 0.]]), mode='nearest_5cm')
    m = voxel_metrics(np.array([42]), q)
    assert m['fn'] == 1 and m['observed_voxel_instance_pq'] == 0


def test_distinct_cg_objects_with_duplicate_export_alias_remain_distinct():
    objects = [dict(object_id=183, points=np.array([[0., 0., 0.]])),
               dict(object_id=183, points=np.array([[1., 0., 0.]]))]
    points, labels, mapping = cg_geometry(objects)
    predicted, _ = transfer_labels(points, labels, points, mode='nearest_5cm')
    assert voxel_metrics(np.array([10, 20]), predicted)['tp'] == 2
    assert [x['export_alias'] for x in mapping] == [183, 183]
    assert objects[0]['object_id'] == objects[1]['object_id'] == 183


def test_unique_alias_order_preserves_nearest_neighbor_vote_ties():
    objects = [dict(object_id=20, points=np.array([[0., 0., 0.]])),
               dict(object_id=3, points=np.array([[0.02, 0., 0.]]))]
    points, labels, mapping = cg_geometry(objects)
    assert labels.tolist() == [2, 1]
    q, _ = transfer_labels(points, labels, np.array([[.01, 0., 0.]]), mode='knn5_unbounded')
    assert q[0] == 1 and mapping[1]['export_alias'] == 3
