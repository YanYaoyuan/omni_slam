import json
from collections import Counter
import numpy as np
from PIL import Image
import pytest
from omni_semantic_map.flat_benchmark import camera_points, instance_metrics, run, evaluate, read_pose
from omni_semantic_map.flat_surface_evaluation import majority_labels


def test_camera_z_backprojection_and_depth_gate():
    depth = np.full((480, 640), 2., np.float32)
    depth[0, 0] = np.nan
    p, valid, _, _ = camera_points(depth, 4)
    np.testing.assert_allclose(p[60, 80], [0, 0, 2])
    np.testing.assert_allclose(p[60, 96], [.4, 0, 2])
    assert not valid[0, 0]


def test_split_and_merge_penalize_instance_metric():
    perfect = instance_metrics(Counter({(1, 1): 100, (2, 2): 100}))
    split = instance_metrics(Counter({(1, 1): 50, (1, 2): 50, (2, 3): 100}))
    merge = instance_metrics(Counter({(1, 1): 100, (2, 1): 100}))
    assert perfect['sequence_pixel_instance_pq'] == 1
    assert split['tp'] == 1 and split['fp'] == 2 and split['fn'] == 1
    assert merge['tp'] == 0 and merge['sequence_pixel_instance_pq'] == 0


def test_missed_and_background_false_objects_remain_in_denominator():
    r = instance_metrics(Counter({(1, 1): 100, (2, 0): 100, (0, 2): 100}))
    assert r['sequence_pixel_instance_pq'] == .5
    assert r['gt_thing_sample_coverage'] == .5
    assert r['fn'] == r['fp'] == 1
    assert instance_metrics(Counter({(0, 0): 100}))['sequence_pixel_instance_pq'] is None


def test_predicted_mapper_runs_without_gt_and_does_not_use_frame_segment_id_as_identity(tmp_path):
    sequence = tmp_path / 'run1'
    sequence.mkdir()
    (sequence / 'timestamps.csv').write_text('ImageID,TimeStamp\n000001,6000000000\n000000,1000000000\n')
    for i, segment in enumerate([1, 19]):
        prefix = sequence / f'{i:06d}'
        Image.fromarray(np.full((480, 640, 3), 100, np.uint8)).save(str(prefix) + '_color.png')
        Image.fromarray(np.full((480, 640), 2., np.float32)).save(str(prefix) + '_depth.tiff')
        mask = np.zeros((480, 640), np.uint8)
        mask[160:320, 240:400] = segment
        Image.fromarray(mask).save(str(prefix) + '_predicted.png')
        np.savetxt(str(prefix) + '_pose.txt', np.eye(4))
        (sequence / f'{i:06d}_labels.json').write_text(json.dumps([
            dict(id=segment, isthing=True, category_id=60, score=.9)]))
    output = tmp_path / 'result'
    result = run(sequence, output)
    assert result['frames_processed'] == 2 and result['object_records'] == 1
    assert result['gt_segmentation_used_by_mapper'] is False
    # Only the separate evaluator is supplied GT.
    (tmp_path / 'groundtruth_labels.csv').write_text(
        'InstanceID,ClassID,PanopticID,Name\n0,0,0,background\n7,2,1,chair\n')
    for i in range(2):
        gt = np.zeros((480, 640), np.uint8)
        gt[160:320, 240:400] = 7
        Image.fromarray(gt).save(sequence / f'{i:06d}_segmentation.png')
    r = evaluate(sequence, output, tmp_path / 'metrics.json')
    assert r['metrics']['sequence_pixel_instance_pq'] == 1
    with np.load(output / 'supports/000000_1.npz') as data:
        assert data['local_points'].shape == (1600, 3)
    with pytest.raises(FileExistsError):
        run(sequence, output)


def test_pose_rejects_reflection(tmp_path):
    p = np.eye(4)
    p[0, 0] = -1
    path = tmp_path / 'pose.txt'
    np.savetxt(path, p)
    with pytest.raises(ValueError, match='invalid camera-to-world'):
        read_pose(path)


def test_surface_vote_keeps_missed_predictions_and_deterministic_ties():
    voxels = np.array([0, 0, 0, 1, 1, 2, 2])
    values = np.array([3, 0, 0, 8, 2, 9, 9])
    np.testing.assert_array_equal(majority_labels(voxels, values, 3), [0, 2, 9])
