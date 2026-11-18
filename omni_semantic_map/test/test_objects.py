import numpy as np
from omni_semantic_map.objects import associate, project, query, reproject_map


def observation(key, x=0.0, stamp=None):
    return dict(
        keyframe_id=key,
        stamp=float(key * 3 if stamp is None else stamp),
        local_position=[x, 0, 2],
        label="car",
        appearance=[1.0, 0.0],
        score=0.9,
        evidence="test.jpg",
    )


def test_projection_rejects_behind_camera_and_applies_mount():
    t = np.eye(4)
    t[0, 3] = 1
    pixels, _, valid = project(
        [[0, 0, 2], [0, 0, -2]], t, [[100, 0, 50], [0, 100, 40], [0, 0, 1]]
    )
    np.testing.assert_allclose(pixels[0], [100, 40])
    assert valid.tolist() == [True, False]


def test_association_requires_repeated_distinct_keyframes_and_query_filters():
    poses = np.repeat(np.eye(4)[None], 4, axis=0)
    objects = []
    for k in range(3):
        associate(objects, [observation(k)], poses)
    assert len(objects) == 1 and objects[0]["confirmed"]
    assert len(query({"objects": objects}, "找车")) == 1
    assert query({"objects": objects}, "unknown") == []


def test_two_detections_in_one_frame_cannot_merge_into_one_object():
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    objects = []
    associate(objects, [observation(0)], poses)
    associate(objects, [observation(1, 0.1), observation(1, 0.3)], poses)
    assert len(objects) == 2
    assert max(len(o["observations"]) for o in objects) == 2


def test_ambiguous_neighbours_remain_separate_proposals():
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    objects = []
    associate(objects, [observation(0, -0.6), observation(0, 0.6)], poses)
    associate(objects, [observation(1, 0)], poses)
    assert len(objects) == 3


def test_ambiguity_ablation_stops_repeated_proposal_creation():
    # Rejected midpoint observations create nearby candidates which then make
    # subsequent observations ambiguous too. The candidate is an ablation only.
    poses = np.repeat(np.eye(4)[None], 6, axis=0)
    baseline, candidate = [], []
    for objects in [baseline, candidate]:
        associate(objects, [observation(0, -.6), observation(0, .6)], poses)
    for k in range(1, 6):
        associate(baseline, [observation(k, 0)], poses)
        associate(candidate, [observation(k, 0)], poses, ambiguity_margin=0.)
    assert len(baseline) > len(candidate)
    assert len(candidate) == 2


def test_pose_revision_moves_objects_without_mutating_anchors_or_ids():
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    objects = []
    for k in range(3):
        associate(objects, [observation(k)], poses)
    document = {"objects": objects, "map_revision": "before"}
    poses[:, :3, 3] = [4, 2, -1]
    revised = reproject_map(document, poses, "after")
    np.testing.assert_allclose(revised["objects"][0]["position"], [4, 2, 1])
    assert revised["objects"][0]["id"] == objects[0]["id"]
    assert document["objects"][0]["position"] == [0.0, 0.0, 2.0]
    assert (
        revised["objects"][0]["observations"] == document["objects"][0]["observations"]
    )
