import numpy as np
from omni_semantic_map.witness_graph import WitnessConfig,WitnessGraph


def obs(frame,x):
    xyz=np.array([[x-.02,0,2],[x,0,2],[x+.02,0,2]])
    return dict(keyframe_id=frame,local_position=[x,0,2],label='object',local_xyz=xyz,
                local_features=np.tile([1.,0.],(3,1)),descriptor=np.array([1.,0.]),support_xyz=xyz.copy())


def setup(policy='witness'):
    poses=np.tile(np.eye(4),(3,1,1));poses[2,0,3]=.15
    tracker=WitnessGraph(conflict_policy=policy)
    tracker.add_frame([obs(0,-.03),obs(0,.03)],poses)
    return tracker,poses


def test_two_independent_witnesses_relax_conflict_only_when_available():
    t,p=setup();early=t.labels()
    t.add_frame([obs(1,0)],p)
    assert t.labels()[0]!=t.labels()[1]
    t.add_frame([obs(2,-.15)],p)
    assert len(set(t.labels()))==1
    assert early.tolist()==[1,2]
    assert any(e['proof'] for e in t.events[-1]['relaxed_conflicts'])


def test_repeated_camera_position_is_not_independent_evidence():
    t,p=setup();p[2,0,3]=0
    t.add_frame([obs(1,0)],p);t.add_frame([obs(2,0)],p)
    assert t.labels()[0]!=t.labels()[1]


def test_surface_contact_is_required_even_with_two_witnesses():
    t,p=setup();t.observations[1]['support_xyz']+=10
    t.add_frame([obs(1,0)],p);t.add_frame([obs(2,-.15)],p)
    assert t.labels()[0]!=t.labels()[1]


def test_pose_revision_can_revoke_the_override():
    t,p=setup();t.add_frame([obs(1,0)],p);t.add_frame([obs(2,-.15)],p)
    assert len(set(t.labels()))==1
    corrected=p.copy();corrected[2,0,3]=.02;t.revise_poses(corrected)
    assert t.labels()[0]!=t.labels()[1]
    t.revise_poses(p)
    assert len(set(t.labels()))==1


def test_hard_and_off_controls_have_expected_distinct_behavior():
    for policy,merged in [('hard',False),('off',True)]:
        t,p=setup(policy);t.add_frame([obs(1,0)],p)
        assert (t.labels()[0]==t.labels()[1])==merged
