import numpy as np
import pytest
from omni_semantic_map.adaptive_fusion import AdaptiveReplayObjectMap, DispatchConfig
from omni_semantic_map.replay_fusion import FusionObservation, ReplayableObjectMap, ConceptGraphsBackend
from test_local_fusion import configuration


def assert_map_equal(a,b):
    assert a.labels() == b.labels()
    x,y=a.export_objects(),b.export_objects()
    assert len(x)==len(y)
    for p,q in zip(x,y):
        assert p['observation_ids']==q['observation_ids']
        for key in ['points','colors','box','image_feature','text_feature']:
            np.testing.assert_allclose(p[key],q[key],rtol=0,atol=1e-10)


@pytest.mark.parametrize('policy',['auto','checkpoint'])
def test_full_to_cold_local_to_full_with_new_observations_and_noop(policy):
    cfg=configuration();state=AdaptiveReplayObjectMap(cfg,DispatchConfig(checkpoint_interval=4),policy)
    full=ReplayableObjectMap(lambda:ConceptGraphsBackend(cfg))
    xyz=np.array(np.meshgrid(*([np.linspace(0,.12,7)]*3))).reshape(3,-1).T
    poses={k:np.eye(4) for k in range(16)}
    for k in range(16):
        o=FusionObservation(k,k,1,xyz,np.ones_like(xyz)*.5,[1,0],[1,0])
        state.add_frame(k,[o],poses[k]);full.add_frame(k,[o],poses[k])
    original=state.causal_labels()
    broad={k:p.copy() for k,p in poses.items()}
    for k in broad:broad[k][0,3]=.01*k+.2
    state.revise_poses(broad);full.revise_poses(broad);assert_map_equal(state,full)
    assert state.events[-1]['update_stats']['route']=='full'
    assert not state.events[-1]['update_stats']['dag_retained']
    tail={k:p.copy() for k,p in broad.items()}
    tail[15][0,3]+=.2
    state.revise_poses(tail);full.revise_poses(tail);assert_map_equal(state,full)
    assert state.events[-1]['update_stats']['route']=='checkpoint'
    assert state.events[-1]['update_stats']['replayed_frames']==4
    assert not state.events[-1]['update_stats']['warm_dag']
    state.revise_poses(poses);full.revise_poses(poses);assert_map_equal(state,full)
    assert state.causal_labels()==original
    state.revise_poses(poses)
    assert state.events[-1]['update_stats']['route']=='noop'
    for k in range(16,20):
        o=FusionObservation(k,k,1,xyz,np.ones_like(xyz)*.5,[1,0],[1,0])
        assert state.add_frame(k,[o],np.eye(4))==full.add_frame(k,[o],np.eye(4))
    assert_map_equal(state,full)


def test_warm_dependency_path_and_invalid_revision_transaction():
    cfg=configuration();state=AdaptiveReplayObjectMap(cfg,DispatchConfig(4,.5,1.))
    full=ReplayableObjectMap(lambda:ConceptGraphsBackend(cfg))
    xyz=np.array(np.meshgrid(*([np.linspace(0,.12,7)]*3))).reshape(3,-1).T
    poses={k:np.eye(4) for k in range(12)}
    for k in range(12):
        points=xyz+(0 if k<2 else 5)
        o=FusionObservation(k,k,1,points,np.ones_like(points)*.5,[1,0],[1,0])
        state.add_frame(k,[o],poses[k]);full.add_frame(k,[o],poses[k])
    changed={k:p.copy() for k,p in poses.items()};changed[0][0,3]=.1
    state.revise_poses(changed);full.revise_poses(changed);assert_map_equal(state,full)
    assert state.events[-1]['update_stats']['route']=='dependency'
    causal=state.causal_labels()
    sizes={}
    for name,target in [('original',poses),('changed',changed),('original',poses),('changed',changed)]:
        state.revise_poses(target);full.revise_poses(target);assert_map_equal(state,full)
        assert state.events[-1]['update_stats']['route']=='dependency'
        assert state.causal_labels()==causal
        engine=state._backend.engine
        size=(len(engine.states),len(engine.pair_scores),len(state._backend.checkpoints))
        if name in sizes:assert size==sizes[name]
        sizes[name]=size
    old_backend=state._backend
    with pytest.raises(ValueError):state.revise_poses({})
    assert state._backend is old_backend
