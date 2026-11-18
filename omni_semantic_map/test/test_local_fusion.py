import numpy as np
import pytest
from omni_semantic_map.local_fusion import interaction_closure
from omni_semantic_map.replay_fusion import FusionObservation, ReplayableObjectMap, ConceptGraphsBackend


def box(x):
    return np.array([x,0,0.]), np.array([x+.1,.1,.1])


def test_closure_includes_new_neighbors_not_just_old_members_and_keeps_distant_objects():
    groups = [[0,1], [2], [3], [4]]
    old = {0:box(0),1:box(0),2:box(1),3:box(1.12),4:box(8)}
    new = dict(old, **{})
    new[1] = box(1)
    selected, iterations = interaction_closure(groups,old,new,{1},.025)
    assert selected == {0,1,2,3}
    assert iterations >= 2


def test_closure_handles_empty_rejected_observation_and_no_change():
    assert interaction_closure([[0],[1]],{0:None,1:box(2)},{0:None,1:box(2)},{0},.025)[0] == {0}
    assert interaction_closure([[0]],{0:box(0)},{0:box(0)},set(),.025)[0] == set()


def configuration():
    pytest.importorskip('conceptgraph')
    from omegaconf import OmegaConf
    return OmegaConf.create(dict(device='cpu',spatial_sim_type='overlap',match_method='sim_sum',
        phys_bias=0.,sim_threshold=1.2,min_points_threshold=16,downsample_voxel_size=.025,
        dbscan_remove_noise=True,dbscan_eps=.1,dbscan_min_points=10,obj_min_points=0,
        obj_min_detections=2,merge_overlap_thresh=.7,merge_visual_sim_thresh=.7,
        merge_text_sim_thresh=.7,denoise_interval=4,filter_interval=-1,merge_interval=-1))


@pytest.mark.parametrize('mode',['local','full','dependency'])
def test_native_exactness_when_replay_crosses_empty_local_frames_and_later_continues(mode):
    from omni_semantic_map.local_fusion import LocalReplayObjectMap
    from omni_semantic_map.dependency_fusion import DependencyReplayObjectMap
    cfg=configuration()
    local=DependencyReplayObjectMap(cfg) if mode=='dependency' else LocalReplayObjectMap(cfg,mode)
    native=ReplayableObjectMap(lambda:ConceptGraphsBackend(cfg))
    xyz=np.array(np.meshgrid(*([np.linspace(0,.12,7)]*3))).reshape(3,-1).T
    poses={};groups=[]
    for k in range(12):
        # Frame 4,8,12 contain only the distant object: global native denoising
        # still applies to the selected earlier object on those exact ticks.
        points=xyz+(0 if k<3 else 5)
        obs=FusionObservation(k,k,1,points,np.ones_like(points)*.5,[1,0],[1,0])
        poses[k]=np.eye(4);groups.append([obs])
        local.add_frame(k,[obs],poses[k]);native.add_frame(k,[obs],poses[k])
    assert local.labels()==native.labels()
    causal=local.causal_labels()
    corrected={k:p.copy() for k,p in poses.items()}
    corrected[0][0,3]=.6
    assert local.revise_poses(corrected)==native.revise_poses(corrected)
    assert local.causal_labels()==causal
    for a,b in zip(local.export_objects(),native.export_objects()):
        assert a['observation_ids']==b['observation_ids']
        for key in ['points','colors','box','image_feature','text_feature']:
            np.testing.assert_allclose(a[key],b[key],rtol=0,atol=1e-10)
    if mode=='local':
        assert local.events[-1]['update_stats']['replayed_observations']==3
        assert local.events[-1]['update_stats']['reused_online_objects']==1
    if mode=='dependency':
        stats=local.events[-1]['update_stats']
        assert stats['fusion_hits']>0 and stats['fusion_misses']>0
        assert stats['obsolete_states_evicted']>0
    assert local.revise_poses(poses)==native.revise_poses(poses)
    obs=FusionObservation(12,12,1,xyz,np.ones_like(xyz)*.5,[1,0],[1,0])
    assert local.add_frame(12,[obs],np.eye(4))==native.add_frame(12,[obs],np.eye(4))
    assert local.labels()==native.labels()
    if mode=='dependency':
        current=dict(poses);current[12]=np.eye(4)
        local.revise_poses(current)
        assert local.events[-1]['update_stats']['fusion_misses']==0
        assert local.events[-1]['update_stats']['denoise_misses']==0


@pytest.mark.parametrize('method',['envelope','dependency'])
def test_native_correction_discovers_previously_absent_cross_object_merge(method):
    from omni_semantic_map.local_fusion import LocalReplayObjectMap
    from omni_semantic_map.dependency_fusion import DependencyReplayObjectMap
    cfg=configuration();local=LocalReplayObjectMap(cfg) if method=='envelope' else DependencyReplayObjectMap(cfg)
    full=ReplayableObjectMap(lambda:ConceptGraphsBackend(cfg))
    xyz=np.array(np.meshgrid(*([np.linspace(0,.12,7)]*3))).reshape(3,-1).T
    poses={k:np.eye(4) for k in range(6)}
    for k in range(6):
        pts=xyz+(np.array([1.,0,0]) if k>=3 else 0)
        o=FusionObservation(k,k,1,pts,np.ones_like(pts)*.5,[1,0],[1,0])
        local.add_frame(k,[o],poses[k]);full.add_frame(k,[o],poses[k])
    assert len(set(local.labels().values()))==2
    corrected={k:p.copy() for k,p in poses.items()}
    for k in range(3,6):corrected[k][0,3]=-1
    assert local.revise_poses(corrected)==full.revise_poses(corrected)
    assert len(set(local.labels().values()))==1
    if method=='envelope':assert local.events[-1]['update_stats']['replayed_observations']==6


def test_dependency_native_full_frame_fallback_bypasses_pair_cache(monkeypatch):
    from omni_semantic_map.dependency_fusion import DependencyReplayObjectMap
    cfg=configuration()
    from conceptgraph.utils import ious
    local=DependencyReplayObjectMap(cfg);native=ReplayableObjectMap(lambda:ConceptGraphsBackend(cfg))
    xyz=np.array(np.meshgrid(*([np.linspace(0,.12,7)]*3))).reshape(3,-1).T
    for k in range(3):
        o=FusionObservation(k,k,1,xyz,np.ones_like(xyz)*.5,[1,0],[1,0])
        local.add_frame(k,[o],np.eye(4));native.add_frame(k,[o],np.eye(4))
    def trigger_guard(*args,**kwargs):raise ValueError('simulate frame-wide numerical fallback')
    monkeypatch.setattr(ious,'compute_3d_iou_accuracte_batch',trigger_guard)
    assert local.revise_poses({k:np.eye(4) for k in range(3)})==native.labels()
    stats=local.events[-1]['update_stats']
    assert stats['native_fallback_frames']==2 and stats['pair_hits']==0
