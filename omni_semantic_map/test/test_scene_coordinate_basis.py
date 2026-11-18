import numpy as np
import pytest
from omni_semantic_map.scene_coordinate_basis import canonicalize_world


def test_fixed_affine_world_restores_metric_and_preserves_projection():
    poses=np.repeat(np.eye(4)[None],3,axis=0)
    poses[1,:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
    poses[2,:3,:3]=[[1,0,0],[0,0,-1],[0,1,0]]
    poses[:,:3,3]=[[1,2,3],[2,-1,1],[-2,3,4]]
    G=np.eye(4);G[:3,:3]=[[1.04,.03,0],[0,1.02,.02],[0,0,.96]]
    raw=G[None]@poses
    result=canonicalize_world(raw)
    Q=result['canonical_from_raw']@G
    np.testing.assert_allclose(Q[:3,:3]@Q[:3,:3].T,np.eye(3),atol=1e-12)
    camera=np.array([.3,-.2,2.,1.])
    for i in range(3):
        world=result['canonical_from_raw']@raw[i]@camera
        np.testing.assert_allclose(np.linalg.inv(result['poses'][i])@world,camera,atol=1e-12)
    a=np.array([1.,2.,3.,1.]);b=np.array([3.,-1.,2.,1.])
    assert np.linalg.norm((Q@a-Q@b)[:3])==pytest.approx(np.linalg.norm((a-b)[:3]))


def test_rigid_input_has_identity_basis():
    poses=np.repeat(np.eye(4)[None],2,axis=0);poses[1,:3,3]=[1,2,3]
    result=canonicalize_world(poses)
    np.testing.assert_allclose(result['canonical_from_raw'],np.eye(4),atol=1e-12)
    np.testing.assert_allclose(result['poses'],poses,atol=1e-12)


def test_varying_deformation_is_not_silently_fitted():
    poses=np.repeat(np.eye(4)[None],2,axis=0);poses[1,0,0]=1.03
    with pytest.raises(ValueError,match='not a single fixed'):canonicalize_world(poses)


def test_reflection_rejected():
    pose=np.eye(4)[None];pose[0,0,0]=-1
    with pytest.raises(ValueError,match='reflected'):canonicalize_world(pose)
