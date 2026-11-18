import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from omni_semantic_map.intrinsic_surface_geometry import geometry_features


@pytest.mark.parametrize('kind',['anisotropic','plane','line','isotropic','singleton'])
def test_rigid_frame_invariance_even_when_principal_axes_are_ambiguous(kind):
    rng=np.random.RandomState(15);c=rng.randn(80,3)*[3,.7,.2]
    if kind=='plane':c[:,2]=0
    if kind=='line':c[:,1:]=0
    if kind=='isotropic':c=np.r_[np.eye(3),-np.eye(3)]
    if kind=='singleton':c=np.array([[1.,2.,3.]])
    p=np.r_[rng.randn(31,3),c.mean(0)[None]];other=rng.randn(41,3)+[8,2,0]
    a=geometry_features(p,c,other,26)
    for rot,t in [(Rotation.from_euler('xyz',[21,37,83],degrees=True).as_matrix(),np.array([100,-300,12.])),
                  (Rotation.from_euler('z',90,degrees=True).as_matrix(),np.zeros(3))]:
        b=geometry_features(p@rot.T+t,c@rot.T+t,other@rot.T+t,26)
        np.testing.assert_allclose(a,b,rtol=2e-5,atol=2e-5)


def test_shape_information_and_core_permutation_are_preserved():
    c=np.array([[-3,0,0],[-1,0,0],[1,0,0],[3,0,0.]])
    p=np.array([[4,0,0],[0,4,0.]])
    a=geometry_features(p,c,np.empty((0,3)),26)
    b=geometry_features(p,c[::-1],np.empty((0,3)),26)
    np.testing.assert_array_equal(a,b)
    assert a[0,4]<a[1,4]  # Equal centroid distance, different relation to shape.
    assert a[0,5]>a[1,5]


def test_empty_queries_and_invalid_core():
    assert geometry_features(np.empty((0,3)),np.zeros((1,3)),np.empty((0,3)),11).shape==(0,12)
    with pytest.raises(ValueError):geometry_features(np.zeros((1,3)),np.empty((0,3)),np.empty((0,3)),11)
