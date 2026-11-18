import numpy as np
import pytest
from omni_semantic_map.semantic_context_evidence import reduce_support,equal_cell_mean,context_relative


def test_query_duplicate_does_not_increase_support_and_unknown_is_explicit():
    m=np.array([[1,0,0],[0,1,0]],bool);scores=np.array([.2,.7])
    value,witness=reduce_support(m,scores)
    np.testing.assert_allclose(value,[.2,.7,0]);np.testing.assert_array_equal(witness,[0,1,-1])
    repeated,_=reduce_support(m[[0,1,1]],scores[[0,1,1]])
    np.testing.assert_array_equal(value,repeated)


def test_context_is_not_weighted_by_measurement_density():
    x=np.array([[.1,.9],[.7,.2]])
    np.testing.assert_allclose(equal_cell_mean(x,[0,1]),equal_cell_mean(x[[0,0,0,1]],[0,0,0,1]))


def test_relative_channels_cancel_common_category_bias():
    x=np.array([[.2,.4],[.7,.3]]);mu=np.array([.1,.2]);bias=np.array([.1,.15])
    a=context_relative(x,mu);b=context_relative(x+bias,mu+bias)
    np.testing.assert_allclose(a[:,2:],b[:,2:])
    assert not np.allclose(a[:,:2],b[:,:2])


def test_invalid_native_scores_rejected():
    with pytest.raises(ValueError):reduce_support(np.ones((1,2),bool),[1.1])
