import numpy as np
import pytest
from omni_semantic_map.consensus_category import classify


def test_one_vote_per_frame_and_no_retention_decision():
    f=np.array([[1.,0],[0,1],[1,0],[1,0]])
    result=classify([0,1,2,3,3],f,np.array([0,0,1,2]),np.array([10,1,10,10]),np.eye(2),
                    {'names':['thing','background'],'replica_class_ids':[7,0]})
    assert result['selected_observation_ids']==[0,2,3]
    assert result['name']=='thing' and 'retained' not in result
    assert len(result['weights'])==3


def test_empty_object_is_not_silently_classified():
    with pytest.raises(ValueError):classify([],np.eye(2),[0,1],[1,1],np.eye(2),{})
