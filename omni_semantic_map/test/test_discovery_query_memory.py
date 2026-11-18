import numpy as np
import pytest
from omni_semantic_map.discovery_query_memory import DiscoveryQueryMemory


def frame(fid,*ids):return dict(frame_id=fid,visible_unit_ids=np.array(ids,dtype=int))


def test_residual_seeks_unexplained_points_while_geometry_control_can_repeat_objects():
    a=DiscoveryQueryMemory();b=DiscoveryQueryMemory('geometric')
    for m in [a,b]:
        assert m.choose([frame(0,0,1,2,3)],0)==0;m.commit(0,np.array([0,1,2,3]))
    candidates=[frame(1,4),frame(2,0,1,2,3)]
    assert a.choose(candidates,2)==1
    assert b.choose(candidates,2)==2


def test_unsuccessful_queries_discount_repetition_without_deleting_support():
    m=DiscoveryQueryMemory();m.choose([frame(0,0,0,1)],0);m.commit(0,np.array([],dtype=int))
    np.testing.assert_array_equal(m.queries,[1,1]);assert not m.explained.any()
    assert m.choose([frame(1,0,1),frame(2,2,3)],2)==2


def test_future_inputs_and_unmeasured_positive_support_are_rejected():
    m=DiscoveryQueryMemory()
    with pytest.raises(ValueError):m.choose([frame(4,0)],3)
    m.choose([frame(1,0)],1)
    with pytest.raises(ValueError):m.commit(1,np.array([1]))
    with pytest.raises(ValueError):m.choose([frame(2,0)],2)
    m.commit(1,np.array([0]))
    with pytest.raises(ValueError):m.choose([frame(1,0)],2)


def test_same_budget_and_empty_depth_fallback_are_deterministic():
    for policy in ['uniform','geometric','residual']:
        m=DiscoveryQueryMemory(policy)
        assert m.choose([frame(0),frame(1)],1)==1
        m.commit(1,np.array([],dtype=int));assert len(m.events)==1 and m.events[0]['queried_units']==0
