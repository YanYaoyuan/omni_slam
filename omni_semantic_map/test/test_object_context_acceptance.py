import numpy as np
from run_object_context_acceptance import simple_veto
from omni_semantic_map.surface_ownership import allocate,append_owned


def test_simple_context_distinguishes_unknown_from_positive_alternative():
    x=np.zeros((3,1,50),np.float32);owners=np.ones((3,1),np.int64);valid=np.ones((3,1),bool)
    x[1,0,12:14]=[.2,.8];x[2,0,12:14]=[.8,.2]
    allowed=simple_veto(x,owners,valid,{1:26},{26:0})
    np.testing.assert_array_equal(allowed[:,0],[True,False,True])
    assigned,_=allocate(np.ones((3,1)),allowed,owners,independent=True)
    np.testing.assert_array_equal(assigned,[1,0,1])


def test_no_valid_owner_abstains_and_old_map_is_preserved():
    owners=np.array([[1,2],[1,2]]);valid=np.array([[0,0],[1,1]],bool)
    assigned,_=allocate(np.array([[3.,4.],[1.,2.]]),valid,owners,independent=True)
    np.testing.assert_array_equal(assigned,[0,2])
    core=np.array([[0.,0,0],[1.,0,0]]);source=np.array([[3.,0,0],[4.,0,0]])
    out=append_owned(core,np.array([1,2]),source,np.array([0,1]),assigned)
    np.testing.assert_array_equal(out['xyz'][:2],core)
    np.testing.assert_array_equal(out['instance'],[1,2,2])
    np.testing.assert_array_equal(out['added_source_ids'],[1])
