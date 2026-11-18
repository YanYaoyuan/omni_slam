import numpy as np
import pytest
from omni_semantic_map.context_residual_memory import ContextObservation,ContextResidualMemory,fit_context,unit


def add(memory,oid,fid,obj,feature,context,area=1.):
    memory.add(ContextObservation(oid,fid,obj,area,np.asarray(feature,float),np.asarray(context,float)))


def test_fixed_effects_remove_object_identity_before_estimating_context():
    gs=[np.array([[1.,0,0],[0,1,0],[0,0,1]]),np.array([[0.,1,0],[1,0,0]])]
    mus=[np.array([3.,2.,1.]),np.array([-2.,1.,4.])]
    fs=[mu+.7*g for mu,g in zip(mus,gs)]
    fit=fit_context(fs,gs);assert fit['beta']==pytest.approx(.7)
    m=ContextResidualMemory();i=0
    for obj,(f,g) in enumerate(zip(fs,gs),1):
        for fid,(a,b) in enumerate(zip(f,g)):
            add(m,i,fid,obj,a,b);i+=1
    r=m.readout()
    for oid,mu in enumerate(mus,1):np.testing.assert_allclose(r['objects'][oid]['descriptor'],unit(mu))


def test_future_evidence_revises_old_object_and_rollback_is_exact():
    m=ContextResidualMemory();mu=np.array([1.,2.,3.]);g=np.array([1.,0.,0.])
    add(m,0,0,1,mu+.8*g,g);before=m.readout()['objects'][1]['descriptor'].copy()
    for i,g in enumerate([np.array([1.,0.,0.]),np.array([0.,1.,0.])],1):add(m,i,i,2,np.array([2.,1.,4.])+.8*g,g)
    after=m.readout();np.testing.assert_allclose(after['objects'][1]['descriptor'],unit(mu))
    assert not np.allclose(before,after['objects'][1]['descriptor'])
    m.remove(2);m.remove(1);np.testing.assert_array_equal(m.readout()['objects'][1]['descriptor'],before)


def test_same_frame_duplicates_do_not_multiply_evidence_and_inputs_are_copied():
    m=ContextResidualMemory();f=np.array([1.,2.,3.]);g=np.array([0.,1.,0.])
    add(m,1,0,1,f,g,area=4);f[:]=9;g[:]=8
    add(m,2,0,1,[8,3,1],[1,0,0],area=2)
    np.testing.assert_allclose(m.readout('none')['objects'][1]['descriptor'],unit([1,2,3]))
    assert m.readout()['objects'][1]['selected_observation_ids']==[1]
    with pytest.raises(ValueError):add(m,1,1,1,[1,2,3],[0,1,0])


def test_constant_context_is_not_identifiable_and_does_not_subtract():
    f=np.array([[1.,0.,2.],[0.,1.,3.]]);g=np.array([[1.,0.,0.],[1.,0.,0.]])
    result=fit_context([f],[g]);assert not result['identifiable'];assert result['beta']==0


def test_group_permutation_and_sampling_duplication_do_not_change_fit():
    f=np.array([[1.,2.,3.],[2.,1.,3.],[1.,1.,4.]])
    g=np.eye(3);fit=fit_context([f,f+2],[g,g])
    other=fit_context([(f+2)[::-1],f[::-1]],[g[::-1],g[::-1]])
    assert fit['beta']==pytest.approx(other['beta'])


def test_invalid_and_degenerate_residual_are_explicit():
    m=ContextResidualMemory()
    with pytest.raises(ValueError):add(m,0,0,1,[float('nan'),1],[1,0])
    add(m,0,0,1,[1,0],[1,0]);r=m.readout('fixed',1.)
    assert r['objects'][1]['degenerate_fallback']
    np.testing.assert_array_equal(r['objects'][1]['descriptor'],[1.,0.])
