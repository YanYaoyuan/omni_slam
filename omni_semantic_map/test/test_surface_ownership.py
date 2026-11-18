import numpy as np
import pytest
import torch
from omni_semantic_map.surface_ownership import allocate, append_owned, grouped_loss


def test_null_and_competition_are_real_alternatives():
    logits=np.array([[2.,2.],[-1.,-2.],[3.,-2.]])
    valid=np.ones_like(logits,bool);ids=np.array([[7,9],[7,9],[7,9]])
    owner,_=allocate(logits,valid,ids)
    np.testing.assert_array_equal(owner,[0,0,7])
    # Two moderately positive claims individually pass, but jointly neither
    # dominates null plus the competing owner. This is a genuine ablation.
    logits=np.array([[.8,.6]])
    assert allocate(logits,valid[:1],ids[:1])[0][0]==0
    assert allocate(logits,valid[:1],ids[:1],independent=True)[0][0]==7


def test_permuting_owner_ids_or_padding_cannot_change_physical_choice():
    z=np.array([[4.,-1.,999.]])
    valid=np.array([[True,True,False]]);ids=np.array([[71,92,0]])
    a=allocate(z,valid,ids)[0]
    order=[2,1,0]
    np.testing.assert_array_equal(a,allocate(z[:,order],valid[:,order],ids[:,order])[0])
    assert a[0]==71


def test_set_likelihood_includes_null_and_accepts_multiple_correct_aliases():
    z=torch.tensor([[1.,2.],[-1.,-2.]],requires_grad=True)
    valid=torch.ones_like(z,dtype=torch.bool)
    positive=torch.tensor([[True,True],[False,False]])
    loss=grouped_loss(z,valid,positive)
    expected=torch.stack([torch.logsumexp(torch.tensor([0.,1.,2.]),0)-torch.logsumexp(z[0],0),
                          torch.logsumexp(torch.tensor([0.,-1.,-2.]),0)])
    torch.testing.assert_close(loss,expected)
    loss.mean().backward();assert torch.isfinite(z.grad).all()


def test_map_preserves_core_and_only_appends_measured_exclusive_sources():
    xyz=np.array([[1.,2.,3.],[4.,5.,6.]])
    source=np.array([[9.,8.,7.],[8.,7.,6.]])
    r=append_owned(xyz,np.array([3,8]),source,np.array([0,1]),np.array([8,0]))
    np.testing.assert_array_equal(r['xyz'][:2],xyz)
    np.testing.assert_array_equal(r['xyz'][2:],source[:1])
    np.testing.assert_array_equal(r['instance'],[3,8,8])
    with pytest.raises(ValueError):append_owned(xyz,np.array([3,8]),source,[0,0],[3,8])
    with pytest.raises(ValueError):append_owned(xyz,np.array([3,8]),source,[0],[99])
