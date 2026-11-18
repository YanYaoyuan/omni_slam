import numpy as np
import pytest
import torch
from omni_semantic_map.provenance_object_memory import (
    ProvenanceObjectMemory,disjoint_inputs,ViewEvidenceAllocator,revision_map)


def test_absence_preserves_identity_and_view_provenance_and_invalid_write_is_atomic():
    bank=ProvenanceObjectMemory([8,19],3)
    p=np.eye(19,dtype=np.float32)[[2,13]]
    bank.observe(0,[1,2],[8,19],[4,5],np.array([[1,0,0],[0,1,0.]]),p)
    old=bank.views[19][0]['descriptor'].copy()
    bank.observe(1,[1],[8],[4],np.array([[.8,.2,0]]),p[:1])
    assert len(bank.views[8])==2 and len(bank.views[19])==1
    np.testing.assert_array_equal(bank.views[19][0]['descriptor'],old)
    np.testing.assert_array_equal(bank.views[8][1]['source_ids'],[1])
    with pytest.raises(ValueError):bank.observe(2,[3],[99],[6],np.array([[1,0,0.]]),p[:1])
    assert bank.last_frame==1 and len(bank.views[8])==2


def test_cell_balancing_does_not_multiply_dense_surface_weight():
    bank=ProvenanceObjectMemory([1],2);p=np.eye(19,dtype=np.float32)[[2]*3]
    bank.observe(0,[1,2,3],[1,1,1],[0,0,1],np.array([[1,0],[1,0],[0,1.]]),p)
    np.testing.assert_allclose(bank.views[1][0]['descriptor'],[2**-.5,2**-.5],rtol=1e-6)


def test_input_separation_handles_early_prefixes_and_unknown():
    q=np.arange(8)[:,None];v=np.arange(8)[None]
    actual=disjoint_inputs(q,v)
    for i in range(8):
        for j in range(8):
            assert actual[i,j]==not_overlap(i,j)
    assert not disjoint_inputs(np.array([-1,0,3]),np.array([7,-1,6])).any()


def not_overlap(i,j):return not(set(range(max(0,i-3),i+1))&set(range(max(0,j-3),j+1)))


def test_allocator_is_object_permutation_equivariant_and_padding_abstains():
    torch.manual_seed(3);torch.set_num_threads(1);m=ViewEvidenceAllocator(4,5)
    x=torch.randn(7,3,4);v=torch.randn(7,3,6,5)
    valid=torch.ones(7,3,dtype=torch.bool);valid[:,-1]=False
    seen=valid[:,:,None].expand(-1,-1,6).clone();seen[:,0,0]=False
    order=torch.tensor([2,0,1])
    for mode in ['mean','retrieval','attention']:
        a=m(x,v,seen,valid,mode);b=m(x[:,order],v[:,order],seen[:,order],valid[:,order],mode)
        torch.testing.assert_close(a[:,order],b)
        assert torch.isneginf(a[:,-1]).all()
        a[valid].square().mean().backward()
        assert m.quality.weight.grad.abs().sum()>0
        m.zero_grad()


def test_revision_changes_history_only_as_named_and_append_control_preserves_it():
    xyz=np.arange(12).reshape(4,3).astype(float);source=xyz+100;ids=np.array([8,8,19,19])
    args=(xyz,ids,source,[1,2],[19,0],[0,3],[8,0],[8,19])
    a=revision_map(*args);b=revision_map(*args,revise=False)
    np.testing.assert_array_equal(a['instance'],[8,19,19,8])
    np.testing.assert_array_equal(a['xyz'],np.r_[xyz[[0,1,3]],source[[0]]])
    np.testing.assert_array_equal(b['xyz'][:4],xyz);np.testing.assert_array_equal(b['instance'][:4],ids)
    with pytest.raises(ValueError):revision_map(xyz,ids,source,[1],[99],[],[],[8,19])
