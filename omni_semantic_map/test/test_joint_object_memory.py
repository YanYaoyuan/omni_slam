import numpy as np
import pytest
import torch
from omni_semantic_map.joint_object_memory import JointObjectMemory,aggregate_source_logits,SourceOwnership,source_instance_loss


def test_new_frame_changes_historical_output_and_gradients_reach_both():
    torch.manual_seed(17);torch.set_num_threads(1)
    m=JointObjectMemory(input_dim=8,width=16,heads=2,layers=1,max_objects=3)
    x=torch.randn(8,8,requires_grad=True);frames=torch.tensor([0]*4+[1]*4)
    relations=torch.zeros(8,8,dtype=torch.int8)
    a=m(x,frames,relations);b=m(torch.cat([x[:4],x[4:]+1]),frames,relations)
    assert not torch.allclose(a['logits'][:4],b['logits'][:4])
    a['logits'][:4].square().mean().backward()
    assert x.grad[:4].abs().sum()>0 and x.grad[4:].abs().sum()>0
    assert a['logits'].shape==(8,4)


def test_row_permutation_equivariance_and_unknown_geometry():
    torch.manual_seed(17)
    m=JointObjectMemory(8,16,2,1,3).eval()
    x=torch.randn(8,8);frames=torch.tensor([0]*4+[1]*4);r=torch.zeros(8,8,dtype=torch.int8)
    with torch.no_grad():m.relation_bias[:]=torch.tensor([[-2.,2.]])
    p=torch.tensor([7,1,3,2,6,4,0,5]);a=m(x,frames,r)
    b=m(x[p],frames[p],r[p][:,p]);torch.testing.assert_close(a['logits'][p],b['logits'],atol=1e-5,rtol=1e-5)
    c=m(x,frames,r,use_geometry=False);torch.testing.assert_close(a['logits'],c['logits'])
    r[0,5]=r[5,0]=1;d=m(x,frames,r)
    assert not torch.allclose(a['logits'],d['logits'])


def test_duplicate_same_frame_cannot_multiply_evidence():
    logits=torch.tensor([[2.,0.],[2.,0.],[0.,4.],[1.,1.]],requires_grad=True)
    ids,mean=aggregate_source_logits(logits,torch.tensor([7,7,7,8]),torch.tensor([0,0,1,1]))
    torch.testing.assert_close(ids,torch.tensor([7,8]))
    torch.testing.assert_close(mean,torch.tensor([[1.,2.],[1.,1.]]))
    mean.sum().backward();assert torch.isfinite(logits.grad).all()


def test_historical_revision_rolls_back_exactly_and_invalid_commit_is_atomic():
    ledger=SourceOwnership([1,2,3],[8,8,9]);old=ledger.owners.copy()
    t=ledger.commit([1,2],[8,10]);assert t['changed']==1 and ledger.owners[2]==10
    with pytest.raises(ValueError):ledger.commit([1,999],[2,2])
    assert ledger.owners=={1:8,2:10,3:9}
    ledger.rollback(t);assert ledger.owners==old and ledger.version==0


def test_persistent_matching_and_occlusion_are_not_false_negatives():
    logits=torch.tensor([[0.,4.,0.,0.],[0.,0.,5.,0.],[2.,0.,0.,0.],[0.,0.,0.,8.]],requires_grad=True)
    target=torch.tensor([18,29,0,-1]);exist=torch.zeros(3,requires_grad=True)
    result=source_instance_loss(logits,target,exist,established={18:1,99:3})
    assert result['binding']=={18:1,99:3,29:2}
    result['loss'].backward()
    assert logits.grad[3].abs().sum()==0 and exist.grad[2]==0
    with pytest.raises(ValueError):source_instance_loss(logits,torch.tensor([18,29,30,31]),exist)
