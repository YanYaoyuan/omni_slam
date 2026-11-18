import numpy as np
import torch
from omni_semantic_map.spatial_acceptance_readout import build_readout, expected_labels
from omni_semantic_map.map_responsibility_supervision import (
    CompiledReadout, select_queries, categorical, source_labels, marginal_nll, normalized_weights)


def sample():
    a = build_readout([[0,0,0]], np.array([1]), [[.1,0,0],[.15,0,0]],
                      [[0,0,0],[.09,0,0],[.14,0,0],[2,0,0]], .2)
    ids = np.array([[1,2],[2,3]])
    return a, ids


def test_compiled_values_gradients_and_arbitrary_query_selection():
    a, ids = sample(); p = torch.tensor([[.3,.2],[.1,.6]], dtype=torch.double, requires_grad=True)
    for rows in [np.arange(4),np.array([3,1,0]),np.array([],dtype=int)]:
        sub = select_queries(a, rows); compiled = CompiledReadout(sub, ids, 4)
        want = expected_labels(sub, p, ids, 4); got = compiled(p)
        torch.testing.assert_close(got, want, rtol=0, atol=0)
        assert torch.autograd.gradcheck(compiled, (p,))


def test_chunk_chain_rule_matches_monolithic_gradient():
    a, ids = sample(); z = torch.tensor([[.3,.8],[-.2,.4]], dtype=torch.double, requires_grad=True)
    valid = torch.ones_like(z,dtype=torch.bool); target = torch.tensor([1,1,2,0]); weight=torch.tensor([.1,.2,.3,.4],dtype=torch.double)
    p=categorical(z,valid); loss=marginal_nll(CompiledReadout(a,ids,4)(p),target,weight); expected=torch.autograd.grad(loss,z)[0]
    p=categorical(z,valid); leaf=p.detach().requires_grad_(True); value=0.
    for ii in [np.array([0,2]),np.array([1,3])]:
        part=marginal_nll(CompiledReadout(select_queries(a,ii),ids,4)(leaf),target[ii],weight[ii]);part.backward();value+=float(part.detach())
    p.backward(leaf.grad)
    torch.testing.assert_close(z.grad,expected,rtol=1e-12,atol=1e-12)
    assert abs(value-float(loss))<1e-12


def test_unmatched_object_and_background_have_nonzero_penalty():
    # Owner 3 has no matched GT: accepting it at a known background location
    # is penalized; an IoU-only empty target would have zero derivative.
    p=torch.tensor([[.2,.4]],dtype=torch.double,requires_grad=True)
    dist=source_labels(p,torch.tensor([[1,3]]),4)
    loss=marginal_nll(dist,torch.tensor([0]),torch.ones(1,dtype=torch.double));loss.backward()
    assert torch.all(p.grad>0)
    torch.testing.assert_close(dist.sum(1),torch.ones(1,dtype=torch.double))


def test_balancing_and_ignored_reference_mass():
    target=np.array([0,0,1,2]);w=np.array([1.,3.,2.,0.]);balanced=normalized_weights(target,w,True)
    np.testing.assert_allclose(balanced,[.125,.375,.5,0])
    np.testing.assert_allclose(normalized_weights(target,w,False),[1/6,3/6,2/6,0])
    p=torch.tensor([[1.,0.,0.],[0.,1.,0.]],requires_grad=True)
    marginal_nll(p,torch.tensor([0,2]),torch.tensor([1.,0.])).backward()
    assert torch.all(p.grad[1]==0)


def test_competing_candidates_and_padding():
    z=torch.tensor([[1.,2.,9.]],requires_grad=True);valid=torch.tensor([[True,True,False]])
    p=categorical(z,valid);dist=source_labels(p,torch.tensor([[1,2,0]]),3)
    assert p[0,2]==0 and p.sum()<1
    torch.testing.assert_close(dist.sum(1),torch.ones(1))


def test_unreachable_target_does_not_reward_diffuse_wrong_labels():
    p=torch.tensor([[.2,.4]],dtype=torch.double,requires_grad=True)
    dist=source_labels(p,torch.tensor([[1,3]]),4)
    marginal_nll(dist,torch.tensor([2]),torch.ones(1,dtype=torch.double)).backward()
    torch.testing.assert_close(p.grad,torch.zeros_like(p))
