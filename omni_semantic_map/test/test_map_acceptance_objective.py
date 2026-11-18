import itertools
import numpy as np
import torch
from omni_semantic_map.spatial_acceptance_readout import hard_readout,expected_labels
from omni_semantic_map.map_acceptance_objective import compile_moments,moments,object_loss


def fixture():
    a=dict(offsets=np.array([0,2,3,3,5]),sources=np.array([0,1,1,2,0]),
           fallback=np.array([1,2,0,2]),candidate_count=3)
    truth=np.array([1001,2001,1001,1001]);w=np.array([1.,.4,2.,.7]);target=np.array([-1,1001,2001])
    owners=np.array([[1,2],[1,2],[1,2]])
    return a,truth,w,target,owners


def test_exact_joint_enumeration_and_mass_conservation():
    a,t,w,target,owners=fixture();p=torch.tensor([[.2,.3],[.4,.1],[.1,.7]],dtype=torch.float64)
    c=compile_moments(a,t,w,target);m,i=moments(c,p,owners)
    wantm=np.zeros(3);wanti=np.zeros(3)
    for assignment in itertools.product(range(3),repeat=3):
        probability=np.prod([1-float(p[j].sum()) if s==0 else float(p[j,s-1]) for j,s in enumerate(assignment)])
        choices=np.array([0 if s==0 else owners[j,s-1] for j,s in enumerate(assignment)])
        labels=hard_readout(a,choices)
        wantm+=probability*np.bincount(labels,weights=w,minlength=3)
        wanti+=probability*np.bincount(labels,weights=w*(t==target[labels]),minlength=3)
    np.testing.assert_allclose(m,wantm,atol=1e-12);np.testing.assert_allclose(i,wanti,atol=1e-12)
    assert abs(float(m.sum())-w.sum())<1e-12
    dense=expected_labels(a,p,owners,3)
    torch.testing.assert_close(m,(dense*torch.tensor(w)[:,None]).sum(0))


def test_hard_endpoints_and_missing_reference_abstention():
    a,t,w,target,owners=fixture();w[1]=0
    c=compile_moments(a,t,w,target)
    for assignment in itertools.product(range(3),repeat=3):
        p=torch.zeros((3,2),dtype=torch.float64);choices=np.zeros(3,int)
        for j,s in enumerate(assignment):
            if s:p[j,s-1]=1;choices[j]=owners[j,s-1]
        m,i=moments(c,p,owners);labels=hard_readout(a,choices)
        np.testing.assert_allclose(m,np.bincount(labels,weights=w,minlength=3),atol=1e-12)
        np.testing.assert_allclose(i,np.bincount(labels,weights=w*(t==target[labels]),minlength=3),atol=1e-12)


def test_mean_neighbor_control_does_not_claim_competition():
    a,t,w,target,owners=fixture();p=torch.tensor([[1.,0],[0,1.],[0,0]],dtype=torch.float64)
    c=compile_moments(a,t,w,target);ordered,_=moments(c,p,owners);mean,_=moments(c,p,owners,'mean_neighbor')
    assert not torch.allclose(mean,ordered)
    assert abs(float(mean.sum())-w.sum())<1e-12


def test_object_loss_gradient_and_unmatched_penalty():
    a,t,w,target,owners=fixture();c=compile_moments(a,t,w,target)
    logits=torch.tensor([[.2,.3],[.4,-.2],[-.7,.4]],dtype=torch.float64,requires_grad=True)
    def loss(z):
        p=torch.cat([torch.zeros((3,1),dtype=z.dtype),z],1).softmax(1)[:,1:]
        m,i=moments(c,p,owners)
        return object_loss(m,i,[1,2],np.array([0,3.7,0]),np.array([1,1,2]))
    assert torch.autograd.gradcheck(loss,(logits,),eps=1e-6,atol=1e-5)
    m=torch.tensor([0.,1.,2.]);i=torch.tensor([0.,1.,0.])
    first=object_loss(m,i,[1,2],[0,2,0],[1,1,2])
    m[2]=3
    assert object_loss(m,i,[1,2],[0,2,0],[1,1,2])>first
