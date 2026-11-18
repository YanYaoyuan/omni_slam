import numpy as np
import pytest
import torch
from omni_semantic_map.memory_resegmentation import CausalTokenMemory,LocalEvidenceReader,partition_candidates


def test_future_memory_rejected_and_radius_missing():
    m=CausalTokenMemory(neighbors=2)
    m.add(np.zeros((1,3)),np.ones((1,384)),3,np.array([0]))
    with pytest.raises(AssertionError):m.query(np.zeros((1,3)),3)
    _,valid,ids=m.query(np.array([[0,0,0],[5,5,5]]),4)
    assert valid.sum()==1 and ids[0,0]==0 and not valid[1].any()


def test_partition_conserves_support_without_overlap():
    original=np.eye(5,dtype=bool)
    parts=partition_candidates(original,np.ones((2,5,5),bool),[.4,.7])
    assert np.array_equal(np.sum(parts,axis=0),original.astype(int))


def test_empty_history_finite_and_attention_trains():
    torch.manual_seed(3)
    m=LocalEvidenceReader()
    x=torch.randn(3,384);h=torch.randn(3,8,384);v=torch.ones(3,8,dtype=torch.bool);v[0]=False
    z=m.encode(x,h,v)
    assert torch.isfinite(z).all()
    m.affinity(z,z.roll(1,0),torch.zeros(3),torch.ones(3)).sum().backward()
    assert m.attention.in_proj_weight.grad.abs().sum()>0
