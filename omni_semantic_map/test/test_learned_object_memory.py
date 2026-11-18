"""Run with the project's torch environment: python -m unittest discover ..."""
import unittest
import torch
from omni_semantic_map.learned_object_memory import ObjectMemoryRanker, positive_mass_loss


class LearnedMemoryTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(7)
        self.q=torch.randn(3,8); self.h=torch.randn(3,5,8)
        self.mask=torch.tensor([[1,1,0,0,0],[1,1,1,1,1],[1,0,0,0,0]],dtype=torch.bool)
        self.native=torch.tensor([1.2,.9,.1])

    def model(self,mode='attention'):
        return ObjectMemoryRanker(mode,input_dim=8,width=16,heads=4)

    def test_zero_initialization_and_paired_common_weights(self):
        torch.manual_seed(3);a=self.model()
        torch.manual_seed(3);b=self.model('mean')
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k])
        for model in [a,b]:torch.testing.assert_close(model(self.q,self.h,self.mask,self.native),self.native)

    def test_padding_and_memory_permutation(self):
        m=self.model();torch.nn.init.normal_(m.score[-1].weight)
        before,w=m(self.q,self.h,self.mask,self.native,True)
        self.assertTrue((w.masked_select(~self.mask[:,None,:].expand_as(w))==0).all())
        self.h[~self.mask]=10000
        torch.testing.assert_close(before,m(self.q,self.h,self.mask,self.native))
        idx=torch.tensor([4,2,1,3,0])
        torch.testing.assert_close(before,m(self.q,self.h[:,idx],self.mask[:,idx],self.native))

    def test_attention_projections_receive_gradients(self):
        m=self.model(); opt=torch.optim.AdamW(m.parameters(),lr=.01)
        labels=torch.tensor([False,True,False])
        for _ in range(3):
            opt.zero_grad();loss=positive_mass_loss(m(self.q,self.h,self.mask,self.native),[0,3],labels).mean();loss.backward();opt.step()
        self.assertGreater(float(m.query.weight.grad.abs().sum()),0)
        self.assertGreater(float(m.key.weight.grad.abs().sum()),0)

    def test_synthetic_training_learns_and_checkpoint_reproduces(self):
        m=self.model();opt=torch.optim.AdamW(m.parameters(),lr=.01)
        labels=torch.tensor([False,True,False])
        initial=float(positive_mass_loss(m(self.q,self.h,self.mask,self.native),[0,3],labels).detach())
        for _ in range(40):
            opt.zero_grad();loss=positive_mass_loss(m(self.q,self.h,self.mask,self.native),[0,3],labels).mean();loss.backward();opt.step()
        scores=m(self.q,self.h,self.mask,self.native)
        self.assertEqual(int(scores.argmax()),1)
        self.assertLess(float(loss.detach()),initial*.1)
        restored=self.model();restored.load_state_dict(m.state_dict())
        torch.testing.assert_close(scores,restored(self.q,self.h,self.mask,self.native))

    def test_multi_positive_and_invalid_memory(self):
        scores=torch.tensor([0.,0.,0.])
        loss=positive_mass_loss(scores,[0,3],torch.tensor([True,True,False]))
        self.assertAlmostEqual(float(loss),float(torch.log(torch.tensor(1.5))),places=6)
        with self.assertRaises(ValueError):self.model()(self.q,self.h,torch.zeros_like(self.mask),self.native)


if __name__=='__main__':unittest.main()
