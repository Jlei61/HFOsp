import math
import numpy as np
import torch
from src.topic5_group_event_state.v039.frozen_transfer import exact_subset_nll,branching_strata,equal_anchor_mean


def test_subset_likelihood_normalises_over_unordered_size_k_sets():
    logits=torch.zeros((2,4),dtype=torch.float64,requires_grad=True)
    target=torch.tensor([[1.,1.,0.,0.],[1.,0.,0.,0.]],dtype=torch.float64)
    available=torch.tensor([[True]*4,[True,True,False,False]])
    loss=exact_subset_nll(logits,target,available)
    torch.testing.assert_close(loss,torch.tensor([math.log(6),math.log(2)],dtype=torch.float64))
    loss.sum().backward();assert logits.grad[1,2:].abs().sum()==0
    # A common logit offset must not change a cardinality-conditioned score.
    torch.testing.assert_close(loss,exact_subset_nll(logits+500,target,available))


def test_same_prefix_and_size_branch_rule_uses_fit_suffixes_only():
    a=[0,1,2,-1];b=[0,1,-1,2];different_size=[0,1,2,2]
    ranks=np.array([a,b,a,b,a,a,different_size]);phases=np.array(['FIT']*5+['SELECTION']*2)
    mask,keys,_=branching_strata(ranks,phases)
    assert len(keys)==1 and mask[5] and not mask[6]
    # Held-out diversity cannot invent a branching stratum absent from FIT.
    ranks[:5]=a
    assert not branching_strata(ranks,phases)[0].any()


def test_equal_anchor_scores_do_not_weight_dense_event_anchors_more():
    values=np.array([0.,0.,0.,4.]);anchor=np.array([1,1,1,2])
    np.testing.assert_array_equal(equal_anchor_mean(values,anchor),[0.,4.])
