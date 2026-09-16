import numpy as np
from scripts.explore_e1146_seizure_interictal_association import exact_rank_test, blocked_rank_test, repeat_excess, bh_adjust


def test_exact_rank_test_ties_and_label_swap():
    r=exact_rank_test([0,0,1,1],[False,False,True,True])
    assert r['auc_tb']==1 and r['rank_biserial']==1
    assert np.isclose(r['p'],1/3)
    rev=exact_rank_test([0,0,1,1],[True,True,False,False])
    assert rev['rank_biserial']==-1 and rev['p']==r['p']


def test_temporal_confound_is_not_tested_as_global_rank_shift():
    # Perfect separation is entirely between pure-label time strata.
    r=blocked_rank_test([0,1,10,11],[False,False,True,True],[0,1,10,11],6)
    assert r['p']==1 and r['n_permutations']==1
    assert r['rank_sum_residual']==0 and r['n_mixed_time_blocks']==0


def test_exact_blocked_test_mixed_strata():
    r=blocked_rank_test([0,2,1,3],[False,True,False,True],[0,1,6,7],6)
    assert r['n_permutations']==4 and r['n_mixed_time_blocks']==2
    assert r['p']==.5


def test_sequence_excess_respects_gaps_and_composition():
    x=np.array([0]*10+[1]*10)
    assert repeat_excess(x,np.zeros(20))>0
    assert repeat_excess(x,np.repeat([0,1],10))==0
    assert repeat_excess(np.tile([0,1],10),np.zeros(20))<0


def test_bh_preserves_whole_family_size():
    np.testing.assert_allclose(bh_adjust([.01,.04,.03]),[.03,.04,.04])


def test_large_permutation_space_is_bounded_and_never_zero_p():
    x=np.arange(24);labels=x>=12
    a=exact_rank_test(x,labels)
    b=blocked_rank_test(x,labels,np.zeros(24),6)
    for result in (a,b):
        assert not result['permutation_exact']
        assert result['n_permutations']==19999
        assert 0<result['p']<.001
