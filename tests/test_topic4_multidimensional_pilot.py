import numpy as np
import pytest
from scipy.sparse import csc_matrix
from src.topic4_multidimensional_parameters import scale_target_pathways, sparse_digest
from src.topic4_interictal_pilot_evaluation import rank_features, validate


def test_pathways_use_postsynaptic_rows_and_preserve_source_graph():
    m=csc_matrix(np.array([[1.,2.],[3.,0.],[4.,5.]]))
    old=m.copy()
    result,audit=scale_target_pathways([m],2,2.,.5)
    np.testing.assert_array_equal(result[0].toarray(),[[2.,4.],[6.,0.],[2.,2.5]])
    np.testing.assert_array_equal(m.toarray(),old.toarray())
    assert sparse_digest([m],topology=True)==sparse_digest(result,topology=True)
    assert audit['after_sum']==[12.,4.5]


def test_parameter_noop_bit_exact():
    m=csc_matrix([[1.,2.],[0.,3.]])
    out,_=scale_target_pathways([m],1,1.,1.)
    assert out[0] is m
    assert sparse_digest(out)==sparse_digest([m])


@pytest.mark.parametrize('bad',[0.,-1.,np.nan,np.inf])
def test_invalid_pathway_scale_rejected(bad):
    with pytest.raises(ValueError):scale_target_pathways([csc_matrix([[1.]])],1,bad,1.)


def test_masked_rank_does_not_encode_missing_as_last():
    x=rank_features(np.array([[3.,np.nan,1.],[1.,2.,3.]]))
    np.testing.assert_allclose(x,[[1.,.5,0.],[0.,.5,1.]])


def test_time_translation_invariant_and_infinity_rejected():
    x=np.array([[0.,3.,np.nan]])
    np.testing.assert_array_equal(rank_features(x),rank_features(x+17000))
    with pytest.raises(ValueError):validate([[np.inf,0.]])


def test_actual_substrate_gaba_tau_and_pathways_are_separate():
    from types import SimpleNamespace
    from src.topic4_multidimensional_parameters import apply_parameters
    a=csc_matrix([[1.,2.],[3.,0.],[4.,5.]])
    g=csc_matrix([[2.],[4.],[6.]])
    s=SimpleNamespace(n_e=2,params=SimpleNamespace(tau_r_GABA=1.,tau_d_GABA=18.),
                      net={'ampa_by_delay':[a],'gaba_by_delay':[g]})
    audit=apply_parameters(s,{'I_to_E_weight_scale':1.5,'tau_d_GABA_ms':12.})
    assert s.params.tau_d_GABA==12.
    np.testing.assert_array_equal(s.net['gaba_by_delay'][0].toarray(),[[3.],[6.],[6.]])
    np.testing.assert_array_equal(s.net['ampa_by_delay'][0].toarray(),a.toarray())
    np.testing.assert_array_equal(g.toarray(),[[2.],[4.],[6.]])
    assert audit['effective']['E_to_I_weight_scale']==1.
    with pytest.raises(ValueError):apply_parameters(s,{'misspelled_tau':12.})
