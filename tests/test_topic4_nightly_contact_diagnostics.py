import numpy as np
from scripts.analyze_topic4_nightly_contacts import features,summarize,covariance

def test_missing_contacts_do_not_acquire_rank_or_duration():
    tab=dict(participation=np.array([[1,1],[1,0],[0,1]]),centroid=np.array([[10.,20.],[30.,np.nan],[np.nan,50.]]),
             recruitment=np.zeros((3,2,1)),local_shape=np.tile(np.arange(19.),(3,2,1)))
    f=features(tab)
    np.testing.assert_array_equal(f['rank'][0],[0,1])
    assert np.isnan(f['rank'][1:, :]).all()
    assert np.isnan(f['local_width_ms'][1,1])
    assert np.isnan(f['local_width_ms'][2,0])
    assert summarize(f['participation'][:,0],np.ones(3))['mean']==2/3
    assert summarize(f['local_width_ms'][:,0],np.ones(3))['n']==2

def test_covariance_uses_pair_support_and_not_imputed_zero():
    x=np.array([[1.,2.],[3.,np.nan],[np.nan,6.]])
    c,n=covariance(x,np.ones(3))
    assert n[0,1]==1 and np.isnan(c[0,1])
    assert n[0,0]==2 and c[0,0]==1
    assert n[1,1]==2 and c[1,1]==4
    assert summarize(np.array([2.,np.nan,4.]),np.array([1.,2.,1.]))['median']==3.
