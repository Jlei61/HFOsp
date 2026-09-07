import numpy as np
from src.topic4_xy_timing_observation_audit import reweight_centroids,paired_order_change


def test_reweighting_keeps_window_participation_and_can_reverse_multi_peak_order():
    env=np.zeros((3,10));env[0,1]=2;env[0,8]=1;env[1,2]=1
    mask=np.array([[True,True,False]])
    a=reweight_centroids(env,1,[[0,10]],mask,power=1)
    b=reweight_centroids(env,1,[[0,10]],mask,power=3)
    np.testing.assert_array_equal(np.isfinite(a),mask)
    np.testing.assert_array_equal(np.isfinite(b),mask)
    assert a[0,0]>a[0,1] and b[0,0]<b[0,1]
    assert paired_order_change(a,b)['strict_order_reversal_fraction']==1


def test_background_centroid_compression_and_recovery():
    env=np.ones((2,20));env[0,3]+=4;env[1,14]+=4
    m=np.ones((1,2),bool)
    raw=reweight_centroids(env,1,[[0,20]],m)
    sub=reweight_centroids(env,1,[[0,20]],m,baseline=np.ones(2))
    assert raw[0,1]-raw[0,0]<11
    np.testing.assert_allclose(sub,[[3,14]])
