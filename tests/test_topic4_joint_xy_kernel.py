import numpy as np
import pytest
from src.topic4_joint_xy_kernel import event_kernel_features,fit_kernel_maps,kernel_map,mapped_distance

XY=np.array([[2,4],[5,4],[8,4],[11,4],[3,12],[9,12]],float)
GROUPS={'ICL':[0,1,2,3],'SCL':[4,5]}


def test_time_stretch_changes_timing_but_preserves_rank_field():
    t=np.array([[0.,10,20,30,40,50],[50,40,30,20,10,0.]])
    a=event_kernel_features(t,XY,GROUPS,20.)
    b=event_kernel_features(t*1.5,XY,GROUPS,20.)
    np.testing.assert_allclose(a['rank_space'],b['rank_space'])
    np.testing.assert_allclose(a['support'],b['support'])
    assert not np.allclose(a['timing_space'],b['timing_space'])


def test_common_time_origin_is_irrelevant_for_every_kernel():
    t=np.array([[0.,10,np.nan,30,40,50],[50,40,30,20,10,0.]])
    a=event_kernel_features(t,XY,GROUPS,20.);b=event_kernel_features(t+1000.,XY,GROUPS,20.)
    for k in a: np.testing.assert_allclose(a[k],b[k])


def test_missing_events_remain_finite_and_are_not_empty_matches():
    f=event_kernel_features(np.full((2,6),np.nan),XY,GROUPS,20.)
    for x in f.values(): assert np.isfinite(x).all()
    assert mapped_distance(np.zeros((0,16)),np.zeros(16)) is None


def test_kernel_distinguishes_mode_mass_without_mode_labels():
    a=np.arange(6)*10.;b=a[::-1]
    t=np.array([a]*24+[b]*8);alter=np.array([a]*8+[b]*24)
    f=event_kernel_features(t,XY,GROUPS,20.);maps=fit_kernel_maps(f,seed=10,n_fourier=256)
    for k in ('rank_space','timing_space','joint'):
        x=kernel_map(f[k],maps[k]);reference=x.mean(axis=0,dtype=float)
        y=kernel_map(event_kernel_features(alter,XY,GROUPS,20.)[k],maps[k])
        assert mapped_distance(x,reference)<1e-12
        assert mapped_distance(y,reference)>.01


def test_maps_reproducible_and_nonpositive_time_scale_rejected():
    t=np.array([np.arange(6),np.arange(6)[::-1]],float)
    f=event_kernel_features(t,XY,GROUPS,20.)
    a=fit_kernel_maps(f,seed=5,n_fourier=64);b=fit_kernel_maps(f,seed=5,n_fourier=64)
    np.testing.assert_array_equal(a['joint']['weights'],b['joint']['weights'])
    with pytest.raises(ValueError):event_kernel_features(t,XY,GROUPS,0.)
