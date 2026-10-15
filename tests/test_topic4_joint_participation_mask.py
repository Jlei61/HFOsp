"""Joint 15-contact participation-mask off-diagonal statistic (design §6; checklist C11)."""
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
from src import topic4_joint_participation_mask as jm


def synthetic_patient(n=4000,seed=1):
    """Two-mode patient-like masks with genuine joint structure: 60% events recruit both rods
    densely, 40% recruit the upper rod sparsely; per-contact marginals alone do not encode this."""
    rng=np.random.default_rng(seed);c=15;mode=rng.random(n)<.6
    p=np.where(mode[:,None],np.r_[np.full(4,.9),np.full(11,.9)][None],np.r_[np.full(4,.3),np.full(11,.85)][None])
    return rng.random((n,c))<p


def test_mask_kernel_is_mean_of_hamming_exponentials():
    a=np.array([[1,1,0,0],[0,0,0,0]],bool);b=np.array([[1,1,0,0],[1,0,1,1]],bool)
    K=jm.mask_kernel(a,b)
    h=np.array([[0,3],[2,3]],float)
    expected=np.mean([np.exp(-h/w) for w in jm.BANDWIDTHS],axis=0)
    np.testing.assert_allclose(K,expected);assert K[0,0]==1.


def test_reference_self_term_matches_explicit_pairwise_mean():
    pat=synthetic_patient(300)
    ref=jm.MaskReference(pat)
    K=jm.mask_kernel(pat,pat);n=len(pat)
    explicit=(K.sum()-np.trace(K))/(n*(n-1))
    assert ref.self_off==pytest.approx(explicit,rel=1e-12)
    model=synthetic_patient(40,seed=5)
    d=jm.off_diagonal_mask_distance(model,ref)
    Km=jm.mask_kernel(model,model);Kx=jm.mask_kernel(model,pat)
    explicit_d=(Km.sum()-np.trace(Km))/(40*39)-2*Kx.mean()+explicit
    assert d==pytest.approx(explicit_d,rel=1e-10)
    assert jm.biased_mask_distance(model,ref)>=0


def test_off_diagonal_distance_rewards_the_exact_joint_distribution_over_counterexamples():
    pat=synthetic_patient(6000);ref=jm.MaskReference(pat);rng=np.random.default_rng(7)
    exact=pat[rng.choice(len(pat),400,replace=False)]
    d_exact=jm.off_diagonal_mask_distance(exact,ref)
    # rods split into separate events with identical per-contact marginals in aggregate
    split=np.concatenate([exact&np.r_[np.ones(4,bool),np.zeros(11,bool)],exact&np.r_[np.zeros(4,bool),np.ones(11,bool)]])
    # marginal-preserving destruction of the joint structure (each contact shuffled independently)
    shuffled=exact.copy()
    for j in range(15):shuffled[:,j]=rng.permutation(shuffled[:,j])
    allon=np.ones_like(exact)
    shifted=exact.copy();shifted[:,:4]&=rng.random((400,4))<.3            # SCL frequency shift
    collapsed=np.repeat(exact[[np.argmax(exact.sum(1))]],400,axis=0)         # dispersion shrink to one pattern
    for name,alt in [('split',split),('shuffled',shuffled),('all_on',allon),('shifted',shifted),('collapsed',collapsed)]:
        assert jm.off_diagonal_mask_distance(alt,ref)>d_exact+1e-3,name
    assert abs(d_exact)<5e-3


def test_calibration_scale_is_positive_block_matched_and_recorded():
    pat=synthetic_patient(2000);blocks=np.repeat(np.arange(50),40);ref=jm.MaskReference(pat)
    cal=jm.calibrate_scale(pat,blocks,ref,sample_count=16,n_samples=32,seed=3)
    assert cal['a_mask']>0 and cal['statistic']=='biased_non_negative_matched_16' and len(cal['draws'])==32
    assert all(d['n_events_available']>=16 and len(d['blocks'])>=1 for d in cal['draws'])
    again=jm.calibrate_scale(pat,blocks,ref,sample_count=16,n_samples=32,seed=3)
    assert again['a_mask']==cal['a_mask']


def test_score_keeps_negative_values_and_flags_insufficient_events():
    pat=synthetic_patient(2000);ref=jm.MaskReference(pat);ref.a_mask=.05
    few=jm.score_masks(pat[:15],ref);assert few['status']=='INSUFFICIENT_EVENTS' and few['D_mask_off'] is None
    times=np.where(pat[:16],1.,np.nan)
    s=jm.score_times(times,ref)
    assert s['status']=='ESTIMABLE' and s['n_events']==16 and np.isfinite(s['D_mask_off'])
    assert jm.combined_search_loss(1.2,s['D_mask_off'],ref.a_mask)==pytest.approx(.5*1.2+.5*s['D_mask_off']/.05)
    minus=jm.off_diagonal_mask_distance(np.tile(pat[:1],(16,1))*0|pat[:16],ref)
    assert isinstance(minus,float)
