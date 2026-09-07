import numpy as np
from src.topic4_observation_repaired import observe,calibrate_reference,order_error


def contract():
    return dict(contact_names=[str(i) for i in range(15)],dt_ms=2.,baseline=[0.]*15,
        threshold=[.2]*15,minimum_detection_ms=4.,extension_ms=30.,
        window_ms=250.,channel_fraction=.5,burnin_ms=500.)


def wave():
    times=np.arange(2000)*2
    return np.array([np.exp(-.5*((times-1000-4*i)/10)**2) for i in range(15)])


def test_two_contact_flutter_cannot_become_population_event():
    e=np.zeros((15,4000))
    for c in range(2,15):e[c,2000+100*c:2002+100*c]=1
    for s in range(500,551,5):e[:2,s:s+2]=1
    assert observe(e,2.,contract())['n_groups']==0


def test_distant_future_peak_preserves_earlier_events_and_times():
    a=wave();b=a.copy();b[0,-10:]=1000
    x,y=observe(a,2.,contract()),observe(b,2.,contract())
    assert x['windows_ms']==y['windows_ms']
    np.testing.assert_array_equal(x['centroid_ms'],y['centroid_ms'])
    np.testing.assert_array_equal(x['recruitment_ms'],y['recruitment_ms'])


def test_same_prefix_invariant_away_from_boundary():
    e=wave();a=observe(e,2.,contract());b=observe(e[:,:1000],2.,contract())
    assert a['windows_ms']==b['windows_ms']
    np.testing.assert_array_equal(a['centroid_ms'],b['centroid_ms'])


def test_known_wave_recovers_order_and_lags():
    r=observe(wave(),2.,contract())
    assert r['n_groups']==1
    np.testing.assert_allclose(np.diff(r['centroid_ms'][0]),4,atol=.05)
    np.testing.assert_allclose(np.diff(r['recruitment_ms'][0]),4,atol=.05)
    assert r['events'][0]['n_unique_contacts']==15


def test_three_states_distinguish_tie_from_reverse():
    assert order_error([-10,-10],[0,0])==1.
    assert order_error([10,-10],[10,-10])==0.


def test_nonvarying_sustained_signal_is_flagged_not_fake_onsets():
    e=np.ones((15,2000))
    r=observe(e,2.,contract())
    assert r['events'][0]['prolonged']
    assert np.isnan(r['centroid_ms']).all()
    assert r['n_primary_events']==0


def test_overlapping_events_are_audited_but_not_double_counted_in_patient_fit():
    e=wave()+np.roll(wave(),90,axis=1)
    r=observe(e,2.,contract())
    assert r['n_groups']==2
    assert r['n_primary_events']==0
    assert all(x['overlap_with_other_windows'] for x in r['events'])


def test_constant_background_correction_preserves_centroid_lags():
    # Keep the absolute crossing safely above background, with a fixed observer.
    e=wave();a=observe(e,2.,contract());b=observe(e+.05,2.,contract())
    np.testing.assert_allclose(np.diff(a['centroid_ms'][0]),np.diff(b['centroid_ms'][0]),atol=.02)
