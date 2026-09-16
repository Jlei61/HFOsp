import numpy as np
from scripts.run_preseizure_broadband_cohort import spectral_diagnostics, frozen_event_indices, run_associations


def _signal(kind,pre=121.):
    fs=512.;t=np.arange(int((pre+11)*fs))/fs-pre;rng=np.random.default_rng(7)
    x=rng.normal(size=(6,len(t)));early=(t>=0)&(t<10)
    if kind=='broad':x[:,early]*=3
    if kind=='narrow':x[:,early]+=12*np.sin(2*np.pi*20*t[early])[None,:]
    return spectral_diagnostics(x,fs,pre,(-120.,-90.))


def test_broad_enhancement_has_multiband_support():
    d=_signal('broad')
    assert sum(d['band_hits'])>=5
    assert np.all(d['band_db'].mean(axis=0)>6)


def test_large_total_power_does_not_make_narrowband_broadband():
    d=_signal('narrow')
    assert np.mean(d['activation'])>0
    assert sum(d['band_hits'])<5
    assert sum((d['band_db'].mean(axis=0)>3))==1


def test_fractional_spectral_grid_still_has_complete_early_window():
    d=_signal('broad',pre=121.25)
    assert d['n_early_frames']==18
    assert sum(d['band_hits'])>=5


def test_all_event_mapping_retains_missing_spatial_view():
    # Four raw events retain four timing labels, even if only two events
    # would survive the older >=3-participating-contact direction filter.
    record={'template_discovery':{'method':'timing_plus_space_all_events_missing_view_v1',
        'sampled_event_indices':[0,1,2,3],'event_labels':[0,1,0,1]}}
    np.testing.assert_array_equal(frozen_event_indices(record,4),[0,1,2,3])


def test_failed_broadband_cannot_supply_association_labels(tmp_path):
    results=[{'sz':i,'provisional_source_label':'TA','qualified_source_label':''} for i in [1,2]]
    result=run_associations({},[],results,tmp_path)
    assert result['status']=='NO_QUALIFIED_SOURCE_INTERVALS'
