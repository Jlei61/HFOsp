import numpy as np
import pytest
from scripts.run_fig5_upper_runaway_map import classify_tail

GATE = {'tail_duration_ms':1000.,'maximum_absolute_half_tail_drift_hz':5.,
        'minimum_population_mean_rate_hz':300.,'minimum_each_regional_rate_hz':250.}


def test_tonic_plateau_does_not_require_oscillation():
    r=classify_tail(np.full(10000,320.),dict(core_a=430.,core_b=430.,surround=310.),dt_ms=.1,gate=GATE)
    assert r['state']=='tonic_runaway'


def test_population_threshold_alone_cannot_define_global_runaway():
    r=classify_tail(np.full(10000,320.),dict(core_a=430.,core_b=430.,surround=210.),dt_ms=.1,gate=GATE)
    assert r['state']=='unresolved'


def test_a_drifting_high_tail_remains_unresolved():
    r=classify_tail(np.linspace(300,350,10000),dict(core_a=430.,core_b=430.,surround=310.),dt_ms=.1,gate=GATE)
    assert r['state']=='unresolved'


def test_short_windows_and_nonfinite_samples_fail_closed():
    for trace in (np.zeros(100),np.full(10000,np.nan)):
        with pytest.raises(ValueError):
            classify_tail(trace,dict(a=0.),dt_ms=.1,gate=GATE)
