import importlib.util
from pathlib import Path
import numpy as np
import sys

spec=importlib.util.spec_from_file_location('burst_metrics',Path(__file__).parents[1]/'scripts/topic4_burst_regime/metrics.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/topic4_burst_regime'))
import metrics_v2

def trace(onsets,duration=30.):
    f=np.zeros(round(duration*100))
    for onset in onsets:
        k=round(onset*100);f[k:k+2]=.25
    return f

def analyze(f):
    return m.summarize(f,np.repeat(f,5)*100/5,100,burnin_s=0.)

def test_regular_and_irregular_with_same_order_of_rate():
    regular=analyze(trace(np.arange(.5,30,.5)))
    rng=np.random.default_rng(58);times=np.cumsum(.10+rng.exponential(.4,100));times=times[times<29.5]
    irregular=analyze(trace(times))
    assert regular['label']=='regular_bursts'
    assert irregular['label']=='irregular_bursts'
    assert irregular['cv2']>regular['cv2']

def test_quiet_sparse_and_tonic_do_not_become_irregular():
    assert analyze(np.zeros(3000))['label']=='background'
    assert analyze(trace([1.,5.,20.]))['label']=='sparse_bursts'
    assert analyze(np.full(3000,.25))['label']=='sustained_activity'

def test_recorder_boundary_is_censored_and_hysteresis_rejoins_short_gap():
    f=np.zeros(100);f[:4]=.2;f[5:9]=.2;f[95:]=.2
    events=m.segments(f)
    assert len(events)==2 and events[0]['left_censored'] and events[1]['right_censored']
    assert events[0]['stop_s']==.09

def test_constant_interval_is_valid_not_nan():
    d=m.intervals(np.arange(20)*.5)
    assert d['cv']==0 and d['cv2']==0 and d['lag1_log_iei_r'] is None

def test_offset_periodic_sources_are_not_called_irregular():
    onsets=np.sort(np.r_[np.arange(.5,29,.5),np.arange(.6,29,.5)])
    old=analyze(trace(onsets))
    assert old['label']=='irregular_bursts'  # explicit counterexample motivating the amendment
    new=metrics_v2.amend(old)
    assert new['ordering_p']<=.05
    assert new['label']=='patterned_bursts'

def test_irregular_renewal_example_survives_ordering_control():
    rng=np.random.default_rng(58);times=np.cumsum(.10+rng.exponential(.4,100));times=times[times<29.5]
    new=metrics_v2.amend(analyze(trace(times)))
    assert new['ordering_p']>.05
    assert new['label']=='irregular_bursts'
