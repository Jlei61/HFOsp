import numpy as np
from scripts.summarize_group_event_state_v039_closure import floored_gain,independent_mask,summarize_values


def test_harmful_control_cannot_inflate_a_state_gain():
    assert floored_gain(1.,1.9,.8)==1.-.8
    assert floored_gain(1.,.7,.8)==.7-.8
    assert floored_gain(1.,None,.8) is None


def test_target_windows_not_seeds_define_nonoverlap_support():
    times=np.arange(12)*300.;valid=np.ones(12,bool);valid[0]=False
    assert np.flatnonzero(independent_mask(times,valid)).tolist()==[1,7]
    assert summarize_values([0.,1e-12,-1e-12])['positive']==0
