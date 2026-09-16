import copy
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fig5_z_retention_dose_response import intervention_state,outcome,wilson


def test_doses_preserve_all_non_z_state_and_pair_future_rng():
    parent={'slow':{'z':np.array([.7,.8,1.]),'m':np.array([4.,5.,0.])},
            'rng_state':np.random.default_rng(3).bit_generator.state,
            'external_drive':{'rng_state':np.random.default_rng(4).bit_generator.state,'field_state':np.array([.1,.2])},
            'V':np.array([2.,3.,4.])}
    early=copy.deepcopy(parent);early['slow']['z'][:2]=[.95,.97]
    states=[intervention_state(parent,early,2,d,7101) for d in [0,.5,1]]
    assert np.array_equal(states[0]['slow']['z'],early['slow']['z'])
    assert np.array_equal(states[2]['slow']['z'],parent['slow']['z'])
    assert np.allclose(states[1]['slow']['z'],(early['slow']['z']+parent['slow']['z'])/2)
    for s in states:
        assert np.array_equal(s['slow']['m'],parent['slow']['m'])
        assert np.array_equal(s['V'],parent['V'])
        assert s['rng_state']==states[0]['rng_state']
        assert s['external_drive']['rng_state']==states[0]['external_drive']['rng_state']
        assert np.array_equal(s['external_drive']['field_state'],parent['external_drive']['field_state'])
    other=intervention_state(parent,early,2,.5,7102)
    assert other['rng_state']!=states[0]['rng_state']


def test_transient_bursts_do_not_count_as_persistent_recruitment():
    r=np.zeros(75);r[:24]=400;r[25:49]=400
    assert not outcome(r)['transition']
    r[50:]=300
    o=outcome(r)
    assert o['transition'] and o['onset_ms']==1000 and o['confirmation_ms']==1500


def test_probability_endpoints_retain_sampling_uncertainty():
    low=wilson(0,4);high=wilson(4,4)
    assert low[0]==0 and low[1]>.45
    assert high[0]<.55 and high[1]==1
