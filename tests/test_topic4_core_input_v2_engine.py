"""kick_probe_core_input_v2: deterministic expected arrivals outside the E-core (checklist C2)."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
from params import Params
from model import build_network
from kick_probe_core_input_v1 import simulate_kick as v1
from kick_probe_core_input_v2 import simulate_kick as v2


class ExtProbe:
    def __init__(self):self.ext=[];self.nu=[]
    def observe(self,t,tm,xi,nu_now,ext,delta_rate,V,I_E,I_I,spk):
        self.ext.append(np.array(ext,float,copy=True))
    def observe_rates(self,t,tm,signal,xi,rates):
        self.nu.append(np.array(rates,float,copy=True))


class RateProbe:
    def __init__(self,sink):self.sink=sink
    def observe(self,t,tm,signal,xi,rates):self.sink.observe_rates(t,tm,signal,xi,rates)


def net_and_params():
    p=Params(L=1.,density=500.,C_EE=50,C_IE=50,C_EI=20,C_II=20,T=60.,dt=.1,seed=57)
    return p,build_network(p,verbose=False)


def test_v2_without_mask_is_byte_identical_to_v1():
    p,net=net_and_params();n=net['NE']+net['NI']
    loading=np.zeros(n);loading[:60]=1.
    def run(engine):
        net['rng']=np.random.default_rng(66)
        return engine(p,net,KICK_BOOST=0.,t_kick=1e9,global_ou_loading=loading)
    a=run(v1);b=run(v2)
    for key in ['rate_E','rate_I','E_spk_bool','initial_V']:
        np.testing.assert_array_equal(a[key],b[key])
    assert b['external_input']['mode']=='legacy_all_poisson'


def test_v2_mask_gives_exact_expected_arrivals_outside_and_poisson_inside():
    p,net=net_and_params();n=net['NE']+net['NI'];ne=net['NE']
    loading=np.zeros(n);loading[:60]=1.
    mask=np.ones(n,bool);mask[:60]=False           # deterministic everywhere except the 60 core E cells
    probe=ExtProbe();net['rng']=np.random.default_rng(66)
    res=v2(p,net,KICK_BOOST=0.,t_kick=1e9,global_ou_loading=loading,deterministic_external_mask=mask,
           step_observer=probe,afferent_rate_observer=RateProbe(probe))
    ext=np.stack(probe.ext);nu=np.stack(probe.nu)
    assert ext.shape==(600,n)
    np.testing.assert_array_equal(ext[:,mask],nu[:,mask]*p.dt)          # exact expectation, every step
    assert np.all(np.mod(ext[:,~mask],1)==0) and ext[:,~mask].max()>0     # integer Poisson counts inside
    # outside cells receive the tonic mean only: no OU fluctuation reaches them
    assert np.ptp(nu[:,mask],axis=0).max()==0
    assert res['external_input']['mode']=='core_poisson_outside_expected'
    assert res['external_input']['n_deterministic']==int(mask.sum()) and res['external_input']['n_stochastic']==60
    np.testing.assert_allclose(res['external_input']['expected_arrivals_per_step_deterministic'],nu[0,mask][0]*p.dt)


def test_v2_mask_changes_only_masked_draws_not_core_innovation_law():
    p,net=net_and_params();n=net['NE']+net['NI']
    loading=np.zeros(n);loading[:60]=1.;mask=np.ones(n,bool);mask[:60]=False
    a=ExtProbe();net['rng']=np.random.default_rng(66)
    v2(p,net,KICK_BOOST=0.,t_kick=1e9,global_ou_loading=loading,deterministic_external_mask=mask,step_observer=a)
    b=ExtProbe();net['rng']=np.random.default_rng(66)
    v2(p,net,KICK_BOOST=0.,t_kick=1e9,global_ou_loading=loading,deterministic_external_mask=mask,step_observer=b)
    np.testing.assert_array_equal(np.stack(a.ext),np.stack(b.ext))     # same seed -> same core innovations
    bad=np.ones(n+1,bool)
    try:
        v2(p,net,KICK_BOOST=0.,t_kick=1e9,deterministic_external_mask=bad);raise AssertionError('shape accepted')
    except ValueError:pass
