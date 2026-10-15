"""Scientific invariants of the isolated split-current experiment."""
import copy, importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scripts.run_topic4_local_width_mechanism_pilot import load_engine, Recorder
from params import Params,compute_nu_theta
from connectivity import place_neurons
from connectivity_rot import build_connectivity_rot
from kick_probe import simulate_kick


def network():
    p=Params(L=6.,density=100.,T=100.,dt=.1,nu_ext_ratio=.6,seed=1)
    rng=np.random.default_rng(1)
    pos,labels,ne,ni=place_neurons(p,rng)
    net=build_connectivity_rot(p,pos,labels,ne,ni,rng,theta_EE=np.pi/4,AR=2.)
    net['rng']=np.random.default_rng(1)
    return p,net


def test_observer_has_exact_spike_parity_and_split_currents_sum():
    p,net=network();a=simulate_kick(copy.deepcopy(p),copy.deepcopy(net),KICK_BOOST=0.,t_kick=1e9)
    m=load_engine()
    sub=SimpleNamespace(net=net,params=p,positions_e=net['pos'][:net['NE']],contact_xy=np.array([[2.,2.],[4.,4.]]),contact_names=['a','b'])
    hook=Recorder(sub);m.LOCAL_WIDTH_HOOK=hook
    b=m.simulate_kick(copy.deepcopy(p),copy.deepcopy(net),KICK_BOOST=0.,t_kick=1e9,dump_pathway_trace=True)
    np.testing.assert_array_equal(a['E_spk_bool'],b['E_spk_bool'])
    np.testing.assert_array_equal(a['rate_I'],b['rate_I'])
    c=np.asarray(hook.current)
    np.testing.assert_allclose(c[:,0],c[:,2]+c[:,3],rtol=1e-12,atol=1e-12)
    assert hook.omitted_mass<1e-6
    prefix=hook.innov.hexdigest()
    for arm in ['recurrent_7ms','external_7ms']:
        m.LOCAL_WIDTH_ARM=arm;m.LOCAL_WIDTH_TAU_MS=7.
        m.LOCAL_WIDTH_HOOK=Recorder(sub)
        m.simulate_kick(copy.deepcopy(p),copy.deepcopy(net),KICK_BOOST=0.,t_kick=1e9,dump_pathway_trace=True)
        assert m.LOCAL_WIDTH_HOOK.innov.hexdigest()==prefix
        c=np.asarray(m.LOCAL_WIDTH_HOOK.current)
        np.testing.assert_allclose(c[:,0],c[:,2]+c[:,3],rtol=1e-12,atol=1e-12)
        assert np.min(c[:,:4])>-1e-10


def test_split_filter_preserves_other_pathway_and_impulse_area():
    dt=.1;tr=.7;base=np.exp(-dt/3.5);slow=np.exp(-dt/14.)
    srec=np.zeros(10000);sext=np.zeros(10000)
    srec[20:]=np.exp(-np.arange(9980)*dt/tr)
    sext[5:]=2*np.exp(-np.arange(9995)*dt/tr)
    def filt(s,a):
        out=[];x=0.
        for v in s:x=v+(x-v)*a;out.append(x)
        return np.array(out)
    total=filt(srec+sext,base);rec=filt(srec,base);alt=filt(srec,slow)
    changed=total+alt-rec
    np.testing.assert_allclose(changed-alt,filt(sext,base),rtol=1e-9,atol=1e-14)
    np.testing.assert_allclose(changed.sum(),total.sum(),rtol=1e-12)
    assert alt.max()<rec.max()
