import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
from params import Params
from model import build_network
from kick_probe import simulate_kick as old
from kick_probe_core_input_v1 import simulate_kick as new
from src.topic4_core_driven_input import lowering_only,MaskedSpatialDrive,RateAudit
from src.topic4_spatial_ou_drive import SpatialOUConfig,SpatialOUDrive


def test_threshold_clipping_retains_only_original_lowering():
    d=np.array([-3.,-.2,0.,.4,2.])
    np.testing.assert_array_equal(lowering_only(d),[-3.,-.2,0,0,0])
    np.testing.assert_array_equal(d,[-3.,-.2,0,.4,2.])


def test_spatial_mask_does_not_change_core_noise_or_recenter_outside():
    p=np.random.default_rng(1).uniform(0,20,(70,2));mask=np.arange(70)<20
    a=SpatialOUDrive(p,20,.1,SpatialOUConfig('local',.1,20,.38,seed=44))
    b=MaskedSpatialDrive(SpatialOUDrive(p,20,.1,SpatialOUConfig('local',.1,20,.38,seed=44)),mask)
    for tm in np.arange(0,100,.5):
        v=a.step(tm);w=b.step(tm)
        np.testing.assert_array_equal(w[mask],v[mask]);assert (w[~mask]==0).all()


def test_engine_default_parity_and_actual_core_only_rates():
    p=Params(L=1.,density=500.,C_EE=50,C_IE=50,C_EI=20,C_II=20,T=100.,dt=.1,seed=57)
    net=build_network(p,verbose=False);n=net['NE']+net['NI'];ne=net['NE']
    def run(engine,**kwargs):
        net['rng']=np.random.default_rng(66)
        return engine(p,net,KICK_BOOST=0.,t_kick=1e9,**kwargs)
    a=run(old);b=run(new);c=run(new,global_ou_loading=np.ones(n))
    for key in ['rate_E','E_spk_bool','initial_V']:
        np.testing.assert_array_equal(a[key],b[key]);np.testing.assert_array_equal(a[key],c[key])
    loading=np.zeros(n);loading[:50]=1
    groups={'coreAE':np.arange(25),'coreBE':np.arange(25,50),'surroundE':np.arange(50,ne),'allI':np.arange(ne,n)}
    audit=RateAudit(groups,n,p.dt);run(new,global_ou_loading=loading,afferent_rate_observer=audit)
    assert audit.calls==1000 and audit.max_outside_deviation==0
    data=audit.arrays();v=data['values'];names=list(data['columns'])
    np.testing.assert_allclose(v[:,names.index('coreAE_mean_rate_per_ms')],np.maximum(v[:,1]+v[:,2],0),rtol=1e-14)
    np.testing.assert_allclose(v[:,names.index('allI_mean_rate_per_ms')],np.maximum(v[:,1],0),rtol=1e-14)
