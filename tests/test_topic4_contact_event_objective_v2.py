import numpy as np
from src.topic4_contact_event_objective_v2 import (
    ContactEventObjectiveV2, VectorKernel, event_representation, stack_events,
    patient_event, u_components,
)


def fixture():
    rng=np.random.default_rng(24);t=np.arange(-124.,126.,2.)
    events=[]
    for i in range(40):
        centers=np.array([-30.,0.,30.])*(1 if i<20 else -1)+rng.normal(0,4,3)
        widths=rng.uniform(9,22,3)
        a=np.exp(-.5*((t[None,:]-centers[:,None])/widths[:,None])**2)
        events.append(event_representation(a,t,[True]*3))
    table=stack_events(events);w=np.ones(40)/40
    return table,w,ContactEventObjectiveV2(table,w,table['participation'],['a','b','c'])


def test_known_distribution_and_mode_frequency():
    x,w,o=fixture()
    assert max(abs(v) for v in o.population(x,w).values())<1e-10
    bad=np.r_[np.full(20,.9/20),np.full(20,.1/20)]
    assert o.population(x,bad)['joint_envelope']>.001


def test_width_collapse_or_expansion_is_not_rewarded():
    x,w,o=fixture()
    for factor in [.25,.5,1.5,2.]:
        changed={**x,'local_shape':x['local_shape']*factor}
        assert o.population(changed,w)['local_shape']>.001


def test_joint_distribution_detects_broken_contact_cooccurrence():
    x,w,o=fixture();z=x['joint_envelope'].reshape(40,3,-1).copy()
    # One channel now belongs to the opposite event mode; all its marginal
    # trajectories are unchanged, but cross-contact cooccurrence is broken.
    z[:,0]=np.roll(z[:,0],20,axis=0)
    assert o.joint.population(z.reshape(40,-1),w)>.001


def test_u_statistic_negative_and_low_n():
    x,w,o=fixture();r=o.score(x)
    assert r['loss']<0 and np.isclose(r['loss'],r['A']-r['B'])
    assert o.score({k:v[:15] for k,v in x.items()})['loss'] is None
    assert o.score(x,physical_status='RUNAWAY')['loss'] is None
    a=np.array([[1.,.1],[.1,1.]])
    assert np.isclose(u_components(a,[.3,.4],.2,minimum_events=2)['D_off'],-.4)


def test_translation_amplitude_and_missing_contact_visibility():
    t=np.arange(-124.,126.,2.)
    a=np.exp(-.5*((t[None,:]-np.array([-20,0,20])[:,None])/12)**2)
    p=event_representation(a,t,[True,True,False])
    q=event_representation(7*a,t+9876,[True,True,False])
    for k in ['local_shape','recruitment','joint_envelope']:
        assert np.allclose(p[k],q[k],atol=1e-10)
    a[2]*=5;r=event_representation(a,t,[True,True,False])
    assert np.array_equal(p['local_shape'],r['local_shape'])
    assert np.linalg.norm(p['joint_envelope']-r['joint_envelope'])>.1


def test_channel_mapping(tmp_path):
    t=np.arange(-124.,126.,2.);a=np.array([np.exp(-.5*((t-c)/15)**2) for c in [-15,0,30]])
    p=tmp_path/'p.npz'
    np.savez(p,contact_names=['b','a','c'],time_ms=t,positive_envelope_mass=a,
             packed_window_mask=np.ones(len(t),bool),participation_mask=[True,False,True])
    x=patient_event(p,['a','b','c']);y=event_representation(a[[1,0,2]],t,[False,True,True])
    assert np.allclose(x['joint_envelope'],y['joint_envelope'])
