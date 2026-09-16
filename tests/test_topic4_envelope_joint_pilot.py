import numpy as np
from src.topic4_envelope_joint_pilot import (
    TemporalDistributionObjective, envelope_descriptor, aligned_packet_event,
)


def fixture():
    rng=np.random.default_rng(2)
    x=np.zeros((40,3,5));x[:,:,0]=1
    x[:,:,1]=rng.normal(0,20,(40,3));x[:,:,1]-=np.median(x[:,:,1],axis=1)[:,None]
    x[:,:,2:4]=rng.uniform(10,50,(40,3,2));x[:,:,4]=rng.normal(0,4,(40,3))
    labels=np.repeat([0,1],20)
    obj=TemporalDistributionObjective(x,labels,[.3,.7],['a','b','c'])
    return x,labels,obj


def test_target_distribution_is_zero_and_other_frequency_costs_more():
    x,labels,obj=fixture()
    assert abs(obj.population_distance(x,obj.patient_weights))<1e-12
    wrong=np.where(labels==1,.1/20,.9/20)
    assert obj.population_distance(x,wrong)>1e-5


def test_off_diagonal_equals_explicit_pair_sum():
    x,_,obj=fixture();s=obj.score(x[:16]);f=obj.features(x[:16])
    k=obj.kernel(f,f)
    manual=sum(k[i,j] for i in range(16) for j in range(16) if i!=j)/(16*15)
    manual-=2*np.mean(obj.kernel(f,obj.reference)@obj.patient_weights)
    manual+=obj.reference_constant
    assert abs(s['D_off']-manual)<1e-12
    assert abs(s['D_off']-(s['A']-s['B']))<1e-12


def test_insufficient_events_and_negative_values_not_clipped():
    x,_,obj=fixture()
    assert obj.score(x[:15])['loss'] is None
    rng=np.random.default_rng(7)
    scores=[obj.score(x[rng.choice(len(x),16,p=obj.patient_weights)])['D_off'] for _ in range(100)]
    assert min(scores)<0


def test_descriptors_preserve_translation_and_detect_width():
    t=np.arange(-100,201,dtype=float)
    a=np.array([np.exp(-.5*((t-c)/8)**2) for c in [0,20,40]])
    b=np.array([np.exp(-.5*((t-c)/30)**2) for c in [0,20,40]])
    x=envelope_descriptor(a,t,[True]*3);y=envelope_descriptor(a,t+1800,[True]*3)
    assert np.allclose(x['values'],y['values'])
    z=envelope_descriptor(b,t,[True]*3)
    assert z['statistics']['contact_width_ms']>3*x['statistics']['contact_width_ms']


def test_patient_contact_identity_and_absence_are_explicit(tmp_path):
    t=np.arange(10.);mass=np.array([t+1,10-t,np.ones(10)])
    p=tmp_path/'packet.npz'
    np.savez(p,contact_names=['b','c','a'],positive_envelope_mass=mass,time_ms=t,
             packed_window_mask=np.ones(10,bool),participation_mask=[True,False,True])
    got=aligned_packet_event(p,['a','b','c'])
    want=envelope_descriptor(mass[[2,0,1]],t,[True,True,False])
    assert np.allclose(got['values'],want['values'])
    assert np.isnan(got['centroid'][2]) and np.all(got['values'][2]==0)
