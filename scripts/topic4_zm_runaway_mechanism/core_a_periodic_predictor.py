"""Secant predictor in full physical state, solely for numerical continuation."""
from common import OUT,np,model,read,write
from onset_variational_return import Coordinates
from onset_poincare_corrector import SectionReturn
from core_a_positive_newton_coordinates import retract
from core_a_equilibrium_branch import Family
from types import SimpleNamespace
from pathlib import Path
import argparse


def main(previous,current,target,name):
    previous=Path(previous).resolve();current=Path(current).resolve()
    for p in [previous,current]:assert read(p/'result.json')['status']=='NUMERICAL_PERIODIC_ROOT'
    contracts=[read(p/'contract.json') for p in [previous,current]]
    assert contracts[0]['dt_ms']==contracts[1]['dt_ms']
    def source(p):return dict(np.load(p/('root_state.npz' if (p/'root_state.npz').exists() else 'latest_state.npz')))
    old=source(previous);base=source(current);s=model(40);family=Family(s)
    d0=1-np.average(old['syn'][5,family.A],weights=s.sizes[family.A])
    d1=1-np.average(base['syn'][5,family.A],weights=s.sizes[family.A])
    assert abs(d1-d0)>1e-6
    assert np.array_equal(old['syn'][5,~family.A],base['syn'][5,~family.A])
    dt=contracts[1]['dt_ms'];c=Coordinates(base,s);x=c.pack(base);before=c.pack(old)
    A=SimpleNamespace(c=c,e=SimpleNamespace(s=s,dt=dt),base=base,xref=x)
    section=contracts[1].get('section','core_A')
    assert section==contracts[0].get('section','core_A') and section in ['core_A','core_B']
    mask=s.E&(s.geo['group_region']==(0 if section=='core_A' else 1))
    w=s.sizes*mask;w=w/w.sum();normal=np.zeros_like(x)
    normal.reshape(-1,s.P)[47]=w*c.scale[47]/c.weight[0];normal/=np.linalg.norm(normal);A.normal=normal
    A.admissible=lambda y:SectionReturn.admissible(A,y)
    factor=float((target-d1)/(d1-d0));delta=factor*(x-before)
    # Enforce the same physical regional rate section; its tiny residual may
    # differ between independently solved roots. All other retained states
    # still use the full secant, not regional averages.
    candidate=retract(A,x,delta,1.);assert candidate is not None
    state=SectionReturn.state(A,candidate);z,tm=family.field(target);state['syn'][5]=z
    out=OUT/'core_a_bifurcation_type_20260924/periodic_predictors'/name
    out.mkdir(parents=True,exist_ok=True);assert not(out/'state.npz').exists()
    np.savez_compressed(out/'state.npz',**state)
    def period(p):
        r=read(p/'result.json');return r.get('period_ms') or r.get('rows',r.get('iterations'))[-1]['period_ms']
    t0,t1=period(previous),period(current);T=t1+factor*(t1-t0)
    write(out/'contract.json',dict(previous=str(previous),current=str(current),D_previous=float(d0),D_current=float(d1),
        target_D_A=target,native_field_coordinate_ms=tm,secant_factor=factor,period_seed_ms=T,dt_ms=dt,section=section,
        old_to_new_phase_difference=float(normal@(x-before)),admissibility='PASS',
        method='Full-state secant in canonical delay coordinates, with positive numerical retraction and restoration of the selected physical regional E-rate section. This is only an initial guess; its actual shooting residual has not been evaluated.',
        all_M_dynamic=True,outside_A_Z_unchanged=True,status='PREDICTOR_NOT_A_PERIODIC_ORBIT',model_promoted=False))
    print(out/'state.npz');print(T)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('previous');p.add_argument('current');p.add_argument('--target-D',type=float,required=True);p.add_argument('--name',required=True)
    a=p.parse_args();main(a.previous,a.current,a.target_D,a.name)
