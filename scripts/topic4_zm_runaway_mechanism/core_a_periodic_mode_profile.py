"""Lift a numerical section eigenvector and inspect its full-cycle support.

A regional rate section removes the time shift at that phase. Its raw spatial
components are therefore not a phase-independent Floquet localization. The
lift restores the autonomous time component, checks the complete fixed-time
eigen-equation, and averages physical rate/M squared amplitudes over a cycle.
Existing physical orbit phase/mesh qualification is still separate.
"""
from common import np,read,write,log
from onset_state_continuation import build
from onset_cubic_section import CubicSectionReturn,CubicSectionDerivative
from onset_poincare_corrector import regional_rate_section
from onset_segment_flow import SegmentDerivative
from onset_period_return import dynamical_state,errors
from fine_rate_frozen_Z_fields import restore,capture
from pathlib import Path
import argparse,os,time


def main(parent,index,device,spectrum_dir='section_spectrum_cached'):
    parent=Path(parent).resolve();spec=parent/spectrum_dir
    r=read(parent/'result.json');assert r['status']=='NUMERICAL_PERIODIC_ROOT'
    verified=read(spec/'verified.json')[index];assert verified['status']=='VERIFIED_NUMERICAL_EIGENPAIR'
    mode=np.load(spec/f'mode{index:02d}.npz');mu=complex(mode['multiplier'])
    assert abs(mu.imag)<1e-10 and abs(mu.real-1)>1e-4
    mu=mu.real;out=spec/f'mode{index:02d}_whole_cycle';out.mkdir(exist_ok=True);assert not(out/'jobs.json').exists()
    write(out/'jobs.json',dict(status='RUNNING',pid=os.getpid()));start=time.time()
    c=read(parent/'contract.json');dt=c['dt_ms'];T=r.get('period_ms') or r.get('rows',r.get('iterations'))[-1]['period_ms']
    numerical_method=c.get('numerical_method','old_endpoint')
    derivative_options={}
    if numerical_method=='exponential_midpoint':
        from onset_exponential_midpoint import ExponentialMidpointEngine
        from onset_midpoint_tangent import MidpointTangent
        e=ExponentialMidpointEngine(dt=dt,device=device);e.graph()
        derivative_options=dict(tangent_class=MidpointTangent)
    else:
        assert numerical_method=='old_endpoint'
        e=build(device,dt)
    base=dict(np.load(parent/('root_state.npz' if(parent/'root_state.npz').exists() else 'latest_state.npz')))
    A=CubicSectionReturn(base,e,T,5.)
    spectrum_contract=read(spec/'contract.json')
    if spectrum_contract.get('cache_segments'):
        from onset_segmented_poincare import cycle_coordinates
        cycle_coordinates(A,T,spectrum_contract['cache_segments'])
    if c.get('section','core_A') in ['core_A','core_B']:
        regional_rate_section(A,0 if c.get('section','core_A')=='core_A' else 1,c.get('section_orientation'))
    x=A.xref;y,meta=A(x);T=meta['period_ms'];slope=A.last_time_slope.copy()
    close=errors(dynamical_state(base),dynamical_state(A.state(y)),e.s.sizes/e.s.sizes.sum())
    assert close['combined_relative_rms']<1e-6
    v=(mode['vector'].real.reshape(-1,e.s.P)*mode['coordinate_scale']/mode['coordinate_weight']*A.c.weight/A.c.scale).ravel();v/=np.linalg.norm(v)
    J=CubicSectionDerivative(A,x,T,slope,**derivative_options);q=J(v);dtau=J.last_return_time_derivative
    eigenerror=float(np.linalg.norm(q-mu*v)/max(1,abs(mu)));assert eigenerror<1e-6
    restore(e,base);points=[x]
    for _ in range(2):
        e.step();e.cp.cuda.get_current_stream().synchronize();points.append(A.c.pack(capture(e)))
    flow=(-3*points[0]+4*points[1]-points[2])/(2*dt)
    lift=-dtau/(mu-1);w=v+lift*flow
    F=SegmentDerivative(A,x,T,cached=False,**derivative_options);neutral=F(flow);fw=F(w)
    neutral_error=float(np.linalg.norm(neutral-flow)/np.linalg.norm(flow))
    lift_error=float(np.linalg.norm(fw-mu*w)/(max(1,abs(mu))*np.linalg.norm(w)))
    qa=dict(section_eigen_residual=eigenerror,neutral_mode_residual=neutral_error,
            lifted_fixed_time_eigen_residual=lift_error,phase_lift_coefficient_ms=lift,return_time_derivative_ms=dtau)
    write(out/'lift_check.json',qa);log('WHOLE CYCLE NUMERICAL MODE LIFT',qa)
    # Keep a diagnostic fail as a fail; no spatial Floquet claim follows it.
    if neutral_error>=1e-3 or lift_error>=1e-4:
        write(out/'result.json',dict(status='NUMERICAL_MODE_LIFT_NOT_VERIFIED',checks=qa,model_promoted=False))
        write(out/'jobs.json',dict(status='COMPLETE'));return
    t=F.t
    with t.stream:
        t.stream.begin_capture()
        for _ in range(round(1/dt)):t.step()
        one=t.stream.end_capture()
    restore(e,base);A.c.set_tangent(t,w);growth=np.log(abs(mu))/T
    rr=[];mm=[];tt=[]
    for tm in range(int(np.floor(T))+1):
        if tm:one.launch(t.stream);t.stream.synchronize()
        tick=int(e.local.clock.get()[0]);rate=t.history[tick%len(base['history'])].get()
        rr.append(rate*rate*np.exp(-2*growth*tm));mm.append(t.syn[4].get()**2*np.exp(-2*growth*tm));tt.append(float(tm))
    actual=fw.reshape(-1,e.s.P)*A.c.scale/A.c.weight
    rr.append(actual[47]**2*np.exp(-2*growth*T));mm.append(actual[4]**2*np.exp(-2*growth*T));tt.append(T)
    rr=np.array(rr);mm=np.array(mm);tt=np.array(tt);rows=[];group_energy=[]
    for name,energy in [('rate',rr),('M',mm)]:
        integrated=np.trapz(energy,tt,axis=0)/T*e.s.sizes
        fractions=[float(integrated[e.s.E&(e.s.geo['group_region']==j)].sum()/integrated.sum()) for j in range(3)]+[float(integrated[~e.s.E].sum()/integrated.sum())]
        rows.append(dict(block=name,cycle_integrated_A_B_surround_I_squared_amplitude=fractions));group_energy.append(integrated)
    np.savez_compressed(out/'mode_profile.npz',time_ms=tt,rate_squared_amplitude=rr,M_squared_amplitude=mm,
                         weighted_integrated_group_energy=np.array(group_energy),Z=base['syn'][5],multiplier=mu,period_ms=T)
    result=dict(status='LIFTED_NUMERICAL_MODE_SPATIAL_PROFILE',checks=qa,multiplier=mu,period_ms=T,dt_ms=dt,rows=rows,
                numerical_method=numerical_method,spectrum_source=str(spec),
                method='For DP_section v=mu v, restore full autonomous phase with w=v-T_derivative(v)*flow/(mu-1). Verify original fixed-time derivative Fw=mu w and neutral Fflow=flow. Integrate squared physical rate/M perturbations times exp(-2 log|mu| t/T), cell-count weighted;1ms sampling with exact cubic endpoint. Negative multiplier phase factor has unit modulus.',
                scope='Localization of the verified lifted numerical mode over this one closed numerical orbit. Existing independent orbit phase/mesh gates remain pending; no critical crossing, complete spectrum, native correspondence or physical Floquet certificate.',seconds=time.time()-start,model_promoted=False)
    write(out/'result.json',result);write(out/'jobs.json',dict(status='COMPLETE'));log('WHOLE CYCLE NUMERICAL MODE REGIONS',rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('parent');p.add_argument('--index',type=int,required=True);p.add_argument('--device',type=int,default=1)
    p.add_argument('--spectrum-dir',default='section_spectrum_cached')
    a=p.parse_args();main(a.parent,a.index,a.device,a.spectrum_dir)
