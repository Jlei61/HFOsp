"""Check the differentiated period-return identity including parameter forcing.

For a family y(t;a), period T(a), and held Z(a), integration of its physical
initial tangent must return x-T'(a)*phase. This is a consistency condition at
any regular family point, not a bifurcation or stability certificate.
"""
from common import *
from native_path import attach_native_path
from streaming_periodic import StreamPeriodic
from orbit_reconstruction import orbit_states_and_derivative
from spectral_grid_sampler import SpectralGridSampler
from physical_cycle_tangent import initial_vector
from parameter_forced_monodromy import ParameterForcedRK4
from scipy.fft import next_fast_len
import argparse,gc


def one(c,dtmax,device,bounded_reconstruction=False):
    s=model();attach_native_path(s);source=OUT/c['source'];z=np.load(source)
    sol=dict(r=z['r'],T=float(z['T']),D=float(z['D']));s.set_D(sol['D'])
    assert np.array_equal(s.Z,z['Z'])
    n=next_fast_len(int(np.ceil(sol['T']/dtmax)),real=True)
    dt=np.nextafter(sol['T']/n,np.inf);depth=int(np.ceil(s.delays[-1]/(sol['T']/n)))+2
    indices=np.unique(np.r_[0,(-2*np.arange(1,depth+1))%(2*n)])
    o=StreamPeriodic(s,len(sol['r']),device);o.cache_mean_operators=False
    log('FAMILY FLOW RECONSTRUCTION',n,'steps',dtmax)
    reconstruction=orbit_states_and_derivative
    if bounded_reconstruction:
        parity=read(OUT/'bounded_reconstruction_parity.json')
        assert parity['status']=='BOUNDED_RECONSTRUCTION_PARITY_PASS'
        assert {row['N']%2 for row in parity['cases']}=={0,1}
        from orbit_reconstruction_bounded import orbit_states_and_derivative_bounded
        reconstruction=orbit_states_and_derivative_bounded
    states,derivatives,_=reconstruction(o,sol,2*n,derivative_indices=indices,include_rate=False)
    sampler=SpectralGridSampler(states,sol['T'],derivative=derivatives,derivative_indices=indices)
    o.sample_state=sampler;o.cache_key=None;o.cache=None;o.cp.get_default_memory_pool().free_all_blocks()
    # Keeping the orbit file-backed makes MemAvailable look large. Do not let
    # the cache heuristic turn the83GB gain array back into anonymous memory.
    m=ParameterForcedRK4(s,o,sol,dtmax=dt,device=device,host_gain_cache=True,
                        host_gain_storage='disk' if bounded_reconstruction else 'auto')
    assert m.n==n and m.Dd==depth
    cp=m.cp
    # Rate-history phase derivative via the original chain rule, independently
    # of the Fourier series used for the family's rate-history derivative.
    times=((-np.arange(1,m.Dd+1))%n)*m.dt
    dy=sampler(times,1);dy[:,11]=0.;past=sampler(times)
    phase_state=sampler(0.,1).copy();phase_state[11]=0.
    hist=cp.empty((m.Dd,s.P));work=cp.empty_like(m.y)
    for j in range(m.Dd):
        m.k['tangent_rhs'](((s.P+127)//128,),(128,),
            (cp.asarray(past[j]),cp.asarray(dy[j]),m.arr,m.pars,m.consts,m.SE,m.SI,m.WE,m.WI,work,hist[j]))
    phase=np.r_[phase_state.ravel(),hist.get().ravel()]
    del hist,past,dy,work,states,derivatives,sampler,o.sample_state
    o.cache_key=None;o.cache=None;cp.get_default_memory_pool().free_all_blocks();gc.collect()
    dest=OUT/'periodic/native_family_flow_consistency';dest.mkdir(exist_ok=True)
    prefix=dest/f'dt{dtmax}'
    mp=m.matvec(phase);phase_error=float(np.linalg.norm(mp-phase)/np.linalg.norm(phase))
    phase_projection=float(phase@mp/(phase@phase))
    write(prefix.with_suffix(prefix.suffix+'.progress.json'),dict(status='PHASE_COMPLETE',phase_error=phase_error,
        phase_projection=phase_projection,dt_ms=m.dt))
    x,meta=initial_vector(o,sol,z['tangent'],m.n,m.Dd,include_parameter=True)
    o.cache_key=None;o.cache=None;cp.get_default_memory_pool().free_all_blocks();gc.collect()
    mx=m.matvec(x);correction=meta['dT_dlog_coordinate']*phase
    residual=mx-x+correction
    denominator=max(np.linalg.norm(x),np.linalg.norm(correction))
    relative=float(np.linalg.norm(residual)/denominator)
    ix=14*s.P;history_error=float(np.linalg.norm(residual[ix:])/max(np.linalg.norm(x[ix:]),np.linalg.norm(correction[ix:]),1e-30))
    a=residual[:ix].reshape(14,s.P);b=x[:ix].reshape(14,s.P);d=correction[:ix].reshape(14,s.P)
    denom=np.maximum(np.linalg.norm(b,axis=1),np.linalg.norm(d,axis=1))
    components=np.linalg.norm(a,axis=1)/np.maximum(denom,1e-30)
    active=denom>1e-8*np.max(denom)
    gate=c['acceptance'];phase_ok=phase_error<gate['phase_relative_max'] and abs(phase_projection-1)<gate['phase_projection_distance_max']
    passed=phase_ok and relative<gate['family_relative_max'] and history_error<gate['family_history_relative_max'] and components[active].max()<gate['active_state_component_max']
    # Projection is descriptive only: with dD != 0 it is not a homogeneous
    # Floquet eigenvector, even if its quotient Rayleigh value is close to one.
    def project(v):return v-phase*(phase@v)/(phase@phase)
    px=project(x);pmx=project(mx)
    result=dict(status='FLOW_CONSISTENCY_PASS' if passed else 'FLOW_CONSISTENCY_FAIL',source=str(source),
        D=sol['D'],T_ms=sol['T'],requested_dt_max_ms=dtmax,dt_ms=m.dt,steps=m.n,
        phase_error=phase_error,phase_projection=phase_projection,phase_pass=bool(phase_ok),
        family_return_relative=relative,family_history_relative=history_error,
        state_component_relative=components.tolist(),active_state_components=np.flatnonzero(active).tolist(),
        projected_forced_return_relative=float(np.linalg.norm(pmx-px)/max(np.linalg.norm(px),1e-30)),
        projected_fraction=float(np.linalg.norm(px)/np.linalg.norm(x)),family=meta,
        identity='ForcedFlow_T(x) - x + dT_da * phase = 0, including the constant full spatial dZ/da.',
        scope=c['scope'],bifurcation_type='NOT_ESTABLISHED',
        reconstruction='disk-backed component-major, independent population FFT blocks' if bounded_reconstruction else 'original dense host states')
    keep=np.unique(np.linspace(0,m.Dd-1,min(m.Dd,256)).astype(int))
    np.savez_compressed(str(prefix)+'.npz',initial_state=b,final_state=mx[:ix].reshape(14,s.P),
                        phase_state=phase_state,state_identity_residual=a,
                        history_sample_indices=keep,history_total_rows=m.Dd,
                        initial_history_sample=x[ix:].reshape(m.Dd,s.P)[keep],
                        final_history_sample=mx[ix:].reshape(m.Dd,s.P)[keep],
                        phase_history_sample=phase[ix:].reshape(m.Dd,s.P)[keep],dt_ms=m.dt)
    write(Path(str(prefix)+'.json'),result);log('FAMILY FLOW RESULT',result)
    del m,o;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    return result


def main(a):
    c=read(OUT/'native_family_flow_consistency_contract.json')
    assert read(OUT/'parameter_forced_monodromy_check.json')['status']=='LOCAL_FORCED_VARIATION_PASS'
    assert read(OUT/'parameter_forced_map_check.json')['status']=='ZERO_PARAMETER_WHOLE_MAP_PARITY_PASS'
    assert read(OUT/'physical_cycle_tangent_check.json')['status']=='PASS'
    assert a.dt in c['allowed_dt_max_ms']
    one(c,a.dt,a.device,a.bounded_reconstruction)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dt',type=float,required=True);p.add_argument('--device',type=int,default=0)
    p.add_argument('--bounded-reconstruction',action='store_true',help='Use independently checked disk-backed state layout and bounded FFT work arrays; identical source/grid/equations/gates')
    main(p.parse_args())
