"""Match the PD1 child's growing return vector to the physical branch tangent.

Compare the complete nine-state plus rate-history state at the common branch
phase. Each secant endpoint retains its own exact J and period. Do not infer
nonlinear criticality from an unidentified positive multiplier.
"""
from complete_rate_positive_stability import DEST, PERIODIC_OUT, RateField, read, write, np, paired_modes, values
from audit_rate_survey_filter_states import fingerprint
from scipy.interpolate import interp1d
from pathlib import Path
import os
import time

FOLDER=DEST/'physical_children/PD_double_low/radial_history_identity'


def state_at_reference(model,r,T,J,ages,name,source_identity):
    """All Fourier harmonics, with the real Nyquist interpolation convention."""
    cache=FOLDER/(name+'.npz');meta=cache.with_suffix('.json')
    expected=dict(source_identity=source_identity,J=J,T=T,N=len(r),history_ages_ms=ages.tolist())
    if cache.exists() and meta.exists():
        assert read(meta)==expected
        with np.load(cache) as z:return z['state'],z['velocity']
    N=len(r);assert N%2==0 and r.shape[1]==935
    rf=np.fft.rfft(r,axis=0)/N
    factors=np.full(len(rf),2.);factors[[0,-1]]=1.
    y=np.zeros((9,model.P));v=np.zeros_like(y)
    history=np.zeros((len(ages),model.P));dh=np.zeros_like(history)
    for start in range(0,len(rf),64):
        stop=min(start+64,len(rf));lam=2j*np.pi*np.arange(start,stop)/T
        for local_i,k in enumerate(range(start,stop)):
            c=model.eigenstate(None,J,lam[local_i],rf[k])
            y+=factors[k]*c.real
            v+=factors[k]*(lam[local_i]*c).real
        kernel=np.exp(-ages[:,None]*lam)
        cf=rf[start:stop]*factors[start:stop,None]
        history+=(kernel@cf).real
        dh+=(kernel@(cf*lam[:,None])).real
        if start%1024==0:
            print('FULL STATE',name,start,len(rf),flush=True)
    assert max(abs(model.output(y)-r[0]))<1e-10
    result=np.r_[y.ravel(),history.ravel()];velocity=np.r_[v.ravel(),dh.ravel()]
    np.savez(cache,state=result,velocity=velocity);write(meta,expected)
    return result,velocity


def main():
    FOLDER.mkdir(parents=True,exist_ok=True)
    write(FOLDER/'worker.json',dict(status='CPU_FULL_STATE_IDENTITY',pid=os.getpid(),timestamp=time.time()))
    source=DEST/'physical_children/PD_double_low/result.json';branch=read(source)
    departure=read(source.parent/'departure_identity.json')
    witnesses=read(DEST/'PD1_parent_witnesses/result.json')
    assert witnesses['status']=='PARENT_SIDES_CHECKED'
    assert departure['status']=='CHILD_DEPARTURE_IDENTITY_CHECKED'
    assert branch['full_physical_child_checks']
    children=sorted(branch['rows'],key=lambda r:r['amplitude_hz']);selected=children[-1]
    assert [r['amplitude_hz'] for r in children]==[10.,20.,40.]
    spectra_paths=[PERIODIC_OUT/f'poincare_floquet/PD1_physical_20260920_child_k6_dt{dt}.json' for dt in ['0.05','0.025']]
    pair=[read(p) for p in spectra_paths];verdict=paired_modes(*pair)
    assert all(Path(q['orbit']).resolve()==Path(selected['orbit']).resolve() for q in pair)
    fine_values=values(pair[-1])
    candidates=np.flatnonzero((abs(fine_values.imag)<1e-10)&(fine_values.real>1)&(fine_values.real<1.1)&
        np.asarray(verdict['outside_unit_disk_mask']))
    assert len(candidates)==1;index=int(candidates[0])
    coarse_index=int(np.argmin(abs(values(pair[0])-fine_values[index])))
    with np.load(spectra_paths[-1].with_suffix('.npz')) as z:
        vector=np.r_[z['local_vectors'][:,index],z['history_vectors'][:,index]]
        assert max(abs(vector.imag))<1e-10;vector=vector.real
        D=z['history_vectors'].shape[0]//935;ages=np.arange(1,D+1)*float(z['dt'])
    model=RateField();assert model.P==935 and len(np.unique(model.geo['group_cell']))==400
    assert ages[-1]>=max(model.delays)
    states=[];phase=None
    for r in children:
        assert r['physical_pass'] and r['physical_check']['filter_state_check']['positive']
        assert r['physical_check']['maximum_group_defect_Hz']<1e-6
        before=fingerprint(r['orbit'])
        with np.load(r['orbit']) as z:
            state,velocity=state_at_reference(model,z['r'],float(z['T']),float(z['J']),ages,
                f'child_a{r["amplitude_hz"]:g}',before)
        assert fingerprint(r['orbit'])==before
        states.append(state);phase=velocity
    parent=branch['parent'];before=fingerprint(parent['mode'])
    with np.load(parent['mode']) as z:mode=z['u'].real.copy()
    seed,_=state_at_reference(model,np.r_[mode,-mode],2*parent['T_ms'],parent['J_EE_core'],ages,
        'critical_antiperiodic_seed',before)
    assert fingerprint(parent['mode'])==before
    phase/=np.linalg.norm(phase)
    w=np.sqrt(model.geo['group_size']/model.geo['group_size'].sum())
    scales=np.r_[(np.array([1000,1000,1,1,1,1,.1,.1,1])[:,None]*w).ravel(),
        np.broadcast_to(1000*w/np.sqrt(D),(D,935)).ravel()]
    def quotient(x):return (x-phase*np.dot(phase,x))*scales
    def cosine(x,y):
        a,b=quotient(x),quotient(y)
        return float(abs(a@b)/(np.linalg.norm(a)*np.linalg.norm(b)))
    assert abs(phase@vector)<1e-6
    comparisons=[dict(earlier_amplitude_Hz=r['amplitude_hz'],earlier_orbit=r['orbit'],
        full_state_history_tangent_cosine=cosine(vector,states[-1]-state))
        for r,state in zip(children[:-1],states[:-1])]
    with np.load(spectra_paths[0].with_suffix('.npz')) as z:
        loc=z['local_vectors'][:,coarse_index].real.reshape(9,935)
        h=z['history_vectors'][:,coarse_index].real.reshape(-1,935)
        h0=model.output(loc)
        times=np.arange(len(h)+1)*float(z['dt']);assert ages[-1]<=times[-1]
        coarse=np.r_[loc.ravel(),interp1d(times,np.r_[h0[None],h],axis=0)(ages).ravel()]
    identity=cosine(vector,seed);mesh_cosine=cosine(vector,coarse)
    passed=identity>.95 and mesh_cosine>.999 and all(q['full_state_history_tangent_cosine']>.95 for q in comparisons)
    result=dict(status='RADIAL_HISTORY_IDENTITY_PASS' if passed else 'RADIAL_HISTORY_IDENTITY_UNRESOLVED',
        source=str(source),parent_witnesses=str(DEST/'PD1_parent_witnesses/result.json'),
        child_orbit=selected['orbit'],child_amplitude_Hz=selected['amplitude_hz'],
        spectra_sources=list(map(str,spectra_paths)),fine_mode_index=index,coarse_mode_index=coarse_index,
        multiplier=float(fine_values[index].real),paired_spectrum=verdict,
        parent_antiperiodic_full_state_history_cosine=identity,paired_time_step_mode_cosine=mesh_cosine,
        tangent_comparisons=comparisons,spatial_cells=400,populations=935,local_states=8415,
        history_samples=D,history_end_ms=float(ages[-1]),
        scope='Complete state and delay-history identity at the common reference phase, after removing the autonomous phase. Parent and secant endpoints use their own unchanged J and period; all Fourier harmonics are retained. Confirms radial-direction identity only when the recorded comparisons pass; no canonical criticality promotion, total unstable-dimension count, or global connection is inferred here.')
    write(FOLDER/'result.json',result)
    write(FOLDER/'worker.json',dict(status=result['status'],pid=os.getpid(),timestamp=time.time()))
    print('RADIAL HISTORY IDENTITY',identity,mesh_cosine,comparisons,flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        FOLDER.mkdir(parents=True,exist_ok=True)
        write(FOLDER/'worker.json',dict(status='COMPUTATION_FAILED',pid=os.getpid(),error=repr(exc)))
        raise
