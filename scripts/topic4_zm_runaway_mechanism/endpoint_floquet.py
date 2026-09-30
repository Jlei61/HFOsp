"""Floquet multipliers with endpoint-correct delay history and explicit phase acceptance."""
from phase_audit import *
from scipy.sparse.linalg import LinearOperator,eigs


def main(a):
    assert not a.family_tangent_only or a.physical_family_tangent
    s=model();{'rate':attach_rate_entry_path,'native':attach_native_path,
               'fine':attach_fine_rate_entry_path}[a.family](s)
    z=np.load(a.orbit);sol=dict(r=z['r'],T=float(z['T']),D=float(z['D']))
    time_grid_n=int(np.ceil(sol['T']/a.dt))
    if a.fast_time_grid:
        from scipy.fft import next_fast_len
        time_grid_n=next_fast_len(time_grid_n,real=True)
    integration_dt=np.nextafter(sol['T']/time_grid_n,np.inf) if a.fast_time_grid else a.dt
    s.set_D(sol['D'])
    if 'Z' in z:assert abs(s.Z-z['Z']).max()<1e-12,'Orbit Z field disagrees with selected parameter path'
    if a.physical_state:
        from types import SimpleNamespace
        from scipy.interpolate import CubicSpline
        import cupy as cp
        import chunk_monodromy
        import floquet_v3
        cp.cuda.Device(a.device).use()
        Y=z['state_cycle'];R=z['rate_cycle'];t=z['cycle_times_ms']
        interpolator=CubicSpline
        if a.local_state:
            from local_cubic import LocalCubic
            assert read(OUT/'local_cubic_check.json')['status']=='PASS'
            interpolator=LocalCubic
        sy=interpolator(t,Y,axis=0);sr=interpolator(t,R,axis=0)
        def reconstruct(o,sol,n):
            times=np.arange(n+1)*sol['T']/n
            return sy(times),sr(times)
        global orbit_states
        orbit_states=reconstruct;chunk_monodromy.orbit_states=reconstruct;floquet_v3.orbit_states=reconstruct
        o=SimpleNamespace(cp=cp,cache_key=None,cache=None)
        if a.stream_orbit:
            assert a.cached
            check='streamed_rk4_monodromy_check.json' if a.method=='rk4' else 'streamed_cached_monodromy_check.json'
            assert read(OUT/check)['status']=='PASS'
            o.sample_state=sy
    elif len(sol['r'])>1024 or a.stream:
        from streaming_periodic import StreamPeriodic
        StreamPeriodic.harmonic_block=a.harmonic_block
        o=StreamPeriodic(s,len(sol['r']),a.device)
        o.cache_mean_operators=False
    else:o=PeriodicV3(s,len(sol['r']),a.device)
    if a.cpu_orbit_fft and not a.physical_state:
        import orbit_reconstruction
        import chunk_monodromy
        import floquet_v3
        orbit_states=orbit_reconstruction.orbit_states
        chunk_monodromy.orbit_states=orbit_states;floquet_v3.orbit_states=orbit_states
    if a.stream_orbit and not a.physical_state:
        from spectral_grid_sampler import SpectralGridSampler
        assert a.cached and a.cpu_orbit_fft
        check='streamed_rk4_monodromy_check.json' if a.method=='rk4' else 'streamed_cached_monodromy_check.json'
        assert read(OUT/check)['status']=='PASS'
        from orbit_reconstruction import orbit_states_and_derivative
        assert read(OUT/'direct_fourier_grid_reconstruction_check.json')['status']=='PASS'
        log('RECONSTRUCTING SPECTRAL GRID',len(sol['r']),a.dt)
        n=time_grid_n*(2 if a.method=='rk4' else 1)
        phase_indices=None
        if a.host_gain_cache:
            assert a.method=='rk4'
            assert read(OUT/'host_gain_rk4_check.json')['status']=='PASS'
            depth=int(np.ceil(s.delays[-1]/(sol['T']/time_grid_n)))+2
            phase_indices=np.unique(np.r_[0,(-2*np.arange(1,depth+1))%n])
        states,dstate,_=orbit_states_and_derivative(o,sol,n,
            derivative_indices=phase_indices,include_rate=False)
        sy=SpectralGridSampler(states,sol['T'],derivative=dstate,
                              derivative_indices=phase_indices);o.sample_state=sy
        del states,dstate
        # The sampled state/derivative is now on the host. The variational
        # operator builds its own physical delay arrays and needs no retained
        # LTI reconstruction kernels beside its much larger gain cache.
        o.cache_key=None;o.cache=None;o.cp.get_default_memory_pool().free_all_blocks()
        log('SPECTRAL GRID READY')
    cls=EndpointMonodromy
    if a.chunk:
        from chunk_monodromy import ChunkEndpointMonodromy
        cls=ChunkEndpointMonodromy
    if a.cached:
        from cached_monodromy import CachedMonodromy
        assert read(OUT/'cached_monodromy_parity.json')['status']=='PASS'
        cls=CachedMonodromy
    if a.method=='rk4':
        import rk4_monodromy
        rk4_monodromy.orbit_states=orbit_states
        assert read(OUT/'rk4_monodromy_check.json')['status']=='PASS'
        cls=rk4_monodromy.RK4Monodromy
    if a.preinterpolate_delays:
        assert a.method=='rk4' and a.host_gain_cache
        assert read(OUT/'preinterpolated_rk4_check.json')['status']=='PASS'
        from preinterpolated_rk4 import PreinterpolatedRK4
        cls=PreinterpolatedRK4
    kwargs=dict(host_gain_cache=True) if a.host_gain_cache else {}
    m=cls(s,o,sol,dtmax=integration_dt,device=a.device,**kwargs)
    m.chunk_progress=a.map_progress
    assert m.n==time_grid_n and m.dt<=a.dt*(1+1e-14)
    log('VARIATIONAL OPERATOR READY',m.n,m.dt)
    if a.map_progress:
        raw_matvec=m.matvec
        def monitored_matvec(x):
            started=time.time();answer=raw_matvec(x)
            log('VARIATIONAL MAP COMPLETE',m.calls,'seconds',round(time.time()-started,1))
            return answer
        m.matvec=monitored_matvec
    full=None
    if not a.stream_orbit:full,_=orbit_states(o,sol,m.n)
    if a.physical_state or a.stream_orbit:
        # Local derivatives of the actually recorded smooth state. A global FFT
        # derivative would amplify the small finite-time closure error at the
        # join into a spurious impulse exactly at the phase-section boundary.
        cp=m.cp
        if a.stream_orbit:
            indices=(-np.arange(1,m.Dd+1))%m.n;times=indices*m.dt
            dY=sy(times,1);dY[:,11]=0.;history_states=sy(times)
            phase_state=sy(0.,1);phase_state[11]=0.
        else:
            times=np.arange(m.n)*m.dt;dY=sy(times,1);dY[:,11]=0.;phase_state=dY[0]
        hist=cp.empty((m.Dd,s.P));work=cp.empty_like(m.y)
        for j in range(1,m.Dd+1):
            idx=j-1 if a.stream_orbit else (-j)%m.n
            state=cp.asarray(history_states[idx]) if a.stream_orbit else m.orbit[idx]
            m.k['tangent_rhs'](((s.P+127)//128,),(128,),
                (state,cp.asarray(dY[idx]),m.arr,m.pars,m.consts,
                 m.SE,m.SI,m.WE,m.WI,work,hist[j-1]))
        ph=np.r_[phase_state.ravel(),hist.get().ravel()]
        m.phase_method=('local cubic state derivative' if a.physical_state else 'spectral state derivative')+'; chain rule for rate history'
        del dY,hist,work
        if a.stream_orbit:del history_states
    else:ph=m.phase_vector(full)
    del full
    if a.stream_orbit:
        del o.sample_state,sy
    o.cache_key=None;o.cache=None;o.cp.get_default_memory_pool().free_all_blocks()
    if a.cached:m.release_full_orbit()
    mp=m.matvec(ph)
    defect=float(np.linalg.norm(mp-ph)/np.linalg.norm(ph))
    out=OUT/'floquet';out.mkdir(parents=True,exist_ok=True)
    stem=Path(a.orbit).stem
    if stem.startswith(('point','eval_')) or stem=='seed':stem=Path(a.orbit).parent.name+'_'+stem
    name=stem+f'_endpoint_dt{a.dt}'+('_quotient' if a.quotient else '')+'_chainphase'
    if a.physical_state:name+='_physical_state_localphase'
    if a.method=='rk4':name+='_rk4_cubic'
    if a.local_state:name+='_local4'
    if a.stream_orbit:name+='_streamed'
    if a.fast_time_grid:name+='_fastgrid'
    if a.host_gain_cache:name+='_hostgains'
    if a.preinterpolate_delays:name+='_preinterp'
    if a.output_tag:
        assert all(c.isalnum() or c in '_-' for c in a.output_tag)
        name+='_'+a.output_tag
    if a.physical_family_tangent:name+='_physicalfamily'
    projection=float(mp@ph/(ph@ph))
    phase_ok=bool(defect<.005 and abs(projection-1)<.003)
    pstate=ph[:NS*s.P].reshape(NS,s.P)
    dstate=(mp-ph)[:NS*s.P].reshape(NS,s.P)
    phase_details=dict(
        state_component_relative_defects=(np.linalg.norm(dstate,axis=1)/
            np.maximum(np.linalg.norm(pstate,axis=1),1e-30)).tolist(),
        state_component_phase_norms=np.linalg.norm(pstate,axis=1).tolist(),
        state_relative_defect=float(np.linalg.norm(dstate)/np.linalg.norm(pstate)),
        history_relative_defect=float(np.linalg.norm((mp-ph)[NS*s.P:])/
            np.linalg.norm(ph[NS*s.P:])),
        propagated_phase_norm_ratio=float(np.linalg.norm(mp)/np.linalg.norm(ph)))
    write(out/f'{name}.progress.json',dict(status='RUNNING',phase_defect=defect,phase_projection=projection,dt=m.dt))
    log('phase',defect,'computing eigenvalues')
    if a.phase_only or (not phase_ok and not a.allow_invalid_phase):
        write(out/f'{name}.phase.json',dict(status='PHASE_ONLY_COMPLETE' if phase_ok else 'PHASE_CHECK_FAILED',
             phase_defect=defect,phase_projection=projection,phase_valid=phase_ok,dt_ms=m.dt,orbit=a.orbit,
             component_diagnostics=phase_details))
        write(out/f'{name}.progress.json',dict(status='PHASE_ONLY_COMPLETE' if phase_ok else 'PHASE_CHECK_FAILED',
             phase_defect=defect,phase_projection=projection,phase_valid=phase_ok,dt_ms=m.dt,orbit=a.orbit))
        log('PHASE GATE',phase_ok,projection);return
    if a.physical_family_tangent:
        from physical_cycle_tangent import flow_diagnostic
        assert read(OUT/'physical_cycle_tangent_check.json')['status']=='PASS'
        source=np.load(a.physical_family_tangent)
        assert np.array_equal(source['r'],z['r']) and np.array_equal(source['Z'],z['Z'])
        assert float(source['T'])==sol['T'] and float(source['D'])==sol['D']
        diagnostic=flow_diagnostic(o,sol,source['tangent'],m,ph)
        diagnostic.update(source=a.physical_family_tangent,phase_defect=defect,
                          phase_projection=projection,phase_valid=phase_ok)
        write(out/f'{name}.family_tangent.json',diagnostic)
        log('PHYSICAL FAMILY TANGENT MAP',diagnostic)
        if a.family_tangent_only:
            write(out/f'{name}.progress.json',dict(status='PHYSICAL_FAMILY_DIAGNOSTIC_COMPLETE',
                phase_valid=phase_ok,orbit=a.orbit,scope='No eigenvalues or bifurcation type assigned'))
            return
    def project(x):return x-ph*(ph@x)/(ph@ph)
    def qmatvec(x):return project(m.matvec(project(x)))
    matvec=qmatvec if a.quotient else m.matvec
    A=LinearOperator((m.dim,m.dim),matvec=matvec,dtype=np.float64)
    v0=None
    seed_refinement=None
    if a.branch_tangent_seed:
        assert not a.eigenvector_seed, 'Choose one Krylov starting-vector source'
        from branch_tangent_guess import history_guess,sanity
        sanity()
        with np.load(a.branch_tangent_seed) as tangent_source:
            v0,seed_refinement=history_guess(tangent_source,z,m.n,m.Dd)
        if a.quotient:v0=project(v0)
        v0/=np.linalg.norm(v0)
        noise=np.random.default_rng(197).normal(size=m.dim);noise/=np.linalg.norm(noise)
        v0+=.01*noise
        seed_refinement['source']=a.branch_tangent_seed
        log('BRANCH TANGENT KRYLOV GUESS',seed_refinement)
    if a.eigenvector_seed:
        from scipy.interpolate import interp1d
        seedpath=Path(a.eigenvector_seed);meta=read(seedpath.with_suffix('.json'))
        if Path(meta['orbit']).resolve()!=Path(a.orbit).resolve():
            assert a.refined_orbit_seed or a.nearby_orbit_seed, 'Different orbit source requires explicit initial-guess validation'
            from scipy.signal import resample
            old_orbit=np.load(meta['orbit'])
            assert float(old_orbit['residual'])<2e-8 and float(z['residual'])<2e-8
            near=a.nearby_orbit_seed
            assert abs(float(old_orbit['T'])-sol['T'])<(sol['T']*.01 if near else 1e-8)
            assert abs(float(old_orbit['D'])-sol['D'])<(.002 if near else 1e-8)
            dz=float(np.max(abs(old_orbit['Z']-z['Z'])));assert dz<(.02 if near else 1e-6)
            old_r=resample(old_orbit['r'],len(sol['r']),axis=0)
            difference=float(np.linalg.norm(old_r-sol['r'])/np.linalg.norm(sol['r']))
            assert difference<(.2 if near else .001),('Initial-guess waveform disagreement',difference)
            seed_refinement=dict(source=meta['orbit'],source_N=len(old_orbit['r']),
                target_N=len(sol['r']),rate_relative_difference=difference,max_Z_difference=dz,
                source_phase_valid=meta.get('phase_valid',False),
                kind='nearby-orbit Krylov guess' if near else 'same-orbit mesh-refinement Krylov guess',
                scope='Krylov initial guess only. Independent refined operator, phase and eigen-residual checks remain required.')
            del old_r,old_orbit
        seed=np.load(seedpath)['vectors'][:,0]
        seed=seed.real if np.linalg.norm(seed.real)>=np.linalg.norm(seed.imag) else seed.imag
        old_depth=len(seed)//s.P-NS
        old=seed[NS*s.P:].reshape(old_depth,s.P)
        old_times=-np.arange(1,old_depth+1)*meta['dt_ms']
        new_times=-np.arange(1,m.Dd+1)*m.dt
        history=interp1d(old_times[::-1],old[::-1],axis=0,bounds_error=False,fill_value='extrapolate')(new_times)
        v0=np.r_[seed[:NS*s.P],history.ravel()]
        if a.quotient:v0=project(v0)
        v0/=np.linalg.norm(v0)
        # Only a Krylov initial guess. The refined operator and independent
        # eigen-residual, phase and time-step criteria remain unchanged.
        noise=np.random.default_rng(197).normal(size=m.dim);noise/=np.linalg.norm(noise)
        # A neighboring-cycle mode is only a numerical predictor. Mix a
        # material generic component so emerging spatial modes can enter the
        # Krylov space; no source stability classification is inherited.
        v0+=(.01 if a.nearby_orbit_seed else 1e-6)*noise
        log('REFINEMENT EIGENVECTOR SEED',seedpath,'history depths',old_depth,m.Dd)
    eig_options=dict(k=a.nev,which='LM',tol=a.eigen_tol,maxiter=1200,ncv=a.ncv,v0=v0)
    if a.ritz_progress:
        from observed_eigs import observed_eigs
        def ritz_observer(values,bounds,nconv,done):
            order=np.argsort(-abs(values))[:min(6,len(values))]
            q=dict(status='UNCONVERGED_RITZ_DIAGNOSTIC',calls=m.calls,
                   values=[[values[j].real,values[j].imag] for j in order],
                   error_estimates=[float(bounds[j]) for j in order],
                   n_converged=nconv,arpack_finished=done,
                   scope='Progress only; independent eigen-residual and phase checks remain required')
            write(out/f'{name}.ritz.json',q);log('RITZ DIAGNOSTIC',q)
        ev,V=observed_eigs(A,ritz_observer,**eig_options)
    else:ev,V=eigs(A,**eig_options)
    idx=np.argsort(-abs(ev));ev=ev[idx];V=V[:,idx]
    residuals=[]
    for i,v in enumerate(V.T):
        mv=matvec(v.real)+1j*matvec(v.imag) if np.any(v.imag) else matvec(v.real)
        residuals.append(float(np.linalg.norm(mv-ev[i]*v)/np.linalg.norm(v)))
    projection=float(mp@ph/(ph@ph))
    if a.quotient:
        j=None;align=None;phase_ok=bool(defect<.005 and abs(projection-1)<.003);remaining=ev
    else:
        j=int(np.argmin(abs(ev-1)));align=float(abs(np.vdot(V[:,j],ph))/(np.linalg.norm(V[:,j])*np.linalg.norm(ph)))
        phase_ok=bool(abs(ev[j]-1)<.003 and defect<.005 and align>.8);remaining=np.delete(ev,j)
    radius=float(max(abs(remaining)))
    row=dict(orbit=a.orbit,N=len(sol['r']),D=sol['D'],T_ms=sol['T'],dt_ms=m.dt,
        multipliers=[[x.real,x.imag] for x in ev],eigen_residuals=residuals,phase_index=j,
        phase_defect=defect,phase_alignment=align,phase_valid=phase_ok,max_transverse_modulus=radius,
        phase_projection=projection,phase_quotient=a.quotient,
        phase_method=m.phase_method,
        graph_schedule='RK4 with cubic delayed history' if a.method=='rk4' else 'reusable blocks with cached exact local gradients' if a.cached else 'reusable blocks' if a.chunk else 'full period',
        status='STEP_REFINEMENT_REQUIRED' if phase_ok else 'PHASE_CHECK_FAILED',
        sampled_stability=('UNSTABLE' if radius>1.001 else 'STABLE' if radius<.999 else 'NEAR_UNIT') if phase_ok else 'UNRESOLVED',
        calls=m.calls,Z='held',M='dynamic',history='instantaneous rate at its labelled endpoint',
        orbit_source='direct fine-time state recording' if a.physical_state else 'spectral BVP LTI reconstruction',
        eigen_tolerance=a.eigen_tol)
    row['state_interpolation']='local four-point cubic' if a.local_state else 'cubic spline' if a.physical_state else 'spectral'
    row['eigenvector_initial_guess']=a.eigenvector_seed
    row['branch_tangent_initial_guess']=a.branch_tangent_seed
    row['eigenvector_refined_orbit_check']=seed_refinement
    row['time_grid']=dict(requested_maximum_dt_ms=a.dt,steps=m.n,fft_friendly=a.fast_time_grid)
    row['gain_cache']='host blocks, float64' if a.host_gain_cache else 'GPU float64'
    row['delay_interpolation_cache']=bool(a.preinterpolate_delays)
    row['phase_component_diagnostics']=phase_details
    row['requested_eigenpairs']=a.nev;row['arnoldi_basis_size']=a.ncv
    np.savez_compressed(out/f'{name}.npz',multipliers=ev,vectors=V,phase=ph)
    write(out/f'{name}.json',row);write(out/f'{name}.progress.json',dict(status='COMPLETE'));log('RESULT',row)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('orbit');p.add_argument('--dt',type=float,default=.05)
    p.add_argument('--nev',type=int,default=3);p.add_argument('--device',type=int,default=1)
    p.add_argument('--family',choices=['native','rate','fine'],default='native')
    p.add_argument('--quotient',action='store_true',help='P M P removes the validated phase direction; eigenvalues are transverse')
    p.add_argument('--chunk',action='store_true')
    p.add_argument('--stream',action='store_true')
    p.add_argument('--physical-state',action='store_true')
    p.add_argument('--cpu-orbit-fft',action='store_true')
    p.add_argument('--cached',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--harmonic-block',type=int,default=33)
    p.add_argument('--phase-only',action='store_true')
    p.add_argument('--allow-invalid-phase',action='store_true')
    p.add_argument('--eigen-tol',type=float,default=1e-8)
    p.add_argument('--method',choices=['heun','rk4'],default='heun')
    p.add_argument('--local-state',action='store_true')
    p.add_argument('--stream-orbit',action='store_true')
    p.add_argument('--ncv',type=int)
    p.add_argument('--map-progress',action='store_true')
    p.add_argument('--eigenvector-seed',help='Accepted spectrum NPZ for this same orbit; interpolate history only to seed step refinement')
    p.add_argument('--branch-tangent-seed',help='Accepted same-period orbit and BVP tangent, used only as a Fourier past-rate Arnoldi guess with generic noise; does not alter phase or eigen-residual acceptance')
    p.add_argument('--physical-family-tangent',help='Exact same-orbit BVP tangent; independently check the physical state/history Jordan relation after the phase gate')
    p.add_argument('--family-tangent-only',action='store_true',help='Stop after the physical family tangent diagnostic; does not assign an eigenvalue or bifurcation type')
    p.add_argument('--refined-orbit-seed',action='store_true',help='Allow a checked refinement of the same periodic solution to provide only the Krylov initial guess')
    p.add_argument('--nearby-orbit-seed',action='store_true',help='Use a close solved orbit only as a Krylov guess, with 1-percent random admixture; does not transfer source stability or relax target acceptance')
    p.add_argument('--fast-time-grid',action='store_true',help='Use the next FFT-friendly number of steps; actual dt remains no larger than requested')
    p.add_argument('--host-gain-cache',action='store_true',help='Exact float64 gain blocks on host; retain only required phase-history derivatives')
    p.add_argument('--preinterpolate-delays',action='store_true',help='Reuse bitwise-verified delay/source interpolation within each RK4 stage; does not change the operator or acceptance gates')
    p.add_argument('--ritz-progress',action='store_true',help='Log unconverged ARPACK Ritz estimates for numerical diagnosis only')
    p.add_argument('--output-tag',default='',help='Distinct output identity for a numerical retry')
    main(p.parse_args())
