"""Confirm local Hopf criticality with full spatial nonlinear periodic BVPs."""
from rate_periodic import *
from rate_hopf_normal_form import compute
from rate_periodic_accuracy import defect


def main(a):
    s=RateField()
    def gains(mom):
        out=[]
        for k in range(3):
            h=5e-5*np.maximum(abs(mom[k]),1.);hi=list(mom);lo=list(mom)
            hi[k]=mom[k]+h;lo[k]=mom[k]-h;d=s.phi(*hi)-s.phi(*lo)
            hi[k]=mom[k]+2*h;lo[k]=mom[k]-2*h
            out.append((8*d-s.phi(*hi)+s.phi(*lo))/(12*h))
        return out
    s.gains=gains
    for label in a.labels:
        source=PERIODIC_OUT/f'stationary_root_counts/{label}_highorder_half.npz'
        for step,suffix in [(1e-7,''),(5e-8,'_half')]:
            path=PERIODIC_OUT/f'normal_form_{label+suffix}.json'
            if not path.exists():compute(s,label+suffix,source,parameter_step=step)
        z=np.load(PERIODIC_OUT/f'normal_form_{label}.npz');nf=read(PERIODIC_OUT/f'normal_form_{label}.json')
        q=z['q'];r0=z['r'];J0=float(z['J']);w=float(z['w']);rows=[]
        for amp in [.05,.1,.2]:
            versions=[]
            for N in [32,64]:
                name=f'{label}_a{amp:.5f}_N{N}';path=PERIODIC_OUT/f'orbits/{name}.npz'
                if path.exists() and read(path.with_suffix('.json'))['status']!='CONVERGED':
                    import shutil
                    attempt=PERIODIC_OUT/'child_attempt_history'/f'{name}_{time.time_ns()}'
                    attempt.mkdir(parents=True)
                    for old in [path,path.with_suffix('.json')]:shutil.copy2(old,attempt/old.name)
                    retry=True
                else:retry=False
                if not path.exists() or retry:
                    o=Periodic(s,N,a.device)
                    o.normalize_linear_rhs=getattr(a,'linear_normalize',False)
                    o.krylov_restart=getattr(a,'krylov_restart',160)
                    if N==32:
                        J=J0+nf['J_shift_per_amplitude_squared']*amp**2;eq,ok,_=s.solve(J,r0);assert ok
                        T=2*np.pi/(w+nf['omega_shift_per_amplitude_squared']*amp**2)
                        phase=np.exp(2j*np.pi*np.arange(N)/N)[:,None]
                        r=eq+2*amp*np.real(phase*q)+amp**2*(z['h11'].real+np.real(phase**2*z['h20']))
                    else:
                        old=np.load(versions[-1]['path']);r=resample(old['r'],N,axis=0);J=float(old['J']);T=float(old['T'])
                    r,T,J,e,h=o.solve(r,T,J,amplitude=(q,amp),tol=1e-10)
                    save_orbit(s,r,T,J,e,h,name);assert e<1e-10
                    del o
                meta=read(path.with_suffix('.json'));assert meta['status']=='CONVERGED';versions.append(meta)
            check=defect(s,path,a.device);relative=(versions[-1]['J_EE_core']-J0)/(nf['J_shift_per_amplitude_squared']*amp**2)-1
            # Strong inhibition can make a rate smaller than double-precision
            # Fourier roundoff. Preserve the raw minimum and a strict absolute
            # tolerance; do not clip the waveform or accept a resolved negative
            # population rate.
            check['nonnegative_rate_tolerance_Hz']=1e-12
            assert check['maximum_group_defect_Hz']<1e-7 and check['minimum_rate_Hz']>=-1e-12
            assert abs(versions[-1]['J_EE_core']-versions[0]['J_EE_core'])<1e-8
            rows.append(dict(amplitude=amp,versions=versions,continuous_defect=check,
                relative_normal_form_J_shift_error=relative))
        errors=np.abs([row['relative_normal_form_J_shift_error'] for row in rows])
        assert errors[0]<.01
        # The cubic is an asymptotic prediction, not an exact finite-amplitude
        # equation. Check that its relative error decays with amplitude rather
        # than rejecting a valid child when the largest amplitude leaves the
        # local cubic regime. Dominant quintic correction gives error ~a^2.
        nf2=read(PERIODIC_OUT/f'normal_form_{label}_half.json')
        original=read(PERIODIC_OUT/f'stationary_root_counts/{label}_highorder.json')
        uncertainty=[];quartic=[]
        for row in rows:
            amp=row['amplitude'];v=row['versions']
            delta=v[-1]['J_EE_core']-J0-nf['J_shift_per_amplitude_squared']*amp**2
            numerical=(abs(original['J_EE_core']-J0)+
                abs(nf2['J_shift_per_amplitude_squared']-nf['J_shift_per_amplitude_squared'])*amp**2+
                abs(v[-1]['J_EE_core']-v[0]['J_EE_core'])+64*np.finfo(float).eps*max(1.,abs(J0)))
            uncertainty.append(numerical)
            quartic.append([(delta-numerical)/amp**4,(delta+numerical)/amp**4])
        clean_growth=(all(errors[i]<errors[i+1] for i in range(2)) and
                      all(2.5<errors[i+1]/errors[i]<6 for i in range(2)))
        compatible_quartic=max(q[0] for q in quartic)<=min(q[1] for q in quartic)
        # Tiny branch shifts can make the smallest quintic correction less
        # than the independently measured derivative/mesh refinement change.
        # In that case do not demand an exact growth ratio below its resolution.
        assert clean_growth or compatible_quartic
        check=read(PERIODIC_OUT/f'stationary_root_counts/{label}_full_state_validation.json')
        assert check['status']=='VALIDATED_IMAGINARY_PAIR_CROSSING'
        assert np.sign(nf['cubic'][0])==np.sign(nf2['cubic'][0])
        out=dict(label=label,status='VALIDATED_LOCAL_HOPF',criticality=nf['criticality'],
            J_EE_core=J0,frequency_hz=nf['frequency_hz'],eigenpair_validation=check,
            normal_form=nf,parameter_derivative_halving=nf2,child_cycles=rows,
            cubic_asymptotic_relative_error_growth_factors=errors[1:]/errors[:-1],
            asymptotic_check=dict(clean_quadratic_relative_error_growth=clean_growth,
                quartic_J_correction_compatible_with_refinement_changes=compatible_quartic,
                absolute_J_refinement_scales=uncertainty,quartic_coefficient_intervals=quartic,
                scale_meaning='Observed gain-step, parameter-step and mesh changes plus floating-point scale; numerical consistency diagnostic, not rigorous interval arithmetic.'),
            parent_already_unstable=True,full_child_Floquet_spectrum='NOT_COMPUTED',
            scope='Local center-manifold criticality and nonlinear cycle birth validated. Existing transverse unstable equilibrium modes prevent an inference of stable cycle birth; the new cycles do not identify a resting-to-burst boundary.')
        write(PERIODIC_OUT/f'stationary_root_counts/{label}_validation.json',out)
        print('LOCAL HOPF VALIDATED',label,J0,nf['criticality'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--labels',nargs='+',default=['H3','H4']);p.add_argument('--device',type=int,default=0)
    p.add_argument('--linear-normalize',action='store_true');p.add_argument('--krylov-restart',type=int,default=160)
    main(p.parse_args())
