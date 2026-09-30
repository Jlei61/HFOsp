"""Local torus direction from converged two-angle solutions, not IEI statistics."""
from plot_rate_periodic_completion import *


def main():
    roots=[read(PERIODIC_OUT/f'TR_A_B_{tag}_N128.json') for tag in ['highorder','highorder_half']]
    root=roots[-1];J0=root['J_EE_core'];slope=root['transversal_slope']['derivative_per_ms_per_J']
    rows=[read(f) for f in sorted((PERIODIC_OUT/'tori').glob('strict_TR_*Hz_N*.json'))]
    rows=[q for q in rows if q['status']=='CONVERGED'];fits=[]
    for mesh in [(64,8),(128,8),(64,16)]:
        subset=sorted([q for q in rows if (q['N_theta'],q['N_psi'])==mesh],key=lambda q:q['amplitude_hz'])
        for cutoff in [.02,.04]:
            rr=[q for q in subset if q['amplitude_hz']<=cutoff]
            if len(rr)<3:continue
            x=np.array([q['amplitude_hz']**2 for q in rr]);y=np.array([q['J_EE_core']-J0 for q in rr])
            # Rescale the amplitude to avoid an ill-conditioned Vandermonde matrix.
            scale=max(x);c=np.polynomial.polynomial.polyfit(x/scale,y,2)
            prediction=np.polynomial.polynomial.polyval(x/scale,c)
            fits.append(dict(mesh=list(mesh),maximum_amplitude_Hz=cutoff,points=len(rr),
                extrapolated_J_minus_TR=float(c[0]),quadratic_J_per_Hz2=float(c[1]/scale),
                quartic_J_per_Hz4=float(c[2]/scale**2),maximum_fit_error_J=float(max(abs(y-prediction)))))
    differences=[]
    for a,b in [((64,8),(128,8)),((64,8),(64,16))]:
        for lo in rows:
            if (lo['N_theta'],lo['N_psi'])!=a:continue
            hi=next((q for q in rows if (q['N_theta'],q['N_psi'])==b and q['amplitude_hz']==lo['amplitude_hz']),None)
            if hi: differences.append(dict(meshes=[list(a),list(b)],amplitude_hz=lo['amplitude_hz'],
                J_absolute_difference=abs(hi['J_EE_core']-lo['J_EE_core']),
                T_ms_absolute_difference=abs(hi['T_ms']-lo['T_ms']),
                modulation_period_ms_absolute_difference=abs(hi['modulation_period_ms']-lo['modulation_period_ms'])))
    defects=[read(f) for f in (PERIODIC_OUT/'torus_accuracy').glob('strict_TR_*json')]
    small=[q for q in fits if q['maximum_amplitude_Hz']==.02]
    assert small and all(q['quadratic_J_per_Hz2']<0 and abs(q['extrapolated_J_minus_TR'])<1e-11 for q in small)
    assert abs(roots[0]['J_EE_core']-J0)<1e-11 and slope>0
    assert any(q['maximum_group_defect_Hz']<1e-9 for q in defects)
    cJ=next(q['quadratic_J_per_Hz2'] for q in small if q['mesh']==[64,16])
    modechecks=[read(f) for f in sorted(PERIODIC_OUT.glob('TR_A_B_highorder_half_monodromy_check_N128_dt*.json'))]
    out=dict(status='LOCALLY_SUBCRITICAL_TORUS_SUPPORTED',label='TR_A_B',
        critical_point=root,root_gain_step_check_J=abs(roots[0]['J_EE_core']-J0),
        independent_full_state_mode_checks=modechecks,
        fits=fits,mesh_comparisons=differences,continuous_defects=defects,torus_solutions=rows,
        amplitude_definition='First slow-angle harmonic projected on the frozen critical rate eigenfunction, normalized to max absolute value 1 at Ntheta=64. Units Hz. Not a core-rate peak.',
        radial_normal_form=dict(equation='da/dt = a * (beta*(J-J_TR) + c*a^2 + higher-order terms)',
            beta_per_ms_per_J=slope,c_inferred_per_ms_per_Hz2=-slope*cJ,
            inference='Negative J(a)-J_TR curvature with beta>0 implies a radially unstable local torus on the transversely stable side of the parent cycle. Coefficient inferred from branch direction, not an independent adjoint normal-form calculation.'),
        interpretation='The first TR is locally subcritical. These weak two-frequency solutions do not establish a stable irregular-burst attractor or its global continuation.',
        stability_scope='Local radial instability inferred from nonzero branch curvature and transverse crossing. Full torus Lyapunov spectrum not computed.',
        reference='https://www.scholarpedia.org/article/Neimark-Sacker_bifurcation')
    write(PERIODIC_OUT/'TR_A_B_validation.json',out)
    print('TORUS VALIDATION',out['status'],'J',J0,'root difference',out['root_gain_step_check_J'],'cJ',cJ,'radial c',-slope*cJ,flush=True)


if __name__=='__main__':main()
