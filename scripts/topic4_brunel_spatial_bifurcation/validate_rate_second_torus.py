"""Nonlinear direction of the second torus bifurcation in full space."""
from plot_rate_periodic_completion import *
from scipy.optimize import linear_sum_assignment


def main():
    crossing=read(PERIODIC_OUT/'TR_A_return_validation.json');root=crossing['critical_point'];J0=root['J_EE_core']
    rows=[read(f) for f in (PERIODIC_OUT/'tori').glob('strict_TR2_*Hz_N*.json')]
    rows=[q for q in rows if q['status']=='CONVERGED'];fits=[]
    for mesh in [[64,8],[128,8],[64,16]]:
        rr=sorted([q for q in rows if [q['N_theta'],q['N_psi']]==mesh and q['amplitude_hz']<=.02],key=lambda q:q['amplitude_hz'])
        assert len(rr)>=3
        x=np.array([q['amplitude_hz']**2 for q in rr]);scale=max(x);y=np.array([q['J_EE_core']-J0 for q in rr])
        c=np.polynomial.polynomial.polyfit(x/scale,y,2)
        fits.append(dict(mesh=mesh,extrapolated_J_minus_TR=c[0],quadratic_J_per_Hz2=c[1]/scale,
            quartic_J_per_Hz4=c[2]/scale**2,amplitudes_hz=[q['amplitude_hz'] for q in rr]))
    assert all(q['quadratic_J_per_Hz2']>0 and abs(q['extrapolated_J_minus_TR'])<1e-10 for q in fits)
    defects=[read(f) for f in (PERIODIC_OUT/'torus_accuracy').glob('strict_TR2_*json')]
    assert defects and any(q['maximum_group_defect_Hz']<1e-8 for q in defects)
    beta=root['transversal_slope']['derivative_per_ms_per_J'];assert beta>0
    spectrum=PERIODIC_OUT/'poincare_floquet/TR_A_return_eval_J0.9457603915254_N64_dt0.05.json'
    rest_stable=None;other=None;spectra=[];matched_change=None
    target=complex(*root['multiplier']);targets=np.array([target,target.conjugate()])
    for dt in ['0.1','0.05']:
        path=spectrum.with_name(spectrum.name.replace('dt0.05','dt'+dt))
        if not path.exists():continue
        q=read(path);mu=np.array([complex(*v) for v in q['multipliers']])
        _,critical=linear_sum_assignment(abs(targets[:,None]-mu[None,:]))
        assert max(abs(targets-mu[critical]))<1e-3
        others=np.delete(mu,critical);rho=q['polynomial_filter_rho']
        # The filter suppresses adaptation multipliers near rho. The largest
        # returned noncritical value is NOT necessarily the largest of all
        # noncritical values. Under numerical LM coverage, unreturned values
        # obey |mu| <= (rho+sqrt(rho^2+4*t))/2, t=smallest returned |mu(mu-rho)|.
        threshold=q['smallest_returned_transformed_modulus']
        unreturned_radius=(rho+np.sqrt(rho*rho+4*threshold))/2
        spectra.append(dict(source=str(path),dt_ms=q['dt_ms'],critical_multipliers=mu[critical],
            other_returned_multipliers=others,largest_other_returned_modulus=float(max(abs(others))),
            unreturned_radius_from_numerical_LM_coverage=unreturned_radius,
            maximum_eigen_residual=max(q['residuals']),phase_defect=q['phase_tangent_relative_defect'],
            coverage_scope='Radius bound assumes numerically converged largest-modulus transformed eigenpairs; not a rigorous Arnoldi enclosure.'))
    if len(spectra)==2:
        a,b=[np.asarray(q['other_returned_multipliers']) for q in spectra]
        ii,jj=linear_sum_assignment(abs(a[:,None]-b[None,:]));matched_change=float(max(abs(a[ii]-b[jj])))
        other=spectra[-1]['largest_other_returned_modulus']
        rest_stable=bool(matched_change<1e-3 and all(max(q['largest_other_returned_modulus'],
            q['unreturned_radius_from_numerical_LM_coverage'])<.9 and q['maximum_eigen_residual']<1e-5 for q in spectra))
    out=dict(status='LOCALLY_SUPERCRITICAL_TORUS_SUPPORTED',label='TR_A_return',critical_point=root,
        fits=fits,torus_solutions=rows,continuous_defects=defects,
        radial_normal_form=dict(beta_per_ms_per_J=beta,c_inferred_per_ms_per_Hz2=-beta*fits[-1]['quadratic_J_per_Hz2'],
            meaning='Positive J(a)-J_TR curvature and beta>0 imply local radial stability. Inferred from branch direction, not an independently calculated adjoint cubic coefficient.'),
        parent_noncritical_spectrum_source=str(spectrum),parent_noncritical_modes_numerically_stable=rest_stable,
        largest_other_returned_multiplier_modulus=other,
        parent_spectrum_step_checks=spectra,matched_noncritical_step_change=matched_change,
        stability_scope='Local torus stability supported when parent noncritical modes are stable; no full torus Lyapunov spectrum or global stability continuation.',
        interpretation='Unlike TR1, TR2 has a supercritical small-amplitude torus branch. It is weak two-frequency activity, not a demonstrated irregular-burst state.')
    write(PERIODIC_OUT/'TR_A_return_nonlinear_validation.json',out)
    print('SECOND TORUS',out['status'],'cJ',fits[-1]['quadratic_J_per_Hz2'],'other spectrum stable',rest_stable,flush=True)


if __name__=='__main__':main()
