"""Check a dominant unstable multiplier without claiming a complete spectrum.

Strong transverse growth can amplify the numerical phase-tangent error by
many orders of magnitude. In that case, paired-step agreement and a relative
eigenpair residual can establish instability, but cannot locate the neutral
multiplier or classify the remaining spectrum.
"""
from rate_periodic import *


def main():
    p=argparse.ArgumentParser()
    p.add_argument('orbit_stem')
    p.add_argument('--label',required=True)
    p.add_argument('--coarse',type=float,default=.1)
    p.add_argument('--fine',type=float,default=.05)
    p.add_argument('--accuracy',required=True)
    a=p.parse_args()
    checks=[]
    for dt in [a.coarse,a.fine]:
        source=PERIODIC_OUT/'floquet'/f'{a.orbit_stem}_dt{dt:g}.json'
        q=read(source);v=np.asarray(q['multipliers'])
        v=v[:,0]+1j*v[:,1] if v.ndim==2 else v.astype(complex)
        i=int(np.argmax(abs(v)));mu=v[i]
        checks.append(dict(source=str(source),J_EE_core=q['J_EE_core'],
            T_ms=q['T_ms'],dt_ms=q['dt_ms'],multiplier=mu,
            relative_eigenpair_residual=q['residuals'][i]/max(1,abs(mu)),
            real_exponent_per_ms=float(np.log(abs(mu))/q['T_ms']),
            phase_tangent_defect=q['phase_tangent_relative_defect'],
            neutral_identified=q.get('identified_neutral_index') is not None))
    assert abs(checks[0]['J_EE_core']-checks[1]['J_EE_core'])<1e-12
    assert abs(checks[0]['T_ms']-checks[1]['T_ms'])<1e-8
    accuracy=read(Path(a.accuracy))
    profile=next(q for q in accuracy['checks'] if Path(q['orbit']).stem==a.orbit_stem)
    orbit_ok=(profile['maximum_group_defect_Hz']<.1 and
              max(profile['regional_defect_Hz'])<.001 and profile['minimum_rate_Hz']>=0)
    change=abs(checks[1]['multiplier']-checks[0]['multiplier'])/abs(checks[1]['multiplier'])
    passed=(orbit_ok and change<.01 and
            all(q['relative_eigenpair_residual']<1e-6 and abs(q['multiplier'])>1.1 for q in checks))
    result=dict(status='VALIDATED_DOMINANT_INSTABILITY' if passed else 'UNRESOLVED',
        orbit_stem=a.orbit_stem,J_EE_core=checks[-1]['J_EE_core'],T_ms=checks[-1]['T_ms'],
        checks=checks,relative_multiplier_step_change=change,
        continuous_orbit_check=profile,
        fine_e_folding_time_ms=1/checks[-1]['real_exponent_per_ms'],
        full_spectrum_validated=False,
        scope='A residual-checked, step-converged multiplier outside the unit circle establishes instability of this orbit only. This leading-mode calculation does not classify near-unit modes, count all unstable modes, or locate another bifurcation. Large phase error, when present, further limits near-neutral inference.')
    write(PERIODIC_OUT/(a.label+'.json'),result)
    print('ENDPOINT',result,flush=True)


if __name__=='__main__':main()
