"""Check the larger PD2 child endpoint without naming a new bifurcation."""
from rate_periodic import *


def main():
    p=PERIODIC_OUT;tag='PDupperextension_a40.00000_N4096'
    sources=[p/'floquet'/f'{tag}_dt{dt:g}.json' for dt in [.1,.05]]
    if not all(f.exists() for f in sources):
        write(p/'PD2_child_extension_stability.json',dict(status='PAIRED_STEP_PENDING',
            sources=[str(f) for f in sources],scope='A coarse-step negative multiplier alone does not confirm a new bifurcation.'))
        return
    checks=[]
    for source in sources:
        q=read(source);vals=np.array([complex(*v) for v in q['multipliers']])
        index=int(np.argmax(abs(vals)));mu=vals[index]
        checks.append(dict(source=str(source),J_EE_core=q['J_EE_core'],T_ms=q['T_ms'],
            multiplier=mu,relative_eigenpair_residual=q['residuals'][index]/max(1,abs(mu)),
            phase_relative_defect=q['phase_tangent_relative_defect']))
    coarse,fine=checks;change=abs(fine['multiplier']-coarse['multiplier'])
    margin=max(.002,4*change,4*fine['phase_relative_defect'])
    accuracy=read(p/'PD2_extended_child_accuracy.json');local=read(p/'PD_upper_child_classification.json')
    passed=(accuracy['status']=='PASS' and abs(fine['multiplier'])>1+margin and
        change/abs(fine['multiplier'])<.01 and
        all(v['relative_eigenpair_residual']<1e-6 for v in checks))
    stable=read(Path(local['child_floquet_source']));q=read(p/'PDupperchild_branch_N4096.json')
    result=dict(status='VALIDATED_ENDPOINT_INSTABILITY' if passed else 'UNRESOLVED',checks=checks,
        multiplier_step_change=change,margin=margin,continuous_accuracy_source=str(p/'PD2_extended_child_accuracy.json'),
        stable_child_J=stable['J_EE_core'],unstable_child_J=fine['J_EE_core'] if passed else None,
        stable_child_source=local['child_floquet_source'],
        candidate_type='Possible subsequent period doubling; antiperiodic crossing not yet solved',
        scope='Numerically stable smaller child and unstable larger child bound a stability change on the followed child family. A negative outside multiplier is a PD search seed, not proof of a unique -1 crossing, its location, or its criticality.')
    write(p/'PD2_child_extension_stability.json',result);print(result,flush=True)


if __name__=='__main__':main()
