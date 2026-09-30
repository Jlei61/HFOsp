"""Verify the short-period H1-return PD independently of long-cycle N gates.

Temporal resolution is certified by two meshes plus a continuous residual,
not by importing the N=2048 threshold used for the 762-ms parent cycle.
The full nine-state/delay-history antiperiodic mode is propagated separately.
No child criticality or global branch connection is inferred here.
"""
from rate_periodic import *


def main():
    label='PD_A_return'
    roots=sorted([read(f) for f in PERIODIC_OUT.glob(label+'_N*.json')],key=lambda q:q['N'])
    assert len(roots)>=2
    lo,q=roots[-2:];dj=abs(lo['J_EE_core']-q['J_EE_core'])
    assert q['N']>=2*lo['N'] and dj<1e-7
    assert q['antiperiodic_relative_residual']<1e-7 and abs(q['dborder_dJ'])>1e-6
    continuous=read(PERIODIC_OUT/f'{label}_continuous_defect_N{q["N"]}.json')
    assert Path(continuous['orbit']).resolve()==Path(q['orbit']).resolve()
    assert continuous['maximum_group_defect_Hz']<.001 and continuous['minimum_rate_Hz']>=0
    direct=sorted([read(f) for f in PERIODIC_OUT.glob(f'{label}_monodromy_check_N{q["N"]}_dt*.json')],key=lambda v:v['dt_ms'],reverse=True)
    assert len(direct)>=3
    for v in direct:assert Path(v['orbit']).resolve()==Path(q['orbit']).resolve()
    errors=np.array([v['minus_one_relative_defect'] for v in direct])
    phase=np.array([v['phase_relative_defect'] for v in direct])
    assert errors[-1]<1e-4 and phase[-1]<1e-4,(errors,phase)
    assert np.all(errors[:-1][-2:]/errors[1:][-2:]>3),errors
    assert np.all(phase[:-1][-2:]/phase[1:][-2:]>3),phase
    witnesses=[]
    for index in [138,142]:
        f=PERIODIC_OUT/f'floquet/H1_followup_arcAreturnStrong_{index:04d}_dt0.05.json'
        v=read(f);mu=np.array([complex(*x) for x in v['multipliers']]);i=np.argmin(abs(mu+1))
        assert abs(mu[i].imag)<1e-8 and v['residuals'][i]<1e-6
        witnesses.append(dict(source=str(f),J_EE_core=v['J_EE_core'],T_ms=v['T_ms'],
            multiplier=float(mu[i].real),eigen_residual=v['residuals'][i],
            largest_returned_multiplier_modulus=float(max(abs(mu)))))
    assert (witnesses[0]['multiplier']+1)*(witnesses[1]['multiplier']+1)<0
    s=RateField();u=np.load(PERIODIC_OUT/f'{label}_mode_N{q["N"]}.npz')['u']
    mass=s.geo['group_size']*s.E;energy=np.mean(u**2,axis=0)*mass
    by=[float(energy[s.geo['group_region']==k].sum()/energy.sum()) for k in range(3)]
    baseline=[float(mass[s.geo['group_region']==k].sum()/mass.sum()) for k in range(3)]
    result=dict(status='VALIDATED_PD',label='PD3',internal_label=label,
        J_EE_core=q['J_EE_core'],T_ms=q['T_ms'],N=q['N'],lower_mesh_N=lo['N'],
        mesh_J_difference=dj,antiperiodic_relative_residual=q['antiperiodic_relative_residual'],
        crossing_border_slope=q['dborder_dJ'],continuous_orbit_check=continuous,
        direct_monodromy_checks=direct,parent_crossing_witnesses=witnesses,
        E_rate_mode_energy_A_B_surround=by,E_population_fraction_A_B_surround=baseline,
        per_E_cell_mean_squared_mode_relative_to_network=[x/y for x,y in zip(by,baseline)],
        criticality='NOT_COMPUTED',parent_stability='ALREADY_UNSTABLE',
        scope='Mesh-converged antiperiodic root, transverse scalar border, and independent full-state -1 mode check with second-order step convergence. The parent is already unstable on both sides. No stable burst onset, child criticality, complete spectrum, or global connection is established.')
    child=PERIODIC_OUT/'PD_return_child_classification.json'
    if child.exists():
        c=read(child)
        assert abs(c['J_EE_core']-result['J_EE_core'])<1e-10
        result.update(criticality=c['status'],child_stability=c['child_stability'],
            child_classification_source=str(child))
        result['scope']=('Mesh-converged antiperiodic root and independently checked full-state -1 mode. '
            'The parent is already unstable; the linked child calculation establishes local criticality '
            'and inherited instability. No stable burst onset, complete spectrum or global connection is established.')
    followup=PERIODIC_OUT/(label+'_filter_state_followup.json')
    if followup.exists():
        f=read(followup)
        if f['status']=='REFINED_PARENT_AND_MODE_CHECKED':
            assert abs(f['J_EE_core']-result['J_EE_core'])<1e-12
            result.update(filter_state_followup=dict(source=str(followup),**f),accepted_parent_orbit=f['orbit'])
    write(PERIODIC_OUT/(label+'_validation.json'),result)
    print('H1 PD VALIDATION',result,flush=True)


if __name__=='__main__':main()
