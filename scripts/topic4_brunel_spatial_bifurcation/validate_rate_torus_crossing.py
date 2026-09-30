"""Mesh, transversality and full-delay eigenfunction checks of a torus crossing."""
from plot_rate_periodic_completion import *


def main(label):
    lo=read(PERIODIC_OUT/f'{label}_N64.json');hi=read(PERIODIC_OUT/f'{label}_N128.json')
    checks=[read(PERIODIC_OUT/f'{label}_monodromy_check_N64_dt{dt}.json') for dt in ['0.1','0.05']]
    delta=abs(lo['J_EE_core']-hi['J_EE_core']);ratio=checks[0]['full_state_mode_relative_defect']/checks[1]['full_state_mode_relative_defect']
    mu=complex(*hi['multiplier']);lam=complex(*hi['lambda_per_ms']);slope=hi['transversal_slope']
    assert delta<1e-9 and abs(lam.real)<1e-10
    assert min(abs(mu**k-1) for k in range(1,5))>1e-3
    assert slope['real_exponents'][0]*slope['real_exponents'][1]<0
    assert checks[1]['full_state_mode_relative_defect']<1e-4 and 3<ratio<5
    row=dict(status='VALIDATED_TORUS_CROSSING',label=label,critical_point=hi,
        temporal_mesh_J_difference=delta,T_ms_difference=abs(lo['T_ms']-hi['T_ms']),
        independent_full_state_mode_checks=checks,step_halving_defect_reduction=ratio,
        nonresonance_distances=[abs(mu**k-1) for k in range(1,5)],
        criticality='NOT_COMPUTED',
        meaning='A second non-real unit-circle crossing on the returned H1 periodic branch after LPC2. Nonlinear torus direction/stability requires a separate calculation.',
        complete_inventory=False)
    write(PERIODIC_OUT/f'{label}_validation.json',row)
    print('CROSSING VALIDATED',label,hi['J_EE_core'],'mesh shift',delta,'mode reduction',ratio,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',default='TR_A_return');a=p.parse_args();main(a.label)
