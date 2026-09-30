"""Apply the same phase and time-refinement gates to one periodic solution."""
from common import *
import argparse


def main(a):
    files=[Path(f).resolve() for f in a.files]
    rows=sorted([read(f) for f in files],key=lambda q:-q['dt_ms'])
    assert len(rows)==2
    assert Path(rows[0]['orbit']).resolve()==Path(rows[1]['orbit']).resolve()
    assert all(q['phase_quotient'] for q in rows)
    stable=all(q['max_transverse_modulus']<.999 for q in rows)
    unstable=all(q['max_transverse_modulus']>1.001 for q in rows)
    gates=dict(phase=all(q['phase_valid'] and q['phase_defect']<.005 and
        abs(q['phase_projection']-1)<.003 for q in rows),
        eigen_residual=all(max(q['eigen_residuals'])<1e-5 for q in rows),
        refined_step=rows[1]['dt_ms']<.51*rows[0]['dt_ms'],
        multiplier_agreement=abs(rows[0]['max_transverse_modulus']-
            rows[1]['max_transverse_modulus'])<.005,
        same_stability_side=stable or unstable)
    status=(('STABLE' if stable else 'UNSTABLE')+'_WITH_STEP_REFINEMENT') if all(gates.values()) else 'ACCEPTANCE_NOT_MET'
    q=dict(status=status,D=rows[0]['D'],T_ms=rows[0]['T_ms'],orbit=rows[0]['orbit'],
        gates=gates,evidence=[str(f) for f in files],multipliers=[r['multipliers'] for r in rows],
        phase_defects=[r['phase_defect'] for r in rows],
        scope='Sampled leading transverse Arnoldi multiplier of this held-Z, dynamic-M periodic solution. No branch completeness or onset type follows from one stable/unstable point.')
    write(OUT/'floquet'/a.output,q);log('CYCLE STABILITY ACCEPTANCE',q)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('files',nargs=2)
    p.add_argument('--output',required=True)
    main(p.parse_args())
