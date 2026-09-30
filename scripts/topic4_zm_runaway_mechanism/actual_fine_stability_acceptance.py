"""Apply existing phase, residual and time-step gates to the actual-Z cycle."""
from pathlib import Path
import json


def main():
    out=Path(__file__).resolve().parents[2]/'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918'
    prefix='actual_fine_Z_fixedD_G8193_M65536_point0004_endpoint_dt'
    suffix='_quotient_chainphase_rk4_cubic_streamed_fastgrid_dominant_ncv12.json'
    files=[out/'floquet'/f'{prefix}{dt}{suffix}' for dt in ['0.025','0.0125']]
    rows=[json.loads(f.read_text()) for f in files]
    assert rows[0]['orbit']==rows[1]['orbit']
    gates=dict(phase=all(q['phase_valid'] and q['phase_defect']<.005 and abs(q['phase_projection']-1)<.003 for q in rows),
               eigen_residual=all(max(q['eigen_residuals'])<1e-5 for q in rows),
               refined_step=rows[1]['dt_ms']<.51*rows[0]['dt_ms'],
               multiplier_agreement=abs(rows[0]['max_transverse_modulus']-rows[1]['max_transverse_modulus'])<.005,
               stable=all(q['max_transverse_modulus']<.999 for q in rows))
    result=dict(status='STABLE_WITH_STEP_REFINEMENT' if all(gates.values()) else 'ACCEPTANCE_NOT_MET',
        D=rows[0]['D'],T_ms=rows[0]['T_ms'],orbit=rows[0]['orbit'],gates=gates,
        evidence=[str(f) for f in files],multipliers=[q['multipliers'] for q in rows],
        phase_defects=[q['phase_defect'] for q in rows],
        scope='Leading transverse Arnoldi multiplier on this periodic solution and this actual fine spatial-Z slice; held Z, dynamic M. Not onset certification or complete branch coverage.')
    (out/'floquet/actual_fine_D14322168_stability_acceptance.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
