"""Collect distinct tests without turning an incomplete test into acceptance.

The unstable-cycle step refinement and the fine-mesh turn geometry may finish
independently. Their execution status is separate from bifurcation acceptance.
"""
from pathlib import Path
import json
import math
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    lower = read(OUT / 'floquet/rate_near_lower_stability_acceptance.json')
    stem = 'rate_upper_G16385_M65536_point0000_endpoint_dt'
    suffix = '_quotient_chainphase_rk4_cubic_streamed_fastgrid.json'
    paths = [OUT / 'floquet' / (stem + dt + suffix) for dt in ['0.003125', '0.0018']]
    spectra = [read(p) for p in paths if p.exists()]
    upper = dict(status='STEP_REFINEMENT_PENDING', sources=[str(p) for p in paths],
                 completed_spectra=len(spectra),
                 scope='This upper orbit only; stability is not copied to unsampled segments')
    failed_phase=paths[-1].with_name(paths[-1].stem+'.phase.json')
    if failed_phase.exists() and len(spectra)<2:
        phase=read(failed_phase)
        if not phase['phase_valid']:
            upper.update(status='PHASE_REFINEMENT_FAILED',failed_refinement=phase,
                         interpretation='Coarse instability is not yet a converged stability result')
    if len(spectra) == 2:
        same = Path(spectra[0]['orbit']).resolve() == Path(spectra[1]['orbit']).resolve()
        valid = same and all(q['phase_valid'] and max(q['eigen_residuals']) < 1e-5
                             for q in spectra)
        radii = [q['max_transverse_modulus'] for q in spectra]
        agreement = abs(radii[0] - radii[1])
        passed = valid and agreement < .005 and all(q > 1.001 for q in radii)
        upper.update(status='UNSTABLE_WITH_STEP_REFINEMENT' if passed else 'NEEDS_REFINEMENT',
                     orbit=spectra[0]['orbit'], D=spectra[0]['D'], T_ms=spectra[0]['T_ms'],
                     max_transverse_moduli=radii, step_refinement_difference=agreement,
                     actual_dt_ms=[q['dt_ms'] for q in spectra],
                     phase_defects=[q['phase_defect'] for q in spectra],
                     time_step_reduction_factor=spectra[0]['dt_ms']/spectra[1]['dt_ms'])
        write(OUT / 'floquet/rate_near_upper_stability_acceptance.json', upper)
    drift = next(q for q in read(OUT / 'cycle_slow_drift.json')['rows'] if q['family'] == 'rate')
    baseline = read(OUT / 'floquet/rate_seed_stability_acceptance.json')
    # Read the spectrum directly so no acceptance-file field alias is assumed.
    spec = read(baseline['sources'][-1])
    relaxation = -(spec['T_ms']/1000) / math.log(spec['max_transverse_modulus'])
    Dc = read(OUT / 'periodic/rate_turn_center_G16385_M65536/point0000.json')['D']
    drift_time = (Dc-drift['D'])/drift['mean_D_drift_per_s']
    scale = dict(baseline_cycle_relaxation_seconds=relaxation,
                 baseline_mean_D_drift_per_second=drift['mean_D_drift_per_s'],
                 baseline_D=drift['D'], affine_turn_candidate_D=Dc,
                 constant_initial_drift_time_to_affine_D_seconds=drift_time,
                 time_ratio=drift_time/relaxation,
                 interpretation='Local timescale comparison only. The drift and its spatial direction change. '
                                'This does not predict the actual onset time and does not establish rate-induced tipping. '
                                'Quasistatic tracking of the conditional branch is not justified by these scales.')
    write(OUT / 'conditional_cycle_drift_timescale.json', scale)
    q = dict(updated_local=datetime.now().astimezone().isoformat(), lower_cycle=lower,
             upper_cycle=upper, conditional_timescale=scale,
             bifurcation_candidate='LIMIT_CYCLE_FOLD',
             bifurcation_acceptance='PENDING_CRITICAL_GEOMETRY_AND_TRANSVERSE_CHECKS',
             fine_actual_Z_path='Separate held-field controls; no certified fold on that curved trajectory',
             native_SNN_bifurcation='NOT_ESTABLISHED', Z='held for conditional spectra', M='dynamic')
    write(OUT / 'current_cycle_evidence.json', q)
    print(json.dumps(q, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
