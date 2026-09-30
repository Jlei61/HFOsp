"""Collect three already running checks without launching numerical experiments.

Only completed result files count; progress/Ritz files never supply scientific
acceptance. Each numerical check keeps its own scope. No LPC is auto-certified.
"""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918'
PREFIX = 'native_T2644_G16875_M262144_point0000_endpoint_dt'
SUFFIX = '_quotient_chainphase_rk4_cubic_streamed_fastgrid_hostgains_preinterp_'
FILES = {
    'fine_spectrum': OUT / 'floquet' / (PREFIX + '0.000390625' + SUFFIX + 'generic3.json'),
    'fine_family': OUT / 'periodic/native_family_flow_consistency/dt0.000390625.json',
    'relocated_tangent': OUT / 'periodic/native_turn_relocated_fine_tangent/result.json',
}
PIDS = {'fine_spectrum': 3875227, 'fine_family': 247078, 'relocated_tangent': 396788}
TOKENS = {'fine_spectrum': 'endpoint_floquet.py',
          'fine_family': 'native_family_flow_consistency.py',
          'relocated_tangent': 'fixed_period_tangent.py'}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def process_state(key):
    registry = OUT / 'pending_critical_process_registry.json'
    override = read(registry).get(key, {}) if registry.exists() else {}
    pid = override.get('pid', PIDS[key])
    try:
        proc = Path(f'/proc/{pid}')
        command = (proc / 'cmdline').read_bytes().decode().replace('\0', ' ')
        if TOKENS[key] in command:
            state = (proc / 'stat').read_text().split(') ', 1)[1].split()[0]
            return dict(status='PAUSED_MEMORY_SCHEDULING' if state in ('T', 't') else 'RUNNING', pid=pid)
        if override.get('queued') and (proc / 'fd/1').resolve() == (OUT / override['log']).resolve():
            return dict(status='QUEUED_MEMORY_RELEASE', pid=pid)
    except (FileNotFoundError, ProcessLookupError):
        pass
    stops = OUT / 'pending_critical_controlled_stops.json'
    stopped = read(stops).get(key, {}) if stops.exists() else {}
    if stopped.get('pid') == pid and stopped.get('status') in ('STOP_REQUESTED', 'STOPPED'):
        return dict(status='INCOMPLETE_CONTROLLED_STOP', pid=pid,
                    reason=stopped['reason'], record=str(stops.relative_to(OUT)))
    return dict(status='PROCESS_ENDED_WITHOUT_RESULT', pid=pid)


def check(key, x):
    if key == 'fine_spectrum':
        coarse = read(OUT / 'floquet' / (PREFIX + '0.00078125' + SUFFIX + 'branchguess.json'))
        c = read(OUT / 'native_T2644_two_step_spectrum_contract.json')['criteria']
        rows = [coarse, x]
        residuals = [v for q in rows for v in q['eigen_residuals']]
        radii = [q['max_transverse_modulus'] for q in rows]
        stable = all(r < .999 for r in radii)
        unstable = all(r > 1.001 for r in radii)
        gates = dict(
            same_orbit=(ROOT / x['orbit']).resolve() == (ROOT / coarse['orbit']).resolve(),
            phase=all(q['phase_quotient'] and q['phase_valid'] and
                      q['phase_defect'] < c['phase_defect'] and
                      abs(q['phase_projection'] - 1) < c['phase_projection_distance_from_one']
                      for q in rows),
            independent_residual=bool(residuals) and all(math.isfinite(v) and v < c['independent_eigen_residual'] for v in residuals),
            step=x['dt_ms'] < c['ratio_actual_dt_less_than'] * coarse['dt_ms'],
            leading_modulus_agreement=abs(radii[0] - radii[1]) < c['absolute_leading_modulus_difference'],
            same_side=stable or unstable,
            fine_requested_three=x['requested_eigenpairs'] == 3,
            fine_returned_at_least_three=len(x['multipliers']) >= 3,
        )
        verdict = ('STABLE' if stable else 'UNSTABLE') + '_WITH_STEP_REFINEMENT' if all(gates.values()) else 'ACCEPTANCE_NOT_MET'
        detail = dict(max_transverse_moduli=radii, phase_defects=[q['phase_defect'] for q in rows],
                      max_independent_residual=max(residuals),
                      scope='Sampled leading transverse modes of this one held-Z/dynamic-M old-v3 orbit, not branch completeness or onset classification.')
    elif key == 'fine_family':
        c = read(OUT / 'native_family_flow_consistency_contract.json')['acceptance']
        components = [x['state_component_relative'][j] for j in x['active_state_components']]
        gates = dict(phase=x['phase_error'] < c['phase_relative_max'] and
                     abs(x['phase_projection'] - 1) < c['phase_projection_distance_max'],
                     family=x['family_return_relative'] < c['family_relative_max'],
                     history=x['family_history_relative'] < c['family_history_relative_max'],
                     components=bool(components) and max(components) < c['active_state_component_max'])
        verdict = 'FLOW_CONSISTENCY_PASS' if all(gates.values()) else 'FLOW_CONSISTENCY_FAIL'
        assert x['status'] == verdict
        detail = dict(phase_error=x['phase_error'], family_return_relative=x['family_return_relative'],
                      history_relative=x['family_history_relative'],
                      scope='Differentiated period-return identity includes constant spatial parameter forcing. Not a homogeneous Floquet eigenvector or a fold certificate.')
    else:
        c = read(OUT / 'native_fine_turn_relocation_contract.json')
        gate = c['acceptance']
        gates = dict(target_period=abs(x['T_ms'] - c['predicted_T_ms']) < 1e-10,
                     root=x['bvp_residual_hz'] < gate['root_residual_hz'],
                     derivative=abs(x['dD_dT']) < gate['target_derivative_abs'],
                     augmented=x['augmented_target_relative_residual'] < gate['full_augmented_tangent_relative'])
        verdict = 'GEOMETRIC_TURN_LOCALIZED' if all(gates.values()) else 'GEOMETRIC_TURN_REFINEMENT_REQUIRED'
        detail = dict(D=x['D'], T_ms=x['T_ms'], dD_dT=x['dD_dT'],
                      augmented_target_relative_residual=x['augmented_target_relative_residual'],
                      scope='One geometric turning point on fixed1mm old-v3 conditional family. A physical cycle fold still needs critical multiplier and nondegeneracy evidence.')
    return dict(status=verdict, gates=gates, **detail)


def collect():
    rows = {}
    for key, path in FILES.items():
        if path.exists():
            try:
                rows[key] = dict(source=str(path.relative_to(OUT)), **check(key, read(path)))
            except json.JSONDecodeError:
                rows[key] = dict(status='RESULT_WRITE_PENDING')
        else:
            rows[key] = process_state(key)
    pending = any(q['status'] in ('RUNNING', 'RESULT_WRITE_PENDING', 'PAUSED_MEMORY_SCHEDULING', 'QUEUED_MEMORY_RELEASE') for q in rows.values())
    incomplete = any(q['status'] == 'INCOMPLETE_CONTROLLED_STOP' for q in rows.values())
    status = 'PENDING_EXISTING_JOBS' if pending else ('BOUNDED_CHECKS_CLOSED_WITH_INCOMPLETE' if incomplete else 'BOUNDED_RESULTS_COLLECTED')
    return dict(updated_local=datetime.datetime.now().astimezone().isoformat(),
                status=status, checks=rows,
                onset_bifurcation_type='NOT_ESTABLISHED', model_promoted=False, launches=0,
                scope='Completed files only. Collects existing contracts; no tolerance changes, model updates, branch launches, or automatic goal completion.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--watch-seconds', type=int, default=0)
    args = p.parse_args()
    assert 0 <= args.watch_seconds <= 21600
    deadline = time.monotonic() + args.watch_seconds
    signature = None
    while True:
        q = collect()
        write(OUT / 'pending_critical_checks_completion.json', q)
        sig = json.dumps(q['checks'], sort_keys=True)
        if sig != signature:
            print(q['updated_local'], {k: v['status'] for k, v in q['checks'].items()}, flush=True)
            signature = sig
        if q['status'] != 'PENDING_EXISTING_JOBS' or time.monotonic() >= deadline:
            return
        time.sleep(min(30, max(0, deadline - time.monotonic())))


if __name__ == '__main__':
    main()
