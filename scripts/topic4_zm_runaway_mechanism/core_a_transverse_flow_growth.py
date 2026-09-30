"""Phase-projected variational growth along an uninterrupted actual flow.

The nominal delayed spatial trajectory is never reset to a return section.
This finite-time diagnostic tests the stable-cycle hypothesis; it cannot name
a bifurcation or establish asymptotic chaos on its own.
"""
from common import np, read, write, log
from onset_state_continuation import build
from onset_variational_return import Coordinates
from onset_tangent_cuda import Tangent
from fine_rate_frozen_Z_fields import capture, restore
from pathlib import Path
import argparse, os, math, time


def flow_direction(e, c):
    base = capture(e); x = c.pack(base); diff = []
    for _ in range(4):
        e.step(); e.cp.cuda.get_current_stream().synchronize()
        diff.append(c.pack(capture(e))-x)
    restore(e, base)
    restored = capture(e)
    assert all(np.array_equal(v, restored[k]) for k, v in base.items())
    velocities = [sum((-1)**(j+1)*math.comb(order, j)/j*diff[j-1]
                      for j in range(1, order+1))/e.dt for order in [2, 4]]
    v = velocities[-1]
    error = float(np.linalg.norm(velocities[0]-v)/np.linalg.norm(v))
    return v/np.linalg.norm(v), error, base


def set_current_tangent(t, c, v):
    # Coordinates.set_tangent uses its stored ring origin. It must follow the
    # actual current clock when renormalizing an uninterrupted trajectory.
    c.tick = int(t.e.local.clock.get()[0])
    c.set_tangent(t, v)
    assert np.linalg.norm(c.tangent(t)-v) < 1e-12


def main(a):
    reference = Path(a.reference).resolve()
    contract = read(reference / 'contract.json')
    assert read(reference / 'jobs.json')['status'] == 'COMPLETE'
    assert contract['target_native_time_ms'] is None, 'Use an exact unmodified source field in this diagnostic'
    assert contract['dt_ms'] == contract['source_dt_ms']
    out = Path(a.destination).resolve(); out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'jobs.json').exists()
    write(out / 'contract.json', dict(reference=str(reference), source=contract['source'],
          dt_ms=contract['dt_ms'], duration_ms=a.duration, discard_ms=2000,
          question='Does the actual short-event flow have persistent transverse expansion, after removing local time-shift perturbations without resetting the nominal network?',
          method='Original full delayed-state variational equations; project onto the plane perpendicular to the fourth-order forward numerical flow direction every100ms in fixed cell-weighted coordinates, then normalize. Recompute only the phase direction; no physical coordinate/state/parameter reduction. Compare second/fourth-order flow directions and exact original saved1s full checkpoints.',
          safeguards='Every phase readout restores the complete nominal state bitwise. Tangent history is reinserted at the current actual ring-clock origin. Existing full derivative validation is retained. Full unprojected gain, removed phase component and projection precision are saved.',
          interpretation='Finite-time phase-orthogonal growth only. A positive value alone proves neither asymptotic chaos nor a crisis. A negative value alone certifies neither a closed cycle nor complete stability. Numerical flow-direction error must be inspected.', model_promoted=False))
    jobs = dict(status='RUNNING', pid=os.getpid()); write(out / 'jobs.json', jobs)
    started = time.time()
    try:
        e = build(a.device, contract['dt_ms']); base = dict(np.load(contract['source']))
        restore(e, base); t = Tangent(e); t.graph(); c = Coordinates(base, e.s)
        # Independent nominal/tangent replay before the phase projections.
        e.chunk(); want = capture(e); restore(e, base); t.chunk(); got = capture(e)
        assert all(np.array_equal(v, got[k]) for k, v in want.items())
        restore(e, base); phase, phase_error, _ = flow_direction(e, c)
        rng = np.random.default_rng(9250739); v = rng.normal(size=c.size)
        v.reshape(-1, c.P)[4, ~e.s.E] = 0
        v -= phase*(phase@v); v /= np.linalg.norm(v); set_current_tangent(t, c, v)
        rows = []; parity = []
        for k in range(a.duration//100):
            for _ in range(10): t.chunk()
            phase, phase_error, state = flow_direction(e, c)
            v = c.tangent(t); raw = float(np.linalg.norm(v)); component = float(phase@v)
            v -= component*phase; projected = float(np.linalg.norm(v))
            assert np.isfinite(projected) and projected > 0
            v /= projected; set_current_tangent(t, c, v)
            row = dict(elapsed_ms=(k+1)*100, raw_gain=raw, projected_gain=projected,
                       removed_phase_component=component, log_gain=float(np.log(projected)),
                       phase_order2_vs4_relative_error=phase_error,
                       projected_phase_residual=float(abs(phase@v)))
            rows.append(row)
            if (k+1)%10 == 0:
                ms=(k+1)*100; exact=dict(np.load(reference/f'checkpoint{ms}.npz'))
                identical=all(np.array_equal(v, exact[key]) for key, v in state.items())
                parity.append(dict(time_ms=ms, full_nominal_bitwise=identical)); assert identical
                write(out/'progress.json', rows); write(out/'nominal_parity.json', parity)
                jobs.update(completed_ms=ms); write(out/'jobs.json', jobs)
                log('UNINTERRUPTED TRANSVERSE GROWTH', ms,
                    sum(q['log_gain'] for q in rows[-10:]), max(q['phase_order2_vs4_relative_error'] for q in rows[-10:]))
        usable = rows[20:]
        write(out/'result.json', dict(status='FINITE_TIME_TRANSVERSE_FLOW_DIAGNOSTIC_COMPLETE',
              rows=rows, full_nominal_checkpoints_bitwise=True,
              finite_time_growth_per_s=sum(q['log_gain'] for q in usable)/(len(usable)*.1),
              max_phase_direction_disagreement=max(q['phase_order2_vs4_relative_error'] for q in rows),
              seconds=time.time()-started, bifurcation_type='NOT_ESTABLISHED', model_promoted=False))
        np.savez_compressed(out/'final_state.npz', **capture(e))
        jobs['status']='COMPLETE'; write(out/'jobs.json', jobs)
    except BaseException as exc:
        jobs.update(status='FAILED', error=repr(exc)); write(out/'jobs.json', jobs); raise


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('reference'); p.add_argument('--destination', required=True)
    p.add_argument('--duration', type=int, default=5000); p.add_argument('--device', type=int, default=1)
    a=p.parse_args(); assert 3000 <= a.duration <= 10000 and a.duration%1000 == 0; main(a)
