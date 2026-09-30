"""Check short-state recurrence without repeatedly resetting a return map.

Each tested return starts at the same actual uninterrupted trajectory state.
This distinguishes a gross doubled-period candidate from interpolation error;
it is not a periodic-orbit correction or a stability certificate.
"""
from common import np, read, write, log
from onset_state_continuation import build
from onset_cubic_section import CubicSectionReturn
from onset_period_return import dynamical_state, errors
from pathlib import Path
import argparse, os


def main(a):
    source = Path(a.source).resolve()
    out = Path(a.destination).resolve(); out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'jobs.json').exists()
    write(out / 'contract.json', dict(source=str(source), dt_ms=a.dt, period_seed_ms=a.period,
          question='Does the actual short-event trajectory repeat after three bursts or only after six, following the candidate negative-mode crossing?',
          method='Two original uninterrupted flow returns from exactly the same full trajectory state, at T and2T. Cubic interpolation only at each terminal crossing; no intermediate state reset. Original full phase plane and full canonical delay/M coordinates.',
          scope='Measured recurrence errors only; phase/interpolation/mesh qualification and Floquet remain required. A closer2T return does not by itself prove period doubling.', model_promoted=False))
    jobs = dict(status='RUNNING', pid=os.getpid()); write(out / 'jobs.json', jobs)
    e = build(a.device, a.dt); base = dict(np.load(source)); rows = []
    for multiple in [1, 2]:
        A = CubicSectionReturn(base, e, multiple*a.period, 3.)
        try:
            y, meta = A(A.xref)
            terminal = A.state(y)
            check = errors(dynamical_state(base), dynamical_state(terminal), e.s.sizes/e.s.sizes.sum())
            row = dict(multiple=multiple, **meta, **check,
                       endpoint_admissible=bool(A.admissible(y)))
            np.savez_compressed(out / f'return{multiple}.npz', **terminal)
        except (RuntimeError, AssertionError) as exc:
            row = dict(multiple=multiple, local_return_error=repr(exc),
                       interpretation='A section-window miss or inadmissible interpolation is not a bifurcation.')
        rows.append(row); write(out / 'progress.json', rows)
        log('NATURAL SHORT RETURN', multiple, row.get('period_ms'), row.get('combined_relative_rms'), row.get('local_return_error'))
    write(out / 'result.json', dict(status='ACTUAL_TRAJECTORY_RECURRENCE_CHECK_COMPLETE', rows=rows,
          physical_periodic_type='NOT_ESTABLISHED', model_promoted=False))
    jobs['status'] = 'COMPLETE'; write(out / 'jobs.json', jobs)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('source'); p.add_argument('--destination', required=True)
    p.add_argument('--period', type=float, required=True); p.add_argument('--dt', type=float, default=.05)
    p.add_argument('--device', type=int, default=1); main(p.parse_args())
