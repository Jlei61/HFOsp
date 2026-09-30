"""Align the period guess of a refined complete-state cycle seed.

Only the scalar initial period guess changes. The full refined starting
state, slow fields and model are identical to the existing fine Newton run.
"""
from common import np, read, write, log
from onset_state_continuation import build
from onset_cubic_section import CubicSectionReturn
from onset_period_return import errors, dynamical_state
from pathlib import Path
import argparse, os, time


def main(a):
    parent=Path(a.parent).resolve();out=parent/'period_alignment';out.mkdir(exist_ok=True)
    assert not (out/'jobs.json').exists()
    contract=read(parent/'contract.json');row=read(parent/'iterations.json')[0]
    source=parent/'iteration00/node00.npz';T=row['period_ms'];dt=contract['dt_ms']
    write(out/'contract.json',dict(source=str(source),dt_ms=dt,period_guess_ms=T,
        search_halfwidth_ms=a.halfwidth,
        question='Does a period shift account for the large refined-mesh return error before full Newton correction?',
        method='One existing full-state cubic Poincare return around the old period, with the identical refined starting state. The return uses actual original integration and the original four-step crossing interpolant. A corrected period guess alone is not a periodic root.',
        decision='Compare original full six-block return residual against the saved initial fixed-T residual. If at least tenfold smaller, use the same starting state with this aligned period for a new declared fine corrector. Otherwise resume the paused original corrector. Do not infer orbit disappearance or a bifurcation from this test.',
        model='Unchanged complete spatial field, all conditional Z held and every E M dynamic, constant original mean input, locked physical private variance and response.',model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid());write(out/'jobs.json',jobs);started=time.time()
    try:
        e=build(a.device,dt);base=dict(np.load(source));A=CubicSectionReturn(base,e,T,a.halfwidth)
        assert A.admissible(A.xref)
        y,meta=A(A.xref);returned=A.state(y)
        check=errors(dynamical_state(base),dynamical_state(returned),e.s.sizes/e.s.sizes.sum())
        ratio=check['combined_relative_rms']/row['whole_cycle']['combined_relative_rms']
        np.savez_compressed(out/'return_state.npz',**returned)
        write(out/'result.json',dict(status='PERIOD_ALIGNMENT_DIAGNOSTIC_COMPLETE',**meta,
            old_period_ms=T,period_shift_ms=meta['period_ms']-T,**check,
            fixed_period_residual=row['whole_cycle']['combined_relative_rms'],residual_ratio=ratio,
            improves_seed_at_least_tenfold=bool(ratio<.1),seconds=time.time()-started,
            bifurcation_type='NOT_ESTABLISHED',model_promoted=False))
        jobs.update(status='COMPLETE');write(out/'jobs.json',jobs);log('FINE PERIOD ALIGNMENT',meta,check,ratio)
    except BaseException as exc:
        jobs.update(status='FAILED',error=repr(exc));write(out/'jobs.json',jobs);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('parent');p.add_argument('--device',type=int,default=0)
    p.add_argument('--halfwidth',type=float,default=50.);main(p.parse_args())
