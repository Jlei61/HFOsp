"""One shared clinical-interval initial mean, with the same between-event OU SDE.

An exponentially decaying deterministic mean plus stationary zero-mean OU is
exactly an OU with a shifted Gaussian initial mean. Clinical offsets are past
observations; no future onset or seizure type determines the mean. This is a
statistical initial-law diagnostic, not a learned biological reset mechanism.
"""
import sys, json, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor, as_completed
from scripts.patient_state_v1.common import RUN, write_json
from scripts.patient_state_v1.model import laplace, filter_adf, slice_data
from scripts.patient_state_v1.advanced_controls import history_data
from scripts.patient_state_v1.analyze_first import best_fits
from scripts.patient_state_v1.review_predictions import selected
from scripts.patient_state_v1.review_activity_clock import paired

OUT = RUN/'learned_initial_offset_v1_36'

def inputs():
    d = dict(np.load(RUN/'observations.npz'))
    origin = float(d['origin_epoch'])
    seizures = json.loads((RUN/'seizures.json').read_text())
    starts = np.array([0.]+[(s['offset']-origin)/3600 for s in seizures])
    d['initial_age'] = d['t']-starts[d['epoch']]
    assert np.all(d['initial_age'] >= -1e-7)
    return d

def design(d, tau):
    q = history_data(d)
    q['x'][:, 2] = np.where(d['epoch'] > 0, np.exp(-np.maximum(d['initial_age'], 0)/tau), 0.)
    return q

def expand(theta, history):
    return np.asarray(theta) if history else np.r_[theta[0], 0., theta[1:]]

def source(scope, history):
    return selected(RUN/'advanced_controls_v1_2/fits', scope, 'ou_history')[0] if history else best_fits()[scope, 'ou']

def worker(job):
    path = OUT/'fits'/f"{job['id']}.json"
    if path.exists(): return json.loads(path.read_text())
    started = time.time()
    try:
        d = inputs(); train = slice_data(d, 0, job['end']); history = job['history']
        old = source(job['scope'], history)['theta']
        initial = np.r_[old[:-2], job['initial_offset'], old[-2:]]
        bounds = [(-8, 8)]*(2 if history else 1)+[(-5, 5), (np.log(1/60), np.log(24)), (np.log(.01), np.log(5))]
        def objective(t):
            ll = laplace(expand(t, history), design(train, np.exp(t[-2])), True)
            return -ll+.5*t[-3]**2
        opt = minimize(objective, initial, method='L-BFGS-B', bounds=bounds, options={'maxiter':180, 'ftol':1e-10, 'eps':2e-5, 'maxls':25})
        trace = filter_adf(expand(opt.x, history), design(d, np.exp(opt.x[-2])), True, order=64)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path.with_suffix('.npz'), **trace)
        result = dict(status='COMPLETE', job=job, theta=opt.x, penalized_loglik=-opt.fun, loglik=-opt.fun+.5*opt.x[-3]**2, success=bool(opt.success), message=str(opt.message), nfev=int(opt.nfev), elapsed=time.time()-started)
    except Exception:
        result = dict(status='FAILED', job=job, traceback=traceback.format_exc(), elapsed=time.time()-started)
    write_json(path, result)
    return result

def review():
    best = {}
    for path in (OUT/'fits').glob('*.json'):
        row = json.loads(path.read_text()); assert row['status'] == 'COMPLETE'
        key = (row['job']['scope'], row['job']['history'])
        if key not in best or row['penalized_loglik'] > best[key][1]['penalized_loglik']: best[key] = path, row
    assert len(best) == 8
    d = inputs(); parameters = []; predictions = []
    for (scope, history), (path, row) in sorted(best.items()):
        theta = expand(row['theta'], history); baseline = source(scope, history)
        parameters.append(dict(scope=scope, history=history, b=theta[0], gamma=theta[1], initial_offset=theta[2], tau_minutes=np.exp(theta[3])*60, sd=np.exp(theta[4]), loglik=row['loglik'], penalized_loglik=row['penalized_loglik'], gain_baseline=row['loglik']-baseline['loglik'], success=row['success'], source=str(path)))
        if scope == 'full': continue
        lo, hi = row['job']['end'], row['job']['test_end']
        p = np.clip(np.load(path.with_suffix('.npz'))['predict_tb'][lo:hi], 1e-12, 1-1e-12); y = d['y'][lo:hi]
        predictions.append(pd.DataFrame(dict(model='initial_offset_history' if history else 'initial_offset', fold=int(scope[4:]), index=np.arange(lo, hi), hour=d['t'][lo:hi], y=y, p_tb=p, score=y*np.log(p)+(1-y)*np.log1p(-p))))
    pd.DataFrame(parameters).to_csv(OUT/'selected_parameters.csv', index=False)
    new = pd.concat(predictions, ignore_index=True); new.to_csv(OUT/'forward_predictions.csv.gz', index=False)
    old = pd.read_csv(RUN/'all_forward_predictions.csv.gz'); rows = []
    for model in new.model.unique():
        for baseline in ['constant', 'ewma', 'ou', 'ou_history']:
            for width in [1, 6]: rows.append(dict(model=model, baseline=baseline, **paired(new[new.model == model], old[old.model == baseline], width)))
    scores = pd.DataFrame(rows); scores.to_csv(OUT/'forward_summary.csv', index=False)
    write_json(OUT/'scientific_audit.json', dict(status='COMPLETE', n_selected_fits=8, all_selected_optimizer_success=all(r['success'] for r in parameters), scope='One shared initial-mean coefficient, Gaussian N(0,1) penalty; model selection uses training penalized Laplace likelihood only; all16157forward events retained', limits='New version uses inherited Laplace/Gaussian numerical approximation. Training improvement alone does not justify a physical reset, nor a posterior interval. Patient seizure types were not used. Initial coordinate is shifted at clinical offsets, not at the first observed event.'))
    print(pd.DataFrame(parameters).to_string(index=False)); print(scores[scores.block_hours == 6].to_string(index=False))

def main():
    d = inputs(); old_history = history_data(d); canaries = []
    for history in [False, True]:
        base = np.array(source('full', history)['theta']); t = np.r_[base[:-2], 0., base[-2:]]
        reference = laplace(np.r_[base[:2], 0., base[2:]], old_history, True) if history else laplace(base, d)
        actual = laplace(expand(t, history), design(d, np.exp(t[-2])), True)
        assert abs(actual-reference) < 1e-7
        assert np.array_equal(design(slice_data(d, 0, 1000), .5)['x'], design(d, .5)['x'][:1000])
        canaries.append(dict(history=history, zero_offset_loglik_error=abs(actual-reference)))
    write_json(OUT/'numerical_canary.json', dict(status='PASS', zero_offset_checks=canaries, causal_design_prefix_identity=True, shift_at_clinical_offset=True, first_record_not_assumed_postictal=True))
    folds = json.loads((RUN/'splits.json').read_text())
    scopes = [dict(scope='full', end=len(d['y']))]+[dict(scope=f"fold{f['fold']}", end=f['train_end'], test_end=f['test_end']) for f in folds]
    jobs = [dict(id=f"{scope['scope']}_h{int(history)}_{i}", history=history, initial_offset=offset, **scope) for scope in scopes for history in [False, True] for i, offset in enumerate([-2., -.5, .5, 2.])]
    write_json(OUT/'contract.json', dict(question='Does the stationary interval-start assumption create the unstable patient time constant?', state='Same OU between clinical intervals; deterministic mean offset*exp(-age/tau) plus zero-mean stationary OU', new_parameter='One shared initial mean in log-odds for every post-clinical interval; bound[-5,5], Normal(0,1) regularization; no episode-specific parameter', initial_distribution='N(shared_offset,stationary_SD^2) at each known clinical offset, propagated through actual time until first observation; initial record uses stationary prior', n_initial_fit_attempts=32, n_selected_fits=8, selection='Training-only penalized Laplace likelihood; three chronological forward folds, identical16157events; seizure types and future onsets not used', boundary='Imposed statistical boundary law, not evidence of biological seizure reset; no IED reset and no SNN/core/ZM changes'))
    write_json(OUT/'queue.json', jobs)
    with ProcessPoolExecutor(max_workers=16) as pool:
        for i, future in enumerate(as_completed([pool.submit(worker, j) for j in jobs])):
            result = future.result(); print(json.dumps(dict(done=i+1, total=len(jobs), id=result['job']['id'], status=result['status'], success=result.get('success'))), flush=True)
    review()
    write_json(OUT/'status.json', dict(status='COMPLETE', n_fits=32, finished_unix=time.time()))

if __name__ == '__main__':
    main()
