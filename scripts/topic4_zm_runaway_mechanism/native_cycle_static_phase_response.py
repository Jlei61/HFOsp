"""Distinguish stationary-transfer error from the nonstationary closure failure.

Each phase is a separate constant-input LIF assay, not a slowed network cycle.
Uses the frozen model and the already selected candidate-cycle groups.
"""
from native_cycle_waveform_response import DEST
from common import *
from lif_mc import condition, run
import argparse


def main(device):
    contract = read(OUT / 'native_cycle_static_phase_contract.json')
    source = np.load(DEST / 'prepared.npz')
    info = read(DEST / 'preparation.json')['groups']
    model_ = model()
    samples = source['wave'].shape[-1]
    phases = (np.arange(contract['phases']) + .5) / contract['phases']
    pos = phases * samples
    lower = np.floor(pos).astype(int) % samples
    upper = (lower + 1) % samples
    alpha = pos - np.floor(pos)
    wave = ((1-alpha)[None, None, :] * source['wave'][:, :, lower]
            + alpha[None, None, :] * source['wave'][:, :, upper])
    parameters, predictions = [], []
    for i, group in enumerate(info):
        mu, ve, vi = wave[i]
        theta = np.full_like(mu, group['theta_mv'])
        predictions.append(model_.spline[group['pop']].evaluate(mu, ve, vi, theta)['rate'] * 1000)
        parameters.extend(condition(m, group['theta_mv'], e, h, group['pop'])
                          for m, e, h in zip(mu, ve, vi))
    duration = contract['duration_ms']
    observations = run(parameters, contract['replicates'], duration,
                       contract['burn_ms'], contract['seed'], device=device, batch=16)
    counts = observations[:, :, 2].reshape(4, contract['phases'], -1)
    rates = counts / duration * 1000
    measured = rates.mean(2)
    sem = rates.std(2, ddof=1) / np.sqrt(contract['replicates'])
    predictions = np.asarray(predictions)
    rows = []
    for i, group in enumerate(info):
        relative = np.linalg.norm(predictions[i]-measured[i]) / max(np.linalg.norm(measured[i]), 1.)
        eligible = measured[i] >= contract['pointwise_min_rate_hz']
        errors = abs(predictions[i]-measured[i]) / np.maximum(measured[i], 1.)
        rows.append(dict(**group, stationary_phase_relative_L2=float(relative),
                         eligible_points=int(eligible.sum()),
                         eligible_relative_error_median=float(np.median(errors[eligible])),
                         eligible_relative_error_max=float(np.max(errors[eligible])),
                         phase_average_measured_hz=float(measured[i].mean()),
                         phase_average_prediction_hz=float(predictions[i].mean()),
                         passed=bool(relative <= contract['acceptance']['relative_L2_max'])))
    np.savez_compressed(DEST / 'static_phase_response.npz', phases=phases, input=wave,
                        counts=counts, measured_hz=measured, sem_hz=sem,
                        predicted_hz=predictions)
    result = dict(status='COMPLETE', rows=rows,
                  verdict='STATIC_SELECTED_PHASES_PASS' if all(r['passed'] for r in rows) else 'STATIC_SELECTED_PHASES_FAIL',
                  scope=contract['scope'], contract=str(OUT/'native_cycle_static_phase_contract.json'))
    write(DEST / 'static_phase_result.json', result)
    log('STATIC PHASE RESPONSE', result)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1)
    main(p.parse_args().device)
