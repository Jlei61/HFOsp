"""Recording-only replay of two existing local response counterfactuals.

This is not a network replacement. Spike counts must match the completed
factorial assay bitwise before any membrane-state interpretation is accepted.
"""
from native_cycle_waveform_response import CODE
from common import OUT, np, read, write, log
from datetime import datetime
import argparse

DEST = OUT / 'membrane_recovery_diagnostic'
NAMES = ['voltage_mv', 'voltage_squared_mv2', 'refractory_fraction',
         'input_current_mv', 'refractory_current_mv', 'below_reset_fraction']


def register():
    path = OUT / 'membrane_recovery_diagnostic_contract.json'
    assert not path.exists(), 'Keep the original preregistered contract'
    source = read(OUT / 'factorial_waveform_contract.json')
    contract = dict(
        created_local=datetime.now().astimezone().isoformat(),
        question='Does the strong-mean local mismatch coincide with suppressed membrane voltage and delayed recovery, rather than only instantaneous refractory occupancy?',
        source='factorial_waveform', selected_indices=[6, 7],
        selected_group=39, conditions=['full', 'mean_only'],
        selection='Previously identified surround E group, with original and constant-variance inputs; no new subgroup or waveform search.',
        observables=NAMES,
        recording='Post-update, post-reset voltage and refractory flag; current is the pre-update input current. All paths, including reset-clamped refractory paths, contribute. Voltage variance includes variation inside each phase bin.',
        statistical_unit='Independent noise path; repeated cycles combined within each path. Two conditions use common random numbers.',
        inherited_simulation={k: source[k] for k in ['dt_ms', 'replicates', 'burn_cycles', 'record_cycles', 'phase_bins', 'seed', 'device']},
        acceptance='All spike counts bitwise equal to the existing factorial assay; finite moments, valid probabilities and variance; phase exposure independently reconstructed.',
        budget='Two recording-only local LIF replays, count parity and moment audit. No fitting, spatial network launch, response equation change or critical-point promotion.',
        scope='Diagnostic of local imposed-input response. A membrane-state association does not establish a unique closure, causal sufficiency, or the native onset bifurcation.')
    write(path, contract)
    log('MEMBRANE DIAGNOSTIC REGISTERED', contract['selected_indices'])


def acquire():
    import cupy as cp
    c = read(OUT / 'membrane_recovery_diagnostic_contract.json')
    sim = c['inherited_simulation']
    source = OUT / c['source']
    assert read(source / 'independent_audit.json')['status'] == 'COUNT_LEVEL_AUDIT_PASS'
    z = np.load(source / 'prepared.npz')
    original = np.load(source / 'response.npz')
    info = read(source / 'preparation.json')
    indices = c['selected_indices']
    assert [info['rows'][i]['condition'] for i in indices] == c['conditions']
    assert all(info['rows'][i]['source_group'] == c['selected_group'] for i in indices)
    pars = np.ascontiguousarray(z['pars'][indices])
    wave = np.ascontiguousarray(z['wave'][indices])
    p, _, w = wave.shape
    r, b, dt = sim['replicates'], sim['phase_bins'], sim['dt_ms']
    period = float(z['T_ms'])
    steps = round(sim['record_cycles'] * period / dt)
    burn = round(sim['burn_cycles'] * period / dt)
    old_signature = 'const double* wave,unsigned int* counts,'
    old_record = 'if(t>=0&&fired){int bin=min((int)(phase*B),B-1);counts[id*B+bin]++;}'
    assert CODE.count(old_signature) == CODE.count(old_record) == 1
    code = CODE.replace(old_signature, 'const double* wave,unsigned int* counts,double* moments,')
    code = code.replace(old_record, '''if(t>=0){
      int bin=min((int)(phase*B),B-1);
      if(fired)counts[id*B+bin]++;
      double* out=moments+((long long)id*B+bin)*6;
      out[0]+=v; out[1]+=v*v; out[2]+=(ref>0);
      out[3]+=cur; out[4]+=(ref>0?cur:0.); out[5]+=(v<p[21]);
    }''')
    cp.cuda.Device(sim['device']).use()
    counts = cp.zeros((p, r, b), dtype=cp.uint32)
    sums = cp.zeros((p, r, b, 6), dtype=cp.float64)
    kernel = cp.RawKernel(code, 'waveform', options=('--fmad=false',))
    log('MEMBRANE REPLAY START', p, r, steps, burn)
    kernel(((p*r+127)//128,), (128,), (cp.asarray(pars), cp.asarray(wave), counts, sums,
        np.int32(p), np.int32(r), np.int32(w), np.int32(b), np.int32(steps), np.int32(burn),
        float(dt), float(period), np.uint64(sim['seed'])))
    counts, sums = counts.get(), sums.get()
    assert np.array_equal(counts, original['counts'][indices]), 'Recording changed spike counts'
    # Independent clock reconstruction, including incomplete final cycle.
    times = (np.arange(steps) + 1) * dt
    bins = np.minimum((((times / period) % 1) * b).astype(int), b-1)
    exposure = np.bincount(bins, minlength=b)
    assert np.array_equal(exposure * dt, original['occupancy_ms'])
    values = sums / exposure[None, None, :, None]
    mean = values.mean(axis=1)
    sem = values.std(axis=1, ddof=1) / np.sqrt(r)
    assert np.isfinite(values).all()
    assert np.min(mean[..., 1] - mean[..., 0]**2) > -1e-8
    assert np.all((values[..., [2, 5]] >= 0) & (values[..., [2, 5]] <= 1))
    assert np.all(mean[..., 0] <= pars[:, 1, None])
    # Refractory cells are reset-clamped, allowing a population conditional mean.
    active_fraction = 1 - mean[..., 2]
    active_voltage = (mean[..., 0] - pars[:, 21, None] * mean[..., 2]) / active_fraction
    assert np.min(active_fraction) > 0
    DEST.mkdir(exist_ok=True)
    np.savez_compressed(DEST / 'response.npz', counts=counts, moment_sums=sums,
        moments_mean=mean, moments_sem=sem, nonrefractory_mean_voltage_mv=active_voltage,
        exposure_steps=exposure, occupancy_ms=exposure*dt, T_ms=period,
        phase_centres=(np.arange(b)+.5)/b, selected_indices=indices,
        measured_hz=original['measured_hz'][indices], sem_hz=original['sem_hz'][indices],
        predicted_hz=original['predicted_hz'][indices], pars=pars)
    rows = []
    for k, index in enumerate(indices):
        pred = original['predicted_hz'][index, 1]
        target = original['measured_hz'][index]
        error = pred - target
        # Associations use all bins; no selected-window fitting or acceptance.
        positive_error = np.maximum(error, 0)**2
        under_reset = active_voltage[k] < pars[k, 21]
        row = dict(condition=c['conditions'][k], source_index=index,
            voltage_min_mv=float(mean[k, :, 0].min()),
            nonrefractory_voltage_min_mv=float(active_voltage[k].min()),
            reset_mv=float(pars[k, 21]), threshold_mv=float(pars[k, 1]),
            max_refractory_fraction=float(mean[k, :, 2].max()),
            max_below_reset_fraction=float(mean[k, :, 5].max()),
            under_reset_phase_fraction=float(np.average(under_reset, weights=exposure)),
            positive_rate_error_energy_under_reset=float(np.sum(positive_error[under_reset]) / np.sum(positive_error)),
            largest_rate_overprediction_hz=float(error.max()),
            largest_overprediction_phase_ms=float((np.argmax(error)+.5)/b*period),
            voltage_at_largest_overprediction_mv=float(mean[k, np.argmax(error), 0]),
            refractory_at_largest_overprediction=float(mean[k, np.argmax(error), 2]))
        rows.append(row)
    q = dict(status='RECORDING_REPLAY_PASS', rows=rows, moment_names=NAMES,
        spike_counts_bitwise=True, original_counts=int(counts.sum()),
        replicates=r, steps=steps, burn_steps=burn, dt_ms=dt,
        scope=c['scope'], model_promoted=False, onset_type='NOT_ESTABLISHED')
    write(DEST / 'result.json', q)
    log('MEMBRANE REPLAY COMPLETE', q)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--register', action='store_true')
    a = parser.parse_args()
    register() if a.register else acquire()
