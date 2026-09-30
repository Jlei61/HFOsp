"""Bounded, recording-only test of a population membrane-balance closure.

The measured spike rate is supplied to this diagnostic. A good voltage
reconstruction would therefore not be an autonomous firing-rate model.
"""
from native_cycle_waveform_response import CODE
from common import OUT, np, read, write, log
from datetime import datetime
from scipy.signal import lfilter
import argparse

DEST = OUT / 'refractory_current_closure'


def register():
    path = OUT / 'refractory_current_closure_contract.json'
    assert not path.exists()
    old = read(OUT / 'factorial_waveform_contract.json')
    write(path, dict(created_local=datetime.now().astimezone().isoformat(),
        question='Does observed firing/refractory occupancy plus mean input suffice for the ensemble voltage balance, or must spike-conditioned colored-current memory also be retained?',
        source='factorial_waveform', indices=[6, 7], source_group=39,
        inherited={k: old[k] for k in ['replicates', 'phase_bins', 'dt_ms', 'burn_cycles', 'record_cycles', 'seed', 'device']},
        record_names=['voltage', 'current', 'clamped_fraction', 'spike_probability', 'reset_impulse', 'clamp_impulse', 'previous_voltage'],
        exact_identity='V_next=a V_previous+(1-a)I-reset_impulse-clamp_impulse',
        approximation='Measured spike_probability*(theta-reset) for reset impulse; (1-a)*(mean_input-reset)*measured_clamped_fraction for clamp impulse.',
        comparisons=['Exact replay', 'Observed reset with factorized clamp', 'Observed clamp with threshold-reset jump', 'Both moment factorizations'],
        checks=['Original per-path per-phase spike counts bitwise', 'Exact full-step voltage identity', 'Clamped occupancy equals delayed measured spike history', 'Independent phase aggregation and filter reconstruction'],
        budget='Two recording-only local replays. No fitted coefficient, response promotion, network run, or further candidate search.',
        statistical_unit='Independent noise path; repeated cycles combined within paths. Conditions paired by seed.',
        interpretation='Descriptive error decomposition with measured rate supplied. No pass of a waveform response gate or autonomous model claim follows from it.'))


def run():
    import cupy as cp
    c = read(OUT / 'refractory_current_closure_contract.json'); sim = c['inherited']
    z = np.load(OUT / c['source'] / 'prepared.npz')
    target = np.load(OUT / c['source'] / 'response.npz')
    pars = np.ascontiguousarray(z['pars'][c['indices']])
    wave = np.ascontiguousarray(z['wave'][c['indices']])
    P, R, B = len(pars), sim['replicates'], sim['phase_bins']
    dt, T, W = sim['dt_ms'], float(z['T_ms']), wave.shape[-1]
    steps, burn = round(sim['record_cycles']*T/dt), round(sim['burn_cycles']*T/dt)
    assert R % 128 == 0
    code = CODE.replace('const double* wave,unsigned int* counts,',
                        'const double* wave,unsigned int* counts,double* records,')
    code = code.replace('double qa=0,ia=0,qg=0,ig=0,v=p[21];int ref=0;',
        'double qa=0,ia=0,qg=0,ig=0,v=p[21];int ref=0; __shared__ double buf[7][128];')
    code = code.replace('double cur=mu+ia-ig;bool fired=false;ref=max(0,ref-1);',
        'double cur=mu+ia-ig;double previous=v;double proposal=p[18]*v+(1-p[18])*cur;bool clamped=ref>1;bool fired=false;ref=max(0,ref-1);')
    original = 'if(t>=0&&fired){int bin=min((int)(phase*B),B-1);counts[id*B+bin]++;}'
    assert code.count(original) == 1
    code = code.replace(original, '''if(t>=0){
      int bin=min((int)(phase*B),B-1);if(fired)counts[id*B+bin]++;
      int lane=threadIdx.x;
      buf[0][lane]=v;buf[1][lane]=cur;buf[2][lane]=clamped;
      buf[3][lane]=fired;buf[4][lane]=fired?proposal-p[21]:0.;
      buf[5][lane]=clamped?proposal-p[21]:0.;buf[6][lane]=previous;
      __syncthreads();
      for(int offset=64;offset>0;offset>>=1){
        if(lane<offset)for(int q=0;q<7;q++)buf[q][lane]+=buf[q][lane+offset];
        __syncthreads();
      }
      if(lane==0){long long base=((long long)blockIdx.x*steps+t)*7;
        for(int q=0;q<7;q++)records[base+q]=buf[q][0];}
      __syncthreads();
    }''')
    cp.cuda.Device(sim['device']).use()
    counts = cp.zeros((P, R, B), dtype=cp.uint32)
    records = cp.empty((P, R//128, steps, 7), dtype=cp.float64)
    k = cp.RawKernel(code, 'waveform', options=('--fmad=false',))
    log('REFRACTORY CURRENT REPLAY', P, R, steps, records.nbytes)
    k((P*R//128,), (128,), (cp.asarray(pars), cp.asarray(wave), counts, records,
        np.int32(P), np.int32(R), np.int32(W), np.int32(B), np.int32(steps), np.int32(burn),
        float(dt), float(T), np.uint64(sim['seed'])))
    counts, raw = counts.get(), records.get()
    assert np.array_equal(counts, target['counts'][c['indices']])
    mean = raw.sum(axis=1)/R
    rows, curves = [], []
    select = np.arange(steps)*dt >= T
    for j, p in enumerate(pars):
        v, current, clamped, fired, reset_q, clamp_q, previous = mean[j].T
        a, vr, theta = p[18], p[21], p[1]
        assert np.max(abs(v-a*previous-(1-a)*current+reset_q+clamp_q)) < 1e-9
        assert np.max(abs(previous[1:]-v[:-1])) < 1e-9
        nref = int(p[19])
        # At t, spikes at t-1 through t-(nref-1) are still clamped.
        history = np.r_[0., np.convolve(fired, np.ones(nref-1))[:steps-1]]
        occupancy_error = float(np.max(abs(history[nref:]-clamped[nref:])))
        assert occupancy_error < 1e-12
        reset_approx = fired*(theta-vr)
        clamp_approx = (1-a)*(current-vr)*clamped
        outputs, summaries = [], []
        for label, rq, cq in [('exact', reset_q, clamp_q),
                             ('factorized_clamp', reset_q, clamp_approx),
                             ('threshold_reset', reset_approx, clamp_q),
                             ('both', reset_approx, clamp_approx)]:
            pred, _ = lfilter([1.], [1., -a], (1-a)*current-rq-cq, zi=[a*previous[0]])
            err = pred-v
            outputs.append(pred)
            summaries.append(dict(variant=label, voltage_RMS_error_mv=float(np.sqrt(np.mean(err[select]**2))),
                maximum_voltage_error_mv=float(abs(err[select]).max()),
                signed_mean_error_mv=float(err[select].mean())))
        assert summaries[0]['maximum_voltage_error_mv'] < 1e-8
        rows.append(dict(condition=['full', 'mean_only'][j], occupancy_error=occupancy_error,
            reset_charge_overshoot_fraction=float((reset_q-reset_approx).sum()/reset_q.sum()),
            clamp_covariance_charge_RMS=float(np.sqrt(np.mean((clamp_q-clamp_approx)[select]**2))),
            summaries=summaries))
        curves.append(outputs)
    DEST.mkdir(exist_ok=True)
    np.savez_compressed(DEST / 'observations.npz', counts=counts, block_sums=raw, ensemble_mean=mean,
        reconstructed_voltage=curves, pars=pars, dt_ms=dt, T_ms=T)
    q = dict(status='REFRACTORY_CURRENT_CLOSURE_DIAGNOSTIC_COMPLETE', rows=rows,
        spike_counts_bitwise=True, measured_rate_supplied=True, fitted_parameters=0,
        model_promoted=False, onset_type='NOT_ESTABLISHED', scope=c['interpretation'])
    write(DEST / 'result.json', q); log('REFRACTORY CURRENT CLOSURE', q)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--register', action='store_true')
    args = parser.parse_args()
    register() if args.register else run()
