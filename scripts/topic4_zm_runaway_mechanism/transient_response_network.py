"""Fixed transient-response correction in the complete spatial feedback loop.

This is a separately registered diagnostic after local validation failed.
It cannot promote the candidate or overwrite the failed local acceptance.
"""
from common import OUT, BASE, np, read, write, log
from physical_delay_count_rate import PhysicalDelayCountEngine, projections, capture, readouts
from refractory_rate_cuda import source
from datetime import datetime
import argparse
import hashlib
import os
import time

LOCAL = OUT / 'transient_response_correction'
DEST = OUT / 'transient_response_network_20260923'
LABEL = 'recorded_drive_binomial_seed1'
PARENT = OUT / 'physical_delay_count_rate' / LABEL


def corrected_source(groups, strength=1.):
    pars = read(LOCAL / 'locked.json')['parameters']
    code = source(groups)
    needle = '  double occupied=0.;int nref=(int)llround(refractory[g]/dt);'
    assert code.count(needle) == 1
    term = '  double H=0.;for(int j=3;j<39;j++)H+=f[j]*f[j];H/=36.;\n'
    term += f'  double b=p==0?{strength*pars["E"]["b"]:.17g}:{strength*pars["I"]["b"]:.17g};\n'
    term += f'  double h=p==0?{pars["E"]["h"]:.17g}:{pars["I"]["h"]:.17g};ell+=b*H/(H+h);\n'
    return code.replace(needle, term + needle)


def install(e, strength=1.):
    e.transient_module = e.cp.RawModule(code=corrected_source(e.s.P, strength),
        options=('--fmad=false',), name_expressions=['local_rate'])
    e.local.kernel = e.transient_module.get_function('local_rate')


def register():
    assert read(LOCAL / 'fresh_validation/independent_audit.json')['status'] == 'PASS'
    assert read(LOCAL / 'fresh_validation/implementation_audit.json')['status'] == 'PASS'
    assert read(LOCAL / 'fresh_validation/result.json')['status'] == 'FRESH_LOCAL_FAIL'
    DEST.mkdir(exist_ok=True)
    assert not (DEST / 'contract.json').exists()
    write(DEST / 'contract.json', dict(
        created_local=datetime.now().astimezone().isoformat(),
        question='Does the locked transient correction improve early self-limited events, spatial recruitment and the Z/M onset path simultaneously in the same whole network?',
        authorization='User continued the autonomous model/onset investigation after being told the remaining local failure and missing whole-network validation. This new diagnostic extends the previous local-only batch; its failed acceptance and no-promotion decision remain unchanged.',
        design_revision='The local-only contract has ended in FAIL. A distinct whole-network sensitivity test now measures the consequence of its residual error. It is not an automatic gate-conditioned launch or model acceptance.',
        local_failure='Early weak surround E group531 fails the original waveform gate; 13/14 fresh local inputs pass. All numerical convergence checks pass. No refit or tolerance change.',
        parameters=str(LOCAL / 'locked.json'),
        parameter_sha256=hashlib.sha256((LOCAL / 'locked.json').read_bytes()).hexdigest(),
        changed='Only delta_log_hazard=b_population*H/(H+h_population), H=mean(f[3:39]^2), using locked coefficients. No new dynamic state.',
        unchanged='Same g40 physical graph, thresholds, cell counts, synaptic and delay operators including repaired private variance, original fine external input, seed1, zero initialization and Z/M laws. Both Z and M dynamic. No native future spikes supplied.',
        stationary='Correction and first derivative vanish at stationary input histories. This does not certify the parent stationary approximation.',
        runs=[dict(label=LABEL, seed=1)], dt_ms=.05, duration_ms=12500,
        checkpoints_ms=[3000, 8000, 9000, 9420, 9870],
        readouts='Unchanged original A4 six gates plus separate early0.5-3s,4-8s,8-9.42s complete-event, quiet, core-order and spatial diagnostics. D=1-cell-weighted mean Z_E; y=Hz per neuron.',
        interpretation='Improved early and late correspondence supports transient response error as a contributor. Early improvement but late failure rejects it as a sufficient repair. New persistent activity in the early interval is a failed regime. No outcome alone proves a bifurcation.',
        budget='One full12.5s trajectory, prefix implementation verification, raw readout audit and comparison figure. No coefficient tuning or branch launch in this batch.',
        local_failure_retained=True, model_promoted=False))
    import audit_fine_forcing_pair as audit
    audit.DEST = DEST
    audit.register()


def check(device):
    assert (DEST / 'contract.json').exists()
    e = PhysicalDelayCountEngine(seed=1, device=device)
    original_weights = e.local.network.get()
    install(e, 0.)
    e.graph()
    x = np.concatenate([e.chunk() for _ in range(10)])
    with np.load(PARENT / 'trajectory.npz') as z:
        assert np.array_equal(x[:, 0].astype('f4'), z['group_rate_hz'][:100])
        assert np.array_equal(x[:, 1].astype('f4'), z['group_expected_rate_hz'][:100])
    install(e)
    e.graph()
    for _ in range(10):
        x = e.chunk()
        assert np.isfinite(x).all() and x[:, 0].min() >= 0
    assert np.array_equal(original_weights, e.local.network.get())
    assert e.local.history.data.ptr == e.transport.history.data.ptr
    counts = e.local.history.get() * e.s.sizes * e.dt
    assert np.max(abs(counts - np.rint(counts))) < 1e-9
    tick = int(e.local.clock.get()[0])
    for mask, ref in [(e.s.E, 2.), (~e.s.E, 1.)]:
        occupied = counts[(tick - np.arange(round(ref/e.dt))) % len(counts)][:, mask].sum(0)
        assert np.all(occupied <= e.s.sizes[mask] + 1e-9)
    write(DEST / 'implementation_check.json', dict(status='PASS',
        zero_correction_100ms_parent_bitwise=True, original_weights_unchanged=True,
        same_history_counts=True, corrected_prefix_ms=100, physical_count_bounds=True,
        local_equation_parity=str(LOCAL / 'fresh_validation/implementation_audit.json'),
        kernel_sha256=hashlib.sha256(corrected_source(e.s.P).encode()).hexdigest(),
        model_promoted=False))
    log('TRANSIENT NETWORK IMPLEMENTATION PASS')


def run(device):
    c = read(DEST / 'contract.json')
    assert hashlib.sha256((LOCAL / 'locked.json').read_bytes()).hexdigest() == c['parameter_sha256']
    assert read(DEST / 'implementation_check.json')['status'] == 'PASS'
    assert not (DEST / 'jobs.json').exists()
    folder = DEST / LABEL
    folder.mkdir()
    jobs = dict(status='RUNNING', pid=os.getpid(), expected=1, completed=[], time_ms=0)
    write(DEST / 'jobs.json', jobs)
    try:
        e = PhysicalDelayCountEngine(seed=1, device=device)
        s = e.s
        install(e)
        e.graph()
        projection = projections(s, e.coarse, e.parent)
        write(folder / 'identity.json', dict(graph=s.prep['graph_identity'], grid=40, groups=s.P,
            response='Locked conditioned39 plus locked transient correction',
            parameters=c['parameters'], parameter_sha256=c['parameter_sha256'],
            Z_and_M_dynamic=c.get('Z_dynamic', True), Z_dynamic=c.get('Z_dynamic', True),
            M_dynamic=True, native_future_spikes_used=False, private_split=e.split_qa))
        R, Z, M = [], [], []
        start = time.time()
        for k in range(1250):
            x = e.chunk()
            assert np.isfinite(x).all() and x.min() >= -1e-9
            R.append(x)
            Z.append(e.syn[5].get())
            M.append(e.syn[4].get())
            assert Z[-1].min() >= 0 and Z[-1].max() <= 1
            now = (k + 1)*10
            if now in c['checkpoints_ms']:
                np.savez_compressed(folder / f'checkpoint{now}.npz', **capture(e))
            if (k+1) % 50 == 0:
                jobs.update(time_ms=now, elapsed_seconds=time.time()-start,
                    D=float(1-Z[-1][s.E]@s.mean_weights))
                write(DEST / 'jobs.json', jobs)
                log('TRANSIENT NETWORK', now, 'ms', round(time.time()-start, 1), 's', 'D', jobs['D'])
            if now in [3000, 8000, 10000]:
                rr = np.concatenate(R)
                field = (projection[20][0]@rr[:, 0].T).T
                np.savez_compressed(folder / f'prefix{now}.npz',
                    time_ms=np.arange(1, now+1.), field_E_hz=field.astype('f4'),
                    cell_counts=projection[20][1], state_time_ms=np.arange(10, now+1., 10),
                    D=1-np.array(Z)[:, s.E]@s.mean_weights)
        r, zs, m = np.concatenate(R), np.array(Z), np.array(M)
        fields = {g:(P@r[:, 0].T).T for g, (P, n) in projection.items()}
        count = projection[20][1]
        whole = r[:, 0, s.E]@s.mean_weights
        assert np.max(abs(fields[20]@(count/count.sum())-whole)) < 1e-8
        t, ts = np.arange(1, 12501.), np.arange(10, 12501., 10)
        D = 1-zs[:, s.E]@s.mean_weights
        events, summary, _, _ = readouts(t, fields[20], count, LABEL)
        np.savez_compressed(folder/'trajectory.npz', time_ms=t, state_time_ms=ts,
            group_rate_hz=r[:, 0].astype('f4'), group_expected_rate_hz=r[:, 1].astype('f4'),
            field_E_hz=fields[20].astype('f4'), field_E_hz_grid40=fields[40].astype('f4'),
            global_E_hz=whole, cell_counts=count, Z=zs.astype('f4'), M_current=m.astype('f4'), D=D,
            parent_g20=e.parent, final_synaptic_slow_state=e.syn.get(), final_local_state=e.local.state.get(),
            final_own_history=e.local.history.get(), final_emitted_history=e.transport.history.get(),
            final_tick=e.local.clock.get(), dt_ms=e.dt)
        summary.update(status='COMPLETE', Z_and_M_dynamic=c.get('Z_dynamic', True),
            Z_dynamic=c.get('Z_dynamic', True), M_dynamic=True, D9870=float(D[986]), D_final=float(D[-1]),
            events=[{k:v for k,v in ev.items() if k!='onset'} for ev in events],
            seconds=time.time()-start, model_promoted=False, local_failure_retained=True)
        write(folder/'result.json', summary)
        jobs.update(status='COMPLETE', completed=[LABEL], time_ms=12500)
        write(DEST/'jobs.json', jobs)
        log('TRANSIENT NETWORK COMPLETE', summary['high_onset_ms'], summary['D9870'])
    except BaseException as error:
        jobs.update(status='FAILED', error=repr(error))
        write(DEST/'jobs.json', jobs)
        raise


def audit():
    import audit_fine_forcing_pair as original
    original.DEST = DEST
    original.audit(partial=False)
    q = read(DEST/'scientific_comparison.json')
    q['rows'] = [r for r in q['rows'] if r['label'] in ['native', LABEL+'_fine_forcing']]
    q['core_event_rows'] = [r for r in q['core_event_rows'] if r['label'] in ['native', LABEL+'_fine_forcing']]
    old = read(OUT/'physical_delay_count_rate/scientific_comparison.json')
    q['rows'].insert(1, next(r for r in old['rows'] if r['label']=='physical_delay_split'))
    q['scope'] = 'Fixed transient correction whole-network diagnostic. Same original A4 gates. Local response FAIL retained; no promotion or bifurcation claim.'
    write(DEST/'scientific_comparison.json', q)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['register', 'check', 'run', 'audit'])
    p.add_argument('--device', type=int, default=1)
    a = p.parse_args()
    {'register':register, 'check':lambda:check(a.device), 'run':lambda:run(a.device), 'audit':audit}[a.command]()
