"""Prescribed native spatial Z(t) isolates slow-path from fast-loop mismatch.

This receives a native slow trajectory, so it is explicitly not autonomous
correspondence or a bifurcation experiment. M and the fast rate field remain
free, with the same exogenous drive and count innovation keys as the free run.
"""
from common import OUT, BASE, ROOT, model, np, read, write, log
from physical_delay_count_rate import PhysicalDelayCountEngine
from fine_rate_frozen_Z_fields import native_field
import transient_response_network as network
from datetime import datetime
import argparse

DEST = OUT/'transient_native_Z_path_20260923'
FREE = network.DEST
SOURCE = ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/runs/eta0.0005_s9108401'


def prepare():
    assert read(FREE/'jobs.json')['status'] == 'COMPLETE'
    DEST.mkdir(exist_ok=True)
    assert not (DEST/'projection_check.json').exists()
    assert not (DEST/'jobs.json').exists()
    c = read(FREE/'contract.json')
    c.update(created_local=datetime.now().astimezone().isoformat(),
        question='Does supplying the native spatial Z history remove the early-entry error in the fixed transient-corrected rate network?',
        design_revision='After the completed free run entered near7.67s, distinguish wrong endogenous Z path from conditional fast/M response. One prescribed-slow-input experiment; no local-response or physical parameter retuning.',
        changed='Replace only the endogenous Z evolution with the original native per-E-cell Z projected onto the same g40 groups, linearly interpolated between actual5/10ms native observations and represented on a5ms grid; original final12500ms checkpoint supplies the endpoint.',
        unchanged='Same physical graph, response correction, mean/variance synapses, delays, fine external drive, countseed1, own fast/refractory histories and dynamicM from physical zero initialization.',
        native_information='Native Z(t) is supplied. Native spikes, rates, M, synaptic currents and membrane histories are not supplied. This is a conditional diagnostic, not autonomous model validation.',
        Z_dynamic=False, M_dynamic=True,
        inference='If recruitment and entry still differ, the fast/M network conditional on native Z remains mismatched. If they recover, the endogenous Z path is a major mediator; this alone does not identify an erroneous Z law or establish a bifurcation.',
        acceptance='Original state/propagation readouts retained as descriptive comparison. D_track is supplied by construction and must never count as a validation pass. Local failed gates remain.',
        budget='One12.5s prescribed-Z trajectory, exact replay and interpolation checks, comparative readout. No refit or branch launch.',
        checkpoints_ms=[3000, 8000, 9000, 9420, 9870],
        sources=str(SOURCE/'fields'), model_promoted=False)
    write(DEST/'contract.json', c)
    if not (DEST/'readout_contract.json').exists():
        import audit_fine_forcing_pair as audit
        audit.DEST = DEST
        audit.register()
    rc = read(DEST/'readout_contract.json')
    rc['role'] = 'Prescribed native Z(t), dynamic M and free fast network; conditional diagnostic only.'
    rc['resource_clock'] = 'D is externally supplied and cannot count as an acceptance criterion.'
    write(DEST/'readout_contract.json', rc)
    s = model(40)
    ids = s.geo['cell_group'][:32000]
    count = np.bincount(ids, minlength=s.P)
    assert np.array_equal(count[s.E], s.sizes[s.E])
    times, fields = [], []
    for path in sorted((SOURCE/'fields').glob('*.npz')):
        with np.load(path) as z:
            for tm, cells in zip(z['zm_step']*.1, z['z']):
                field = np.ones(s.P)
                field[s.E] = np.bincount(ids, weights=cells.astype(float), minlength=s.P)[s.E]/count[s.E]
                assert abs(field[s.E]@s.mean_weights-cells.mean(dtype=float)) < 1e-14
                times.append(tm)
                fields.append(field)
    times.append(12500.)
    fields.append(native_field(s, 12500))
    times, fields = np.array(times), np.array(fields)
    assert times[0] == 0 and times[-1] == 12500
    assert set(np.diff(times)) == {5., 10.}
    source_times = times.copy()
    uniform = np.arange(0, 12501., 5.)
    lo = np.clip(np.searchsorted(times, uniform, side='right')-1, 0, len(times)-2)
    alpha = (uniform-times[lo])/(times[lo+1]-times[lo])
    interpolated = (1-alpha[:, None])*fields[lo]+alpha[:, None]*fields[lo+1]
    assert np.array_equal(interpolated[(source_times/5).astype(int)], fields)
    times, fields = uniform, interpolated
    assert fields.min() >= 0 and fields.max() <= 1 and np.array_equal(fields[0], np.ones(s.P))
    error = max(float(np.max(abs(fields[t//5]-native_field(s, t)))) for t in [8000, 9000, 9420, 9870, 10370, 12500])
    assert error < 1e-6
    np.savez_compressed(DEST/'native_Z_path.npz', time_ms=times, source_time_ms=source_times, Z=fields,
        D=1-fields[:, s.E]@s.mean_weights)
    write(DEST/'projection_check.json', dict(status='PASS', groups=s.P, samples=len(times),
        range_ms=[0, 12500], independently_saved_checkpoint_error=error,
        original_E_count=32000, cell_weighted_mean_preserved=True,
        actual_native_intervals_ms=[5, 10], interpolation='linear between actual native samples; exact5ms representation'))
    log('NATIVE Z PATH PREPARED', error)


class PrescribedZEngine(PhysicalDelayCountEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Z_path = self.cp.asarray(np.load(DEST/'native_Z_path.npz')['Z'])
        self.Z_kernel = self.cp.RawKernel(r'''
extern "C" __global__ void prescribed_Z(double* Z,const double* path,const int* clock,int P,double dt){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=P)return;
 double a=fmin(2500.,clock[0]*dt/5.);int lo=min((int)floor(a),2499);a-=lo;
 Z[g]=(1.-a)*path[(long long)lo*P+g]+a*path[(long long)(lo+1)*P+g];
}''', 'prescribed_Z', options=('--fmad=false',))
        self.transport.pars[19].fill(0)

    def load_Z(self):
        self.Z_kernel(((self.s.P+127)//128,), (128,),
            (self.syn[5], self.Z_path, self.local.clock, np.int32(self.s.P), self.dt))

    def step(self):
        super().step()
        self.load_Z()


def check(device):
    assert read(DEST/'projection_check.json')['status'] == 'PASS'
    e = PrescribedZEngine(seed=1, device=device)
    network.install(e)
    s, cp = e.s, e.cp
    data = np.load(DEST/'native_Z_path.npz')
    path_time, path_Z = data['time_ms'], data['Z']
    errors = []
    for tm in [0., 10., 1234.55, 9000., 9420.05, 12500.]:
        e.local.clock.fill(round(tm/e.dt))
        e.load_Z()
        expected = np.array([np.interp(tm, path_time, path_Z[:, g]) for g in range(s.P)])
        error = float(np.max(abs(e.syn[5].get()-expected)))
        assert error < 1e-12
        errors.append(error)
    # Disabling the new Z prescription recovers the free model exactly.
    e.step = lambda:PhysicalDelayCountEngine.step(e)
    e.transport.pars[19].fill(1)
    e.graph()
    x = np.concatenate([e.chunk() for _ in range(10)])
    with np.load(FREE/network.LABEL/'trajectory.npz') as z:
        assert np.array_equal(x[:, 0].astype('f4'), z['group_rate_hz'][:100])
        assert np.array_equal(x[:, 1].astype('f4'), z['group_expected_rate_hz'][:100])
    del e.step
    e.transport.pars[19].fill(0)
    e.graph()
    for k in range(10):
        x = e.chunk()
        assert np.isfinite(x).all()
        assert np.max(abs(e.syn[5].get()-path_Z[2*(k+1)])) < 1e-12
    assert np.all(e.transport.pars[20].get() == 1)
    write(DEST/'implementation_check.json', dict(status='PASS',
        interpolation_errors=errors, free100ms_bitwise=True,
        prescribed100ms_matches_native_Z=True, M_dynamic=True, model_promoted=False))
    log('NATIVE Z PATH IMPLEMENTATION PASS')


def run(device):
    network.DEST = DEST
    network.PhysicalDelayCountEngine = PrescribedZEngine
    network.run(device)


def audit():
    import audit_fine_forcing_pair as original
    original.DEST = DEST
    if not (DEST/'readout_contract.json').exists():
        original.register()
    rc = read(DEST/'readout_contract.json')
    rc['role'] = 'Prescribed native Z(t), dynamic M and free fast network; conditional diagnostic only.'
    rc['resource_clock'] = 'D is externally supplied and cannot count as an acceptance criterion.'
    write(DEST/'readout_contract.json', rc)
    original.audit(partial=False)
    label = network.LABEL+'_fine_forcing'
    q = read(DEST/'scientific_comparison.json')
    q['rows'] = [r for r in q['rows'] if r['label'] in ['native', label]]
    q['core_event_rows'] = [r for r in q['core_event_rows'] if r['label'] in ['native', label]]
    free = read(FREE/'scientific_comparison.json')
    q['rows'].insert(1, dict(next(r for r in free['rows'] if r['label']==label), label='free_Z_transient_correction'))
    for r in q['rows']:
        if r['label'] == label:
            r['label'] = 'prescribed_native_Z'
            r['original_six_checks']['D_track'] = None
            r['applicable_five_criteria_passed'] = sum(v for v in r['original_six_checks'].values() if v is not None)
            r.pop('original_six_passed')
            r['D_note'] = 'Supplied by construction; not a model prediction or validation pass.'
    for r in q['core_event_rows']:
        if r['label'] == label:
            r['label'] = 'prescribed_native_Z'
    q['scope'] = 'Native Z(t) prescribed, M and fast rates free. Original D acceptance is inapplicable. Conditional diagnosis, not autonomous equivalence or bifurcation.'
    q['Z_dynamic'] = False
    q['M_dynamic'] = True
    path = np.load(DEST/'native_Z_path.npz')
    z = np.load(DEST/network.LABEL/'trajectory.npz')
    index = np.searchsorted(path['time_ms'], z['state_time_ms'])
    error = float(np.max(abs(path['Z'][index]-z['Z'])))
    assert error < 1e-7
    q['prescribed_Z_saved_max_error'] = error
    q['original_local_failure_retained'] = True
    write(DEST/'scientific_comparison.json', q)
    raw = read(DEST/'independent_comparison.json')
    raw['scope'] = q['scope']
    for r in raw['rows']:
        if r['label'] != 'native':
            r['qa'].update(Z_and_M_dynamic=False, Z_prescribed=True, M_dynamic=True,
                prescribed_Z_saved_max_error=error)
    write(DEST/'independent_comparison.json', raw)
    log('PRESCRIBED NATIVE Z COMPARISON', [(r['label'], r['high_entry_ms'], r['tail_global_hz'], r['physical_core_order']['original_events']) for r in q['rows']])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'check', 'run', 'audit'])
    p.add_argument('--device', type=int, default=1)
    a = p.parse_args()
    {'prepare':prepare, 'check':lambda:check(a.device), 'run':lambda:run(a.device), 'audit':audit}[a.command]()
