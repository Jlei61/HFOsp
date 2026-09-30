#!/usr/bin/env python3
"""Historical manual field: native Z, manual refill, and simultaneous checkpoint capture."""
from validate_topic4_fixed_rate_base import ROOT, SOURCE, read, write, make_external_drive, spatial_cell_index
from topic4_historical_manual_z_common import OUT, setup, TIMES_MS, ARM
from checkpoint import save as save_checkpoint
from src.topic4_raster_protocol_engine import simulate_kick
from src.snn_engine.mz_slow_vars import MZSlowVars, MZSlowVarsConfig
import numpy as np
import time
import hashlib
import resource

REFERENCE = OUT / 'reference_samples'


class EndObservation(Exception):
    pass


class RefilledZ(MZSlowVars):
    """Original Z ODE up to a scheduled intervention; no M or added biology."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.restore_ms = None
        self.restore_from = None
        self.last_t_ms = 0.

    def _record_trace(self, spikes, dt):
        # A compact recorder below replaces only the inherited trace allocation.
        pass

    def apply_currents(self, ie, ii, labels=None, rec=None):
        tm = self._step_index * .1
        self.last_t_ms = tm
        if self.restore_ms is not None and tm >= self.restore_ms:
            if self.restore_from is None:
                self.restore_from = self.z[:self.NE].copy()
            alpha = min(1., (tm - self.restore_ms) / 1000.)
            self.z[:self.NE] = self.restore_from + alpha * (1. - self.restore_from)
        return super().apply_currents(ie, ii, labels, rec)

    def step(self, spk, labels, dt):
        if self.restore_from is None:
            super().step(spk, labels, dt)
        else:
            self._step_index += 1


def verify_slow():
    cfg = MZSlowVarsConfig(use_z=True, use_m=False, tau_z=5000., I_th_EI=95.19851312666987)
    original = MZSlowVars(8, 18., cfg, NE=6)
    extended = RefilledZ(8, 18., cfg, NE=6)
    rng = np.random.default_rng(37)
    for step in range(400):
        ie = rng.uniform(0, 300, 8); ii = rng.uniform(0, 300, 8); sp = rng.random(8) < .2
        assert np.array_equal(original.apply_currents(ie, ii), extended.apply_currents(ie, ii))
        original.step(sp, None, .1); extended.step(sp, None, .1)
        assert np.array_equal(original.z, extended.z)
    extended.restore_ms = 40.
    initial = extended.z[:6].copy()
    for index, tm in enumerate([40., 540., 1040., 2040.]):
        extended._step_index = round(tm / .1)
        extended.apply_currents(np.zeros(8), np.ones(8))
        expected = initial + min(1., (tm - 40.) / 1000.) * (1. - initial)
        assert np.array_equal(extended.z[:6], expected)
        assert np.array_equal(extended.z[6:], np.ones(2))
    return {'status': 'PASS', 'native_Z_before_refill': 'bitwise equal over 400 supplied-current steps',
            'refill': 'linear neuron-wise restoration checked at start/middle/end and hold; I targets stay 1'}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / 'slow_qa.json', verify_slow())
    started = time.time(); seed = 9108401
    write(OUT / 'status.json', {'status': 'BUILDING', 'started_unix': started})
    s, tr, frozen, identity = setup(seed)
    assert read(REFERENCE / 'engine_qa.json')['status'] == 'PASS'
    checkpoint_records = []
    def capture(step, state):
        tm = round(state['absolute_time_ms'])
        path = OUT / 'checkpoints' / f't{tm}ms.npz'
        digest = save_checkpoint(state, path)
        checkpoint_records.append({'time_ms': tm, 'path': str(path), 'sha256': digest})
        write(OUT / 'checkpoint_index.json', checkpoint_records)
    legacy_path = SOURCE / 'config/topic4_data_driven_zm_ictal_transition_v1.json'
    legacy = read(legacy_path)['zm']
    cfg = MZSlowVarsConfig(use_z=True, use_m=False, tau_z=legacy['tau_z'],
                          I_th_EI=legacy['I_th_EI'], trace_stride_steps=100)
    protocol = {'status': 'DEFINED_BEFORE_TRAJECTORY', 'seed': seed,
                'carrier': str(OUT / 'substrate.json'), 'manual_arm': ARM,
                'frozen_checkpoint_times_ms': TIMES_MS,
                'frozen_identity': identity, 'max_duration_ms': 20000., 'dt_ms': .1,
                'Z_equation': 'tau_z dz_i/dt = 1[I_GABA,i < I_th_EI] - z_i; E only; I z=1',
                'initial_Z': 1., 'tau_z_ms': cfg.tau_z, 'I_th_EI': cfg.I_th_EI,
                'legacy_parameter_source': str(legacy_path),
                'legacy_parameter_source_sha256': hashlib.sha256(legacy_path.read_bytes()).hexdigest(),
                'parameter_scope': 'Transferred historical Z settings on current carrier, not a previously calibrated Z working point',
                'M': 'off to isolate the C mechanism', 'noise': 'original global and spatial OU plus neuron Poisson',
                'manual_intervention_trigger': 'All-E 10-ms rate >=200 Hz for 200 consecutive ms, then wait 500 ms',
                'manual_intervention': 'Suspend only Z ODE; refill linearly over 1000 ms, then hold at 1 for at least 2000 ms, extending to 10.69 s if needed to capture all requested checkpoints',
                'state_preservation': 'No membrane, refractory, synapse, delay or RNG reset; I targets remain 1',
                'no_trigger': 'Stop at 20 s, with no forced depletion or restoration; latency is right-censored',
                'trigger_scope': 'Operational sustained-high-activity trigger, not seizure classifier',
                'new_biological_variables': False, 'workers': 1, 'automatic_next_round': False}
    write(OUT / 'protocol.json', protocol)
    p = s.params; p.T = 20000.; ne, ni = s.net['NE'], s.net['NI']; dt = p.dt
    assert dt == .1
    slow = RefilledZ(ne + ni, p.V_th, cfg, NE=ne)
    ref = np.load(REFERENCE / 'runs/z_current_e_seed9108401.npz')
    sample = ref['sample_ids']; sample_groups = ref['sample_groups']
    cells = spatial_cell_index(s.positions_e, n_grid=20, sheet_l_mm=p.L)
    counts = np.bincount(cells, minlength=400)
    centers = np.asarray(frozen['candidate']['node_field']['centers_mm'])
    dist = np.linalg.norm(s.positions_e[:, None] - centers[None], axis=2)
    groups = np.full(ne, 2); groups[dist[:, 0] < 1.75] = 0
    groups[(dist[:, 1] < 1.75) & (dist[:, 1] < dist[:, 0])] = 1
    region_counts = np.bincount(groups, minlength=3)
    steps = round(p.T / dt); frames = round(p.T)
    spikes = np.zeros((steps, len(sample)), bool)
    rates = np.zeros((steps, 2)); fields = np.zeros((frames, 400), np.uint16)
    regions = np.zeros((frames, 3), np.uint32)
    # All Z/current records describe the values applied to this membrane step.
    z_stats = []; z_fields = []; currents = []; sample_z = []; z_times = []
    seen = 0; recent_count = 0; high_ms = 0; detected_ms = None

    def observe_current(tm, ie, ii, v):
        step = round(tm / dt)
        if step % 100:
            return
        z = slow.z[:ne]; raw = ii[:ne]
        z_times.append(tm)
        z_stats.append([z.mean(), z.std(), *np.quantile(z, [.1, .5, .9]),
                        *[z[groups == k].mean() for k in range(3)],
                        float(np.mean(raw >= cfg.I_th_EI)),
                        float(np.mean(z * raw) / max(np.mean(raw), 1e-12))])
        z_fields.append(np.bincount(cells, weights=z, minlength=400) / counts)
        sample_z.append(z[sample[:240]].copy())
        currents.append([ie[:ne].mean(), raw.mean(), np.mean(z * raw), ie[ne:].mean(), ii[ne:].mean()])

    # Engine calls current_observer before apply_currents, so record from the
    # slow hook after applying the external refill to avoid a one-step offset.
    original_apply = slow.apply_currents
    def apply_record(ie, ii, labels=None, rec=None):
        out = original_apply(ie, ii, labels, rec)
        observe_current(slow.last_t_ms, ie, ii, None)
        return out
    slow.apply_currents = apply_record

    def observe_spikes(tm, spk):
        nonlocal seen, recent_count, high_ms, detected_ms
        step = round(tm / dt); seen = step + 1; frame = step // 10
        spikes[step] = spk[sample]; ec = int(spk[:ne].sum()); ic = int(spk[ne:].sum())
        rates[step] = [ec / ne / dt * 1000., ic / ni / dt * 1000.]
        ids = np.flatnonzero(spk[:ne]); fields[frame] += np.bincount(cells[ids], minlength=400).astype(np.uint16)
        regions[frame] += np.bincount(groups[ids], minlength=3).astype(np.uint32)
        recent_count += ec
        if seen % 100 == 0:
            rate10 = recent_count / ne / .01; recent_count = 0
            high_ms = high_ms + 10 if rate10 >= 200 else 0
            if detected_ms is None and high_ms >= 200:
                detected_ms = seen * dt; slow.restore_ms = detected_ms + 500.
        if seen % 1000 == 0:
            write(OUT / 'status.json', {'status': 'RUNNING', 'time_ms': seen * dt,
                  'elapsed_s': time.time() - started, 'z_mean': float(slow.z[:ne].mean()),
                  'z_std': float(slow.z[:ne].std()), 'E_rate_last10ms_hz': float(rates[max(0, seen-100):seen, 0].mean()),
                  'sustained_high_detected_ms': detected_ms, 'restore_start_ms': slow.restore_ms})
        if slow.restore_ms is not None and seen * dt >= max(slow.restore_ms + 3000., TIMES_MS[-1] + 10.):
            raise EndObservation()

    s.net['rng'] = np.random.default_rng(seed)
    drive = make_external_drive(s, tr['spatial_ou'], seed)
    try:
        simulate_kick(p, s.net, KICK_BOOST=0., V_th_per_neuron=s.vtheta, slow=slow,
                      external_e_rate_drive=drive, early_stop_runaway=False,
                      checkpoint_steps=[round(t / dt) for t in TIMES_MS], checkpoint_sink=capture,
                      spike_observer=observe_spikes, record_dense_spikes=False,
                      fast_scatter=True, verbose=True)
    except EndObservation:
        pass
    nframes = seen // 10
    assert seen % 10 == 0
    actual_counts = np.rint(rates[:seen, 0] * ne * dt / 1000).astype(np.int64).reshape(-1, 10).sum(1)
    assert np.array_equal(fields[:nframes].sum(1), actual_counts)
    assert np.array_equal(regions[:nframes].sum(1), actual_counts)
    np.savez_compressed(OUT / 'trajectory.npz', sample_spikes=spikes[:seen], sample_ids=sample,
                        sample_groups=sample_groups, rate_e_hz=rates[:seen, 0], rate_i_hz=rates[:seen, 1],
                        field_e_count_1ms=fields[:nframes], region_spikes_1ms=regions[:nframes],
                        region_counts=region_counts, cell_e_counts=counts, positions_e=s.positions_e,
                        cell_e=cells, dt_ms=dt, z_time_ms=np.array(z_times), z_stats=np.array(z_stats),
                        z_field_10ms=np.array(z_fields), sample_z_10ms=np.array(sample_z), currents_10ms=np.array(currents))
    write(OUT / 'run.json', {'status': 'COMPLETE', 'duration_ms': seen * dt,
          'sustained_high_detected_ms': detected_ms, 'restore_start_ms': slow.restore_ms,
          'restore_end_ms': None if slow.restore_ms is None else slow.restore_ms + 1000.,
          'Z_stats_columns': ['mean', 'std', 'p10', 'p50', 'p90', 'core_A_mean', 'core_B_mean', 'surround_mean', 'fraction_GABA_above_threshold', 'current_weighted_Z'],
          'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
          'seconds': time.time() - started, 'spatial_count_conservation': True,
          'frozen_identity': identity, 'seed': seed})
    import plot_topic4_autonomous_z_manual_restore as plot
    plot.OUT = OUT
    plot.render()
    write(OUT / 'status.json', {'status': 'COMPLETE_PENDING_SCIENTIFIC_AND_VISUAL_REVIEW',
          'duration_ms': seen * dt, 'seconds': time.time() - started,
          'figure': str(OUT / 'figures/autonomous_z_manual_restore.png'), 'automatic_next_round': False})


if __name__ == '__main__':
    try:
        run()
    except Exception as exc:
        write(OUT / 'status.json', {'status': 'FAILED', 'error': repr(exc)})
        raise
