#!/usr/bin/env python3
"""Exact replay of the autonomous-Z segment; capture all native SNN state."""
from run_topic4_autonomous_z_manual_restore import (
    ROOT, SOURCE, setup, read, write, make_external_drive, RefilledZ,
    MZSlowVarsConfig, simulate_kick)
import numpy as np
import time
from checkpoint import save

OUT = ROOT / 'results/topic4_sef_hfo/z_transition_bifurcation_audit_v1'
REFERENCE = ROOT / 'results/topic4_sef_hfo/autonomous_z_manual_restore_v1'
TIMES_MS = [8000, 9400, 9800, 10180, 10680]


class ReplayComplete(Exception):
    pass


def main():
    started = time.time(); seed = 9108401
    write(OUT / 'replay_status.json', {'status': 'BUILDING', 'started_unix': started})
    s, tr, frozen, identity = setup(seed)
    reference = np.load(REFERENCE / 'trajectory.npz')
    reference_spikes = reference['sample_spikes']
    reference_e = reference['rate_e_hz']
    reference_i = reference['rate_i_hz']
    meta = read(REFERENCE / 'run.json')
    assert meta['frozen_identity'] == identity
    cfg = read(REFERENCE / 'protocol.json')
    slow = RefilledZ(s.net['NE'] + s.net['NI'], s.params.V_th,
                    MZSlowVarsConfig(use_z=True, use_m=False, tau_z=cfg['tau_z_ms'],
                                     I_th_EI=cfg['I_th_EI'], trace_stride_steps=100), NE=s.net['NE'])
    p = s.params; p.T = TIMES_MS[-1] + p.dt
    s.net['rng'] = np.random.default_rng(seed)
    drive = make_external_drive(s, tr['spatial_ou'], seed)
    samples = reference['sample_ids']; ne = s.net['NE']; ni = s.net['NI']
    seen = 0; records = []

    def observe(tm, spk):
        nonlocal seen
        step = round(tm / p.dt)
        assert np.array_equal(spk[samples], reference_spikes[step]), ('raster mismatch', step)
        e = int(round(reference_e[step] * ne * p.dt / 1000))
        i = int(round(reference_i[step] * ni * p.dt / 1000))
        assert int(spk[:ne].sum()) == e and int(spk[ne:].sum()) == i, ('rate mismatch', step)
        seen = step + 1
        if seen % 1000 == 0:
            write(OUT / 'replay_status.json', {'status': 'RUNNING', 'time_ms': seen * p.dt,
                  'all_observed_steps_identical': True, 'saved_checkpoints': records,
                  'elapsed_s': time.time() - started})

    def capture(step, state):
        tm = round(state['absolute_time_ms'])
        file = OUT / 'checkpoints' / f't{tm}ms.npz'
        digest = save(state, file)
        records.append({'time_ms': tm, 'path': str(file), 'sha256': digest,
                        'Z_mean': float(state['slow']['z'][:ne].mean()),
                        'Z_std': float(state['slow']['z'][:ne].std())})
        write(OUT / 'checkpoint_index.json', records)
        if tm == TIMES_MS[-1]:
            raise ReplayComplete()

    try:
        simulate_kick(p, s.net, KICK_BOOST=0., V_th_per_neuron=s.vtheta, slow=slow,
                      external_e_rate_drive=drive, early_stop_runaway=False,
                      checkpoint_steps=[round(t / p.dt) for t in TIMES_MS], checkpoint_sink=capture,
                      spike_observer=observe, record_dense_spikes=False, fast_scatter=True, verbose=True)
    except ReplayComplete:
        pass
    assert seen == round(TIMES_MS[-1] / p.dt) and len(records) == len(TIMES_MS)
    write(OUT / 'replay_status.json', {'status': 'COMPLETE', 'time_ms': seen * p.dt,
          'all_observed_steps_identical': True, 'saved_checkpoints': records,
          'seconds': time.time() - started})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write(OUT / 'replay_status.json', {'status': 'FAILED', 'error': repr(exc)})
        raise
