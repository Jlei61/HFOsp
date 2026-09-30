#!/usr/bin/env python3
"""Resume the exact native prefix, capture dense Z/current and full checkpoints."""
from topic4_spatial_boundary_common import OUT, OLD, REFERENCE, read, write
from run_topic4_autonomous_z_manual_restore import setup, make_external_drive, RefilledZ, MZSlowVarsConfig, simulate_kick, spatial_cell_index
from checkpoint import load, save
import numpy as np
import time


class Finished(Exception):
    pass


def main():
    start = time.time(); lo = 8000; hi = 9400; seed = 9108401
    write(OUT / 'replay_status.json', {'status': 'BUILDING', 'start_ms': lo, 'end_ms': hi})
    s, tr, frozen, identity = setup(seed)
    ref = np.load(REFERENCE / 'trajectory.npz'); cfg = read(REFERENCE / 'protocol.json')
    assert identity == cfg['frozen_identity']
    state = load(OLD / 'checkpoints/t8000ms.npz')
    ne, ni = s.net['NE'], s.net['NI']; p = s.params; p.T = hi-lo+p.dt
    slow = RefilledZ(ne+ni, p.V_th, MZSlowVarsConfig(use_z=True, use_m=False,
        tau_z=cfg['tau_z_ms'], I_th_EI=cfg['I_th_EI'], trace_stride_steps=100), NE=ne)
    drive = make_external_drive(s, tr['spatial_ou'], seed)
    cell_e = ref['cell_e']; cell_i = spatial_cell_index(s.positions_i, n_grid=20, sheet_l_mm=p.L)
    e_counts = np.zeros((hi-lo, 400), np.uint16); i_counts = np.zeros_like(e_counts)
    snapshots = (hi-lo)//10
    z = np.empty((snapshots, ne), np.float32); raw = np.empty_like(z); excitation = np.empty_like(z)
    samples = ref['sample_ids']; ref_sp = ref['sample_spikes']; ref_e = ref['rate_e_hz']; ref_i = ref['rate_i_hz']
    seen = 0; saved = []

    def observe_current(tm, ie, ii, v):
        step = round((tm-lo)/p.dt)
        if step % 100 == 0:
            k = step//100
            z[k] = slow.z[:ne]; raw[k] = ii[:ne]; excitation[k] = ie[:ne]

    def observe_spikes(tm, spk):
        nonlocal seen
        step = round(tm/p.dt); k = step-round(lo/p.dt); frame = k//10
        assert np.array_equal(spk[samples], ref_sp[step]), ('sample raster', step)
        e = int(spk[:ne].sum()); i = int(spk[ne:].sum())
        assert e == round(ref_e[step]*ne*p.dt/1000) and i == round(ref_i[step]*ni*p.dt/1000), ('population count', step)
        e_counts[frame] += np.bincount(cell_e[spk[:ne]], minlength=400).astype(np.uint16)
        i_counts[frame] += np.bincount(cell_i[spk[ne:]], minlength=400).astype(np.uint16)
        seen = k+1
        if seen % 1000 == 0:
            write(OUT/'replay_status.json', {'status':'RUNNING','time_ms':tm+p.dt,'elapsed_s':time.time()-start,
                  'all_observed_steps_identical':True, 'saved_checkpoints':saved})

    def capture(step, payload):
        tm = round(payload['absolute_time_ms']); path = OUT/'checkpoints'/f't{tm}ms.npz'
        digest = save(payload, path)
        saved.append({'time_ms':tm,'path':str(path),'sha256':digest,'mean_Z':float(payload['slow']['z'][:ne].mean())})
        write(OUT/'checkpoint_index.json',saved)
        if tm == hi: raise Finished()

    try:
        simulate_kick(p,s.net,KICK_BOOST=0.,V_th_per_neuron=s.vtheta,slow=slow,
            external_e_rate_drive=drive,resume_state=state,time_offset_ms=lo,early_stop_runaway=False,
            current_observer=observe_current,spike_observer=observe_spikes,
            checkpoint_steps=[round(t/p.dt) for t in range(8200,9401,200)],checkpoint_sink=capture,
            record_dense_spikes=False,fast_scatter=True,verbose=False)
    except Finished:
        pass
    assert seen == round((hi-lo)/p.dt)
    assert np.array_equal(e_counts,ref['field_e_count_1ms'][lo:hi])
    assert np.allclose(z.mean(1),ref['z_stats'][lo//10:hi//10,0],atol=2e-7,rtol=0)
    np.savez_compressed(OUT/'native_boundary_moments.npz',time_ms=np.arange(lo,hi,10),z_e=z,
        raw_gaba_e=raw,raw_ampa_e=excitation,field_e_count_1ms=e_counts,field_i_count_1ms=i_counts,
        cell_e=cell_e,cell_i=cell_i,cell_e_counts=np.bincount(cell_e,minlength=400),
        cell_i_counts=np.bincount(cell_i,minlength=400),vtheta_e=s.vtheta[:ne],positions_e=s.positions_e,
        threshold=cfg['I_th_EI'],tau_z_ms=cfg['tau_z_ms'])
    write(OUT/'replay_status.json',{'status':'COMPLETE','elapsed_s':time.time()-start,
        'all_observed_steps_identical':True,'field_counts_identical':True,'saved_checkpoints':saved,
        'frozen_identity':identity,'raw_current_snapshot_step_ms':10})


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        write(OUT/'replay_status.json',{'status':'FAILED','error':repr(exc)}); raise
