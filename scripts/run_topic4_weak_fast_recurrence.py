#!/usr/bin/env python3
"""One fixed weak-fast M trajectory, externally refill Z from an exact saved high state."""
from validate_topic4_fixed_rate_base import ROOT, read, write, make_external_drive, spatial_cell_index
from topic4_historical_manual_z_common import setup, OUT as PREVIOUS
from src.topic4_raster_protocol_engine import simulate_kick
from src.snn_engine.mz_slow_vars import MZSlowVars, MZSlowVarsConfig
from lfp import LFPRecorder
import numpy as np
import argparse
import time
import resource
import os
import sys
import subprocess
import pickle
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = ROOT / 'results/topic4_sef_hfo/fig5_manual_core_release_v1'
OUT = Path(os.environ.get('TOPIC4_M_RETURN_OUT',str(BASE / 'weak_fast_z_refill_recurrence_v2')))
THRESHOLD = 95.19851312666987


class CheckpointPause(Exception):
    pass


class EndObservation(Exception):
    pass


class ReleaseZ(MZSlowVars):
    """Suspend only Z for one 1-s refill, then resume the exact original update."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.restore_ms = None
        self.restore_from = None
        self.last_t_ms = 0.
        self.refilling = False

    def _record_trace(self, spikes, dt):
        pass

    def apply_currents(self, ie, ii, labels=None, rec=None):
        tm = self._step_index * .1
        self.last_t_ms = tm
        self.refilling = self.restore_ms is not None and self.restore_ms <= tm < self.restore_ms + 1000.
        if self.restore_ms is not None and self.restore_ms <= tm <= self.restore_ms + 1000.:
            if self.restore_from is None:
                self.restore_from = self.z[:self.NE].copy()
            alpha = min(1., (tm - self.restore_ms) / 1000.)
            self.z[:self.NE] = self.restore_from + alpha * (1. - self.restore_from)
        return super().apply_currents(ie, ii, labels, rec)

    def step(self, spk, labels, dt):
        if self.refilling:
            # The external intervention fixes only Z. M must keep evolving.
            z_fixed = self.z[:self.NE].copy()
            super().step(spk, labels, dt)
            self.z[:self.NE] = z_fixed
        else:
            super().step(spk, labels, dt)


def verify_slow():
    cfg = MZSlowVarsConfig(use_z=True, use_m=False, tau_z=5000., I_th_EI=THRESHOLD)
    orig = MZSlowVars(8, 18., cfg, NE=6)
    new = ReleaseZ(8, 18., cfg, NE=6)
    rng = np.random.default_rng(51)
    for k in range(400):
        ie, ii = rng.uniform(0, 300, (2, 8)); sp = rng.random(8) < .2
        assert np.array_equal(orig.apply_currents(ie, ii), new.apply_currents(ie, ii))
        orig.step(sp, None, .1); new.step(sp, None, .1)
        assert np.array_equal(orig.z, new.z)
    new.restore_ms = 40.; z0 = new.z[:6].copy()
    for tm in (40., 540., 1039.9, 1040.):
        new._step_index = round(tm / .1)
        new.apply_currents(np.zeros(8), np.full(8, 200.))
        assert np.allclose(new.z[:6], z0 + min(1., (tm - 40.) / 1000.) * (1. - z0), atol=1e-15)
    orig.z[:] = new.z; orig._step_index = new._step_index
    for k in range(200):
        ie, ii = rng.uniform(0, 300, (2, 8)); sp = rng.random(8) < .2
        assert np.array_equal(orig.apply_currents(ie, ii), new.apply_currents(ie, ii))
        orig.step(sp, None, .1); new.step(sp, None, .1)
        assert np.array_equal(orig.z, new.z)
    assert new.z[:6].mean() < 1 and np.array_equal(new.z[6:], np.ones(2))
    cfg_m=MZSlowVarsConfig(use_z=True,use_m=True,tau_z=5000.,I_th_EI=THRESHOLD,tau_adp=2000.,eta_m=.2)
    orig=MZSlowVars(8,18.,cfg_m,NE=6);new=ReleaseZ(8,18.,cfg_m,NE=6)
    new.restore_ms=10.
    for k in range(12000):
        ie,ii=rng.uniform(0,300,(2,8));sp=rng.random(8)<.1
        new.apply_currents(ie,ii);orig.apply_currents(ie,ii)
        before=new.z.copy();new.step(sp,None,.1);orig.step(sp,None,.1)
        assert np.array_equal(new.m,orig.m)
        if new.refilling:assert np.array_equal(new.z,before)
    assert np.max(new.m[:6])>0 and np.all(new.m[6:]==0)
    return dict(status='PASS',native_before_refill_bitwise_steps=400,native_after_release_bitwise_steps=200,
                M_carries_through_Z_refill_steps=12000,I_cell_M_zero=True)


def worker(job):
    name = job['name']; folder = OUT / 'runs'; folder.mkdir(parents=True, exist_ok=True)
    result_file = folder / (name + '.json')
    if result_file.exists() and read(result_file).get('status') == 'COMPLETE':
        return read(result_file)
    assert job['refill'] and job['eta_m']==.02 and job['tau_adp_ms']==2000.
    assert job['restore_start_ms'] >= job['branch_time_ms']
    started = time.time()
    checkpoint_path=OUT/'checkpoints'/(name+'.pkl')
    checkpoint_path.parent.mkdir(parents=True,exist_ok=True)
    engine_hash=hashlib.sha256((ROOT/'src/topic4_raster_protocol_engine.py').read_bytes()).hexdigest()
    progress = OUT / 'progress' / (name + '.json')
    write(progress, dict(status='BUILDING', job=job))
    s, tr, frozen, identity = setup(job['seed'])
    assert identity == read(PREVIOUS / 'substrate.json')['identity']
    assert np.max(s.vtheta[:s.n_e]) <= 18.
    p = s.params; p.T = job['duration_ms']; dt = p.dt; ne, ni = s.n_e, s.n_i
    cfg = MZSlowVarsConfig(use_z=True,use_m=True,tau_z=job['tau_z_ms'],I_th_EI=job['threshold'],
                          tau_adp=job['tau_adp_ms'],eta_m=job['eta_m'])
    slow = ReleaseZ(ne + ni, p.V_th, cfg, NE=ne)
    detail = job.get('detail', False)
    ref = np.load(PREVIOUS / 'reference_samples/runs/z_current_e_seed9108401.npz')
    samples = ref['sample_ids']; sample_groups = ref['sample_groups']
    centers = np.asarray(frozen['candidate']['node_field']['centers_mm'])
    def regions(pos):
        d = np.linalg.norm(pos[:, None] - centers[None], axis=2)
        g = np.full(len(pos), 2); g[d[:, 0] < 1.75] = 0
        g[(d[:, 1] < 1.75) & (d[:, 1] < d[:, 0])] = 1
        return g
    ge, gi = regions(s.positions_e), regions(s.positions_i)
    nreg = np.r_[np.bincount(ge, minlength=3), np.bincount(gi, minlength=3)]
    cells = spatial_cell_index(s.positions_e, n_grid=20, sheet_l_mm=p.L)
    ncell = np.bincount(cells, minlength=400)
    steps = round(p.T / dt); frames = round(p.T)
    rates = np.empty((steps, 2), np.float64)
    raster = np.zeros((steps, len(samples)), bool) if detail else None
    fields = np.zeros((frames, 400), np.uint16) if detail else None
    region_counts = np.zeros((frames, 6), np.uint16) if detail else None
    zt, zs, zf, currents, zrhs, inputs, mstats = [], [], [], [], [], [], []
    lfp_time, lfp_raw, lfp_effective = [], [], []
    recorder = LFPRecorder(p, s.net['pos'], s.net['labels'], sites=s.contact_xy) if detail else None
    history=[]
    seen = recent = high_ms = 0
    triggers = []; high_active = False; low_ms = 0
    prefix_qa = None
    old = np.load(BASE/'runs/continuous_refill_release.npz')
    resume_state=None;resume_step=0;previous_seconds=0.
    if checkpoint_path.exists():
        with checkpoint_path.open('rb') as stream:checkpoint=pickle.load(stream)
        assert checkpoint['job']==job and checkpoint['identity']==identity
        assert checkpoint['engine_hash']==engine_hash
        resume_state=checkpoint['engine'];resume_step=int(resume_state['step']);seen=resume_step
        o=checkpoint['observations'];nf=seen//10
        rates[:seen]=o['rates']
        if detail:
            raster[:seen]=o['raster'];fields[:nf]=o['fields'];region_counts[:nf]=o['region_counts']
        zt,zs,zf,currents,zrhs,inputs,mstats=[o[k] for k in ['zt','zs','zf','currents','zrhs','inputs','mstats']]
        lfp_time,lfp_raw,lfp_effective=[o[k] for k in ['lfp_time','lfp_raw','lfp_effective']]
        h=checkpoint['tracker'];history=h['history'];recent=h['recent'];high_ms=h['high_ms'];low_ms=h['low_ms']
        triggers=h['triggers'];high_active=h['high_active'];prefix_qa=h['prefix_qa'];previous_seconds=h['seconds']
        slow.restore_ms=job['restore_start_ms']
        slow.restore_from=h.get('restore_from')
        del checkpoint,o,h
    interrupted=OUT/'interrupted_attempt_20260911'/(name+'.json')
    original_history={v['time_s']:v for v in read(interrupted).get('history',[])} if interrupted.exists() else {}
    replay_checked=[]
    original_apply = slow.apply_currents
    def apply(ie, ii, labels=None, rec=None):
        result = original_apply(ie, ii, labels, rec)
        k = slow._step_index; tm = k * dt; z = slow.z[:ne]
        if detail and k % 5 == 0:
            # Original Eq 9-11 readout, plus a separately named applied-inhibition proxy.
            lfp_time.append(tm); lfp_raw.append(recorder.sample(ie, ii))
            applied = ii.copy(); applied[:ne] *= z
            lfp_effective.append(recorder.sample(ie, applied))
        if k % 50 == 0:
            raw = ii[:ne]; duty = np.mean(raw >= cfg.I_th_EI)
            zt.append(tm)
            zs.append([z.mean(), z.std(), *np.quantile(z, [.1, .5, .9]),
                       *[z[ge == g].mean() for g in range(3)], duty,
                       float(np.mean(z * raw) / max(np.mean(raw), 1e-12)), float(slow.m[:ne].mean())])
            m=slow.m[:ne]
            mstats.append([m.mean(),m.std(),*np.quantile(m,[.1,.5,.9]),
                           *[m[ge==g].mean() for g in range(3)]])
            zrhs.append([(1. - duty - z.mean()) / (cfg.tau_z / 1000.),
                         float(slow.refilling)])
            if detail:
                zf.append(np.bincount(cells, weights=z, minlength=400) / ncell)
                currents.append([ie[:ne].mean(), raw.mean(), np.mean(z * raw), ie[ne:].mean(), ii[ne:].mean()])
        return result
    slow.apply_currents = apply
    def observe_input(tm, nu, xi):
        if round(tm / dt) % 1000 == 0:
            inputs.append([tm, xi, nu[:ne].mean(), nu[ne:].mean()])
    def observe(tm, spk):
        nonlocal seen, recent, high_ms, low_ms, high_active, prefix_qa
        k = round(tm / dt); seen = k + 1; frame = k // 10
        ec = int(spk[:ne].sum()); ic = int(spk[ne:].sum())
        rates[k] = [ec / ne / dt * 1000., ic / ni / dt * 1000.]
        if detail:
            raster[k] = spk[samples]
            fields[frame] += np.bincount(cells[spk[:ne]], minlength=400).astype(np.uint16)
            region_counts[frame, :3] += np.bincount(ge[spk[:ne]], minlength=3).astype(np.uint16)
            region_counts[frame, 3:] += np.bincount(gi[spk[ne:]], minlength=3).astype(np.uint16)
            if seen == 6000:
                assert np.array_equal(np.asarray(inputs),old['input_summary'][:len(inputs)])
                prefix_qa = '600 ms external drive exactly matches M-off; recurrent spikes may differ with M'
        recent += ec
        if seen % 100 == 0:
            rate10 = recent / ne / .01; recent = 0
            high_ms = high_ms + 10 if rate10 >= 200. else 0
            low_ms = low_ms + 10 if rate10 < 200. else 0
            if low_ms >= 200: high_active = False
            if high_ms >= 200 and not high_active:
                high_active = True; triggers.append(seen * dt)
                if job['refill'] and slow.restore_ms is None:
                    slow.restore_ms = seen * dt + 500.
        if seen % 5000 == 0:
            window=rates[max(0,seen-5000):seen,0].reshape(-1,100).mean(1)
            state=dict(time_s=seen*dt/1000,mean_Z=float(slow.z[:ne].mean()),
                       mean_M=float(slow.m[:ne].mean()),mean_adaptation_current=float(cfg.eta_m*slow.m[:ne].mean()),
                       E_mean_hz=float(window.mean()),quiet_fraction=float(np.mean(window<5)),
                       high_fraction=float(np.mean(window>=200)))
            if state['time_s'] in original_history:
                prev=original_history[state['time_s']]
                for key in ['mean_Z','mean_M','mean_adaptation_current','E_mean_hz','quiet_fraction','high_fraction']:
                    assert abs(state[key]-prev[key])<1e-10,(key,state[key],prev[key])
                replay_checked.append(state['time_s'])
            history.append(state)
            write(progress, dict(status='RUNNING', seconds=previous_seconds+time.time()-started,job=job,
                                 worker_pid=os.getpid(),resumed_at_s=resume_step*dt/1000,checkpoint_file=str(checkpoint_path),
                                 first_trigger_ms=triggers[0] if triggers else None,
                                 all_trigger_times_ms=triggers,**state,history=history))
        # Complete the fixed horizon even after high activity: recovery is an outcome.
    def checkpoint_sink(k,state):
        # Always carry the complete fast state, both OU generators, Z/M, and observations together.
        if not job.get('qa_pause') and not high_active and k % 20000 and not (job['restore_start_ms'] <= k*dt <= job['restore_start_ms']+1500):
            return
        assert k==seen and k%100==0
        observation=dict(rates=rates[:seen],raster=None if raster is None else raster[:seen],
            fields=None if fields is None else fields[:seen//10],
            region_counts=None if region_counts is None else region_counts[:seen//10],
            zt=zt,zs=zs,zf=zf,currents=currents,zrhs=zrhs,inputs=inputs,mstats=mstats,
            lfp_time=lfp_time,lfp_raw=lfp_raw,lfp_effective=lfp_effective)
        tracker=dict(history=history,recent=recent,high_ms=high_ms,low_ms=low_ms,triggers=triggers,
            high_active=high_active,prefix_qa=prefix_qa,seconds=previous_seconds+time.time()-started,
            restore_from=slow.restore_from)
        package=dict(schema='native_m_complete_state_v1',job=job,identity=identity,engine_hash=engine_hash,
            engine=state,observations=observation,tracker=tracker)
        tmp=checkpoint_path.with_suffix('.tmp')
        with tmp.open('wb') as stream:
            pickle.dump(package,stream,protocol=5);stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,checkpoint_path)
        write(checkpoint_path.with_suffix('.json'),dict(status='SAVED',time_s=k*dt/1000,
            bytes=checkpoint_path.stat().st_size,worker_pid=os.getpid(),includes_observations=True))
        if len(triggers)>1 and k*dt >= triggers[1]+job['stop_after_second_high_ms']:
            raise EndObservation()
        if job.get('qa_pause') and resume_step==0:raise CheckpointPause()
    s.net['rng'] = np.random.default_rng(job['seed'])
    drive = make_external_drive(s, tr['spatial_ou'], job['seed'])
    p.T=job['duration_ms']-resume_step*dt
    assert p.T>0
    try:
        simulate_kick(p, s.net, KICK_BOOST=0., V_th_per_neuron=s.vtheta, slow=slow,
                      external_e_rate_drive=drive, early_stop_runaway=False,
                      spike_observer=observe, input_observer=observe_input,
                      record_dense_spikes=False, fast_scatter=True, verbose=False,
                      resume_state=resume_state,time_offset_ms=resume_step*dt,
                      checkpoint_steps=range((resume_step//5000+1)*5000,steps,5000),checkpoint_sink=checkpoint_sink)
    except CheckpointPause:
        r=dict(status='CHECKPOINTED_FOR_QA',time_s=seen*dt/1000);write(progress,r);return r
    except EndObservation:
        pass
    payload = dict(rate_e_hz=rates[:seen, 0], rate_i_hz=rates[:seen, 1], dt_ms=dt,
                   z_time_ms=np.asarray(zt), z_stats=np.asarray(zs), z_rhs=np.asarray(zrhs),
                   input_summary=np.asarray(inputs),final_z_e=slow.z[:ne],
                   m_stats=np.asarray(mstats),final_m_e=slow.m[:ne])
    if detail:
        nf = seen // 10
        counts = np.rint(rates[:seen, 0] * ne * dt / 1000).astype(np.int64).reshape(-1, 10).sum(1)
        assert np.array_equal(counts, fields[:nf].sum(1))
        assert np.array_equal(counts, region_counts[:nf, :3].sum(1))
        icount = np.rint(rates[:seen, 1] * ni * dt / 1000).astype(np.int64).reshape(-1, 10).sum(1)
        assert np.array_equal(icount, region_counts[:nf, 3:].sum(1))
        zerror = float(np.max(np.abs(np.average(zf, axis=1, weights=ncell)-np.asarray(zs)[:, 0])))
        assert zerror < 1e-12
        if slow.restore_ms is not None and slow.restore_ms+1000. < seen*dt:
            release = slow.restore_ms + 1000.
            ix = np.flatnonzero(np.asarray(zt) == release)[0]
            assert np.all(np.asarray(zf)[ix] == 1.)
            assert np.any(np.asarray(zs)[ix+1:, 0] < .999)
        payload.update(sample_spikes=raster[:seen], sample_ids=samples, sample_groups=sample_groups,
                       field_e_count_1ms=fields[:nf], region_spikes_1ms=region_counts[:nf], region_counts=nreg,
                       cell_e_counts=ncell, positions_e=s.positions_e, cell_e=cells, centers_mm=centers,
                       z_field_5ms=np.asarray(zf), currents_5ms=np.asarray(currents),
                       lfp_time_ms=np.asarray(lfp_time), lfp_raw=np.asarray(lfp_raw),
                       lfp_effective=np.asarray(lfp_effective), contact_names=np.asarray(s.contact_names),
                       contact_xy=s.contact_xy, valid_contacts=s.valid_contacts,
                       shaft_ids=np.asarray(s.shaft_ids))
    nmatch=min(len(inputs),len(old['input_summary']))
    assert np.array_equal(np.asarray(inputs)[:nmatch],old['input_summary'][:nmatch])
    baseline_prefix=None
    if cfg.eta_m == 0:
        npre=round(11180./dt)
        assert np.array_equal(rates[:npre,0],old['rate_e_hz'][:npre])
        assert np.array_equal(rates[:npre,1],old['rate_i_hz'][:npre])
        baseline_prefix='M current off exactly reproduces both E and I rates until old external refill at11.18s'
    assert np.max(slow.m[:ne])>0 and np.all(slow.m[ne:]==0)
    np.savez_compressed(folder / (name + '.npz'), **payload)
    result = dict(status='COMPLETE', job=job, duration_ms=seen*dt, first_trigger_ms=triggers[0] if triggers else None,
                  all_trigger_times_ms=triggers, restore_start_ms=slow.restore_ms,
                  release_ms=None if slow.restore_ms is None else slow.restore_ms+1000.,
                  observation_horizon_ms=job['duration_ms'], event_observed=bool(triggers),
                  peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                  seconds=previous_seconds+time.time()-started, frozen_identity=identity,
                  resumed_at_s=resume_step*dt/1000,replayed_old_progress_points=len(replay_checked),
                  runner_path=str(Path(__file__).resolve()),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  prefix_qa=prefix_qa,M_enabled=True,eta_m=cfg.eta_m,tau_adp_ms=cfg.tau_adp,
                  final_mean_Z=float(slow.z[:ne].mean()),final_mean_M=float(slow.m[:ne].mean()),
                  external_input_prefix_matches_M_off=True,external_input_match_samples=nmatch,
                  external_input_match_duration_s=nmatch*.1,baseline_rate_prefix=baseline_prefix,
                  adaptation_gain_per_hz=cfg.eta_m*cfg.tau_adp/1000,history=history,
                  neuron_counts={'E':ne,'I':ni},Vth_counts_E={'lower':int((s.vtheta[:ne]<18).sum()),
                    'equal':int((s.vtheta[:ne]==18).sum()),'raised':int((s.vtheta[:ne]>18).sum())})
    if detail: result.update(spatial_count_conservation=True, spatial_Z_reconstruction_error=zerror)
    write(result_file, result); write(progress, result)
    return result


if __name__=='__main__':
    import fcntl
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);args=parser.parse_args()
    job=read(args.job);(OUT/'locks').mkdir(parents=True,exist_ok=True)
    with (OUT/'locks'/(job['name']+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:worker(job)
        except Exception as exc:
            write(OUT/'progress'/(job['name']+'.json'),dict(status='FAILED',job=job,error=repr(exc),worker_pid=os.getpid()))
            raise
