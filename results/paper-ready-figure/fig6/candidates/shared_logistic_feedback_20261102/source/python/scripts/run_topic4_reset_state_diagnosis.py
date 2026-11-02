#!/usr/bin/env python3
"""Bounded, conditional long-run diagnosis of Z-only versus Z/M restoration.

All interventions act on state, never on weights, thresholds or noise amplitudes.
Ten-second recording blocks and state-only checkpoints bound RAM and disk use.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import numpy as np
import run_topic4_weak_fast_recurrence as old

ROOT = old.ROOT
OUT = ROOT / 'results/topic4_sef_hfo/reset_state_diagnosis_20260911'
PREVIOUS = old.OUT
RELEASE_MS = 76500.
HORIZON_MS = 1000000.
SEED = 9108401
ENGINE = ROOT / 'src/topic4_raster_protocol_engine.py'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    old.write(path, obj)


def save_pickle(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        pickle.dump(obj, stream, protocol=5)
    tmp.replace(path)


def load_pickle(path):
    with Path(path).open('rb') as stream:
        return pickle.load(stream)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class StopRun(Exception):
    pass


class ReleaseZ(old.ReleaseZ):
    # Keep the checkpoint protocol's existing class name. The only extra action
    # is one exactly timed M reset, before the first post-refill current update.
    def __init__(self, *args, reset_m=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_m = reset_m
        self.reset_audit = None

    def apply_currents(self, ie, ii, labels=None, rec=None):
        if self.reset_m and self._step_index == 765000:
            before = self.m[:self.NE].copy()
            self.m[:self.NE] = 0.
            self.reset_audit = dict(time_s=76.5, before_mean=float(before.mean()),
                                   after_max=float(self.m[:self.NE].max()),
                                   I_cells_zero=bool(np.all(self.m[self.NE:] == 0)))
        return super().apply_currents(ie, ii, labels, rec)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {
        'high_parent': old.BASE / 'weak_fast_z_refill_v1/parent_state.pkl',
        'extension_parent': PREVIOUS / 'checkpoints/weak_fast_z_refill_recurrence.pkl',
    }
    records = {}
    for label, path in sources.items():
        dest = OUT / 'parents' / (label + '.pkl')
        if not dest.exists():
            p = load_pickle(path)
            assert p['engine_hash'] == sha(ENGINE)
            compact = {k: p[k] for k in ('schema', 'identity', 'engine_hash', 'engine')}
            compact['restore_from'] = p['tracker'].get('restore_from')
            compact['source'] = str(path)
            save_pickle(dest, compact)
            del p, compact
        p = load_pickle(dest)
        records[label] = dict(path=str(dest), time_s=p['engine']['absolute_time_ms']/1000)
        del p
    plan = dict(
        status='PREPARED', substrate='historical manual dual core on fixed C fast carrier',
        eta_M=.02, tau_M_s=2., tau_Z_s=5., topology_seed=6101, original_noise_seed=SEED,
        original_trajectory_horizon_s=1000., parents=records,
        M_reset='At 76.5 s, set all E-cell M to zero once, with Z=1; resume native M immediately',
        fast_state='Preserve V, refractory, AMPA/GABA states and delay rings unless an explicitly named reset arm',
        noise='No replay or reseeding in original continuation; paired new-future arms share both RNG states and OU field states',
        recurrence='After >=2 s recovered activity (each 1 s mean E<50 Hz and quiet-bin fraction>=0.2), E>=200 Hz in 10-ms bins for >=200 ms',
        stopping='Stop 5 s after confirmed recurrence; otherwise right-censor at horizon. Never infer permanent inability from censoring.',
        stages=[
            'A: Continue original Z-only trajectory to absolute 1000 s.',
            'B: If A censored, start at the original release state and reset M once; same future noise, absolute 1000 s.',
            'C: If B recurs, run three paired new-future-noise Z-only versus Z+M comparisons, 500 s after release each.',
            'D: If A and B censored, test full fast-state clearing with Z/M reset, plus original full-initialization replay.',
            'E: If fast-state clearing permits entry, separate voltage/refractory clearing from synaptic/delay clearing; same horizon.',
        ],
        max_workers=3, parameter_search=False, patient_fit_acceptance=False,
        source_hashes={str(p): sha(p) for p in [ENGINE, Path(old.__file__),
                       ROOT/'src/snn_engine/mz_slow_vars.py', ROOT/'scripts/topic4_historical_manual_z_common.py']})
    if (OUT/'plan.json').exists():
        assert read(OUT/'plan.json')['source_hashes'] == plan['source_hashes']
    else:
        write(OUT/'plan.json', plan)
    write(OUT/'status.json', dict(status='PREPARED', phase='QA', plan=str(OUT/'plan.json')))


def worker(job):
    folder = OUT/'runs'/job['name']
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder/'result.json'
    if result_path.exists():
        return read(result_path)
    for path, digest in read(OUT/'plan.json')['source_hashes'].items():
        assert sha(path) == digest, 'Changed physical source: ' + path
    import fcntl
    lock = (folder/'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.time()
    s, tr, frozen, identity = old.setup(SEED)
    ne = s.n_e
    p = s.params
    assert ne == 32000 and s.n_i == 8000 and p.dt == .1
    checkpoint = folder/'checkpoint.pkl'
    tracker = dict(recovered=bool(job.get('already_recovered', False)), recovery_s=None,
                   history=[], recurrence_onset_s=None, recurrence_confirmation_s=None,
                   high_run_bins=0, rolling_rates=[], n_high_bins=0,
                   prior_wall_s=0., reset_audit=None)
    state = None
    resume_from_checkpoint = checkpoint.exists()
    if resume_from_checkpoint:
        package = load_pickle(checkpoint)
        assert package['job'] == job
        state = package['engine']; tracker = package['tracker']
        prior_wall = tracker['prior_wall_s']
        restore_from = package.get('restore_from')
    elif job.get('parent'):
        package = load_pickle(job['parent'])
        assert package['identity'] == identity
        assert package['engine_hash'] == sha(ENGINE)
        state = package['engine']; restore_from = package.get('restore_from')
        prior_wall = 0.
    else:
        restore_from = None
        prior_wall = 0.
    if state is not None and not resume_from_checkpoint:
        if job.get('future_seed'):
            state['rng_state'] = np.random.default_rng(job['future_seed']).bit_generator.state
            state['external_drive']['rng_state'] = np.random.default_rng(job['future_seed']+100000).bit_generator.state
        reset = job.get('fast_reset', '')
        if reset in ('all', 'voltage'):
            state['V'].fill(p.V_reset); state['ref'].fill(0)
        if reset in ('all', 'synapse'):
            for key in ['s_E', 'I_E', 's_I', 'I_I', 'ring_sE', 'ring_sI']:
                state[key].fill(0)
            state['slow']['I_I_last'].fill(0)
        write(folder/'initial_state_audit.json', dict(
            absolute_time_s=state['absolute_time_ms']/1000,
            M_reset_s=76.5 if job.get('reset_m') else None,
            fast_reset=reset or 'none', future_seed=job.get('future_seed'),
            mean_Z=float(state['slow']['z'][:ne].mean()),
            mean_M=float(state['slow']['m'][:ne].mean()),
            zero_delay_history=bool(not np.any(state['ring_sE']) and not np.any(state['ring_sI'])),
            V_sd=float(state['V'].std()), parameters_unchanged=True))
    offset = 0. if state is None else state['absolute_time_ms']
    first_step = round(offset/.1)
    end_step = round(job['horizon_ms']/.1)
    cfg = old.MZSlowVarsConfig(use_z=True, use_m=True, tau_z=5000.,
                              I_th_EI=old.THRESHOLD, tau_adp=2000., eta_m=.02)
    slow = ReleaseZ(ne+s.n_i, p.V_th, cfg, NE=ne, reset_m=job.get('reset_m', False))
    slow.restore_ms = None if job.get('fresh') else 75500.
    slow.restore_from = restore_from
    centers = np.asarray(frozen['candidate']['node_field']['centers_mm'])
    def groups(pos):
        d = np.linalg.norm(pos[:, None] - centers[None], axis=2)
        g = np.full(len(pos), 2); g[d[:,0] < 1.75] = 0
        g[(d[:,1]<1.75) & (d[:,1]<d[:,0])] = 1
        return g
    ge, gi = groups(s.positions_e), groups(s.positions_i)
    nr = np.r_[np.bincount(ge,minlength=3), np.bincount(gi,minlength=3)]
    cells = old.spatial_cell_index(s.positions_e, n_grid=20, sheet_l_mm=p.L)
    nc = np.bincount(cells,minlength=400)
    sample_source = np.load(old.PREVIOUS/'reference_samples/runs/z_current_e_seed9108401.npz')
    sel = np.r_[np.arange(0,60,3),np.arange(60,120,3),np.arange(120,240,6),np.arange(240,300,3)]
    samples = sample_source['sample_ids'][sel]
    recorder = old.LFPRecorder(p,s.net['pos'],s.net['labels'],sites=s.contact_xy)
    geometry_path = OUT/'geometry.npz'
    if not geometry_path.exists():
        temp=folder/'geometry.tmp.npz'
        np.savez_compressed(temp,region_counts=nr,cell_e_counts=nc,centers_mm=centers,
                            contact_names=s.contact_names,contact_xy=s.contact_xy,
                            sample_ids=samples,sample_source_indices=sel,positions_e=s.positions_e)
        temp.replace(geometry_path)
    data = {}; block_start = first_step; local_inputs = hashlib.sha256()
    recent = 0
    def empty():
        data.clear()
        data.update(time_ms=[],spikes_1ms=[],regions_1ms=[],field_5ms=[],field_time_ms=[],
                    raster=[],slow_time_ms=[],Z=[],M=[],Z_field=[],currents=[],lfp_time_ms=[],lfp_raw=[],inputs=[])
    empty()
    counts1=np.zeros(2,dtype=np.uint32); reg1=np.zeros(6,dtype=np.uint32); field5=np.zeros(400,dtype=np.uint32)
    original_apply=slow.apply_currents
    def apply(ie,ii,labels=None,rec=None):
        value=original_apply(ie,ii,labels,rec)
        k=slow._step_index;tm=k*.1
        if k%5==0:
            data['lfp_time_ms'].append(tm);data['lfp_raw'].append(recorder.sample(ie,ii))
        if k%50==0:
            z=slow.z[:ne];m=slow.m[:ne]
            data['slow_time_ms'].append(tm)
            data['Z'].append([z.mean(),z.std(),*np.quantile(z,[.1,.5,.9]),*[z[ge==g].mean() for g in range(3)],np.mean(ii[:ne]>=old.THRESHOLD)])
            data['M'].append([m.mean(),*[m[ge==g].mean() for g in range(3)]])
            data['Z_field'].append(np.bincount(cells,weights=z,minlength=400)/nc)
            data['currents'].append([ie[:ne].mean(),ii[:ne].mean(),np.mean(z*ii[:ne])])
        return value
    slow.apply_currents=apply
    def input_observer(tm,nu,xi):
        if round(tm/.1)%1000==0:
            data['inputs'].append([tm,xi,nu[:ne].mean(),nu[ne:].mean()])
            local_inputs.update(np.asarray(nu).tobytes());local_inputs.update(np.float64(xi).tobytes())
    def observe(tm,spk):
        nonlocal recent
        k=round(tm/.1);ec=int(spk[:ne].sum());ic=int(spk[ne:].sum())
        counts1[0]+=ec;counts1[1]+=ic;recent+=ec
        reg1[:3]+=np.bincount(ge[spk[:ne]],minlength=3).astype(np.uint32)
        reg1[3:]+=np.bincount(gi[spk[ne:]],minlength=3).astype(np.uint32)
        field5[:]+=np.bincount(cells[spk[:ne]],minlength=400).astype(np.uint32)
        data['raster'].append(spk[samples])
        if (k+1)%10==0:
            data['time_ms'].append((k+1)*.1-.5);data['spikes_1ms'].append(counts1.copy());counts1.fill(0)
            data['regions_1ms'].append(reg1.copy());reg1.fill(0)
        if (k+1)%50==0:
            data['field_time_ms'].append((k+1)*.1-2.5);data['field_5ms'].append(field5.copy());field5.fill(0)
        if (k+1)%100==0:
            rate=recent/ne/.01;recent=0;sec=(k+1)*.0001
            rr=tracker['rolling_rates'];rr.append(rate)
            if len(rr)>200:del rr[0]
            eligible=job.get('fresh',False) or sec>=76.5
            if eligible and len(rr)==200 and not tracker['recovered']:
                v=np.asarray(rr).reshape(2,100)
                if np.all(v.mean(1)<50) and np.all((v<5).mean(1)>=.2):
                    tracker['recovered']=True;tracker['recovery_s']=sec-2.
            tracker['high_run_bins']=tracker['high_run_bins']+1 if rate>=200 else 0
            if rate>=200:tracker['n_high_bins']+=1
            if (eligible and tracker['recovered'] and tracker['high_run_bins']>=20
                    and tracker['recurrence_onset_s'] is None):
                tracker['recurrence_confirmation_s']=sec
                tracker['recurrence_onset_s']=sec-.2
        if (k+1)%10000==0:
            rr=np.asarray(tracker['rolling_rates'][-100:])
            summary=dict(time_s=(k+1)*.0001,mean_Z=float(slow.z[:ne].mean()),
                         Z_cores=[float(slow.z[:ne][ge==g].mean()) for g in [0,1]],
                         adaptation_current=float(.02*slow.m[:ne].mean()),E_rate=float(rr.mean()),
                         quiet_fraction=float((rr<5).mean()))
            tracker['history'].append(summary)
            write(folder/'progress.json',dict(status='RUNNING',job=job,pid=os.getpid(),
                  wall_s=prior_wall+time.time()-started,recurrence_onset_s=tracker['recurrence_onset_s'],
                  recovered=tracker['recovered'],**summary))
    def checkpoint_sink(k,engine):
        nonlocal block_start,local_inputs
        tm=k*.1
        if k==765000 and job.get('save_release'):
            save_pickle(OUT/'parents/release.pkl',dict(schema='native_m_complete_state_v1',
                identity=identity,engine_hash=sha(ENGINE),engine=engine,restore_from=slow.restore_from))
        confirmation=tracker['recurrence_confirmation_s']
        stop=(k>=end_step or (confirmation is not None and tm/1000>=confirmation+5))
        if (k-block_start)>=100000 or stop:
            arrays={key:np.asarray(value) for key,value in data.items()}
            assert arrays['spikes_1ms'][:,0].sum()==arrays['regions_1ms'][:,:3].sum()==arrays['field_5ms'].sum()
            assert arrays['spikes_1ms'][:,1].sum()==arrays['regions_1ms'][:,3:].sum()
            arrays['raster']=arrays['raster'].astype(bool)
            for key in ['spikes_1ms','regions_1ms','field_5ms']:arrays[key]=arrays[key].astype(np.uint16)
            arrays['start_step']=np.asarray(block_start);arrays['end_step']=np.asarray(k)
            arrays['input_digest']=np.asarray(local_inputs.hexdigest())
            chunks=folder/'chunks';chunks.mkdir(exist_ok=True)
            chunk_path=chunks/f'{block_start:010d}_{k:010d}.npz'
            tmp=chunk_path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arrays);tmp.replace(chunk_path)
            tracker['prior_wall_s']=prior_wall+time.time()-started
            if slow.reset_audit is not None:tracker['reset_audit']=slow.reset_audit
            save_pickle(checkpoint,dict(job=job,identity=identity,engine_hash=sha(ENGINE),
                engine=engine,tracker=tracker,restore_from=slow.restore_from))
            empty();block_start=k;local_inputs=hashlib.sha256()
        if stop:raise StopRun()
    s.net['rng']=np.random.default_rng(SEED)
    drive=old.make_external_drive(s,tr['spatial_ou'],SEED)
    p.T=(end_step-first_step+1)*.1
    checkpoints=set(range((first_step//5000+1)*5000,end_step+1,5000))|{end_step}
    if first_step<765000<=end_step:checkpoints.add(765000)
    try:
        old.simulate_kick(p,s.net,KICK_BOOST=0.,V_th_per_neuron=s.vtheta,slow=slow,
            external_e_rate_drive=drive,early_stop_runaway=False,spike_observer=observe,
            input_observer=input_observer,record_dense_spikes=False,fast_scatter=True,
            resume_state=state,time_offset_ms=offset,checkpoint_steps=checkpoints,
            checkpoint_sink=checkpoint_sink)
    except StopRun:
        pass
    result=dict(status='COMPLETE',job=job,end_s=block_start*.0001,
                recurrence_observed=tracker['recurrence_onset_s'] is not None,
                interpretation=('RECURRENT_HIGH_AFTER_RETURN' if tracker['recurrence_onset_s'] is not None
                                else 'RIGHT_CENSORED_WITHOUT_RECURRENT_HIGH' if tracker['recovered']
                                else 'LOW_ACTIVITY_RETURN_NOT_ESTABLISHED'),
                wall_s=prior_wall+time.time()-started,**tracker)
    write(result_path,result);write(folder/'progress.json',result)
    return result


def launch(job):
    path=OUT/'jobs'/(job['name']+'.json');write(path,job)
    log=OUT/'logs'/(job['name']+'.log');log.parent.mkdir(exist_ok=True)
    with log.open('a') as stream:
        return subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker','--job',str(path)],
                                stdout=stream,stderr=subprocess.STDOUT)


def job(name,parent,**kw):
    return dict(name=name,parent=str(OUT/'parents'/(parent+'.pkl')) if parent else None,
                horizon_ms=HORIZON_MS,**kw)


def run_group(jobs,phase):
    pending=list(jobs);running=[]
    while pending or running:
        while pending and len(running)<3:
            j=pending.pop(0);p=launch(j);running.append((p,j))
        for p,j in list(running):
            rc=p.poll()
            if rc is not None:
                if rc:raise RuntimeError(f'{j["name"]} failed, exit {rc}; see logs')
                running.remove((p,j))
        write(OUT/'status.json',dict(status='RUNNING',phase=phase,supervisor_pid=os.getpid(),
              running=[j['name'] for _,j in running],pending=[j['name'] for j in pending]))
        analyze()
        if running:time.sleep(20)
    return [read(OUT/'runs'/j['name']/'result.json') for j in jobs]


def qa():
    tests=[job('qa_extension','extension_parent',horizon_ms_override=240000.,already_recovered=True),
           job('qa_refill','high_parent',horizon_ms_override=76600.,save_release=True)]
    for j in tests:j['horizon_ms']=j.pop('horizon_ms_override')
    run_group(tests,'CONTINUITY_QA')
    with np.load(PREVIOUS/'runs/weak_fast_z_refill_recurrence.npz') as f:
        source={k:f[k] for k in ['rate_e_hz','rate_i_hz','sample_spikes','region_spikes_1ms',
            'field_e_count_1ms','lfp_raw','z_stats','m_stats','input_summary']}
    sel=np.load(OUT/'geometry.npz')['sample_source_indices']
    checks={}
    for j in tests:
        for path in sorted((OUT/'runs'/j['name']/'chunks').glob('*.npz')):
            with np.load(path) as a:
                lo=int(a['start_step']);hi=int(a['end_step'])
                expected=np.rint(np.column_stack([source['rate_e_hz'][lo:hi].reshape(-1,10).mean(1)*32,
                                                source['rate_i_hz'][lo:hi].reshape(-1,10).mean(1)*8])).astype(np.uint16)
                row=dict(spikes=np.array_equal(a['spikes_1ms'],expected),
                    raster=np.array_equal(a['raster'],source['sample_spikes'][lo:hi,sel]),
                    regions=np.array_equal(a['regions_1ms'],source['region_spikes_1ms'][lo//10:hi//10]),
                    field=np.array_equal(a['field_5ms'],source['field_e_count_1ms'][lo//10:hi//10].reshape(-1,5,400).sum(1)),
                    lfp=np.array_equal(a['lfp_raw'],source['lfp_raw'][lo//5:hi//5]),
                    Z=np.array_equal(a['Z'],source['z_stats'][lo//50:hi//50,:9]),
                    M=np.array_equal(a['M'],source['m_stats'][lo//50:hi//50][:,[0,5,6,7]]))
                checks[j['name']]=row;assert all(row.values()),row
    # Algebraic application check: only E M is changed at the intervention.
    p=load_pickle(OUT/'parents/release.pkl');st=p['engine']
    from checkpoint import restore_slow
    a=ReleaseZ(40000,18.,old.MZSlowVarsConfig(use_z=True,use_m=True,tau_z=5000.,I_th_EI=old.THRESHOLD,tau_adp=2000.,eta_m=.02),NE=32000)
    b=ReleaseZ(40000,18.,a.cfg,NE=32000,reset_m=True)
    for slow in [a,b]:
        restore_slow(st,slow);slow.restore_ms=75500.;slow.restore_from=p['restore_from']
    va=a.apply_currents(st['I_E'],st['I_I']);vb=b.apply_currents(st['I_E'],st['I_I'])
    checks['M_reset_only']=dict(Z_identical=bool(np.array_equal(a.z,b.z)),
        M_all_E_zero=bool(np.all(b.m[:32000]==0)),I_unchanged=bool(np.array_equal(va[32000:],vb[32000:])),
        current_delta_is_removed_M=bool(np.allclose(vb[:32000]-va[:32000],.02*a.m[:32000],atol=1e-12)))
    assert all(checks['M_reset_only'].values())
    write(OUT/'qa.json',dict(status='PASS',checks=checks))


def analyze():
    rows=[]
    for p in sorted((OUT/'runs').glob('*/result.json')):
        r=read(p)
        if r['job']['name'].startswith('qa_'):continue
        rows.append({k:r[k] for k in ['job','end_s','recurrence_observed','interpretation',
                                      'recovery_s','recurrence_onset_s','recurrence_confirmation_s','reset_audit']})
    write(OUT/'analysis/summary.json',dict(runs=rows,statistical_unit='one initial state x one future-noise realization',
        repeated_original_prefix_not_independent=True,permanent_nonrecurrence_not_inferred=True))
    lines=['# Zreset后再次进入高态：条件实验进度与分析','',
           '固定历史手放双核；ηM=0.02，τM=2秒，τZ=5秒。高态判据和恢复判据预先固定；有限观察无事件按右删失处理。',
           '', '| 条件 | 实际终点(s) | 恢复后再次高态 | 起始(s) |','|---|---:|---|---:|']
    for r in rows:
        lines.append(f'| {r["job"]["name"]} | {r["end_s"]:.2f} | {r["interpretation"]} | {r["recurrence_onset_s"]} |')
    if not rows:lines+=['','正式长程结果尚未完成。']
    lines+=['','M清零改变一条配对轨迹，说明该初始状态干预影响该噪声实现下的演化；不能直接证明形成永久不发作状态。',
            'Z/M清零仍保留膜电位、突触、延迟及随机历史；完整快状态清零与原始初始化回放承担不同的定位任务。']
    (OUT/'analysis/scientific_review.md').write_text('\n'.join(lines)+'\n')


def supervise():
    import fcntl
    lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    qa()
    a=run_group([job('z_only_long','extension_parent',already_recovered=True)],'A_EXTEND_Z_ONLY')[0]
    if not a['recurrence_observed']:
        b=run_group([job('z_m_reset_long','release',reset_m=True)],'B_RESET_M')[0]
        if b['recurrence_observed']:
            pairs=[]
            for seed in [9109401,9109402,9109403]:
                for m in [False,True]:
                    j=job(f'paired_{seed}_'+('ZM' if m else 'Z'),'release',reset_m=m,future_seed=seed)
                    j['horizon_ms']=576500.;pairs.append(j)
            run_group(pairs,'C_PAIRED_M_EFFECT')
        else:
            fast=job('reset_ZM_all_fast','release',reset_m=True,fast_reset='all')
            restart=job('original_initialization_replay',None,fresh=True)
            restart['horizon_ms']=90000.
            d=run_group([fast,restart],'D_FAST_STATE_AND_FULL_INITIALIZATION')
            if d[0]['recurrence_observed']:
                run_group([job('reset_ZM_voltage','release',reset_m=True,fast_reset='voltage'),
                           job('reset_ZM_synapse_delay','release',reset_m=True,fast_reset='synapse')],
                          'E_SPLIT_FAST_STATE')
    analyze()
    write(OUT/'status.json',dict(status='COMPLETE_PENDING_SCIENTIFIC_REVIEW',phase='DONE',
                               analysis=str(OUT/'analysis/scientific_review.md')))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','worker','supervise','analyze'])
    parser.add_argument('--job');args=parser.parse_args()
    try:
        if args.command=='prepare':prepare()
        elif args.command=='worker':worker(read(args.job))
        elif args.command=='analyze':analyze()
        else:supervise()
    except Exception as e:
        if args.command=='worker':
            j=read(args.job);write(OUT/'runs'/j['name']/'progress.json',dict(status='FAILED',error=repr(e),job=j))
        else:write(OUT/'status.json',dict(status='FAILED',error=repr(e)))
        raise


if __name__=='__main__':main()
