#!/usr/bin/env python3
"""Fixed-substrate continuation with terminal observation and resumable site jobs."""
import argparse
import gc
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import run_fig5_two_gif_bases as first
old=first.old
from src.topic4_xy_fig5_followup import read,write,sha,array_sha,critical_state
from src.topic4_multidimensional_parameters import apply_parameters
from src.topic4_forced_source_capacity import exclude_injected_packet_frame

DATA=ROOT/'results/topic4_sef_hfo/fig5_two_gif_bases_recovery'
FIG=ROOT/'results/paper-ready-figure/fig5_two_gif_bases_recovery/figures'
IDS=first.IDS


def recruitment_start(a):
    """Earliest start of terminal uninterrupted >=250 Hz segment across E regions.

    This locates recruitment; it does not certify a stable terminal plateau.
    Require at least 1 s of terminal high activity in every region.
    """
    r=a['rates_hz'];t=a['time_ms'];starts=[]
    for col in range(4):
        below=np.flatnonzero(r[:,col]<250)
        start=int(below[-1]+1) if len(below) else 0
        if len(r)-start<50:raise ValueError('no 1 s terminal recruitment in every region')
        starts.append(float(t[start]-10))
    return min(starts),starts


def build(cid):
    p=read(first.OUTPUT/cid/'protocol.json')
    config=read(first.SOURCE/'execution/paired_round1/execution_config.json')
    transition=ROOT/config['inputs']['transition_config']['path']
    h=dict(candidate=p['candidate'],transition_config=str(transition),
        artifact_root='/home/honglab/leijiaxin/HFOsp',network_cache=config['network_cache'],
        networks=config['corrected_networks'],input_hashes={str(transition):config['inputs']['transition_config']['sha256']})
    job=dict(p['job'],duration_ms=15000.)
    s,cfg,regions,f=old.build(h,job)
    f['dynamic_parameter_audit']=apply_parameters(s,p['candidate']['dynamic_parameters'])
    expected=read(first.OUTPUT/cid/'substrate_verification.json')['fingerprint']
    if f!=expected:raise RuntimeError('fixed GIF substrate changed')
    return s,cfg,regions,f,job,p


def prepare(cid):
    out=DATA/cid;out.mkdir(parents=True,exist_ok=True)
    s,cfg,regions,f,job,p=build(cid)
    with np.load(first.OUTPUT/cid/'trajectory.npz') as original:
        prior={k:original[k] for k in original.files}
    onset,regional=recruitment_start(prior)
    states={'reference':1000.,'pre_onset':onset-250.}
    if states['pre_onset']<=1200.:raise RuntimeError('no separated pretransition checkpoint')
    p.update(job=job,states_ms=states,
        state_rule='Earliest terminal continuous >=250 Hz regional recruitment minus 250 ms; label pre-onset only after unchanged stable-plateau criterion passes on extended trajectory.',
        observation_amendment='15 s fixed horizon, engine early stop disabled; all physical and Z/M parameters unchanged.')
    write(out/'protocol.json',p)
    checkpoints={};steps={int(round(t/.1)):key for key,t in states.items()}
    def sink(step,state):
        path=out/f'checkpoint_{steps[step]}.npz'
        checkpoints[steps[step]]=dict(path=str(path),sha256=old.save_checkpoint(state,path))
    native=old.simulate_kick
    def no_early_stop(*args,**kwargs):
        kwargs['early_stop_runaway']=False
        return native(*args,**kwargs)
    old.simulate_kick=no_early_stop
    write(out/'status.json',dict(stage='EXTENDING_FIXED_TRAJECTORY',pid=os.getpid()))
    result,slow=old.simulate(s,cfg,regions,job,checkpoint_steps=steps,checkpoint_sink=sink)
    a=old.trajectory_arrays(s,regions,result,slow)
    for key in ['rates_hz','lfp','region_z','region_m']:
        if not np.array_equal(a[key][:len(prior[key])],prior[key]):
            raise RuntimeError('extended trajectory changed retained original prefix: '+key)
    outcome=old.classify_trajectory(a['time_ms'],a['rates_hz'],15000.)
    actual,regional_actual=recruitment_start(a)
    if actual!=onset:raise RuntimeError('recruitment onset changed under extended observation')
    outcome['regional_recruitment_ms']=regional_actual
    outcome['earliest_regional_recruitment_ms']=actual
    outcome['population_plateau_onset_ms']=outcome['onset_ms']
    # Keep primary classification unchanged; expose separate reference for probe timing.
    outcome['qualified_pretransition']=bool(outcome['runaway'] and actual>=2000.)
    for name,t in states.items():
        start=int(round(t/.1))
        old.save_arrays(out/f'expected_sham_{name}.npz',dict(rate=result['rate_E'][start:start+2000],lfp=result['lfp_trace'][start:start+2000]))
    record=dict(candidate_id=cid,job=job,substrate_fingerprint=f,trajectory=outcome,
        arrays=old.save_arrays(out/'trajectory.npz',a),checkpoints=checkpoints,
        original_retained_prefix_bitwise_equal=True,states_ms=states,
        critical_state=critical_state(a,actual),prior_trajectory_sha256=sha(first.OUTPUT/cid/'trajectory.json'))
    write(out/'trajectory.json',record)
    write(out/'status.json',dict(stage='PREPARED' if outcome['qualified_pretransition'] else 'NO_QUALIFIED_PLATEAU',trajectory=outcome))


def probe(cid,state_name,index):
    out=DATA/cid
    record=read(out/'trajectory.json')
    if not record['trajectory']['qualified_pretransition']:raise RuntimeError('no qualified pretransition')
    s,cfg,regions,f,job,p=build(cid)
    ck=record['checkpoints'][state_name]
    if sha(ck['path'])!=ck['sha256']:raise RuntimeError('checkpoint changed')
    state=old.load_checkpoint(ck['path'])
    sham,slow=old.simulate(s,cfg,regions,job,resume=state,duration=200.)
    with np.load(out/f'expected_sham_{state_name}.npz') as expected:
        if not np.array_equal(sham['rate_E'],expected['rate']) or not np.array_equal(sham['lfp_trace'],expected['lfp']):
            raise RuntimeError('sham does not match parent continuation')
    sites=np.array([[x,y] for y in [4.,10.,16.] for x in [4.,10.,16.]])
    ids=np.argsort(((s.positions_e-sites[index])**2).sum(axis=1),kind='stable')[:16]
    result,slow=old.simulate(s,cfg,regions,job,resume=state,forced_ids=ids,duration=200.)
    mask=np.zeros(s.n_e,bool);mask[ids]=True
    descendant=exclude_injected_packet_frame(result['E_spk_bool'],sham['E_spk_bool'],mask,trigger_step=0)
    full=descendant.sum(0,dtype=np.int32)-sham['E_spk_bool'].sum(0,dtype=np.int32)
    early=descendant[:500].sum(0,dtype=np.int32)-sham['E_spk_bool'][:500].sum(0,dtype=np.int32)
    count=descendant.sum(1,dtype=np.int32)-sham['E_spk_bool'].sum(1,dtype=np.int32)
    stem=out/'sites'/f'{state_name}_{index}'
    stem.parent.mkdir(exist_ok=True)
    arr=old.save_arrays(stem.with_suffix('.npz'),dict(full=full,early=early,curve=count.reshape(200,10).sum(1)))
    write(stem.with_suffix('.json'),dict(state=state_name,site_index=index,
        extra_spikes_200ms=int(full.sum()),extra_spikes_50ms=int(early.sum()),
        collision_count=int(result['forced_spike_collision_count']),arrays=arr,
        sham_continuation_exact=True,checkpoint_sha256=ck['sha256']))


def collect():
    reports=[]
    for cid in IDS:
        out=DATA/cid;r=read(out/'trajectory.json')
        if not r['trajectory']['qualified_pretransition']:
            reports.append(dict(candidate_id=cid,trajectory=r['trajectory'],probe_status='NOT_ESTIMABLE'));continue
        with np.load(out/'trajectory.npz') as a:
            arrays={k:a[k] for k in ['positions_E','contact_xy','region_E']}
        arrays['sites_mm']=np.array([[x,y] for y in [4.,10.,16.] for x in [4.,10.,16.]])
        rows=[]
        for name in r['states_ms']:
            values={k:[] for k in ['full','early','curve']}
            for index in range(9):
                row=read(out/'sites'/f'{name}_{index}.json')
                if sha(row['arrays']['path'])!=row['arrays']['sha256']:raise RuntimeError('site result changed')
                rows.append(row)
                with np.load(row['arrays']['path']) as a:
                    for k in values:values[k].append(a[k])
            for k,suffix in [('full','full_field'),('early','early_field'),('curve','extra_spikes_per_ms')]:arrays[name+'_'+suffix]=np.asarray(values[k])
        write(out/'probe.json',dict(candidate_id=cid,states_ms=r['states_ms'],rows=rows,
            arrays=old.save_arrays(out/'probe.npz',arrays),qualified_pretransition=True,
            sham_continuation_exact=True,injected_frame_excluded=True,all_sites_retained=True))
        first_response=[x['extra_spikes_200ms'] for x in rows[:9]]
        second_response=[x['extra_spikes_200ms'] for x in rows[9:]]
        reports.append(dict(candidate_id=cid,trajectory=r['trajectory'],probe_status='COMPLETE',
            reference=first_response,pre_onset=second_response,
            sites_with_increase=sum(b>a for a,b in zip(first_response,second_response))))
    subprocess.run([sys.executable,str(ROOT/'scripts/paper_figures/plot_fig5_two_gif_bases.py'),'--data',str(DATA),'--out',str(FIG)],check=True)
    write(DATA/'analysis.json',dict(status='COMPLETE_PENDING_VISUAL_REVIEW',results=reports,author_accepted=False))


def batch(tasks,maximum):
    pending=list(tasks);running=[];failures=[]
    while pending or running:
        while pending and len(running)<maximum:
            args=pending.pop(0);tag='_'.join(args)
            log=open(DATA/(tag+'.log'),'a')
            process=subprocess.Popen([sys.executable,__file__,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            running.append((process,args,log))
        for item in list(running):
            proc,args,log=item
            if proc.poll() is not None:
                log.close();running.remove(item)
                if proc.returncode:failures.append(dict(args=args,returncode=proc.returncode));pending=[]
        write(DATA/'service_status.json',dict(running=[dict(pid=p.pid,args=a) for p,a,_ in running],pending=len(pending),failures=failures))
        if running:time.sleep(10)
    if failures:raise RuntimeError(str(failures))


def orchestrate():
    DATA.mkdir(parents=True,exist_ok=True)
    batch([['prepare',c] for c in IDS if not (DATA/c/'trajectory.json').exists()],2)
    tasks=[]
    for c in IDS:
        r=read(DATA/c/'trajectory.json')
        if r['trajectory']['qualified_pretransition']:
            tasks.extend([['probe',c,s,str(i)] for s in r['states_ms'] for i in range(9) if not (DATA/c/'sites'/f'{s}_{i}.json').exists()])
    batch(tasks,6);collect()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','probe','orchestrate','collect']);parser.add_argument('candidate',nargs='?',choices=IDS);parser.add_argument('state',nargs='?');parser.add_argument('site',nargs='?',type=int)
    a=parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(24*1024**3,24*1024**3))
    if a.stage=='prepare':prepare(a.candidate)
    elif a.stage=='probe':probe(a.candidate,a.state,a.site)
    elif a.stage=='collect':collect()
    else:orchestrate()
