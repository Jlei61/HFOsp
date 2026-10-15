"""Bounded core-led input correction. Existing experiments remain untouched."""
from pathlib import Path
import argparse,copy,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
from scripts import pilot_topic4_core_extent as pilot
from scripts import run_topic4_continuous_core_state_r1 as r1
from src.topic4_streaming_spike_readout import simulate_streaming
from src.topic4_core_driven_input import lowering_only,MaskedSpatialDrive,RateAudit
from src.topic4_zm_ictal_transition import make_external_drive
from kick_probe_core_input_v1 import simulate_kick
rt=pilot.rt
OUT=Path('/data/hfosp/topic4_sef_hfo/core_driven_input_pilot_20260910')
SCRIPT=Path(__file__).resolve()
ORIGINAL_BUILD=pilot.build_modified
SOURCE_FILES=[SCRIPT,ROOT/'src/topic4_core_driven_input.py',ROOT/'src/snn_engine/kick_probe_core_input_v1.py',
    ROOT/'scripts/run_topic4_continuous_core_state_r1.py',ROOT/'scripts/pilot_topic4_core_extent.py',ROOT/'src/topic4_streaming_spike_readout.py']


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'plan.json').exists():return
    old=rt.read('/data/hfosp/topic4_sef_hfo/core_extent_long_propagation_20260909/plan.json')
    base=copy.deepcopy(old['candidates'][0]);cs=[]
    specs=[('legacy_mixed_full','历史混合阈值＋全场OU',False,'full','full',True),
           ('lower_full','仅修正为降阈值',True,'full','full',True),
           ('lower_local_core','局部OU限制核内',True,'full','core',True),
           ('core_ou_plus_local','两种OU均限核内',True,'core','core',True),
           ('core_ou_only','核心方案：核内OU、局部OU关闭',True,'core','off',True),
           ('core_ou_only_Istate_off','核心方案＋关闭慢I状态',True,'core','off',False)]
    for cid,label,lower,global_scope,local_scope,state in specs:
        cs.append(dict(base,id=cid,label=label,lowering_only=lower,global_ou_scope=global_scope,spatial_ou_scope=local_scope,retain_I_state=state))
    design=copy.deepcopy(old['parent_design']);seeds=[847101,847102]
    design['jobs']=[dict(id=f'ou_dyn{s}',kind='ou',z=None,duration_ms=20000.,dynamics_seed=s,coupling_seed=s+100000,state_seed=s+10000) for s in seeds]
    plan=dict(version='core_driven_input_v1',output_root=str(OUT),parent_design=design,candidates=cs,seeds=seeds,duration_ms=20000.,
        canary_duration_ms=500.,max_parallel=6,formal_runs=12,analysis_burnin_ms=1500.,
        source_sha256=rt.sha(pilot.SCRIPT),source_snapshot={str(p):rt.sha(p) for p in SOURCE_FILES},
        default_candidate='core_ou_only',legacy_control_only='legacy_mixed_full',
        noise_contract='Core global OU loading is 1 on geometric E-core support, 0 on all other E and I. Fixed tonic Poisson input remains everywhere. Spatial OU is off by default; optional core mask preserves each retained innovation without recentering or amplification.',
        threshold_contract='Clip positive E threshold shifts to zero, retain all original negative shifts; no dose compensation. Threshold-only arm isolates this change.',
        stop='12 fixed 20-second runs plus automated full-output analysis; no adaptive search, no mode-based early stopping, no model freeze or Figure5',
        role='single topology 2511, two paired seed replays; same seed is not a claim of identical Poisson innovations after changed rates')
    rt.write(OUT/'plan.json',plan)
    (OUT/'candidates').mkdir(exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    for c in cs:rt.write(OUT/'candidates'/f'{c["id"]}.json',c)


def worker(cid,seed,canary=False):
    plan=rt.read(OUT/'plan.json')
    for p,h in plan['source_snapshot'].items():
        if rt.sha(p)!=h:raise RuntimeError(f'frozen source changed: {p}')
    candidate=next(c for c in plan['candidates'] if c['id']==cid)
    duration=plan['canary_duration_ms'] if canary else plan['duration_ms']
    runroot=OUT/('canary' if canary else 'formal');runroot.mkdir(parents=True,exist_ok=True)
    unit=runroot/'units'/cid/str(seed)
    local=copy.deepcopy(plan)
    if not candidate['retain_I_state']:local['parent_design']['state']['amplitude']=0.
    rt.write(runroot/f'plan_{cid}.json',local)
    # Each process uses its own immutable plan path; avoid shared-plan races across conditions.
    ownroot=unit/'runner';ownroot.mkdir(parents=True,exist_ok=True)
    rt.write(ownroot/'plan.json',local);pilot.OUT=ownroot
    actualunit=ownroot/'units'/cid/str(seed)
    def build(design,s,c):
        built,control,details=ORIGINAL_BUILD(design,s,c)
        sub,cand,transition,execution,audit,network=built
        old_delta=sub.delta_vtheta.copy()
        if c['lowering_only']:
            sub.delta_vtheta=lowering_only(old_delta)
            sub.vtheta[:sub.n_e]=sub.engine['v_base']+sub.delta_vtheta
        d=sub.delta_vtheta;h=sub.h_e>0
        if c['lowering_only'] and np.any(d>0):raise RuntimeError('core raising remains')
        if np.any(d[~h]!=0):raise RuntimeError('outside threshold changed')
        identity=rt.static_identity(sub,audit)
        assert all(identity[k]==details['baseline_identity'][k] for k in identity if k.startswith(('positions_','ampa_','gaba_')))
        details.update(applied_identity=identity,n_threshold_raised=int((d>0).sum()),n_threshold_lowered=int((d<0).sum()),
            positive_clipped_n=int((old_delta>0).sum()) if c['lowering_only'] else 0,
            new_net_lowering_mV=float(-d.sum()),old_net_lowering_mV=float(-old_delta.sum()),
            core_driven_input_contract={k:c[k] for k in ['lowering_only','global_ou_scope','spatial_ou_scope','retain_I_state']})
        return built,control,details
    pilot.build_modified=build
    def simulate(sub,transition,design,job,control,observer=None):
        sub.net['rng']=np.random.default_rng(job['dynamics_seed'])
        h=np.asarray(sub.h_e)>0;n=sub.n_e+sub.n_i
        drive=None
        if candidate['spatial_ou_scope']!='off':
            drive=make_external_drive(sub,transition['spatial_ou'],job['dynamics_seed'])
            if candidate['spatial_ou_scope']=='core':drive=MaskedSpatialDrive(drive,h)
        loading=None
        if candidate['global_ou_scope']=='core':loading=np.r_[h.astype(float),np.zeros(sub.n_i)]
        groups=observer.groups
        audit=RateAudit({k:groups[k] for k in ['coreAE','coreBE','surroundE','allI']},n,sub.params.dt)
        result=simulate_streaming(simulate_kick,sub.params,sub.net,KICK_BOOST=0.,t_kick=1e9,V_th_per_neuron=sub.vtheta,slow=None,
            early_stop_runaway=True,external_e_rate_drive=drive,global_ou_loading=loading,
            afferent_rate_observer=audit,step_observer=observer,external_i_state=control,positions=sub.positions_e,montage=sub.montage)
        (actualunit/'workers').mkdir(exist_ok=True)
        rt.atomic_npz(actualunit/'workers/input_rates.npz',**audit.arrays(),global_ou_loading=np.ones(n) if loading is None else loading,local_ou_loading=np.zeros(sub.n_e) if drive is None else h.astype(float) if candidate['spatial_ou_scope']=='core' else np.ones(sub.n_e))
        rt.write(actualunit/'input_application_audit.json',dict(maximum_outside_E_rate_deviation_from_tonic_per_ms=audit.max_outside_deviation,steps=audit.calls,
            expected_outside_OU_zero=candidate['global_ou_scope']=='core',contract=candidate))
        if candidate['global_ou_scope']=='core' and audit.max_outside_deviation!=0:raise RuntimeError('outside OU leakage')
        return result,drive
    r1.simulate=simulate
    r1.sheet_activity_movie=lambda spikes,*args,**kwargs:spikes.native()
    r1.snn_event_envelope=lambda spikes,*args,**kwargs:spikes.envelope()
    pilot.run_worker(OUT/'candidates'/f'{cid}.json',seed,duration)
    # Stable public unit links; actual runner products and original metadata stay intact.
    for name in ['workers','applied_geometry.json','input_application_audit.json','design.json']:
        dest=unit/name;source=actualunit/name
        if source.exists() and not dest.exists():dest.symlink_to(source,target_is_directory=source.is_dir())
    if canary and cid=='legacy_mixed_full':
        ref=Path('/data/hfosp/topic4_sef_hfo/core_extent_long_propagation_20260909/duration_90000/units/baseline')/str(seed)/'workers/trajectory.npz'
        with np.load(unit/'workers/trajectory.npz') as a,np.load(ref) as b:
            checks={k:np.array_equal(a[k],b[k][:len(a[k])]) for k in ['rate_E','state_z','sheet_activity_counts']}
        rt.write(unit/'legacy_prefix_audit.json',checks)
        if not all(checks.values()):raise RuntimeError(f'legacy prefix mismatch {checks}')


def result_path(cid,s,canary=False):return OUT/('canary' if canary else 'formal')/'units'/cid/str(s)/'workers/trajectory.json'


def queue(canary=False):
    import psutil
    plan=rt.read(OUT/'plan.json');seeds=plan['seeds'][:1] if canary else plan['seeds']
    jobs=[(c['id'],s) for c in plan['candidates'] for s in seeds];pending=[];active={};failures=[]
    for cid,s in jobs:
        f=result_path(cid,s,canary)
        if f.exists():
            r=rt.read(f)
            if r['status']!='COMPLETE' or rt.sha(f.with_suffix('.npz'))!=r['arrays_sha256']:raise RuntimeError('invalid resume artifact')
        else:pending.append((cid,s))
    while pending or active:
        rss={}
        for pid,(proc,cid,s,log) in list(active.items()):
            if proc.poll() is not None:
                log.close();del active[pid]
                if proc.returncode or not result_path(cid,s,canary).exists():failures.append(dict(candidate=cid,seed=s,exit=proc.returncode))
                continue
            try:
                tree=[psutil.Process(pid)]+psutil.Process(pid).children(recursive=True);rss[pid]=sum(p.memory_info().rss for p in tree)/2**30
                if rss[pid]>18 or rt.available_gib()<40:
                    for p in reversed(tree):p.terminate()
                    failures.append(dict(candidate=cid,seed=s,error='RESOURCE_GUARD'))
            except psutil.NoSuchProcess:pass
        if failures:pending=[]
        reserve=sum(max(0,8-rss.get(pid,0)) for pid in active)
        while pending and len(active)<plan['max_parallel'] and rt.available_gib()>60+reserve+8:
            cid,s=pending.pop(0);log=(OUT/'logs'/f'{"canary" if canary else "formal"}_{cid}_{s}.log').open('a')
            cmd=[rt.PYTHON,'-u',str(SCRIPT),'worker','--candidate',cid,'--seed',str(s)]+(['--canary'] if canary else [])
            proc=subprocess.Popen(cmd,cwd=ROOT,env=rt.ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[proc.pid]=(proc,cid,s,log);reserve+=8
        rt.write(OUT/'status.json',dict(status='CANARY_RUNNING' if canary else 'PILOT_RUNNING',complete=sum(result_path(c,s,canary).exists() for c,s in jobs),total=len(jobs),queued=len(pending),
            active=[dict(pid=pid,candidate=cid,seed=s,rss_gib=rss.get(pid)) for pid,(pr,cid,s,f) in active.items()],failures=failures,updated_unix=time.time()))
        if pending or active:time.sleep(5)
    if failures:raise RuntimeError(str(failures))


def controller():
    queue(True)
    # All six canaries must physically complete; source localization is not a canary gate.
    p=rt.read(OUT/'plan.json')
    for c in p['candidates']:
        path=result_path(c['id'],p['seeds'][0],True);r=rt.read(path)
        if r['actual_duration_ms']!=p['canary_duration_ms']:raise RuntimeError('canary stopped early')
    rt.write(OUT/'canary_audit.json',dict(status='PASS',n=6,scope='applied thresholds/input locality, completed 500 ms, baseline prefix parity; not propagation success'))
    queue(False)
    rt.write(OUT/'status.json',dict(status='ANALYZING',updated_unix=time.time()))
    subprocess.run([rt.PYTHON,str(ROOT/'scripts/analyze_topic4_core_driven_input_pilot.py')],cwd=ROOT,env=rt.ENV,check=True)
    rt.write(OUT/'status.json',dict(status='COMPLETE_PENDING_SCIENTIFIC_REVIEW',formal_runs=12,updated_unix=time.time()))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','worker','controller']);ap.add_argument('--candidate');ap.add_argument('--seed',type=int);ap.add_argument('--canary',action='store_true');a=ap.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='worker':worker(a.candidate,a.seed,a.canary)
    else:
        try:controller()
        except Exception as exc:rt.write(OUT/'status.json',dict(status='FAILED',error=repr(exc),updated_unix=time.time()));raise
