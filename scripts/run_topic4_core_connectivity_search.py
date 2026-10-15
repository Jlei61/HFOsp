"""core_connectivity_v2 search: frozen candidate table, application canary, paired screen,
bounded DE and new-seed confirmation (design 2026-09-10). Historical runners and physics
snapshots are untouched; new physics lives in src/topic4_core_connectivity_v2.py and
src/snn_engine/kick_probe_core_input_v2.py.
"""
from pathlib import Path
import argparse,copy,hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
from src import topic4_initial_state_runtime as rt
from src import topic4_core_connectivity_v2 as v2
from src.topic4_core_connectivity import _hash_sparse_bins
from src.topic4_core_driven_input import RateAudit
from src.topic4_initial_state import RunObserver,array_sha256,event_times_ms
from src.topic4_streaming_spike_readout import simulate_streaming
from src.topic4_observation_repaired import observe
from kick_probe_core_input_v2 import simulate_kick

OUT=Path('/data/hfosp/topic4_sef_hfo/core_connectivity_search_20260910')
DESIGN=Path('/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/core_connectivity_search_design_20260910/search_design.json')
PARENT=ROOT/'config/topic4_continuous_core_state_r1.json'
BRIDGE=Path('/data/hfosp/topic4_sef_hfo/core_position_scl_response_20260910')
SCRIPT=Path(__file__).resolve()
SOURCE_FILES=[SCRIPT,ROOT/'src/topic4_core_connectivity_v2.py',ROOT/'src/snn_engine/kick_probe_core_input_v2.py',ROOT/'src/topic4_streaming_spike_readout.py',
    ROOT/'src/topic4_initial_state_runtime.py',ROOT/'src/topic4_initial_state.py',ROOT/'src/topic4_observation_repaired.py',ROOT/'src/snn_engine/connectivity.py',
    ROOT/'src/snn_engine/connectivity_rot.py',ROOT/'src/snn_engine/params.py',ROOT/'src/topic4_core_field.py',ROOT/'src/topic4_core_driven_input.py',ROOT/'src/topic4_core_connectivity.py']
CANARY=['endpoint__baseline','endpoint__EE_core_to_out_scale_1.25','endpoint__IE_same_core_scale_0.75','endpoint__EE_core_to_out_degree_scale_1.5',
        'endpoint__EE_kernel_perp_scale_1.5','near_upper__radius_A_mm_2.5']
RUNAWAY=dict(es_thresh_hz=120.,es_dur_ms=100.,post_runaway_record_ms=500.)


def canonical_sha(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def plan_from_design(design):
    """C10: the screen table is the design's table, verbatim, frozen before any dispatch."""
    if design['schema']!='topic4.core_connectivity_search.DESIGN_ONLY.v1':raise RuntimeError('unexpected design schema')
    cands=[]
    for d in design['screen_candidates']:
        prm=dict(d['parameters'])
        cands.append(dict(id=d['id'],layout=d['layout'],centers_mm=copy.deepcopy(d['centers_mm']),radii_mm=[prm['radius_A_mm'],prm['radius_B_mm']],
            parameters=prm,changed_parameter=d['changed_parameter'] or 'baseline',changed_value=d['changed_value'],adjacency_changes=bool(d['adjacency_changes']),stage='screen'))
    seeds=[u['dynamics_seed'] for u in design['screen_units'][:2]]
    physics=dict(design['physics'],version=v2.PHYSICS_VERSION,engine='src/snn_engine/kick_probe_core_input_v2.py',module='src/topic4_core_connectivity_v2.py',
        threshold='Vth=18-a_k*max(18-Vraw_i,0) inside core k, latent Vraw_i per neuron identity (quantile_seed), floor 11 mV recorded, overlap=min, outside/I=18',
        external_input='shared OU + independent Poisson only on E in the union of both cores; outside E and all I receive exact expected arrivals nu_signal*dt into the unchanged AMPA filter',
        weights='six blocks scaled on explicit pre/post type and A/B/O membership; edges and delays preserved',
        degree='EE core->out block only; per outside E target round(baseline*factor) with fixed topology random keys; distance delays; zero-baseline targets recorded',
        kernel='all E->E in-edges resampled per E target under (l_par*s_par, l_perp*s_perp, theta+offset) keeping baseline in-degree; distance delays; other pathways preserved',
        composition='kernel baseline -> core->out degree -> block weights, each from the immutable cached graph')
    return dict(version=v2.PHYSICS_VERSION,schema='topic4.core_connectivity_search.RUN.v1',design_sha256=canonical_sha(design),physics=physics,
        layouts=design['layouts'],axes=design['axes'],baseline_parameters=design['baseline_parameters'],candidates=cands,seeds=seeds,topology_seed=int(design['screen_units'][0]['topology_seed']),
        duration_ms=float(design['screen_units'][0]['duration_ms']),canary_duration_ms=float(design['budget']['canary_duration_ms']),canary_candidates=list(CANARY),
        screen_units=[dict(candidate=c['id'],seed=s) for c in cands for s in seeds],runaway=dict(RUNAWAY),analysis=design['analysis'],adaptive=design['adaptive'],
        confirmation=design['confirmation'],resources=design['resources'],budget=design['budget'],historical_bridge=design['historical_bridge'],output_root=str(OUT),
        seed_contract='same dynamics seed = same core Poisson/OU innovation law, not identical innovations after any rate change; weight scans keep the cached adjacency; density/kernel candidates share only the topology random identity',
        stop=design['stop'])


class CoreInputObserver(RunObserver):
    """RunObserver with float64 external-arrival digests (deterministic expected arrivals are fractional)."""
    def _reset_segment(self,index):
        super()._reset_segment(index);self._segment_ext_sum=np.zeros(self.n_total,np.float64)
    def observe(self,t,tm,xi,nu_now,ext,delta_rate,V,I_E,I_I,spk):
        if t!=self.n_observed_steps:raise RuntimeError('observer steps must be contiguous from 0')
        if (t//self.segment_steps)!=self._segment_index:
            self._finalize_segment();self._reset_segment(t//self.segment_steps)
        arrivals=np.asarray(ext,np.float64)
        self.maximum_poisson_count=max(self.maximum_poisson_count,int(arrivals.max()) if arrivals.size else 0)
        digest=self._segment_hash;digest.update(np.float64(xi).tobytes());digest.update(np.float64(nu_now).tobytes());digest.update(arrivals.tobytes())
        self._segment_ext_sum+=arrivals;self._segment_xi_sum+=float(xi);self._segment_xi_sumsq+=float(xi)**2
        self._segment_steps_seen+=1
        self._spike_accumulator+=spk
        if (t+1)%self.trace_steps==0 or t+1==self.n_steps:
            k=self._trace_count;trace=self.trace
            trace['time_ms'][k]=tm;trace['xi'][k]=xi;trace['nu_now'][k]=nu_now
            trace['E_spikes'][k]=int(self._spike_accumulator[:self.n_e].sum());trace['I_spikes'][k]=int(self._spike_accumulator[self.n_e:].sum())
            trace['E_mean_V'][k]=float(V[:self.n_e].mean())
            for name,index in self.groups.items():
                if index.size:
                    trace[f'{name}_mean_V'][k]=float(V[index].mean());trace[f'{name}_spikes'][k]=int(self._spike_accumulator[index].sum())
                    trace[f'{name}_mean_I_E'][k]=float(I_E[index].mean());trace[f'{name}_mean_I_I'][k]=float(I_I[index].mean())
            self._spike_accumulator[:]=0;self._trace_count+=1
        self.n_observed_steps+=1
    def _finalize_segment(self):
        if self._segment_steps_seen==0:return
        start=self._segment_index*self.segment_steps*self.dt_ms
        self.segments.append(dict(index=int(self._segment_index),start_ms=float(start),end_ms=float(start+self._segment_steps_seen*self.dt_ms),n_steps=int(self._segment_steps_seen),
            complete=bool(self._segment_steps_seen==self.segment_steps),stream_sha256=self._segment_hash.hexdigest(),ext_sum_sha256=array_sha256(self._segment_ext_sum),
            delta_sum_sha256=None,ext_total=float(self._segment_ext_sum.sum()),delta_abs_total=None,xi_sum=float(self._segment_xi_sum),xi_sumsq=float(self._segment_xi_sumsq)))
        self.ext_segment_sums.append(self._segment_ext_sum.copy());self.delta_segment_sums.append(self._segment_delta_sum.copy())


class ProgressObserver(CoreInputObserver):
    def __init__(self,progress_path,job,**kwargs):
        super().__init__(**kwargs);self.progress_path,self.job=progress_path,job
    def observe(self,t,tm,*args):
        super().observe(t,tm,*args)
        if (t+1)%round(1000./self.dt_ms)==0:rt.write(self.progress_path,dict(status='SIMULATING',job=self.job,simulated_ms=tm+self.dt_ms,updated_unix=time.time()))


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    design=rt.read(DESIGN);plan=plan_from_design(design)
    plan['source_snapshot']={str(p):rt.sha(p) for p in SOURCE_FILES};plan['design_file_sha256']=rt.sha(DESIGN);plan['parent_design']=str(PARENT);plan['parent_design_sha256']=rt.sha(PARENT)
    if (OUT/'plan.json').exists():
        old=rt.read(OUT/'plan.json')
        if old['design_sha256']!=plan['design_sha256'] or old['candidates']!=plan['candidates']:raise RuntimeError('frozen candidate table changed')
        return old
    rt.write(OUT/'plan.json',plan);(OUT/'candidates').mkdir(exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    for c in plan['candidates']:rt.write(OUT/'candidates'/f'{c["id"]}.json',c)
    (OUT/'execution_plan.md').write_text('\n'.join([
        '# 端点先验双核的局部连接与传播范围搜索：执行合同','',
        '物理版本 core_connectivity_v2：两核只降E阈值（Vth=18−a_k·d_i，d_i=max(18−Vraw_i,0)，潜在样本按神经元身份固定，11 mV下界记录截断）；随机输入只加载在两核E上（共享OU＋独立Poisson），核外E与全部I改为确定的期望到达量nu_signal·dt进入同一突触滤波器；空间OU、慢I状态、Z/M、kick全部关闭。',
        '','连接干预按真实前/后突触类型与A/B/O成员施加：EE同核、EE核→外、EE外→外、E→I同核、I→E同核、I→I同核六个权重块；EE核→外真实入度增删；E→E横向/纵向核尺度与方向重采样整个EE介质并按距离重算时延。联合候选按“核基线→密度→权重”顺序从不可变缓存图计算。',
        '','第一阶段：3布局×(基线+19单参数探针)×2噪声=120条20秒，拓扑2511、噪声847101/847102，候选表在派发前冻结自search_design.json；不按标签或患者相似度停批。runaway冻结定义：20 ms EMA≥120 Hz持续100 ms提前结束并再记录500 ms；OOM/工程错误另计并先修复。',
        '','最多6条500 ms应用检查只核对阈值、输入、局部权重、真实加边/重采样是否按候选值落地，不以传播卡住第一阶段。资源：初始2 worker，测得峰值后最多8；机器保留≥40 GiB；进程树≥18 GiB或余量不足时终止并记录。',
        '','同一动力学seed只保证核内随机创新的规律相同，不声称改变率或图后逐时刻相同；权重扫描保持缓存邻接，密度/范围候选只共享拓扑随机身份。分析窗口从1.5秒起；完整检测、连续包络与原生2 ms活动全部保留。',
        '','后续：联合阶段最多16条件×2噪声（两布局人口、DE/rand/1/bin、F=0.6、CR=0.7、rng 934711、不polish），确认阶段最多4条件×2新拓扑×2新噪声×60秒；正式总上限168；完成后停在科学审阅，不自动冻结模型或进入Fig5。'])+'\n')
    rt.write(OUT/'status.json',dict(status='PREPARED_NOT_DISPATCHED',updated_unix=time.time()))
    return plan


def unit_dir(stage,cid,topology_seed,seed):return OUT/stage/'units'/cid/f'{topology_seed}_{seed}'
def result_path(stage,cid,topology_seed,seed):return unit_dir(stage,cid,topology_seed,seed)/'workers/trajectory.json'


def complete(path):
    if not path.exists():return False
    r=rt.read(path)
    return r.get('status')=='COMPLETE' and rt.sha(path.with_suffix('.npz'))==r['arrays_sha256']


def build_unit(candidate,topology_seed,dynamics_seed,network_record=None):
    parent=rt.read(PARENT)
    sub,cand,transition,execution,audit,network=rt.build_frozen_substrate(parent,topology_seed,dynamics_seed,frozen_manifest=network_record)
    baseline_identity=rt.static_identity(sub,audit);n_e,n_i=sub.n_e,sub.n_i
    reg=sub.extras['placement'];reference=v2.ee_kernel(sub.params,theta_deg=float(reg['theta_deg']),ar=float(sub.engine['AR']))
    prm=candidate['parameters'];centers=candidate['centers_mm'];radii=[prm['radius_A_mm'],prm['radius_B_mm']]
    field=v2.threshold_field(sub.positions_e,centers,radii,[prm['depth_A_scale'],prm['depth_B_scale']],n_total=n_e+n_i,quantile_seed=sub.stage['quantile_seed'],
        core_mean=sub.engine['core_mean'],core_std=sub.engine['core_std'],v_base=sub.engine['v_base'],sheet_mm=float(sub.engine['L']))
    e_core=field['core_index'];i_core=v2.core_index_for(sub.positions_i,centers,radii)
    net,graph_audit=v2.build_candidate_network(sub.net,np.asarray(sub.net['pos'],float),n_e,e_core,i_core,prm,topology_seed=topology_seed,p=sub.params,reference_kernel=reference)
    sub.net=net;sub.vtheta=field['vtheta'];sub.delta_vtheta=field['delta_vtheta'];sub.h_e=(e_core>=0).astype(float);sub.h_i=(i_core>=0).astype(float)
    groups=dict(coreAE=np.flatnonzero(e_core==0),coreBE=np.flatnonzero(e_core==1),coreAI=n_e+np.flatnonzero(i_core==0),coreBI=n_e+np.flatnonzero(i_core==1),
                surroundE=np.flatnonzero(e_core<0),allI=np.arange(n_e,n_e+n_i))
    loading=np.r_[(e_core>=0).astype(float),np.zeros(n_i)];deterministic=~(loading>0)
    identity=dict(positions_E_sha256=array_sha256(np.asarray(sub.positions_e,np.float32)),vtheta_sha256=array_sha256(np.asarray(sub.vtheta,np.float32)),
        vtheta_float64_sha256=array_sha256(np.asarray(sub.vtheta,np.float64)),h_sha256=array_sha256(np.asarray(sub.h_e,np.float32)),
        core_index_sha256=array_sha256(np.asarray(e_core,np.int64)),ampa_topology_sha256=_hash_sparse_bins(net['ampa_by_delay'],include_data=False),
        ampa_values_sha256=_hash_sparse_bins(net['ampa_by_delay']),gaba_topology_sha256=_hash_sparse_bins(net['gaba_by_delay'],include_data=False),
        gaba_values_sha256=_hash_sparse_bins(net['gaba_by_delay']),identity_scope='positions, thresholds, core membership, delay-binned edges/weights after candidate application')
    applied=dict(physics=v2.PHYSICS_VERSION,candidate=candidate,topology_seed=int(topology_seed),dynamics_seed=int(dynamics_seed),reference_kernel=reference,
        threshold=field['audit'],graph=graph_audit,identity=identity,baseline_identity=baseline_identity,network_cache=sub.network_cache,
        input=dict(mode='core_poisson_outside_expected',n_stochastic=int((loading>0).sum()),n_deterministic=int(deterministic.sum()),ou_loading='1 on E in either core, 0 elsewhere',
                   outside='exact expected arrivals nu_signal*dt each step',spatial_ou='off',slow_I_state='off',ZM='off',kick='off'),
        group_counts={k:int(len(v)) for k,v in groups.items()},I_core_members=[int((i_core==k).sum()) for k in range(2)])
    return sub,groups,loading,deterministic,applied,(np.asarray(e_core,np.int8),np.asarray(i_core,np.int8))


def worker(stage,cid,topology_seed,seed,duration):
    plan=rt.read(OUT/'plan.json')
    for p,h in plan['source_snapshot'].items():
        if rt.sha(p)!=h:raise RuntimeError(f'frozen source changed: {p}')
    candidate=rt.read(OUT/'candidates'/f'{cid}.json');unit=unit_dir(stage,cid,topology_seed,seed);unit.mkdir(parents=True,exist_ok=True)
    network_record=None
    if str(topology_seed)!=str(plan['topology_seed']):network_record=rt.read(OUT/'confirmation/frozen_networks.json')
    design=dict(candidate=candidate,stage=stage,topology_seed=int(topology_seed),dynamics_seed=int(seed),duration_ms=float(duration),runaway=plan['runaway'],
                physics=plan['physics'],source_snapshot=plan['source_snapshot'],analysis_burnin_ms=plan['analysis']['burnin_ms'])
    path=unit/'design.json'
    if path.exists() and rt.read(path)!=rt.json_safe(design):raise RuntimeError('resume changes physical unit')
    rt.write(path,design);work=unit/'workers';work.mkdir(exist_ok=True);stem='trajectory'
    if (work/(stem+'.json')).exists():
        if complete(work/(stem+'.json')):return
        raise RuntimeError('existing worker result is not a verified completion')
    start=time.time();rt.write(work/(stem+'.progress.json'),dict(status='BUILDING',started_unix=start))
    sub,groups,loading,deterministic,applied,(core_e,core_i)=build_unit(candidate,int(topology_seed),int(seed),network_record)
    rt.write(unit/'applied_physics.json',applied)
    sub.params.T=float(duration);dt=sub.params.dt;n=round(sub.params.T/dt);n_total=sub.n_e+sub.n_i
    sub.net['rng']=np.random.default_rng(int(seed))
    job=dict(candidate=cid,stage=stage,topology_seed=int(topology_seed),dynamics_seed=int(seed),duration_ms=float(duration))
    observer=ProgressObserver(work/(stem+'.progress.json'),job,dt_ms=dt,n_e=sub.n_e,n_total=n_total,groups=groups,n_steps=n,segment_ms=1000.,trace_ms=1.)
    audit=RateAudit({k:groups[k] for k in ['coreAE','coreBE','surroundE','allI']},n_total,dt)
    rt.write(work/(stem+'.progress.json'),dict(status='SIMULATING',job=job,started_unix=time.time()))
    rw=plan['runaway']
    result=simulate_streaming(simulate_kick,sub.params,sub.net,KICK_BOOST=0.,t_kick=1e9,V_th_per_neuron=sub.vtheta,slow=None,early_stop_runaway=True,
        es_thresh_hz=rw['es_thresh_hz'],es_dur_ms=rw['es_dur_ms'],post_runaway_record_ms=rw['post_runaway_record_ms'],global_ou_loading=loading,
        deterministic_external_mask=deterministic,step_observer=observer,afferent_rate_observer=audit,positions=sub.positions_e,montage=sub.montage)
    if audit.max_outside_deviation!=0:raise RuntimeError('outside OU leakage')
    spikes=result['E_spk_bool'];actual_ms=len(spikes)*dt
    rt.write(work/(stem+'.progress.json'),dict(status='OBSERVING',job=job,actual_duration_ms=actual_ms))
    movie=spikes.native();env,envdt,_=spikes.envelope();del spikes,result['E_spk_bool']
    observed=observer.finish();trace=observed['trace']
    parent=rt.read(PARENT);contract=rt.load_observation_contract(parent);observation=observe(env,float(envdt),contract)
    mu=np.asarray(observation['centroid_ms'],float).reshape(-1,len(contract['contact_names']))
    evaluator=rt.load_evaluator(parent);objective=rt.load_objective(parent)
    labels,support,dist=rt.classify_with_both_modes(evaluator,mu);times,primary=event_times_ms(observation)
    features=np.full((len(mu),len(objective.target_global)),np.nan,np.float32);readable=np.isfinite(mu).sum(1)>=2
    if readable.any():features[readable]=objective.embedding(mu[readable]).astype(np.float32)
    events=[dict(event_index=i,event_time_ms=float(times[i]),qualifying_start_ms=float(e['qualifying_interval_ms'][0]),qualifying_interval_ms=e['qualifying_interval_ms'],
        window_ms=e['window_ms'],primary_eligible=e['primary_eligible'],exclusion_reasons=e['primary_exclusion_reasons'],prolonged=e['prolonged'],
        mode=int(labels[i]),support=int(support[i]),distance_modes=dist[i].tolist(),n_contacts=int(np.isfinite(mu[i]).sum())) for i,e in enumerate(observation['events'])]
    steps_seen=max(int(observed['n_observed_steps']),1)
    rt.atomic_npz(work/(stem+'.npz'),contact_envelope=np.asarray(env,np.float32),contact_envelope_dt_ms=np.asarray(envdt),contact_names=np.asarray(sub.contact_names),
        contact_xy_mm=sub.contact_xy,positions_E=np.asarray(sub.positions_e,np.float32),positions_I=np.asarray(sub.positions_i,np.float32),
        h=np.asarray(sub.h_e,np.float32),vtheta=np.asarray(sub.vtheta,np.float32),core_index_E=core_e,core_index_I=core_i,
        sheet_activity_counts=movie['activity_counts'],sheet_activity_frame_ms=np.asarray(movie['frame_ms']),
        centroid_ms=mu,recruitment_ms=np.asarray(observation['recruitment_ms'],float),primary_event_indices=np.asarray(primary,int),event_mode=labels,event_support=support,
        event_distance_modes=dist,event_time_ms=times,event_phi=features,rate_E=np.asarray(result['rate_E'],np.float32),rate_I=np.asarray(result['rate_I'],np.float32),
        input_expected_per_step=(observed['ext_segment_sums'].sum(0)/steps_seen).astype(np.float32),input_is_stochastic=~deterministic,
        input_rate_audit_columns=audit.arrays()['columns'],input_rate_audit_values=audit.arrays()['values'],
        **{f'trace_{k}':v for k,v in trace.items()},**{f'group_{k}':v for k,v in groups.items()})
    payload=dict(status='COMPLETE',job=job,candidate_id=cid,stage=stage,design_sha256=rt.sha(path),applied_physics_sha256=rt.sha(unit/'applied_physics.json'),
        actual_duration_ms=actual_ms,runaway_early_stop_ms=result.get('runaway_early_stop_ms'),physical_status='RUNAWAY' if result.get('runaway_early_stop_ms') is not None else 'COMPLETE_NO_RUNAWAY',
        static_array_identity=applied['identity'],external_input=result['external_input'],input_segments=observed['segments'],events=events,
        group_counts=applied['group_counts'],n_detected=len(events),n_primary=len(primary),observation_boundary=observation['boundary_or_low_window_support'],
        maximum_outside_rate_deviation=audit.max_outside_deviation,wall_seconds=time.time()-start,peak_rss_gib=rt.peak_rss_gib(),arrays_sha256=rt.sha(work/(stem+'.npz')))
    rt.write(work/(stem+'.json'),payload);rt.write(work/(stem+'.progress.json'),dict(status='COMPLETE',job=job,n_primary=len(primary)))
    print(json.dumps({k:payload[k] for k in ['status','job','physical_status','n_primary','wall_seconds','peak_rss_gib']}),flush=True)


def queue(stage,units,duration,label):
    """C16: resource-aware dispatch. units: list of (candidate, topology_seed, seed)."""
    import psutil
    plan=rt.read(OUT/'plan.json');res=plan['resources']
    pending=[u for u in units if not complete(result_path(stage,*u))];active={};failures=[];peak_seen=[]
    for u in units:
        p=result_path(stage,*u)
        if p.exists() and not complete(p):raise RuntimeError(f'invalid resume artifact {p}')
    def reserve():return max(8.,1.5*max(peak_seen)) if peak_seen else 8.
    while pending or active:
        rss={}
        for pid,(proc,u,log) in list(active.items()):
            if proc.poll() is not None:
                log.close();del active[pid]
                if proc.returncode or not complete(result_path(stage,*u)):failures.append(dict(unit=u,exit=proc.returncode,log=log.name))
                else:peak_seen.append(float(rt.read(result_path(stage,*u))['peak_rss_gib'] or 0))
                continue
            try:
                tree=[psutil.Process(pid)]+psutil.Process(pid).children(recursive=True);rss[pid]=sum(x.memory_info().rss for x in tree)/2**30
                if rss[pid]>18 or rt.available_gib()<res['min_available_GiB']:
                    for x in reversed(tree):x.terminate()
                    failures.append(dict(unit=u,error='RESOURCE_GUARD',tree_rss_gib=rss[pid],available_gib=rt.available_gib()))
            except psutil.NoSuchProcess:pass
        if failures:pending=[]
        max_workers=res['initial_workers'] if not peak_seen else res['max_workers']
        while pending and len(active)<max_workers and rt.available_gib()>res['min_available_GiB']+reserve()*(len(active)+1):
            u=pending.pop(0);cid,topo,seed=u;log=(OUT/'logs'/f'{label}_{cid}_{topo}_{seed}.log').open('a')
            cmd=[rt.PYTHON,'-u',str(SCRIPT),'worker','--stage',stage,'--candidate',cid,'--topology',str(topo),'--seed',str(seed),'--duration',str(duration)]
            proc=subprocess.Popen(cmd,cwd=ROOT,env=rt.ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[proc.pid]=(proc,u,log)
            time.sleep(2)
        rt.write(OUT/'status.json',dict(status=f'{label.upper()}_RUNNING',stage=stage,complete=sum(complete(result_path(stage,*u)) for u in units),total=len(units),queued=len(pending),
            max_workers=max_workers,peak_rss_seen_gib=max(peak_seen) if peak_seen else None,available_gib=rt.available_gib(),
            active=[dict(pid=pid,candidate=u[0],topology=u[1],seed=u[2],rss_gib=rss.get(pid)) for pid,(pr,u,f) in active.items()],failures=failures,updated_unix=time.time()))
        if pending or active:time.sleep(10)
    return failures


def canary_audit(plan):
    rows=[];topo=plan['topology_seed']
    for cid in plan['canary_candidates']:
        unit=unit_dir('canary',cid,topo,plan['seeds'][0]);ap=rt.read(unit/'applied_physics.json');r=rt.read(result_path('canary',cid,topo,plan['seeds'][0]))
        prm=ap['candidate']['parameters'];th=ap['threshold'];g=ap['graph'];w=g['stage_audits']['weights']['blocks']
        checks=dict(duration_500ms=r['actual_duration_ms']==plan['canary_duration_ms'],no_raised_threshold=th['n_raised']==0,depth_scales=th['depth_scales']==[prm['depth_A_scale'],prm['depth_B_scale']],
            radii=th['radii_mm']==[prm['radius_A_mm'],prm['radius_B_mm']],lowered_members=th['n_lowered']>0,
            engine_mode=r['external_input']['mode']=='core_poisson_outside_expected',engine_stochastic_equals_core=r['external_input']['n_stochastic']==ap['input']['n_stochastic']==th['n_members'],
            outside_rate_fixed=r['maximum_outside_rate_deviation']==0,
            weight_blocks_applied=all(abs(w[k]['weight_after']-w[k]['weight_before']*prm[k])<1e-6*max(1,w[k]['weight_before']) for k in w),
            adjacency_flag=g['adjacency_changes']==ap['candidate']['adjacency_changes'],
            weight_only_keeps_baseline_topology=(ap['identity']['ampa_topology_sha256']==_baseline_topology(plan) if not ap['candidate']['adjacency_changes'] else True))
        if 'degree' in g['stage_audits']:
            d=g['stage_audits']['degree'];checks['degree_target_met']=d['final_block_edges']==d['requested_block_edges'] and d['shortfall_targets']==0
        if 'kernel' in g['stage_audits']:
            k=g['stage_audits']['kernel'];checks['kernel_indegree_preserved']=k['indegree_preserved'] and not k['adjacency_identical_to_baseline']
        rows.append(dict(candidate=cid,checks=checks,physical_status=r['physical_status'],n_detected=r['n_detected'],members=th['members'],total_lowering_mV=th['total_lowering_mV'],
            floor_clipped=th['floor_clipped_count'],wall_seconds=r['wall_seconds'],peak_rss_gib=r['peak_rss_gib']))
    status='PASS' if all(all(x['checks'].values()) for x in rows) else 'FAIL'
    rt.write(OUT/'canary_audit.json',dict(status=status,units=rows,scope='applied thresholds, input locality, block weights, real edge changes, 500 ms completion; not propagation success'))
    return status


def _baseline_topology(plan):
    unit=unit_dir('canary','endpoint__baseline',plan['topology_seed'],plan['seeds'][0])
    return rt.read(unit/'applied_physics.json')['identity']['ampa_topology_sha256']


def canary():
    plan=prepare();units=[(cid,plan['topology_seed'],plan['seeds'][0]) for cid in plan['canary_candidates']]
    failures=queue('canary',units,plan['canary_duration_ms'],'canary')
    if failures:rt.write(OUT/'status.json',dict(status='CANARY_ENGINEERING_FAILURE',failures=failures,updated_unix=time.time()));raise RuntimeError(str(failures))
    status=canary_audit(plan);rt.write(OUT/'status.json',dict(status=f'CANARY_{status}',updated_unix=time.time()))
    if status!='PASS':raise RuntimeError('canary application audit failed')


def screen():
    plan=prepare()
    if rt.read(OUT/'canary_audit.json')['status']!='PASS':raise RuntimeError('canary not passed')
    units=[(u['candidate'],plan['topology_seed'],u['seed']) for u in plan['screen_units']]
    failures=queue('screen',units,plan['duration_ms'],'screen')
    if failures:rt.write(OUT/'status.json',dict(status='SCREEN_ENGINEERING_FAILURE',failures=failures,updated_unix=time.time()));raise RuntimeError(str(failures))
    rt.write(OUT/'status.json',dict(status='SCREEN_COMPLETE_ANALYSIS_PENDING',formal_runs=len(units),updated_unix=time.time()))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','worker','canary','screen'])
    ap.add_argument('--stage');ap.add_argument('--candidate');ap.add_argument('--topology',type=int);ap.add_argument('--seed',type=int);ap.add_argument('--duration',type=float)
    a=ap.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='worker':worker(a.stage,a.candidate,a.topology,a.seed,a.duration)
    else:
        try:globals()[a.action]()
        except Exception as exc:rt.write(OUT/'status.json',dict(status='FAILED',action=a.action,error=repr(exc),updated_unix=time.time()));raise
