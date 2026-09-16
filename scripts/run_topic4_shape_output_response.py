"""Bounded paired shape/output experiment. Existing physical executors remain frozen."""
from pathlib import Path
import argparse,copy,fcntl,hashlib,json,os,pickle,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
from scripts import run_topic4_core_ou_correlation_recovery as run
from src import topic4_core_shape_output as shape
base=run.base;rt=base.rt;v2=base.v2
OUT=Path('/data/hfosp/topic4_sef_hfo/core_shape_output_response_20260911')
OLD=Path('/data/hfosp/topic4_sef_hfo/core_connectivity_search_20260910')
SCRIPT=Path(__file__).resolve()


def build_unit(candidate,topology_seed,dynamics_seed,network_record=None):
    parent=rt.read(base.PARENT)
    sub,_,_,_,audit,_=rt.build_frozen_substrate(parent,topology_seed,dynamics_seed,frozen_manifest=network_record)
    baseline_identity=rt.static_identity(sub,audit);ne,ni=sub.n_e,sub.n_i;prm=candidate['parameters'];pos=np.asarray(sub.net['pos'],float)
    reg=sub.extras['placement'];reference=v2.ee_kernel(sub.params,theta_deg=float(reg['theta_deg']),ar=float(sub.engine['AR']))
    field=shape.shape_field(sub.positions_e,sub.positions_i,candidate['centers_mm'],[prm['radius_A_mm'],prm['radius_B_mm']],
        [prm['depth_A_scale'],prm['depth_B_scale']],candidate.get('shape',{}),n_total=ne+ni,quantile_seed=sub.stage['quantile_seed'],
        core_mean=sub.engine['core_mean'],core_std=sub.engine['core_std'],v_base=sub.engine['v_base'],sheet_mm=float(sub.engine['L']))
    ec,ic=field['core_index'],field['i_core_index']
    kernel=v2.ee_kernel(sub.params,theta_deg=reference['theta_deg'],ar=float(sub.engine['AR']),perp_scale=prm['EE_kernel_perp_scale'],
        parallel_scale=prm['EE_kernel_parallel_scale'],angle_offset_deg=prm['EE_angle_offset_deg'])
    # Cache only the global EE baseline. Local mutations always start from its immutable bytes.
    key=base.canonical_sha(dict(topology=topology_seed,baseline=baseline_identity,kernel=kernel,version=shape.VERSION))
    cache=OUT/'global_graph_cache';cache.mkdir(exist_ok=True)
    with (cache/(key+'.lock')).open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);path=cache/(key+'.pkl')
        if path.exists():
            with path.open('rb') as f:ampa,ka=pickle.load(f)
        else:
            ampa=sub.net['ampa_by_delay'];ka=dict(exact_noop=True)
            if not kernel['is_reference']:ampa,ka=v2.resample_ee_kernel(ampa,pos,ne,topology_seed=topology_seed,kernel=kernel,p=sub.params)
            tmp=path.with_suffix('.tmp')
            with tmp.open('wb') as f:pickle.dump((ampa,ka),f,pickle.HIGHEST_PROTOCOL)
            tmp.replace(path)
    ampa,local=shape.local_outgoing(ampa,pos,ne,ec,kernel,candidate.get('outgoing',{}),sub.params,topology_seed)
    da=dict(exact_noop=True)
    if prm['EE_core_to_out_degree_scale']!=1:
        ampa,da=v2.apply_core_to_out_degree(ampa,pos,ne,ec,prm['EE_core_to_out_degree_scale'],topology_seed=topology_seed,kernel=kernel,p=sub.params)
    ampa,gaba,wa=v2.scale_pathway_blocks(ampa,sub.net['gaba_by_delay'],ne,ec,ic,prm)
    length=max(len(ampa),len(gaba));ampa=v2._pad(ampa,length,ampa[0].shape);gaba=v2._pad(gaba,length,gaba[0].shape)
    net={k:v for k,v in sub.net.items() if not (k.startswith('ampa_') or k.startswith('_ampa') or k.startswith('gaba_') or k.startswith('_gaba')) or k in ('ampa_by_delay','gaba_by_delay')}
    net.update(ampa_by_delay=ampa,gaba_by_delay=gaba,max_delay_steps=length-1)
    sub.net=net;sub.vtheta=field['vtheta'];sub.delta_vtheta=field['delta_vtheta'];sub.h_e=(ec>=0).astype(float);sub.h_i=(ic>=0).astype(float)
    groups=dict(coreAE=np.flatnonzero(ec==0),coreBE=np.flatnonzero(ec==1),coreAI=ne+np.flatnonzero(ic==0),coreBI=ne+np.flatnonzero(ic==1),surroundE=np.flatnonzero(ec<0),allI=np.arange(ne,ne+ni))
    loading=np.r_[(ec>=0).astype(float),np.zeros(ni)];deterministic=~(loading>0)
    identity=dict(positions_E_sha256=base.array_sha256(np.asarray(sub.positions_e,np.float32)),vtheta_sha256=base.array_sha256(np.asarray(sub.vtheta,np.float32)),
        vtheta_float64_sha256=base.array_sha256(np.asarray(sub.vtheta,np.float64)),h_sha256=base.array_sha256(np.asarray(sub.h_e,np.float32)),
        core_index_sha256=base.array_sha256(np.asarray(ec,np.int64)),ampa_topology_sha256=base._hash_sparse_bins(ampa,include_data=False),
        ampa_values_sha256=base._hash_sparse_bins(ampa),gaba_topology_sha256=base._hash_sparse_bins(gaba,include_data=False),gaba_values_sha256=base._hash_sparse_bins(gaba))
    applied=dict(physics=shape.VERSION,candidate=candidate,topology_seed=topology_seed,dynamics_seed=dynamics_seed,reference_kernel=reference,
        threshold=field['audit'],graph=dict(kernel=kernel,global_cache=str(path),stage_audits=dict(kernel=ka,local_output=local,degree=da,weights=wa),
        block_summary=v2.pathway_block_summary(ampa,gaba,ne,ec,ic),max_delay_steps=length-1),identity=identity,baseline_identity=baseline_identity,network_cache=sub.network_cache,
        input=dict(mode='core_poisson_outside_expected',n_stochastic=int((loading>0).sum()),n_deterministic=int(deterministic.sum()),outside='exact expected arrivals nu_signal*dt',spatial_ou='off',slow_I_state='off',ZM='off',kick='off'),
        group_counts={k:int(len(v)) for k,v in groups.items()},I_core_members=[int((ic==k).sum()) for k in [0,1]])
    assert applied['threshold']['n_raised']==0
    return sub,groups,loading,deterministic,applied,(ec.astype(np.int8),ic.astype(np.int8))


def prepare():
    OUT.mkdir(exist_ok=True,parents=True)
    if (OUT/'plan.json').exists():return rt.read(OUT/'plan.json')
    source=OLD/'candidates/coreOU_mid_EE085_mean095_rho000.json';c0=rt.read(source)
    raw=OLD/'recovery_core_OU_correlation_20260911/units/coreOU_mid_EE085_mean095_rho000/2511_847101/workers/trajectory.npz'
    with np.load(raw) as z:
        names=z['contact_names'].tolist();xy=z['contact_xy_mm'];endpoints=xy[[names.index(n) for n in ['SCL9','ICL11','ICL9']]]
    val,vec=np.linalg.eigh(np.cov(endpoints.T));angle=float(np.rad2deg(np.arctan2(vec[1,-1],vec[0,-1]))%180)
    candidates=[];r=c0['parameters']['radius_A_mm']
    for layout,dy in [('endpoint',-3.),('up3',0.)]:
        def add(label,sh=None,output=None,params=None,compare='circle'):
            c=copy.deepcopy(c0);c.update(id=layout+'__'+label,layout=layout,shape=sh or {},outgoing=output or {},stage='shape_range',contrast=label,comparison=layout+'__'+compare)
            c['centers_mm'][0][1]+=dy
            if params:c['parameters'].update(params)
            c['radii_mm']=[c['parameters']['radius_A_mm'],c['parameters']['radius_B_mm']];candidates.append(c)
        add('circle')
        for ar in [4.,9.]:add('ellipse'+str(int(ar)),dict(aspect_A=ar,angle_A_deg=angle,match_dose=True))
        add('ellipse4_orthogonal',dict(aspect_A=4,angle_A_deg=angle+90,match_dose=True))
        add('radius25',params=dict(radius_A_mm=2.5))
        add('radius25_dose_matched',dict(match_dose=True,dose_reference_radius_mm=r),params=dict(radius_A_mm=2.5))
        add('out_reference',output=dict(resample=True))
        for key,value,label in [('perp_scale',1.5,'out_perp15'),('perp_scale',2.,'out_perp20'),('parallel_scale',1.5,'out_parallel15'),('angle_offset_deg',-15.,'out_angle_m15'),('angle_offset_deg',15.,'out_angle_p15')]:
            add(label,output=dict(resample=True,**{key:value}),compare='out_reference')
        add('ellipse4_out_perp20',dict(aspect_A=4,angle_A_deg=angle,match_dose=True),dict(resample=True,perp_scale=2.),compare='ellipse4')
    files=list(dict.fromkeys(base.SOURCE_FILES+[run.SCRIPT,SCRIPT,ROOT/'src/topic4_core_ou_correlation.py',ROOT/'src/topic4_core_shape_output.py']))
    plan=dict(schema='topic4.core_shape_output_response.v1',physics=dict(version=shape.VERSION),source_snapshot={str(p):rt.sha(p) for p in files},
        patient_prior=dict(source=str(source),source_sha256=rt.sha(source),anchor_names=['SCL9','ICL11','ICL9'],anchor_xy_mm=endpoints.tolist(),
            angle_A_deg=angle,interpretation='already adopted coarse endpoint geometry; not independent validation or a prescribed route'),
        topology_seed=2511,seeds=[847101,847102],duration_ms=60000.,analysis=dict(burnin_ms=1500.,primary_min_events_per_run=16),runaway=base.RUNAWAY,
        candidates=candidates,resources=dict(initial_workers=2,max_workers=20,min_available_GiB=50,per_tree_limit_GiB=18,launch_reserve_GiB=9),
        response=dict(anchors=['up3__circle','up3__ellipse4'],axes=[['EE_same_core_scale',.75],['EE_same_core_scale',1.],['EE_core_to_out_scale',.75],['EE_core_to_out_scale',1.25],
            ['EI_same_core_scale',.75],['EI_same_core_scale',1.25],['IE_same_core_scale',1.25],['II_same_core_scale',.75],['depth_A_scale',.75],['depth_A_scale',1.25],
            ['core_mean_rate_scale',1.],['core_ou_correlation',.5],['core_ou_correlation',1.],['EE_core_to_out_degree_scale',1.5]]),
        confirmation=dict(candidates=['endpoint__circle','endpoint__ellipse4','up3__circle','up3__ellipse4','up3__out_reference','up3__out_perp20','up3__radius25','up3__radius25_dose_matched'],topology_seeds=[2611,2612],noise_seeds=[847201,847202]),
        budget=dict(shape_range=52,response=56,confirmation=32,total_formal=140,application_canary=4,canary_duration_ms=500),
        contract='No new route/label/width/rotation loss. Paired parameter responses, frozen patient features plus participation loss for ranking only. New topologies test predeclared contrasts irrespective of sign. Complete bounded 140 formal units, then scientific review; no automatic model freeze/Fig5.',
        spiral='Operational rotational candidates with native ring and phase sensitivity; not sustained rotors; diagnostic only; synthetic superposition false-positive control retained.')
    (OUT/'candidates').mkdir(exist_ok=True);(OUT/'logs').mkdir(exist_ok=True);(OUT/'confirmation').mkdir(exist_ok=True)
    rt.write(OUT/'confirmation/frozen_networks.json',rt.read(OLD/'confirmation/frozen_networks.json'))
    for c in candidates:rt.write(OUT/'candidates'/f'{c["id"]}.json',c)
    rt.write(OUT/'plan.json',plan);return plan


def configure():
    run.PLAN=OUT/'plan.json';base.OUT=OUT;base.build_unit=build_unit


def result_path(stage,u):return OUT/stage/'units'/u[0]/f'{u[1]}_{u[2]}'/'workers/trajectory.json'


def queue(stage,units,duration,workers):
    import psutil
    pending=[]
    for u in units:
        p=result_path(stage,u)
        if p.exists() and not base.complete(p):raise RuntimeError(f'invalid existing result: {p}')
        if not p.exists():pending.append(u)
    active={};failures=[];peak=0.
    while pending or active:
        rss={}
        for pid,(proc,u,log) in list(active.items()):
            if proc.poll() is not None:
                log.close();del active[pid]
                if proc.returncode or not base.complete(result_path(stage,u)):failures.append(dict(unit=u,exit=proc.returncode,log=log.name))
                else:peak=max(peak,rt.read(result_path(stage,u))['peak_rss_gib'])
                continue
            try:
                tree=[psutil.Process(pid)]+psutil.Process(pid).children(recursive=True);rss[pid]=sum(p.memory_info().rss for p in tree)/2**30
                if rss[pid]>18 or rt.available_gib()<30:
                    for p in reversed(tree):p.terminate()
                    failures.append(dict(unit=u,error='RESOURCE_GUARD',rss_gib=rss[pid]))
            except psutil.NoSuchProcess:pass
        if failures:pending=[]
        # Available memory already excludes live RSS; reserve remaining growth and one new tree.
        growth=sum(max(0,9-rss.get(pid,0)) for pid in active)
        if pending and len(active)<workers and rt.available_gib()>50+growth+max(9,peak*1.4):
            u=pending.pop(0);log=(OUT/'logs'/f'{stage}_{u[0]}_{u[1]}_{u[2]}.log').open('a')
            cmd=[rt.PYTHON,'-u',str(SCRIPT),'worker','--stage',stage,'--candidate',u[0],'--topology',str(u[1]),'--seed',str(u[2]),'--duration',str(duration)]
            p=subprocess.Popen(cmd,cwd=ROOT,env=rt.ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[p.pid]=(p,u,log)
        rt.write(OUT/'status.json',dict(status=stage.upper()+'_RUNNING' if not failures else 'ENGINEERING_FAILURE_DRAINING',stage=stage,total=len(units),
            complete=sum(result_path(stage,u).exists() for u in units),queued=len(pending),active=[dict(pid=pid,unit=u,rss_gib=rss.get(pid)) for pid,(p,u,f) in active.items()],
            max_workers=workers,peak_rss_gib=peak,available_gib=rt.available_gib(),failures=failures,updated_unix=time.time()))
        time.sleep(5)
    if failures:raise RuntimeError(json.dumps(failures))


def controller():
    plan=prepare();configure()
    with (OUT/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        canary=[('up3__'+c,2511,847101) for c in ['circle','ellipse4','radius25_dose_matched','out_perp20']]
        queue('canary',canary,500.,2)
        validate_canary(plan,canary)
        queue('shape_range',[(c['id'],2511,s) for c in plan['candidates'] for s in plan['seeds']],plan['duration_ms'],20)
        response=[]
        for anchor in plan['response']['anchors']:
            for key,value in plan['response']['axes']:
                c=copy.deepcopy(rt.read(OUT/'candidates'/f'{anchor}.json'));c['id']=anchor+'__'+key+'_'+str(value);c['comparison']=anchor;c['contrast']=key+'='+str(value);c['stage']='response'
                if key.startswith('core_'):c[key]=value
                else:c['parameters'][key]=value
                path=OUT/'candidates'/f'{c["id"]}.json'
                if path.exists() and rt.read(path)!=c:raise RuntimeError('response candidate changed on restart')
                rt.write(path,c);response.extend((c['id'],2511,s) for s in plan['seeds'])
        rt.write(OUT/'response_units.json',response)
        queue('response',response,plan['duration_ms'],20)
        conf=plan['confirmation'];units=[(c,t,s) for c in conf['candidates'] for t in conf['topology_seeds'] for s in conf['noise_seeds']]
        queue('confirmation',units,plan['duration_ms'],20)
        rt.write(OUT/'simulation_complete.json',dict(status='SIMULATIONS_COMPLETE_ANALYSIS_PENDING',formal_units=140,updated_unix=time.time()))
        rt.write(OUT/'status.json',dict(status='SIMULATIONS_COMPLETE_ANALYSIS_PENDING',formal_units=140,updated_unix=time.time()))


def validate_canary(plan,units):
    audits={u[0]:rt.read(result_path('canary',u).parents[1]/'applied_physics.json') for u in units}
    baseline=audits['up3__circle'];old=rt.read(OLD/'recovery_core_OU_correlation_20260911/units/coreOU_mid_EE085_mean095_rho000/2511_847101/applied_physics.json')
    for k,v in baseline['identity'].items():
        if old['identity'].get(k)!=v:raise RuntimeError(f'circle static identity mismatch {k}')
    reference=OLD/'recovery_core_OU_correlation_20260911/units/coreOU_mid_EE085_mean095_rho000/2511_847101/workers/trajectory.npz'
    with np.load(reference) as ref,np.load(result_path('canary',units[0]).with_suffix('.npz')) as z:
        for key in ['trace_E_spikes','trace_I_spikes','trace_coreAE_mean_V','trace_coreBE_mean_V','sheet_activity_counts']:
            if not np.array_equal(z[key],ref[key][:len(z[key])]):raise RuntimeError('circle dynamic prefix mismatch '+key)
    for cid,a in audits.items():
        assert a['threshold']['n_raised']==0
        assert a['input']['ZM']=='off' and a['input']['spatial_ou']=='off'
        if a['threshold']['dose_target_mV'] is not None:assert abs(a['threshold']['dose_error_mV'])<1e-7
    assert audits['up3__ellipse4']['threshold']['members'][0]==baseline['threshold']['members'][0]
    rt.write(OUT/'canary_validation.json',dict(status='PASS',circle_static_identity_matches_old=True,circle_500ms_dynamic_prefix_matches_old=True,
        actual_application_audits=audits,note='Application and identity check; no propagation acceptance criterion.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','worker','controller']);p.add_argument('--stage');p.add_argument('--candidate');p.add_argument('--topology',type=int);p.add_argument('--seed',type=int);p.add_argument('--duration',type=float)
    a=p.parse_args()
    if a.action=='prepare':print(prepare()['budget'])
    elif a.action=='worker':configure();run.worker(a.stage,a.candidate,a.topology,a.seed,a.duration)
    else:controller()
