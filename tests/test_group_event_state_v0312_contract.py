"""Integration contracts on measured events, not labels on empty run cards."""
import copy
from dataclasses import replace
import itertools
import numpy as np
import pytest
import torch
from src.topic5_group_event_state.v0312 import data as D
from src.topic5_group_event_state.v0312.train import RunConfig,load_run,loss_for_targets,target_units,training_ids,training_normalizers,update,optimizer_for,evaluate
from src.topic5_group_event_state.v0312.engine import infer_asof,predict,build_model,apply_ablations
from src.topic5_group_event_state.v0312.prepare import Prepared
from src.topic5_group_event_state.v0312.frozen import conditional_set_lp,identity_units,calibration
from src.topic5_group_event_state.v0312.seizure import controls_mask,available_recent
from src.topic5_group_event_state.v0312.synthetic import make_world
from src.topic5_group_event_state.v0311.packets import block_event_tables

@pytest.fixture(scope='module')
def real():
    torch.set_num_threads(1)
    return load_run(RunConfig(device='cpu'))


def test_optimization_seed_cannot_change_split_or_coordinates(real):
    _,p=real
    m,b=load_run(RunConfig(device='cpu',seed=20260908))
    assert p.split['split_id']==b.split['split_id']
    assert p.scaling['transform_id']==b.scaling['transform_id']
    assert np.array_equal(D.target_table(p.payload,p.split,'inner'),D.target_table(b.payload,b.split,'inner'))


def test_sid_hidden_poison_cannot_enter_fit_coordinates(real):
    _,p=real;pay=copy.deepcopy(p.payload);s=D.build_split(pay,p.split['subject'],stage='sid',protocol='S-ID')
    px,pt,_=D.packet_tables(pay,s);sc=D.fit_scaling(pay,s,px,pt)
    bad=~s['train_packet'][D.event_packets(pay)]
    assert bad.any()
    pay['contact_tokens'][bad]*=1000;pay['event_features'][bad]*=1000
    for k in ('band_ratio','signed_xlag','delay_iqr'):pay['targets'][k][bad]*=1000
    px,pt,_=D.packet_tables(pay,s);after=D.fit_scaling(pay,s,px,pt)
    assert sc['transform_id']==after['transform_id']


def test_common_parameters_identical_with_rich_input_added(real):
    _,p=real;a=build_model(p,'P_stats');b=build_model(p,'P_marks')
    shared=set(a.state_dict())&set(b.state_dict())
    for k in shared:assert torch.equal(a.state_dict()[k],b.state_dict()[k]),k


def test_half_hour_is_per_query_and_no_observation_is_explicit(real):
    m,p=real;qs=np.array([2275,2305]);s=infer_asof(m,p,qs,'outer',.5)
    assert s.readable_events[0]==0 and s.readable_events[1]>0
    assert s.metadata()['last_observed_source_time'][0] is None
    assert s.metadata()['has_observation']==[False,True]
    assert np.nanmax(s.release_time-s.query_time)<=0


def test_query_is_propagated_to_query_time_and_chunk_invariant(real):
    from src.topic5_group_event_state.v0312.numerics import propagate_moments
    m,p=real
    a=infer_asof(m,p,[2305,2310],'outer',2)
    b=infer_asof(m,p,[2310],'outer',2)
    # Strict windows have different starts, but a batch must not change a query.
    assert torch.allclose(a.m[1],b.m[0],atol=1e-7)
    assert np.all(a.information_age_minutes>0)
    c=infer_asof(m,p,[2305,2310],'outer')
    # No new source has been published during this gap.
    assert c.source_time[0]==c.source_time[1]
    mm,PP=propagate_moments(m.dynamics,c.m[:1],c.P[:1],5/60)
    assert torch.allclose(mm,c.m[1:],atol=1e-7)
    assert torch.allclose(PP,c.P[1:],atol=1e-7)


def test_unreleased_values_cannot_change_query(real):
    m,p=real;p2=copy.deepcopy(p);q=2305
    bad=np.flatnonzero(p.packet_release>p.packet_end[q]);ei=np.isin(p2.segment.numpy(),bad)
    p2.tokens[ei]=100.;p2.stats[bad]=100.
    a=infer_asof(m,p,[q],'outer');b=infer_asof(m,p2,[q],'outer')
    assert torch.equal(a.m,b.m) and torch.equal(a.P,b.P)


def test_sid_long_forecast_is_target_indexed(real):
    _,p=real;s=D.build_split(p.payload,p.split['subject'],stage='sid',protocol='S-ID')
    table=D.target_table(p.payload,s,'outer',1,(120,))
    assert len(table)>0
    assert s['outer_packet'][table[:,0]].all()
    assert np.all(table[:,0]-table[:,2]==120)
    assert not s['train_packet'][s['inner_packet']|s['outer_packet']].any()


def test_mc_paths_are_query_keyed_and_horizon_consistent(real):
    m,p=real;s=infer_asof(m,p,[2305,2310],'outer',.5)
    a=predict(m,p,s,(1,5),paths=3,seed=55);b=predict(m,p,s.subset([1]),(5,),paths=3,seed=55)
    assert torch.allclose(a[5].grid[:,:,1],b[5].grid[:,:,0],atol=1e-7)
    c=predict(m,p,s,(1,5,30,120),paths=3,seed=55)
    assert torch.equal(a[5].grid,c[5].grid)


def test_accumulation_preserves_physical_target_gradient(real):
    _,p=real;cfg=RunConfig(device='cpu',batch_size=4,microbatch=4,train_paths=2,history_hours=.5)
    ids=training_ids(p)[200:204];norm=training_normalizers(p,4)
    a=build_model(p);b=build_model(p);oa=optimizer_for(a,cfg);ob=optimizer_for(b,cfg)
    la,ga=update(a,p,cfg,oa,ids,norm,7,4);lb,gb=update(b,p,cfg,ob,ids,norm,7,2)
    assert abs(la-lb)<1e-5
    va=torch.cat([v.grad.reshape(-1) for v in a.parameters() if v.grad is not None]);vb=torch.cat([v.grad.reshape(-1) for v in b.parameters() if v.grad is not None])
    assert float((va-vb).norm()/va.norm())<2e-5


def test_set_likelihood_normalizes_conditional_identity():
    logits=torch.tensor([[.2,-.3,.8,1.]],dtype=torch.double);members=[]
    for c in itertools.combinations(range(4),2):
        y=torch.zeros(4,dtype=torch.double);y[list(c)]=1;members.append(y)
    y=torch.stack(members);lp=conditional_set_lp(logits.expand(len(y),-1),y,torch.zeros(4,dtype=torch.long))
    assert torch.allclose(lp.exp().sum(),torch.ones((),dtype=torch.double),atol=1e-12)
    assert not identity_units(torch.ones(1,4),torch.zeros(4,dtype=torch.long)).any()


def test_seizure_controls_exclude_entire_preinterval_and_availability(real):
    sz=[dict(onset_epoch=10000,offset_epoch=10010)]
    times=np.array([2700.,2800.,3000.,9999.,10000.,13610.,14000.])
    assert controls_mask(times,sz).tolist()==[True,False,False,False,False,False,True]
    _,p=real;r=available_recent(p,2275)
    assert r['exposure_seconds']==0 and r['rate_per_hour'] is None


def test_raw_synthetic_targets_really_follow_the_human_measurement():
    p=make_world(seed=2,n_blocks=3,morph_gain=1.5)
    tok,ev,tgt,part=block_event_tables(p['raw_measurement'],p['raw_manifest'],p['n_contacts'])
    assert np.array_equal(tok,p['contact_tokens'],equal_nan=True)
    for k,v in tgt.items():assert np.array_equal(v,p['targets'][k],equal_nan=True),k
    assert np.unique(p['targets']['shaft_count'],axis=0).tolist()==[[2.,2.]]
    pi=((p['event_time']-p['observed_support'][0,0])//60).astype(int)
    assert np.corrcoef(p['slow_state'][pi],p['targets']['band_ratio'][:,0])[0,1]>.9


def test_crossview_input_and_target_channels_both_removed(real):
    _,p=real;q=copy.deepcopy(p);apply_ablations(q,True)
    assert q.xlag_valid.sum()==0 and q.iqr_valid.sum()==0
    for a,b in ((0,3),(8,13),(18,28)):assert q.tokens[:,:,a:b].count_nonzero()==0
    assert q.event[:,4:6].count_nonzero()==0


def test_calibration_uses_same_g_distribution(real):
    m,p=real;cfg=RunConfig(device='cpu',history_hours=.5,eval_stride=180,eval_paths=4,eval_chunk=8)
    c=calibration(m,p,cfg,'inner')
    assert len(c['rows'])>0
    assert max(abs(r['logp']-r['scipy_logp']) for r in c['rows'])<1e-4


def test_persistent_credit_is_microbatch_invariant(real):
    _,p=real;cfg=RunConfig(device='cpu',batch_size=4,train_paths=2,history_hours=None)
    ids=training_ids(p)[200:204];norm=training_normalizers(p,4)
    a=build_model(p);b=build_model(p)
    la,ga=update(a,p,cfg,optimizer_for(a,cfg),ids,norm,7,4);lb,gb=update(b,p,cfg,optimizer_for(b,cfg),ids,norm,7,2)
    va=torch.cat([v.grad.reshape(-1) for v in a.parameters() if v.grad is not None]);vb=torch.cat([v.grad.reshape(-1) for v in b.parameters() if v.grad is not None])
    assert abs(la-lb)<1e-5 and float((va-vb).norm()/va.norm())<2e-5


def test_queue_lock_survives_dispatch_and_releases_after_crash(tmp_path):
    import subprocess,sys,time
    from src.topic5_group_event_state.v0312.queue import acquire_lock
    path=tmp_path/'lock'
    process=subprocess.Popen([sys.executable,'-c',
        'import fcntl,time,sys; f=open(sys.argv[1],"w"); fcntl.flock(f,fcntl.LOCK_EX); print("owned",flush=True); time.sleep(30)',str(path)],stdout=subprocess.PIPE,text=True)
    try:
        assert process.stdout.readline().strip()=='owned'
        assert acquire_lock(path) is None
        process.kill();process.wait(timeout=5)
        f=acquire_lock(path);assert f is not None;f.close()
    finally:
        if process.poll() is None:process.kill();process.wait()


def test_both_inner_origins_required_for_outer_recipe(tmp_path):
    import json
    from src.topic5_group_event_state.v0312.queue import make_recipe
    from src.topic5_group_event_state.v0312.train import source_digest
    p=tmp_path/'one.json';p.write_text(json.dumps(dict(status='COMPLETE',source_digest=source_digest()[0],config=dict(stage='inner0'))))
    with pytest.raises(ValueError,match='both temporal'):make_recipe([p],tmp_path/'recipe.json')


def test_training_interpretation_separates_origin_outer_plateau_and_budget():
    from src.topic5_group_event_state.v0312.train import interpret_training
    assert 'no learned component' in interpret_training('inner0',0,'plateau',{'drops':2})
    assert 'not convergence evidence' in interpret_training('outer',100,'fixed_inner_recipe',{'drops':0})
    assert 'local optimization evidence' in interpret_training('inner0',100,'plateau',{'drops':2})
    assert 'convergence remains unresolved' in interpret_training('inner0',100,'budget',{'drops':1})


def test_recipe_requires_both_inner_origins_to_select_nonzero_checkpoint(tmp_path):
    import json
    from src.topic5_group_event_state.v0312.queue import make_recipe
    from src.topic5_group_event_state.v0312.train import source_digest
    common=dict(subject='x',protocol='S-E',inputs='P_marks',family='I-L-G1',arm='constant_state',
                history_hours=None,crossview=False,old_targets=False,lr=.001,dynamics_lr=.0003,
                seed=1,split_seed=2,train_paths=4,batch_size=32)
    paths=[]
    for stage,selected in [('inner0',0),('inner1',100)]:
        p=tmp_path/f'{stage}.json';p.write_text(json.dumps(dict(status='COMPLETE',source_digest=source_digest()[0],
            config=common|{'stage':stage},selected_updates=selected,stop_reason='plateau',plateau={'drops':2},curve=[])))
        paths.append(p)
    recipe=make_recipe(paths,tmp_path/'recipe.json')
    assert recipe['updates']==50 and recipe['learned_state_eligible'] is False
    assert recipe['training_adequacy'].startswith('both temporal INNER origins reached')


def test_next_plan_admits_only_one_training_process_per_gpu(tmp_path):
    from src.topic5_group_event_state.v0312.queue import build_plan
    plan=build_plan(tmp_path/'plan.json',tmp_path/'results',quick=True)
    assert plan['workers_per_gpu']==1
    assert 'concurrent same-GPU' in plan['gpu_concurrency_reason']


def test_reporting_refuses_duplicate_physical_scores():
    from src.topic5_group_event_state.v0312.report import flatten
    row=dict(horizon=1,packet=[5],logp={'count':torch.tensor([-2.])},units={'count':torch.tensor([1.])})
    with pytest.raises(ValueError,match='twice'):flatten({'rows':[row,row]},'count')


def test_frozen_query_cache_matches_uncached_and_refuses_training_model(real):
    m,p=real;q=copy.deepcopy(p);q.frozen_query_cache={}
    with pytest.raises(ValueError,match='frozen'):infer_asof(m,q,[2305],'outer')
    frozen=copy.deepcopy(m)
    for v in frozen.parameters():v.requires_grad_(False)
    uncached=infer_asof(frozen,p,[2305,2310],'outer')
    a=infer_asof(frozen,q,[2305,2310],'outer');b=infer_asof(frozen,q,[2310,2305],'outer')
    assert torch.equal(uncached.m,a.m) and torch.equal(uncached.P,a.P)
    assert torch.equal(a.m,b.m.flip(0)) and torch.equal(a.P,b.P.flip(0))


def test_ictal_cache_join_uses_identity_and_records_two_time_origins():
    from src.topic5_group_event_state.v0312.spatial_transfer import inventory_crosswalk
    rows=[dict(subject='1125',seizure_id='b',eeg_onset_epoch='200',clin_onset_epoch='240'),dict(subject='1125',seizure_id='a',eeg_onset_epoch='100',clin_onset_epoch='120')]
    seizures=[dict(seizure_id='b',onset_epoch=200)]
    r=inventory_crosswalk('epilepsiae_1125',rows,seizures)
    assert r[0]['source_index']==1 and r[0]['clinical_onset']-r[0]['eeg_onset']==40
    with pytest.raises(ValueError,match='disagrees'):inventory_crosswalk('epilepsiae_1125',rows,[dict(seizure_id='b',onset_epoch=201)])


def test_wrong_time_control_retains_recipient_time_and_clock(real):
    from src.topic5_group_event_state.v0312.engine import wrong_time_control
    m,p=real;s=infer_asof(m,p,[2305],'outer',.5)
    ref=dict(donor_time=s.query_time-86400,donor_age=s.information_age_minutes,donor_m=s.m+1,donor_P=s.P*2)
    z=wrong_time_control(s,ref)
    assert np.array_equal(z.query_time,s.query_time) and np.array_equal(z.query_packet,s.query_packet)
    assert torch.equal(z.m,s.m+1) and torch.equal(z.P,s.P*2)


def test_zero_innovation_preserves_prior_location_but_corrects_covariance():
    from src.topic5_group_event_state.v0312.model import SlowState
    from src.topic5_group_event_state.v0312.numerics import evidence_update
    model=SlowState(4)
    with torch.no_grad():model.a_head.weight.zero_();model.a_head.bias.zero_()
    m=torch.arange(24,dtype=torch.float32)[None,:]/10;P=torch.eye(24)[None,:,:]
    a,R,_=model.evidence(torch.zeros(1,64),m,P)
    mm,PP,_,_=evidence_update(m.double(),P.double(),a.double(),R.double())
    assert torch.allclose(mm,m.double(),atol=1e-12)
    assert torch.all(PP.diagonal(dim1=-1,dim2=-2)<1.)


def test_adapter_deterministic_sets_are_not_a_successful_zero_loss():
    from src.topic5_group_event_state.v0312.frozen import fit_adapter
    d=dict(clock=torch.zeros(3,2),state=torch.ones(3,24),identity=torch.ones(3,4),history=torch.zeros(3,4),event_ids=torch.arange(3),packet=np.arange(3))
    result=fit_adapter(d,d,d,torch.zeros(4,dtype=torch.long),kind='state',steps=1)
    assert result['status']=='NOT_ESTIMABLE'


def test_short_targets_are_not_gated_by_a_120min_origin(real):
    from src.topic5_group_event_state.v0312.train import training_table
    _,p=real;table=training_table(p);short=np.unique(table[table[:,1]==1,0]);long=np.unique(table[table[:,1]==120,0])
    only_short=np.setdiff1d(short,long)
    assert len(only_short)>0 and np.isin(only_short,training_ids(p)).all()
    norm=training_normalizers(p,32)
    assert norm[120]['count']<norm[1]['count']
    assert abs(norm[1]['count']-32)<1e-10


def test_all_frozen_adapter_arms_include_known_target_clock(real):
    from src.topic5_group_event_state.v0312.frozen import fit_adapter
    d=dict(clock=torch.tensor([[0.,1.],[1.,0.],[-1.,0.],[0.,-1.]]),state=torch.ones(4,24),
           history=torch.zeros(4,4),identity=torch.eye(4),event_ids=torch.arange(4),packet=np.arange(4))
    for kind in ('trait','history','state','state_history'):
        r=fit_adapter(d,d,d,torch.zeros(4,dtype=torch.long),kind=kind,steps=1)
        assert r['status']=='COMPLETE' and r['fitted']['feature_names'][-1]=='clock'
        assert r['fitted']['state_dict']['weight'].shape[1]=={'trait':2,'history':6,'state':26,'state_history':30}[kind]
