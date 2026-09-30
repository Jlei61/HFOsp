#!/usr/bin/env python3
"""Keep native Fig5 interictal physics; recruit extra feedback only at high global rate.

This is a declared new nonlinear response, not a redistribution of native GABA.
All parameters and equations are fixed from t=0; no event-triggered intervention.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import argparse
import copy
import fcntl
import json
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import psutil
from scipy.signal import lfilter
import run_topic4_z_recovery_exploration as budget

previous=budget.previous
fixed=budget.fixed
carrier=budget.carrier
base=budget.base
audit=budget.audit
SOURCE=budget.OUT
REF=Path('/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/fig5_preentry_event_audit_20260914')
OUT=Path('/data/hfosp/topic4_sef_hfo/fig5_rhythm_preserving_feedback_20260916')
SEED=9108401
RATE_START=200.
RATE_FULL=500.
EK=-30.
EG=-17.662847938268442
REGIONS=budget.REGIONS
write=budget.write


class RhythmSlow(budget.RecoveryBudgetSlow):
    """Native local Z*II + additive Z-gated G, and high-rate-gated intrinsic K."""
    def __init__(self,*args,**kw):
        assert kw['gamma']==0., 'Do not redistribute local inhibition'
        super().__init__(*args,**kw)
        self.gate=0.
        self.feedback_records=[]
        self.first_feedback_s=None

    def uses_shunt(self):
        # Literal native membrane path while added conductances are exactly zero.
        return bool(np.any(self.g_global) or np.any(self.g_k))

    def shunt_g_at_E(self):
        return self.g_global+self.g_k

    def apply_currents(self,ie,ii,labels=None,rec=None):
        self.gate=float(np.clip((self.r_global-RATE_START)/(RATE_FULL-RATE_START),0.,1.))
        raw=self.global_gain*self.gate
        self.g_global[:]=raw*self.z[:self.NE]
        total=ii
        if raw:
            total=ii.copy();total[:self.NE]+=raw*(18.-self.global_reversal)
        self.delivered=total;self._I_I_last=total
        self.raw_mean=float(ii[:self.NE].mean())
        # Preserve every operation of the old current-based native local equation.
        value=ie-self.z*ii-self.cfg.eta_m*self.m
        if np.any(self.g_k):
            value[:self.NE]+=self.g_k*(EK-self.global_reversal)
        if self.gate>0 and (self.global_gain>0 or self.sahp_gain>0) and self.first_feedback_s is None:
            self.first_feedback_s=self._step_index*.0001
        v=self.voltage[:self.NE] if self.voltage is not None else np.zeros(self.NE)
        global_current=self.g_global*(v-self.global_reversal)
        if self._step_index%10==0:
            self.extra_records.append([self._step_index*.1,self.r_global,raw,
                self.g_global.mean(),0.,0.,global_current.mean()])
            self.k_records.append([self._step_index*.1,self.g_k.mean(),self.g_k.max(),
                                   (self.g_k*(v-EK)).mean()])
        if self._step_index%20==0 and self.current_recorder is not None:
            absolute=np.abs(ie[:self.NE])+np.abs(self.z[:self.NE]*ii[:self.NE])+np.abs(global_current)
            self.contact_records.append(np.array([np.dot(w,absolute[ix]) for ix,w in zip(self.current_recorder._idx,self.current_recorder._w)]))
            self.field_records.append(np.bincount(self.current_cells,weights=absolute,minlength=400)/self.current_cell_counts)
            self.current_record_times.append(self._step_index*.1)
        if self._step_index%200==0:
            row=[self._step_index*.1]
            for ix in self.region_groups():
                row.extend([self.g_k[ix].mean(),self.g_global[ix].mean(),v[ix].mean(),
                    (self.g_k[ix]*(v[ix]-EK)).mean(),global_current[ix].mean(),
                    ie[ix].mean(),(self.z[:self.NE][ix]*ii[:self.NE][ix]).mean(),
                    self.cfg.eta_m*self.m[:self.NE][ix].mean(),self.z[:self.NE][ix].mean(),self.m[:self.NE][ix].mean()])
            self.regional_records.append(row)
            self.feedback_records.append([self._step_index*.1,self.r_global,self.gate,raw,
                                          self.g_global.mean(),self.g_k.mean()])
        return value

    def step(self,spk,labels,dt):
        if self.budget_ms==0:self.budget_start=self.means(self.z[:self.NE])
        target=self._I_I_last[:self.NE]<self.cfg.I_th_EI
        dz=(target.astype(float)-self.z[:self.NE])*(dt/self.cfg.tau_z)
        self.recovery_sum+=np.maximum(dz,0.);self.consumption_sum+=np.maximum(-dz,0.)
        self.budget_ms+=dt
        # Native M/Z and the existing15ms global rate; bypass old ungated K step.
        NativeTerminationStep(self,spk,labels,dt)
        self.g_k*=np.exp(-dt/self.sahp_tau_ms)
        self.g_k[spk[:self.NE]]+=.01*self.sahp_gain*self.gate
        if self._step_index%200==0:
            end=self.means(self.z[:self.NE])
            gain,loss=self.means(self.recovery_sum),self.means(self.consumption_sum)
            error=end-self.budget_start-(gain-loss)
            self.max_balance_error=max(self.max_balance_error,float(np.max(abs(error))))
            assert self.max_balance_error<1e-11
            sec=self.budget_ms/1000.
            row=np.stack([self.budget_start,end,gain/sec,loss/sec,(gain-loss)/sec,error,
                          self.means(target),self.means(self._I_I_last[:self.NE])],axis=-1)
            self.budget_records.append((self._step_index*.1,row))
            self.recovery_sum.fill(0.);self.consumption_sum.fill(0.);self.budget_ms=0.


def reference(seed):
    dest=OUT/'references'/f'native_s{seed}.npz'
    if dest.exists():return dest
    folder=REF/'runs'/f'eta0.0005_s{seed}'
    if not folder.exists():return None
    p=base.read(SOURCE/'protocol.json');result=base.read(folder/'result.json')
    assert result['identity']==p['identity']
    keys=['time_ms','spikes_1ms','regions_1ms','raster','slow_time_ms','Z','M','inputs','field_1ms','population_0p1ms']
    parts={k:[] for k in keys};end=0
    for file in sorted((folder/'chunks').glob('*.npz')):
        if '.tmp.' in file.name:continue
        with np.load(file) as a:
            assert int(a['start_step'])==end;end=int(a['end_step'])
            for k in keys:parts[k].append(a[k])
        if end>=100000:break
    d={k:np.concatenate(v) for k,v in parts.items()}
    # Same fixed reference interval for waveform/state identity, never selected by new outcomes.
    spikes=d['population_0p1ms'][:100000,0]
    rr=lfilter([1000/15/32000],[1,-np.exp(-.1/15)],spikes)
    rr=np.r_[0.,rr[:-1]];rt=np.arange(len(rr))*.0001
    d['R15_time_s']=rt;d['R15_Hz']=rr
    d['field_5ms']=d.pop('field_1ms')[:10000].reshape(-1,5,400).sum(1)
    dest.parent.mkdir(exist_ok=True)
    np.savez_compressed(dest,**d)
    write(dest.with_suffix('.json'),dict(source=str(folder),seed=seed,
        preevent_R15_max_0p5_9p42_s=float(rr[(rt>=.5)&(rt<9.42)].max()),
        early_R15_max_0p5_8_s=float(rr[(rt>=.5)&(rt<8)].max()),
        first_R15_above200_s=float(rt[np.flatnonzero(rr>200)[0]]) if np.any(rr>200) else None,
        identity=result['identity'],source_result_sha256=base.sha(folder/'result.json')))
    return dest


def make_job(g,k500,tau,seed,index,stage='initial',horizon=60.):
    return dict(name=f'G{g:g}_K{k500:g}_tau{tau:g}_s{seed}',seed=seed,eta_m=.0005,tau_M_s=1.,tau_Z_s=5.,
        threshold=95.19851312666987,mode='native',gamma=0.,global_gain=g,global_resource='native_z',
        phi_jump=0.,horizon_s=horizon,checkpoint_s=2.,device=index%2,qa=False,
        stop_after_second_entry=False,stage=stage,C_R=0.,feedback_form='conductance',
        global_reversal_mV=EG,sahp_gain=k500/(5*tau),sahp_tau_s=tau,k100=k500/5,
        K500=k500,G500=g,rate_start_Hz=RATE_START,rate_full_Hz=RATE_FULL,
        added_K_increment_when_gate1=k500/(500*tau),local_inhibition_multiplier=1.,
        mechanism='Fixed high-global-rate gate, unchanged native local inhibition; no scheduled switch')


def prepare():
    if (OUT/'protocol.json').exists():return base.read(OUT/'protocol.json')
    OUT.mkdir(parents=True,exist_ok=True)
    p0=base.read(SOURCE/'protocol.json')
    # Full factorial G comparison for three declared K kinetics plus both mechanism controls.
    settings=[(0.,0.,1.),(0.,40.,.5),(15.,40.,.5),(15.,0.,1.),
              (0.,40.,2.),(15.,40.,2.),(0.,80.,1.),(15.,80.,1.)]
    jobs=[make_job(g,k,t,SEED,i) for i,(g,k,t) in enumerate(settings)]
    p={k:copy.deepcopy(p0[k]) for k in ['identity','source_hashes','baseline','reference_current_scale']}
    p.update(status='PRESERVE_NATIVE_RHYTHM_FIRST',created_epoch=time.time(),deadline_epoch=time.time()+8*3600,
        initial_jobs=jobs,branch_jobs=[],max_workers=4,min_available_memory_GiB=70.,disk_reserve_GiB=50.,
        producer_sha256=base.sha(carrier.__file__),wrapper_sha256=base.sha(fixed.__file__),
        exploration_sha256=base.sha(__file__),protected_global_authorized=False,
        reference_root=str(REF),source_round=str(SOURCE),same_substrate=True,
        local_inhibition='Exactly original Z_i*II_i, coefficient1 at all times; gamma=0, no local/global redistribution.',
        new_fixed_equation=dict(q='clip((R_G-200Hz)/300Hz,0,1)',R_G='Existing15ms causal E population rate',
           G_raw='G500*q',G_applied='Z_i*G_raw',J='Original II_i + (18-EG)*G_raw for E; I unchanged',
           K='gK decays with tauK; each E spike adds K500/(500*tauK_seconds)*q',
           K500='gK/gL steady level for a neuron firing500Hz while q=1. k100=K500/5 is conditional reference strength, not homogeneous100Hz equilibrium (q=0 there).'),
        new_hypothesis='High-rate-selective recruitment of additional feedback. Not claimed to be the original Liou equation or an established biological mechanism.',
        unchanged='Native Z/M functions and parameters, graph, threshold field, input noise, local GABA, no extra10s pool or Z recovery supply.',
        preservation_gate='Each seed9108401 run must exactly reproduce native spikes, regional spikes, raster, fields, Z, M and input in first8s; extra G/K=0 there. Failure stops that run and further dispatch. The trace before first feedback activation is also compared where reference exists.',
        temporal_rule='Short20-200ms events bounded by >=20ms with all E AND both cores <5Hz, all-E event peak>=20Hz. Preserve native interictal rhythm before entry; postexit events must also resemble native duration/peak/interval and regional recruitment.',
        stage_policy='8paired-noise60s cold starts including native and global-only controls. At most1preserved-rhythm autonomous-exit+Z-recovery condition continues to120s and receives two extra noise seeds. Add same-seed native8s references only where missing. No other parameter expansion.8h wall bound, four workers.',
        acceptance='Native preentry rhythm preserved -> high-state entry -> autonomous exit with core Z recovery -> recurrent native-like brief population events. Reentry reported separately; human Fig5 morphology remains mandatory.',
        figure_contract='Latest user Fig5 A-E semantics. Reference-preservation overlay is mandatory. Do not label non-interictal initial strong activity Pre-ictal. F remains pending source-matched dense spectral validation.',
        human_review='PENDING',adaptive_selected=[])
    for f in [__file__,budget.__file__,previous.__file__,fixed.__file__,carrier.__file__,audit.__file__]:
        p['source_hashes'][str(Path(f).resolve())]=base.sha(f)
    impl=OUT/'implementation';impl.mkdir(exist_ok=True)
    for f in [__file__,budget.__file__,previous.__file__,fixed.__file__,carrier.__file__,audit.__file__]:shutil.copy2(f,impl/Path(f).name)
    shutil.copy2(SOURCE/'geometry.npz',OUT/'geometry.npz')
    with np.load(OUT/'geometry.npz') as new,np.load(REF/'geometry.npz') as old:
        for k in ['positions_e','sample_ids','region_counts']:assert np.array_equal(new[k],old[k]),k
    for j in jobs:write(OUT/'jobs'/f"{j['name']}.json",j)
    write(OUT/'protocol.json',p)
    reference(SEED);reference(SEED+1)
    (OUT/'design.md').write_text('''# 保留原Fig5间期节律的反馈实验

用户要求先保住原图已有的间期群体短事件，再得到进入、终止和返回。本轮不是继续调上一轮gamma分配比例：gamma固定0，原局部ZI抑制系数始终为1，原Z/M与快网络、背景噪声不变。统计单位为一个条件的一条噪声轨迹；初筛均使用同一9108401噪声配对。

新增机制是预先固定的高群体率响应q=clip((R_G-200)/300,0,1)，R_G沿用15ms因果E平均率。原参考0.5–9.42s的R_G最高约140Hz，因此原间期段新增反馈严格为0。G_raw=G500*q，G_applied=Z_i*G_raw；不使用protected通路。新增全局反馈在18mV参考电压的资源负荷加入J，原局部II不减小；Z仍按原方程演化。K只在q>0时按每E spike累积K500/(500*tauK_s)*q，始终按tauK衰减。K500表示q=1、单细胞500Hz时的稳态gK/gL。此为新的非线性机制假设，不是Liou原方程的复现。

8条60s条件：原生无新增反馈；仅G500=15；K500/tauK=(40,0.5s)、(40,2s)、(80,1s)各配G500=0或15。K单独组区分慢适应与快全局分流的贡献，G单独组检验无慢记忆能否终止；两种时间尺度检验终止后是否被K尾部压成静默。所有参数从t=0固定，无定时刺激、检测到onset后切参数、reset或强制回补Z。

第一硬门：配对主种子前8s的spikes、两核/核外计数、固定细胞raster、原生空间场、Z/M与外部输入逐位复现原Fig5；这段新增G/K必须严格为0。任何不符按错误停止该条与后续派发。首次反馈激活前的更长前缀也比较。不能只保事件数而改变原生节律。其他噪声逐种子与原生对照比较，固定8s窗若提前触发而受扰则不通过保留门。

短事件审核同时要求全E与两核低于5Hz至少20ms作为边界，事件全E峰>=20Hz、持续20–200ms。退出后至少5个短事件跨越2s，短事件占比>=80%；持续时间、峰值、间隔中位数须在该种子原生间期参考的0.5–2倍内，作为宽松探索门，仍需原生空间/参与和人工目视。不把quiet、Z回升或高低高切换单独算返回。

第二阶段最多选择1条同时保留间期、自主退出、两核Z净恢复的轨迹，以原完整状态续至120s，配两条独立噪声；优先已出现短事件返回者。缺同种子原生对照时先跑8s原生对照。最多11条冷启动（含至多1条缺失参考），4并发，8小时墙钟；无候选即停止，不自动扩展其他方程或参数。无退出只能表述已观察窗口；截止时保留截尾，不冒充完整120s。

交付原图参考对齐图、原生Z收支、参数控制表、Fig5候选；A原生raster、B Z/M、C真实状态、D meanZ/IE/rE、E本轮实测条件，F暂不冒充已核验患者能量。参考间期门未过者只能作为诊断图；不会将其标为Pre-ictal成功候选。最终停在作者审阅点。
''')
    return p


def qa():
    p=prepare();previous.OUT=OUT
    cfg=base.old.MZSlowVarsConfig(use_z=True,use_m=True,tau_z=5000.,I_th_EI=95.19851312666987,tau_adp=1000.,eta_m=.0005)
    rng=np.random.default_rng(16091607);checks=[]
    for g,k,tau in [(0.,0.,1.),(0.,40.,.5),(15.,80.,1.)]:
        RhythmSlow.C_R=0.;RhythmSlow.feedback_form='conductance';RhythmSlow.sahp_gain=k/(5*tau);RhythmSlow.sahp_tau_ms=1000*tau
        o=RhythmSlow(12,18,cfg,NE=10,mode='native',gamma=0.,global_gain=g,global_resource='native_z',phi_jump=0.)
        o._groups=[np.arange(3),np.arange(3,6),np.arange(6,10)];o.voltage=np.full(12,5.);o.global_reversal=EG
        native=carrier.RecoverySlow(12,18,cfg,NE=10,mode='native',gamma=0.)
        for _ in range(600):
            ie,ii=rng.uniform(0,180,(2,12));sp=rng.random(12)<.005
            o.r_global=139.96620253452687
            assert np.array_equal(o.apply_currents(ie,ii),native.apply_currents(ie,ii))
            assert np.array_equal(o._I_I_last,ii) and not o.uses_shunt()
            o.step(sp,None,.1);native.step(sp,None,.1)
            assert np.array_equal(o.z,native.z) and np.array_equal(o.m,native.m)
            assert np.all(o.g_k==0.) and np.all(o.g_global==0.)
        o.r_global=500.;o.z[:10]=.6;o.g_k[:]=2.
        ie,ii=rng.uniform(0,800,(2,12));v=o.apply_currents(ie,ii)
        rhs=v[:10]+o.shunt_g_at_E()*EG
        expected=ie[:10]-.6*ii[:10]-.0005*o.m[:10]+(.6*g)*EG+2.*EK
        assert np.allclose(rhs,expected,rtol=0,atol=1e-12)
        assert np.allclose(o._I_I_last[:10],ii[:10]+g*(18-EG))
        before=o.g_k.copy();o.step(np.ones(12,bool),None,.1)
        assert np.allclose(o.g_k,before*np.exp(-.1/(tau*1000))+k/(500*tau),rtol=0,atol=1e-14)
        o.r_global=100.;o.apply_currents(ie,ii);before=o.g_k.copy();o.step(np.ones(12,bool),None,.1)
        assert np.array_equal(o.g_k,before*np.exp(-.1/(tau*1000)))
        checks.append(dict(G500=g,K500=k,tauK=tau,low_rate_native_bitwise=True,
                           high_rate_rhs=True,resource_load=True,K_decay_below_gate=True))
    for path,h in p['source_hashes'].items():assert base.sha(path)==h,path
    audit.OUT=OUT;audit.qa()
    write(OUT/'mechanism_qa.json',dict(status='PASS',checks=checks,local_GABA_multiplier=1.,
          no_Z_reset=True,no_M_change=True,new_gate_is_fixed_equation=True))
    print(json.dumps(checks))


def flush(folder,step):
    budget.flush(folder,step)
    o=budget.RecoveryBudgetSlow.instance
    if o is None:return
    if o.first_feedback_s is not None and not (folder/'feedback_activation.json').exists():
        write(folder/'feedback_activation.json',dict(first_feedback_s=o.first_feedback_s))
    if o.feedback_records:
        arr=np.asarray(o.feedback_records);d=folder/'feedback_chunks';d.mkdir(exist_ok=True)
        path=d/f'{round(arr[0,0]*10):010d}_{step:010d}.npz';tmp=path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,time_ms=arr[:,0],R_global_Hz=arr[:,1],gate=arr[:,2],
                            G_raw=arr[:,3],G_applied_mean=arr[:,4],K_mean=arr[:,5])
        tmp.replace(path);o.feedback_records.clear()


def strict_events(rates,end):
    events=audit.old.event_audit.events(rates[:,:3].max(1),end=end)
    return [e for e in events if rates[round(e['start_s']/.01):round(e['end_s']/.01),0].max()>=20]


def rate_data(folder):
    d=audit.old.load(folder,keys=['spikes_1ms','regions_1ms'])
    if not d:return None
    with np.load(OUT/'geometry.npz') as g:nr=np.r_[32000,g['region_counts'][:3]]
    n=len(d['spikes_1ms'])//10
    return np.column_stack([d['spikes_1ms'][:n*10,0],d['regions_1ms'][:n*10,:3]]).reshape(n,10,4).sum(1)/nr/.01


def prefix_check(name):
    folder=OUT/'runs'/name;job=base.read(OUT/'jobs'/f'{name}.json')
    if job.get('stage')=='native_reference':return
    saved=base.read(folder/'rhythm_preservation.json') if (folder/'rhythm_preservation.json').exists() else {}
    if saved.get('status')=='PASS' and saved.get('activation_prefix_complete'):return
    ref=reference(job['seed'])
    if ref is None:
        native=next(j for j in prepare()['initial_jobs'] if j['seed']==job['seed'] and j['stage']=='native_reference')
        ref_folder=OUT/'runs'/native['name']
        reference_data=audit.old.load(ref_folder,keys=['spikes_1ms','regions_1ms','raster','slow_time_ms','Z','M','inputs','field_5ms'])
    else:
        with np.load(ref) as a:reference_data={k:a[k] for k in ['spikes_1ms','regions_1ms','raster','slow_time_ms','Z','M','inputs','field_5ms']}
    keys=list(reference_data)
    d=audit.old.load(folder,keys=keys)
    if not d:return
    observed=len(d['spikes_1ms'])/1000
    fb=audit.old.load(folder,'feedback_chunks',keys=['time_ms','G_raw','K_mean'])
    def compare(end):
        # Exclude partial count bins and the first step with active feedback.
        ns=int(np.floor(end*1000+1e-7));nt=int(np.floor(end*10000+1e-7));nf=int(np.floor(end*200+1e-7))
        checks={k:np.array_equal(d[k][:length],reference_data[k][:length]) for k,length in [
            ('spikes_1ms',ns),('regions_1ms',ns),('raster',nt),('field_5ms',nf)]}
        sel=d['slow_time_ms']<end*1000-1e-8;tm=d['slow_time_ms'][sel]
        ids=np.searchsorted(reference_data['slow_time_ms'],tm)
        assert np.array_equal(reference_data['slow_time_ms'][ids],tm)
        for key in ['Z','M']:checks[key]=np.array_equal(d[key][sel],reference_data[key][ids])
        x=d['inputs'][d['inputs'][:,0]<end*1000-1e-8]
        y=reference_data['inputs'][reference_data['inputs'][:,0]<end*1000-1e-8]
        checks['inputs']=np.array_equal(x,y)
        mask=fb['time_ms']<end*1000-1e-8
        checks['extra_feedback_zero']=bool(np.all(fb['G_raw'][mask]==0) and np.all(fb['K_mean'][mask]==0))
        return checks
    end=min(8.,observed);checks=compare(end)
    activation=base.read(folder/'feedback_activation.json')['first_feedback_s'] if (folder/'feedback_activation.json').exists() else None
    extended_end=min(observed,len(reference_data['spikes_1ms'])/1000,activation if activation is not None else np.inf)
    extended=compare(extended_end)
    passed=all(checks.values()) and all(extended.values())
    status='PASS' if passed and observed>=8 else 'PREFIX_PASS_INCOMPLETE' if passed else 'FAIL'
    prefix_complete=bool(observed>=min(activation if activation is not None else np.inf,len(reference_data['spikes_1ms'])/1000))
    write(folder/'rhythm_preservation.json',dict(status=status,compared_s=end,checks=checks,reference=str(ref or ref_folder),
        activation_prefix_s=extended_end,activation_prefix_checks=extended,activation_prefix_complete=prefix_complete,
        first_feedback_s=activation))
    if not passed:
        write(folder/'preservation_failure.json',dict(checks=checks,observed_s=observed))
        raise RuntimeError('Native interictal prefix changed; do not trade rhythm for termination')


def analyze(name):
    budget.OUT=OUT
    row=budget.analyze(name)
    if row is None:return None
    folder=OUT/'runs'/name;rr=rate_data(folder);end=len(rr)*.01;pp=row['primary']
    events=strict_events(rr,end)
    first=pp['entries'][0]['onset_s'] if pp['entries'] else end
    pre=audit.interval_events(events,.5,first)
    post=None
    if pp['low_activity_exits']:
        lo=pp['low_activity_exits'][-1]['confirmation_s']
        hi=next((e['onset_s'] for e in pp['entries'] if e['onset_s']>lo),end)
        post=audit.interval_events(events,lo,hi)
    preservation=base.read(folder/'rhythm_preservation.json') if (folder/'rhythm_preservation.json').exists() else {'status':'PENDING'}
    def features(part,rates=rr):
        ev=part['brief_events'] if part else []
        return dict(duration_ms=float(np.median([e['duration_s']*1000 for e in ev])) if ev else None,
                    interval_ms=float(np.median(np.diff([e['start_s'] for e in ev]))*1000) if len(ev)>1 else None,
                    peak_Hz=float(np.median([rates[round(e['start_s']/.01):round(e['end_s']/.01),0].max() for e in ev])) if ev else None)
    before,after=features(pre),features(post)
    ref=reference(row['job']['seed'])
    if ref:
        with np.load(ref) as data:
            with np.load(OUT/'geometry.npz') as geo:nr=np.r_[32000,geo['region_counts'][:3]]
            ref_rates=np.column_stack([data['spikes_1ms'][:8000,0],data['regions_1ms'][:8000,:3]]).reshape(800,10,4).sum(1)/nr/.01
    else:
        candidates=[j for j in prepare()['initial_jobs'] if j['seed']==row['job']['seed'] and j['stage']=='native_reference']
        ref_rates=rate_data(OUT/'runs'/candidates[0]['name']) if candidates else rr[:800]
    ref_part=audit.interval_events(strict_events(ref_rates,len(ref_rates)*.01),.5,min(8.,len(ref_rates)*.01))
    reference_features=features(ref_part,ref_rates)
    morphology=bool(post and audit.qualifies(post))
    ratios={}
    if morphology:
        for key in reference_features:
            ratio=after[key]/reference_features[key] if reference_features[key] and after[key] is not None else None
            ratios[key]=ratio
            morphology=bool(morphology and ratio is not None and .5<=ratio<=2.)
    row['native_rhythm']=dict(preservation=preservation,strict_pre=pre,strict_post=post,
        pre_features=before,post_features=after,reference_features=reference_features,
        post_to_native_reference_ratios=ratios,native_reference_window_s=[.5,min(8.,len(ref_rates)*.01)],
        native_like_return_screen=morphology,strict_pre_pass=audit.qualifies(pre))
    row['candidate_for_extension']=bool(preservation['status']=='PASS' and audit.qualifies(pre)
       and pp['low_activity_exits'] and row['Z_recovery']['core_recovery_screen'])
    row['full_sequence_screen']=bool(row['candidate_for_extension'] and morphology)
    row['classification']=('NATIVE_RHYTHM_RETURN_CANDIDATE' if row['full_sequence_screen'] else
                           'PRESERVED_RHYTHM_EXIT_NO_MATCHED_RETURN' if row['candidate_for_extension'] else
                           'PRESERVED_RHYTHM_'+pp['classification'] if preservation['status']=='PASS' else
                           'PRESERVATION_'+preservation['status'])
    # Replace observer's event list for figures with the stricter native accounting.
    pp['preentry']=pre;pp['preentry_brief_screen']=audit.qualifies(pre);pp['latest_postexit']=post;pp['events']=events
    strict_pairs=[]
    for old_pair in pp['interhigh_intervals']:
        part=audit.interval_events(events,*old_pair['window_s'])
        part.update({k:old_pair[k] for k in ['first_entry_s','second_entry_s','low_activity_confirmation_s']})
        part['temporal_pass']=audit.qualifies(part);strict_pairs.append(part)
    pp['interhigh_intervals']=strict_pairs;pp['temporal_loop_pass']=any(v['temporal_pass'] for v in strict_pairs)
    write(OUT/'analysis'/f'{name}.json',row)
    write(folder/'live_status.json',dict(time_s=end,classification=row['classification'],
        preservation=preservation['status'],preentry_brief=pre['brief_count'],
        postexit_brief=post['brief_count'] if post else None,entries=pp['entries'],exits=pp['low_activity_exits'],
        core_recovery_screen=row['Z_recovery']['core_recovery_screen'],updated_epoch=time.time()))
    return row


def worker(name):
    p=prepare();assert base.sha(__file__)==p['exploration_sha256']
    assert base.read(OUT/'mechanism_qa.json')['status']=='PASS'
    job=base.read(OUT/'jobs'/f'{name}.json');folder=OUT/'runs'/name
    previous.OUT=OUT;budget.OUT=OUT
    RhythmSlow.C_R=0.;RhythmSlow.feedback_form='conductance';RhythmSlow.sahp_gain=job['sahp_gain']
    RhythmSlow.sahp_tau_ms=job['sahp_tau_s']*1000.;RhythmSlow.record_regions=True
    RhythmSlow.z_gate_off=False;RhythmSlow.k_freeze=False
    old=(fixed.OUT,fixed.prepare,fixed.TerminationSlow,fixed.observation_sink)
    sink0=fixed.observation_sink
    def sink_factory(sink,j,deadline):
        full=sink0(sink,j,deadline)
        def observe(step,state):
            try:return full(step,state)
            finally:
                config=base.read(folder/'applied_configuration.json')
                config.update(native_local_GABA_unchanged=True,native_Z_function_preserved=True,
                    native_Z_input='Original local GABA plus (18-EG)*G500*q when the fixed high-rate gate is active',
                    native_GABA_and_Z_unchanged=job['G500']==0.,new_fixed_feedback_law=p['new_fixed_equation'])
                write(folder/'applied_configuration.json',config)
                flush(folder,step);prefix_check(name);analyze(name)
        return observe
    fixed.OUT=OUT;fixed.prepare=lambda:p;fixed.TerminationSlow=RhythmSlow;fixed.observation_sink=sink_factory
    try:fixed.worker(name)
    finally:fixed.OUT,fixed.prepare,fixed.TerminationSlow,fixed.observation_sink=old
    result=base.read(folder/'result.json');result['display_stop_s']=result['end_s']
    if result['status']!='CENSORED_WALL_DEADLINE':result['tracker']['stop_reason']='SIMULATION_HORIZON'
    result['new_fixed_feedback_law']=p['new_fixed_equation']
    write(folder/'result.json',result);write(folder/'progress.json',result)
    analyze(name)


# fixed.worker temporarily changes fixed.TerminationSlow; never resolve its step dynamically.
NativeTerminationStep=fixed.TerminationSlow.step


def select_extensions():
    p=prepare();eligible=[]
    for j in p['initial_jobs']:
        path=OUT/'analysis'/f"{j['name']}.json"
        if path.exists():
            row=base.read(path)
            if row['candidate_for_extension']:eligible.append(row)
    eligible.sort(key=lambda r:(r['full_sequence_screen'],r['primary']['temporal_loop_pass'],
                               r['native_rhythm']['strict_post']['brief_count'] if r['native_rhythm']['strict_post'] else 0,
                               r['Z_recovery']['maximum_shared_core_1s_rise']),reverse=True)
    selected=eligible[:1];p['adaptive_selection_done']=True;p['adaptive_selected']=[r['job']['name'] for r in selected]
    for row in selected:
        j0=row['job'];name=j0['name'];folder=OUT/'runs'/name
        archive=folder/'stage1';archive.mkdir(exist_ok=True)
        for file in ['result.json','rhythm_preservation.json']:shutil.copy2(folder/file,archive/file)
        shutil.copy2(OUT/'analysis'/f'{name}.json',archive/'analysis.json')
        j=copy.deepcopy(j0);j.update(horizon_s=120.,stage='continuation')
        with (folder/'checkpoint.pkl').open('rb') as h:saved=pickle.load(h)
        assert saved['job']==j0;saved['job']=j;base.save_pickle(folder/'checkpoint.pkl',saved)
        (folder/'result.json').unlink();write(OUT/'jobs'/f'{name}.json',j)
        p['initial_jobs']=[j if q['name']==name else q for q in p['initial_jobs']]
        for seed in [SEED+1,SEED+2]:
            ref=reference(seed)
            dependency=None
            if ref is None:
                control=make_job(0.,0.,1.,seed,len(p['initial_jobs']),stage='native_reference',horizon=8.)
                p['initial_jobs'].append(control);write(OUT/'jobs'/f"{control['name']}.json",control)
                dependency=control['name']
            confirm=make_job(j['G500'],j['K500'],j['sahp_tau_s'],seed,len(p['initial_jobs']),stage='independent_noise',horizon=120.)
            confirm['depends_on']=dependency;confirm['independent_noise_seed_of']=name
            p['initial_jobs'].append(confirm);write(OUT/'jobs'/f"{confirm['name']}.json",confirm)
    assert len(p['initial_jobs'])<=11
    write(OUT/'protocol.json',p)
    write(OUT/'selection.json',dict(selected=p['adaptive_selected'],reason='Native rhythm preserved first; then exit and native Z recovery. Maxone condition.',rows=selected))


def supervise():
    p=prepare();lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert base.read(OUT/'mechanism_qa.json')['status']=='PASS'
    if 'launched_epoch' not in p:
        p['launched_epoch']=time.time();p['deadline_epoch']=time.time()+8*3600;write(OUT/'protocol.json',p)
    running={};failures=[];logs=OUT/'logs';logs.mkdir(exist_ok=True)
    while True:
        p=prepare()
        for name,(proc,h) in list(running.items()):
            if proc.poll() is not None:
                h.close();del running[name]
                if proc.returncode:failures.append(dict(name=name,exit_code=proc.returncode))
        done=[j['name'] for j in p['initial_jobs'] if (OUT/'runs'/j['name']/'result.json').exists()]
        failed={f['name'] for f in failures}
        pending=[j for j in p['initial_jobs'] if j['name'] not in done and j['name'] not in running and j['name'] not in failed]
        can_start=time.time()<p['deadline_epoch']-900 and not failures
        dispatchable=[j for j in pending if not j.get('depends_on') or j['depends_on'] in done]
        while dispatchable and len(running)<p['max_workers'] and can_start:
            if psutil.virtual_memory().available/2**30<p['min_available_memory_GiB'] or shutil.disk_usage(OUT).free/2**30<p['disk_reserve_GiB']:break
            j=dispatchable.pop(0);pending.remove(j)
            h=(logs/f"{j['name']}.log").open('a')
            proc=subprocess.Popen([sys.executable,'-u',__file__,'worker','--name',j['name']],stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
            running[j['name']]=(proc,h);print('START',j['name'],proc.pid,flush=True);time.sleep(2)
        write(OUT/'status.json',dict(updated_epoch=time.time(),pid=os.getpid(),running={n:pr.pid for n,(pr,h) in running.items()},
            queued=[j['name'] for j in pending],finished=done,failures=failures,deadline_epoch=p['deadline_epoch']))
        if not running:
            if failures or not can_start:break
            if not pending:
                if not p.get('adaptive_selection_done'):select_extensions();continue
                break
        time.sleep(15)
    p=prepare();missing=[j['name'] for j in p['initial_jobs'] if not (OUT/'runs'/j['name']/'result.json').exists()]
    write(OUT/'batch_complete.json',dict(status='FAILED' if failures else 'FINISHED' if not missing else 'WALL_BUDGET_EXHAUSTED',
        finished_epoch=time.time(),failures=failures,missing_results=missing,human_review='PENDING'))
    with (logs/'report.log').open('a') as h:
        code=subprocess.call([sys.executable,str(Path(__file__).with_name('report_topic4_rhythm_preserving_feedback.py'))],stdout=h,stderr=subprocess.STDOUT)
    write(OUT/'report_status.json',dict(exit_code=code,time=time.time(),human_review='PENDING'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','qa','worker','supervise','analyze']);ap.add_argument('--name');a=ap.parse_args()
    try:
        if a.action in ['worker','analyze']:globals()[a.action](a.name)
        else:globals()[a.action]()
    except Exception as exc:
        dest=OUT/'runs'/a.name/'failure.json' if a.name else OUT/f'{a.action}_failure.json'
        write(dest,dict(error=repr(exc),time=time.time()));raise
