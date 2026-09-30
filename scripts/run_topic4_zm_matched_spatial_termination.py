#!/usr/bin/env python3
"""Matched local/global feedback on the fixed Fig5 Z/M-on native SNN."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import argparse,copy,time,subprocess,sys
from pathlib import Path
import numpy as np
import run_topic4_fixed_zm_termination as fixed

PARENT=fixed.OUT
OUT=Path(os.environ.get('TOPIC4_TERMINATION_OUT',str(PARENT/'matched_spatial_round2')))
GAP=7.  # reference threshold18 minus native shunt reversal11, mV

class MatchedSlow(fixed.TerminationSlow):
    C_R=None
    feedback_form=None
    sahp_gain=0.
    sahp_tau_ms=5000.
    sahp_reversal_mV=-30.
    dense_contact=False
    def __init__(self,*args,**kw):
        self.spatial_fraction=kw['gamma'];kw['gamma']=0.;kw['mode']='native'
        super().__init__(*args,**kw)
        assert self.global_resource=='native_z'
        assert self.feedback_form in ['current','conductance']
        assert self.cfg.use_z and self.cfg.use_m
        self.g_k=np.zeros(self.NE);self.k_records=[];self.dense_times=[];self.dense_values=[]
    def uses_shunt(self):return self.sahp_gain>0 or (self.feedback_form=='conductance' and self.spatial_fraction>0)
    def shunt_g_at_E(self):return self.g_global+self.g_k if self.sahp_gain else self.g_global
    def apply_currents(self,ie,ii,labels=None,rec=None):
        fraction=self.spatial_fraction
        local=ii.copy();local[:self.NE]*=(1-fraction)
        glob_equivalent=fraction*self.C_R*self.r_global
        total=local.copy();total[:self.NE]+=glob_equivalent
        self.delivered=total;self._I_I_last=total;self.raw_mean=float(ii[:self.NE].mean())
        global_conductance=self.feedback_form=='conductance'
        self.g_global[:]=self.global_gain*self.r_global*self.z[:self.NE] if global_conductance else 0.
        value=ie-self.z*(local if global_conductance else total)-self.cfg.eta_m*self.m
        if self.sahp_gain:
            # The engine has one shunt reversal. This numerator correction gives
            # exactly g_G E_G + g_K E_K with denominator 1+g_G+g_K.
            value[:self.NE]+=self.g_k*(self.sahp_reversal_mV-self.global_reversal)
        if self.voltage is None:
            global_current=self.z[:self.NE]*glob_equivalent
        elif global_conductance:
            global_current=self.g_global*(self.voltage[:self.NE]-self.global_reversal)
        else:global_current=self.z[:self.NE]*glob_equivalent
        if self._step_index%10==0:
            self.extra_records.append([self._step_index*.1,self.r_global,
                self.global_gain*self.r_global if self.uses_shunt() else 0.,self.g_global.mean(),
                self.phi.mean(),self.phi.max(),global_current.mean()])
            if self.sahp_gain:
                k_current=self.g_k*(self.voltage[:self.NE]-self.sahp_reversal_mV)
                self.k_records.append([self._step_index*.1,self.g_k.mean(),self.g_k.max(),k_current.mean()])
        if (self._step_index%20==0 or self.dense_contact) and self.current_recorder is not None:
            absolute=np.abs(ie[:self.NE])+np.abs(self.z[:self.NE]*local[:self.NE])+np.abs(global_current)
            contact=np.array([np.dot(w,absolute[ix]) for ix,w in zip(self.current_recorder._idx,self.current_recorder._w)])
            if self.dense_contact:
                self.dense_times.append(self._step_index*.1);self.dense_values.append(contact)
            if self._step_index%20==0:
                self.contact_records.append(contact)
                self.field_records.append(np.bincount(self.current_cells,weights=absolute,minlength=400)/self.current_cell_counts)
                self.current_record_times.append(self._step_index*.1)
        return value
    def step(self,spk,labels,dt):
        super().step(spk,labels,dt)
        if self.sahp_gain:
            self.g_k*=np.exp(-dt/self.sahp_tau_ms)
            # Source40/(tau_K5000 * fmax0.2/ms * gL4) =0.01 per spike.
            self.g_k[spk[:self.NE]]+=self.sahp_gain*.01

def prepare():
    if (OUT/'protocol.json').exists():return fixed.carrier.base.read(OUT/'protocol.json')
    import analyze_topic4_fixed_zm_termination as analysis
    parent=fixed.prepare();source=PARENT/'runs/native_s9108401'
    data=analysis.load(source,keys=['slow_time_ms','currents'])
    mech=analysis.load(source,'mechanism_chunks')
    times=data['slow_time_ms']/1000;sel=(times>=.5)&(times<8.)
    assert np.count_nonzero(sel)==375
    raw=data['currents'][sel,1];rate=np.interp(times[sel],mech['time_ms']/1000,mech['global_E_rate_Hz'])
    C_R=float(raw.mean()/rate.mean())
    p=copy.deepcopy(parent)
    jobs=[]
    for phi,form in [(0.,'current'),(0.,'conductance'),(2.5,'conductance')]:
        for fraction in [1/6,.5]:
            name=f'{form}_g{fraction:.6g}_phi{phi:g}_s9108401'
            j=dict(name=name,seed=9108401,eta_m=.0005,tau_M_s=1.,tau_Z_s=5.,threshold=95.19851312666987,
                mode='native',gamma=fraction,global_gain=fraction*C_R/GAP,global_resource='native_z',
                phi_jump=phi,horizon_s=30.,checkpoint_s=1.,device=len(jobs)%2,qa=False,
                feedback_form=form,C_R=C_R,reference_voltage_mV=18.,reference_gap_mV=GAP)
            jobs.append(j);fixed.carrier.base.write(OUT/'jobs'/f'{name}.json',j)
    p.update(initial_jobs=jobs,matched_producer_sha256=fixed.carrier.base.sha(__file__),wrapper_sha256=fixed.carrier.base.sha(fixed.__file__),
        status='DEFINED_BEFORE_ROUND2',reference_current_scale=C_R,source_round1=str(PARENT),
        stage_policy='Six30s single-noise diagnostic conditions, maximum4 workers across both rounds. Current/conductance comparison at gamma1/6 and1/2, with source fast threshold as a separate third row. No blind expansion.',
        question='At matched reference-voltage feedback, does conductance feedback change termination versus current feedback, and does source100ms threshold adaptation interact with it?',
        native_Z_policy='Same native first-order equation, now sensing the actual redistributed total resource-use equivalent J=(1-gamma)I_I+gamma*C_R*R_G; all local/global effects carry the same Z_i.',
        calibration=dict(source=str(source),window_s=[.5,8.],measurement_step_ms=20,raw_GABA_mean=float(raw.mean()),global_rate_mean_Hz=float(rate.mean()),C_R=C_R,
            rule='Ratio of paired baseline means; fixed before new outcomes. Exactly mean-preserving at the18mV reference voltage over this calibration window, not pointwise or at arbitrary voltages.',
            reference_voltage_mV=18.,global_reversal_mV=11.,difference_mV=GAP),
        no_new_M_or_Z_equation=True,protected_global_authorized=False,
        important_difference='Global inhibition is redistributed from local inhibition, not added on top. Current and conductance use the same total J to drive Z. The original graph and E/I cell dynamics remain; no source-model replacement.',
        parameter_panel='Rows: current withphi0, conductance withphi0, conductance withphi2.5. Columns: global fraction1/6,1/2. Categories are measured outcomes, not a continuous bifurcation surface.',
        actual_current_observable='|I_E|+|Z(1-gamma)I_I|+|I_global_actual|. Inherited currents column2 contains Z*J, the resource-current equivalent; add actualglobal minus referenceglobal for physical total inhibition.')
    fixed.carrier.base.write(OUT/'protocol.json',p)
    return p

def qa():
    p=prepare();MatchedSlow.C_R=p['reference_current_scale'];rng=np.random.default_rng(607)
    MatchedSlow.sahp_gain=0.
    cfg=fixed.carrier.base.old.MZSlowVarsConfig(use_z=True,use_m=True,tau_z=5000,I_th_EI=95.19851312666987,tau_adp=1000,eta_m=.0005)
    reversals=sorted(set([11.]+[j.get('global_reversal_mV',11.) for j in p['initial_jobs']]))
    checks=[(frac,reversal) for reversal in reversals for frac in [0.,1/6,.5]]
    for frac,reversal in checks:
        objects=[]
        for form in ['current','conductance']:
            MatchedSlow.feedback_form=form
            obj=MatchedSlow(10,18,cfg,NE=8,mode='native',gamma=frac,global_gain=frac*MatchedSlow.C_R/(18.-reversal),global_resource='native_z',phi_jump=0)
            obj.feedback_form=form;obj.global_reversal=reversal;obj.voltage=np.full(10,18.);obj.r_global=35.;objects.append(obj)
        z=rng.uniform(.2,1,8);m=rng.uniform(0,10,8);ie,ii=rng.uniform(0,400,(2,10))
        for obj in objects:obj.z[:8]=z;obj.m[:8]=m
        current,shunt=objects;f1=current.apply_currents(ie,ii);f2=shunt.apply_currents(ie,ii)
        actual2=f2.copy();actual2[:8]+=shunt.g_global*(reversal-18.)
        assert np.allclose(f1,actual2,rtol=0,atol=1e-12)
        assert np.array_equal(current._I_I_last,shunt._I_I_last)
        sp=rng.random(10)<.2;current.step(sp,None,.1);shunt.step(sp,None,.1)
        assert np.array_equal(current.z,shunt.z) and np.array_equal(current.m,shunt.m)
        assert np.all(current.z[8:]==1) and np.all(current.m[8:]==0)
        if frac==0:
            assert np.array_equal(f1,ie-np.r_[z,np.ones(2)]*ii-np.r_[m,np.zeros(2)]*.0005)
    for job in p['initial_jobs']:
        assert np.isclose(job['global_gain'],job['gamma']*job['C_R']/(18.-job.get('global_reversal_mV',11.)),rtol=1e-12,atol=1e-12)
    k_checks=[]
    for gain in sorted(set(j.get('sahp_gain',0.) for j in p['initial_jobs'])):
        if not gain:continue
        MatchedSlow.sahp_gain=gain;MatchedSlow.feedback_form='conductance'
        reversal=p['initial_jobs'][0].get('global_reversal_mV',11.)
        obj=MatchedSlow(10,18,cfg,NE=8,mode='native',gamma=.5,global_gain=.5*MatchedSlow.C_R/(18-reversal),global_resource='native_z',phi_jump=0.)
        obj.global_reversal=reversal;obj.voltage=rng.uniform(-20,18,10);obj.r_global=50.;obj.g_k[:]=rng.uniform(0,15,8)
        obj.z[:8]=rng.uniform(.2,1,8);obj.m[:8]=rng.uniform(0,20,8)
        ie,ii=rng.uniform(0,400,(2,10));net=obj.apply_currents(ie,ii)
        got=net[:8]+obj.shunt_g_at_E()*reversal
        want=ie[:8]-obj.z[:8]*.5*ii[:8]-.0005*obj.m[:8]+obj.g_global*reversal+obj.g_k*(-30.)
        assert np.allclose(got,want,rtol=0,atol=1e-12)
        before=obj.g_k.copy();sp=rng.random(10)<.2;obj.step(sp,None,.1)
        assert np.array_equal(obj.g_k,before*np.exp(-.1/5000.)+sp[:8]*(gain*.01))
        assert np.all(obj.z[8:]==1) and np.all(obj.m[8:]==0)
        k_checks.append(gain)
    MatchedSlow.sahp_gain=0.
    fixed.carrier.base.write(OUT/'equation_qa.json',dict(status='PASS',reversals_tested_mV=reversals,checks=checks,same_membrane_rhs_at_reference=True,same_resource_input_J=True,same_Z_M_update_given_same_spikes=True,zero_fraction_native_parity=True,no_I_cell_modification=True,all_job_gains_matched=True,sahp_gains_tested=k_checks,two_reversal_numerator_verified=True,sahp_source_update_verified=True))

def worker(name):
    p=prepare();assert p['matched_producer_sha256']==fixed.carrier.base.sha(__file__)
    assert fixed.carrier.base.read(OUT/'equation_qa.json')['status']=='PASS'
    job=fixed.carrier.base.read(OUT/'jobs'/f'{name}.json')
    MatchedSlow.C_R=job['C_R'];MatchedSlow.feedback_form=job['feedback_form']
    MatchedSlow.sahp_gain=job.get('sahp_gain',0.)
    MatchedSlow.sahp_tau_ms=1000.*float(job.get('sahp_tau_s',5.))
    assert MatchedSlow.sahp_tau_ms>0
    MatchedSlow.dense_contact=job.get('dense_contact',False)
    old=(fixed.OUT,fixed.prepare,fixed.TerminationSlow)
    fixed.OUT,fixed.prepare,fixed.TerminationSlow=OUT,lambda:p,MatchedSlow
    try:fixed.worker(name)
    finally:fixed.OUT,fixed.prepare,fixed.TerminationSlow=old

def supervise():
    import psutil
    p=prepare();qa();logs=OUT/'logs';logs.mkdir(exist_ok=True)
    pending=[j for j in p['initial_jobs'] if not (OUT/'runs'/j['name']/'result.json').exists()]
    running={};failed=[]
    def total_workers():
        n=0
        for proc in psutil.process_iter(['cmdline']):
            args=proc.info['cmdline'] or []
            if 'worker' in args and any(Path(a).name in ['run_topic4_fixed_zm_termination.py',Path(__file__).name] for a in args):n+=1
        return n
    while pending or running:
        for name,(proc,handle) in list(running.items()):
            code=proc.poll()
            if code is not None:
                handle.close();del running[name]
                if code:failed.append(dict(name=name,exit_code=code))
        if failed or time.time()>=p['deadline_epoch']-300:pending=[]
        while pending and total_workers()<p['max_workers'] and psutil.virtual_memory().available/2**30>80:
            j=pending.pop(0);handle=(logs/(j['name']+'.log')).open('a')
            proc=subprocess.Popen([sys.executable,'-u',__file__,'worker','--name',j['name']],stdout=handle,stderr=subprocess.STDOUT)
            running[j['name']]=(proc,handle)
        fixed.carrier.base.write(OUT/'supervisor_status.json',dict(time=time.time(),running={n:proc.pid for n,(proc,h) in running.items()},queued=[j['name'] for j in pending],failures=failed))
        if pending or running:time.sleep(10)
    fixed.carrier.base.write(OUT/'batch_complete.json',dict(time=time.time(),status='FAILED' if failed else 'FINISHED',failures=failed))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','qa','worker','supervise']);ap.add_argument('--name');args=ap.parse_args()
    if args.action=='worker':worker(args.name)
    else:globals()[args.action]()
