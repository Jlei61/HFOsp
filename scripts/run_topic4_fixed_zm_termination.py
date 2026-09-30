#!/usr/bin/env python3
"""Termination mechanism on the actual fixed Fig5 Z/M SNN, never LAS replacement.

Native graph, inputs, threshold substrate and Z/M equations are unchanged.
Global conductance is an explicit new pathway; its resource policy must be
approved in protocol.json before a protected-pathway simulation is dispatched.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import argparse,copy,json,time,subprocess,sys
from pathlib import Path
import numpy as np
import run_topic4_autonomous_recovery as carrier
import checkpoint
from run_topic4_nativeZ_continuation import checkpoint_policy

ROOT=carrier.ROOT
OUT=ROOT/'results/topic4_sef_hfo/fig5_fixed_zm_termination_20260915'
REF=ROOT/'results/topic4_sef_hfo/fig5_preentry_event_audit_20260914'

def observation_sink(sink,job,deadline):
    """Keep the carrier's recurrence endpoint unless full duration is explicit."""
    if job.get('qa') or job.get('stop_after_second_entry',True):return sink
    return checkpoint_policy(sink,round(job['horizon_s']*10000),deadline,[])

class TerminationSlow(carrier.RecoverySlow):
    def __init__(self,*args,global_gain=0.,global_resource='native_z',phi_jump=0.,**kwargs):
        super().__init__(*args,**kwargs)
        self.nE=self.NE;self.global_gain=float(global_gain);self.global_resource=global_resource
        self.phi_jump=float(phi_jump);self.phi=np.zeros(self.NE);self.r_global=0.
        self.extra_records=[];self.g_global=np.zeros(self.NE);self.voltage=None
        self.field_records=[];self.contact_records=[];self.current_record_times=[];self.current_recorder=None
        assert global_resource in ['native_z','protected']
    def uses_shunt(self):return self.global_gain>0
    def shunt_g_at_E(self):return self.g_global
    def threshold(self,base):
        if self.phi_jump==0:return base
        value=np.broadcast_to(np.asarray(base,dtype=float),(self.N,)).copy()
        value[:self.NE]+=self.phi
        return value
    def apply_currents(self,ie,ii,labels=None,rec=None):
        value=super().apply_currents(ie,ii,labels,rec)
        raw=self.global_gain*self.r_global
        self.g_global[:]=raw*(self.z[:self.NE] if self.global_resource=='native_z' else 1.)
        # Z senses its existing local GABA pathway; the explicit global conductance
        # has the selected multiplicative resource policy, no hidden Z source term.
        if self._step_index%10==0:
            current=(self.g_global*(self.voltage[:self.NE]-self.global_reversal)).mean() if self.voltage is not None else 0.
            self.extra_records.append([self._step_index*.1,self.r_global,raw,self.g_global.mean(),
                self.phi.mean(),self.phi.max(),current])
        if self._step_index%20==0 and self.current_recorder is not None:
            local=self.z[:self.NE]*ii[:self.NE]
            glob=self.g_global*(self.voltage[:self.NE]-self.global_reversal)
            absolute_current=np.abs(ie[:self.NE])+np.abs(local)+np.abs(glob)
            # Observe actual synaptic currents; M is intrinsic adaptation, not a
            # synaptic current. Keep the inherited raw pre-Z proxy separately.
            contacts=np.array([np.dot(w,absolute_current[ix]) for ix,w in zip(self.current_recorder._idx,self.current_recorder._w)])
            field=np.bincount(self.current_cells,weights=absolute_current,minlength=400)/self.current_cell_counts
            self.current_record_times.append(self._step_index*.1)
            self.contact_records.append(contacts);self.field_records.append(field)
        return value
    def step(self,spk,labels,dt):
        super().step(spk,labels,dt)
        self.r_global=self.r_global*np.exp(-dt/15.)+np.count_nonzero(spk[:self.NE])/self.NE*1000/15.
        if self.phi_jump:
            self.phi*=np.exp(-dt/100.);self.phi[spk[:self.NE]]+=self.phi_jump

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():return carrier.base.read(OUT/'protocol.json')
    p=copy.deepcopy(carrier.base.read(carrier.OUT/'protocol.json'))
    ref=carrier.base.read(REF/'protocol.json');r=carrier.base.read(REF/'runs/eta0.0005_s9108401/result.json')
    assert p['identity']==r['identity']
    j=dict(name='baseline_qa',seed=9108401,eta_m=.0005,tau_M_s=1.,tau_Z_s=5.,threshold=95.19851312666987,
        mode='native',gamma=0.,global_gain=0.,global_resource='native_z',phi_jump=0.,horizon_s=1.,checkpoint_s=.5,device=0,qa=True)
    p.update(identity=r['identity'],reference=str(REF),reference_job=r['job'],source_hashes=ref['source_hashes'],
        producer_sha256=carrier.base.sha(carrier.__file__),wrapper_sha256=carrier.base.sha(__file__),
        baseline=dict(eta_M=.0005,tau_M_s=1.,tau_Z_s=5.,Ith=95.19851312666987,Z_on=True,M_on=True),
        initialized_epoch=time.time(),deadline_epoch=time.time()+8*3600,
        protected_global_authorized=False,authorization_status='Awaiting answer only for the new protected pathway; native baseline checks proceed',
        initial_jobs=[j],max_workers=4,min_available_memory_GiB=80.,
        no_10s_pool=True,no_added_Z_recovery=True,no_external_reset=True,
        source_mechanisms=dict(global_synapse_ms=15.,fast_threshold_ms=100.,source_spike_threshold_jump_mV=2.5),
        exact_scope='Current fixed manual-core SNN; optional new global shunt and fast spike adaptation. Not a replacement with the LAS model.',
        native_Z_policy='Unchanged local raw GABA drives native Z. If global_resource=native_z, global conductance is also multiplied by this Z; protected is a separately declared pathway hypothesis.',
        figure_acceptance='Actual Fig5 style: continuous E/I raster and two-core zooms; aligned Z/M; native spatial states through autonomous return; trajectory and matched mechanism controls. No recycled parameter heatmap from another equation.')
    carrier.base.write(OUT/'protocol.json',p);carrier.base.write(OUT/'jobs/baseline_qa.json',j)
    return p

def worker(name):
    p=prepare();assert p['wrapper_sha256']==carrier.base.sha(__file__)
    job=carrier.base.read(OUT/'jobs'/f'{name}.json')
    if job['global_resource']=='protected' and job['global_gain']>0:assert p['protected_global_authorized']
    folder=OUT/'runs'/name;box={};capture0,restore0,wrap0=checkpoint.capture,checkpoint.restore_slow,carrier.wrap_simulator
    def factory(*a,**kw):
        obj=TerminationSlow(*a,**kw,global_gain=job['global_gain'],global_resource=job['global_resource'],phi_jump=job['phi_jump']);box['slow']=obj;return obj
    def capture(**kw):
        state=capture0(**kw);o=kw['slow'];state['termination_mechanism']=dict(r_global=o.r_global,phi=o.phi.copy())
        if hasattr(o,'g_k'):
            state['termination_mechanism']['sahp_g']=o.g_k.copy()
            if o.dense_times:
                dest=folder/'dense_contact_chunks';dest.mkdir(exist_ok=True)
                path=dest/f'{round(o.dense_times[0]*10):010d}_{int(kw["step"]):010d}.npz'
                np.savez_compressed(path,time_ms=o.dense_times,contact_current=o.dense_values)
                o.dense_times.clear();o.dense_values.clear()
            if o.k_records:
                rec_k=np.asarray(o.k_records);dest=folder/'intrinsic_adaptation_chunks';dest.mkdir(exist_ok=True)
                path=dest/f'{round(rec_k[0,0]*10):010d}_{int(kw["step"]):010d}.npz'
                np.savez_compressed(path,time_ms=rec_k[:,0],sahp_mean_conductance_ratio=rec_k[:,1],sahp_max_conductance_ratio=rec_k[:,2],sahp_outward_current_mV_equiv=rec_k[:,3])
                o.k_records.clear()
        rec=np.asarray(o.extra_records)
        if len(rec):
            dest=folder/'mechanism_chunks';dest.mkdir(exist_ok=True);path=dest/f'{round(rec[0,0]*10):010d}_{int(kw["step"]):010d}.npz'
            keys=['time_ms','global_E_rate_Hz','global_raw_conductance_ratio','global_applied_conductance_ratio','phi_mean_mV','phi_max_mV','global_outward_current_mV_equiv']
            np.savez_compressed(path,**{key:rec[:,i] for i,key in enumerate(keys)});o.extra_records.clear()
        if o.current_record_times:
            dest=folder/'actual_current_chunks';dest.mkdir(exist_ok=True)
            path=dest/f'{round(o.current_record_times[0]*10):010d}_{int(kw["step"]):010d}.npz'
            np.savez_compressed(path,time_ms=o.current_record_times,contact_current=o.contact_records,field_current=o.field_records)
            o.current_record_times.clear();o.contact_records.clear();o.field_records.clear()
        return state
    def restore(state,obj):
        restore0(state,obj);extra=state['termination_mechanism'];obj.r_global=float(extra['r_global']);obj.phi[:]=extra['phi']
        if hasattr(obj,'g_k'):
            if obj.sahp_gain:assert 'sahp_g' in extra
            obj.g_k[:]=extra.get('sahp_g',0.)
    def wrap(fn,device_index):
        gpu=wrap0(fn,device_index=device_index)
        def invoke(params,net,*a,**kw):
            o=box['slow'];o.global_reversal=float(job.get('global_reversal_mV',params.E_gaba))
            kw['e_gaba']=o.global_reversal
            with np.load(OUT/'geometry.npz') as geo:
                o.current_recorder=carrier.base.old.LFPRecorder(params,net['pos'],net['labels'],sites=geo['contact_xy'])
                o.current_cells=carrier.base.old.spatial_cell_index(geo['positions_e'],n_grid=20,sheet_l_mm=params.L)
                o.current_cell_counts=geo['cell_e_counts'].copy()
            assert np.all(o.current_cell_counts>0)
            def current(tm,ie,ii,v):o.voltage=v
            kw['current_observer']=current
            kw['checkpoint_sink']=observation_sink(kw['checkpoint_sink'],job,p['deadline_epoch'])
            carrier.base.write(folder/'applied_configuration.json',dict(job=job,identity=p['identity'],
                native_Z_on=o.cfg.use_z,native_M_on=o.cfg.use_m,eta_M=o.cfg.eta_m,tau_M_ms=o.cfg.tau_adp,
                global_reversal_mV=o.global_reversal,membrane_time_constants={key:float(value) for key,value in vars(params).items() if 'tau' in key and isinstance(value,(int,float))},
                no_reset=True,global_path_resource=job['global_resource'],
                native_GABA_and_Z_unchanged=('feedback_form' not in job or job['gamma']==0),
                native_Z_function_preserved=True,
                native_Z_input='Redistributed J' if 'feedback_form' in job and job['gamma'] else 'Original local GABA',
                added_sahp_gain=job.get('sahp_gain',0.),added_sahp_tau_ms=float(o.sahp_tau_ms) if job.get('sahp_gain',0.) else None))
            return gpu(params,net,*a,**kw)
        return invoke
    old=(carrier.OUT,carrier.prepare,carrier.RecoverySlow,carrier.DEADLINE)
    carrier.OUT,carrier.prepare,carrier.RecoverySlow,carrier.DEADLINE=OUT,lambda:p,factory,p['deadline_epoch']
    checkpoint.capture,checkpoint.restore_slow,carrier.wrap_simulator=capture,restore,wrap
    try:carrier.worker(name)
    finally:
        carrier.OUT,carrier.prepare,carrier.RecoverySlow,carrier.DEADLINE=old
        checkpoint.capture,checkpoint.restore_slow,carrier.wrap_simulator=capture0,restore0,wrap0

def supervise():
    import psutil
    p=prepare()
    assert carrier.base.read(OUT/'baseline_qa.json')['status']=='PASS'
    assert carrier.base.read(OUT/'equation_qa.json')['status']=='PASS'
    pending=[j for j in p['initial_jobs'] if not (OUT/'runs'/j['name']/'result.json').exists()]
    running={};failed=[];logs=OUT/'logs';logs.mkdir(exist_ok=True)
    while pending or running:
        for name,(proc,handle) in list(running.items()):
            code=proc.poll()
            if code is not None:
                handle.close();del running[name]
                if code:failed.append(dict(name=name,exit_code=code))
        if failed or time.time()>=p['deadline_epoch']-300:pending=[]
        while pending and len(running)<p['max_workers'] and psutil.virtual_memory().available/2**30>p['min_available_memory_GiB']:
            j=pending.pop(0);handle=(logs/(j['name']+'.log')).open('a')
            proc=subprocess.Popen([sys.executable,'-u',__file__,'worker','--name',j['name']],stdout=handle,stderr=subprocess.STDOUT)
            running[j['name']]=(proc,handle)
        carrier.base.write(OUT/'supervisor_status.json',dict(time=time.time(),running={n:proc.pid for n,(proc,h) in running.items()},queued=[j['name'] for j in pending],failures=failed))
        if pending or running:time.sleep(10)
    carrier.base.write(OUT/'batch_complete.json',dict(time=time.time(),status='FAILED' if failed else 'FINISHED',failures=failed))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','worker','supervise']);ap.add_argument('--name');args=ap.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='worker':worker(args.name)
    else:supervise()
