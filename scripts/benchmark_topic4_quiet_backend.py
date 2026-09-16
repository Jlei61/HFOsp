#!/usr/bin/env python3
"""Paired continuation audit, not a new scientific parameter condition."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import copy,json,time
import numpy as np
import run_topic4_fig5_log_m_scan as run

def main():
    source=run.OUT/'runs/eta1_tau100_s9108401/checkpoint.pkl'
    saved=run.base.load_pickle(source);job=saved['job'];state=saved['engine']
    s,tr,frozen,identity=run.base.old.setup(job['seed']);assert identity==saved['identity']
    first=state['step'];duration=2000;last=first+duration
    outputs={};timings={}
    for mode in ['cpu','gpu']:
        cfg=run.base.old.MZSlowVarsConfig(use_z=True,use_m=True,tau_z=5000.,I_th_EI=job['threshold'],tau_adp=job['tau_M_s']*1000,eta_m=job['eta_m'])
        slow=run.base.old.ReleaseZ(s.n_e+s.n_i,s.params.V_th,cfg,NE=s.n_e)
        s.net['rng']=np.random.default_rng(job['seed'])
        drive=run.base.old.make_external_drive(s,tr['spatial_ou'],job['seed'])
        captured={};spikes=[];times={}
        def observe(tm,spk):
            k=round(tm/.1)-first
            spikes.append(spk.copy())
            if k in [200,1800]:times[k]=time.perf_counter()
        def capture(k,engine):captured.update(engine)
        p=copy.deepcopy(s.params);p.T=(duration+1)*.1
        fn=run.base.old.simulate_kick if mode=='cpu' else run.wrap_simulator(run.base.old.simulate_kick,device_index=job['device'])
        tic=time.perf_counter()
        fn(p,s.net,KICK_BOOST=0.,slow=slow,V_th_per_neuron=s.vtheta,external_e_rate_drive=drive,
            early_stop_runaway=False,spike_observer=observe,record_dense_spikes=False,fast_scatter=True,
            resume_state=copy.deepcopy(state),time_offset_ms=first*.1,checkpoint_steps={last},checkpoint_sink=capture,verbose=False)
        outputs[mode]=(captured,np.asarray(spikes))
        timings[mode]=dict(total_wall_s=time.perf_counter()-tic,middle160ms_wall_s=times[1800]-times[200])
        print(mode,timings[mode],flush=True)
    a,sa=outputs['cpu'];b,sb=outputs['gpu'];assert np.array_equal(sa,sb)
    matched=[]
    for key in ['V','ref','s_E','I_E','s_I','I_I','ring_sE','ring_sI']:
        assert np.array_equal(a[key],b[key]),key;matched.append(key)
    for key in ['z','m']:assert np.array_equal(a['slow'][key],b['slow'][key]),key
    def equal(x,y,path='checkpoint'):
        assert type(x) is type(y),(path,type(x),type(y))
        if isinstance(x,np.ndarray):assert np.array_equal(x,y,equal_nan=True),path
        elif isinstance(x,dict):
            assert x.keys()==y.keys(),path
            for k in x:equal(x[k],y[k],path+'.'+str(k))
        elif isinstance(x,(list,tuple)):
            assert len(x)==len(y),path
            for i,(v,w) in enumerate(zip(x,y)):equal(v,w,path+'.'+str(i))
        else:assert x==y,path
    equal(a,b)
    out=run.ROOT/'results/topic4_sef_hfo/autonomous_recovery_exploration_20260914'
    run.base.write(out/'quiet_backend_benchmark.json',dict(status='PASS',source=str(source),source_step=first,
        duration_ms=200,full_spike_matrix_bitwise=True,fast_state_arrays_bitwise=matched,Z_M_RNG_bitwise=True,
        entire_checkpoint_recursive_bitwise=True,including_external_spatial_OU_and_delay_history=True,
        timings=timings,speed_ratio_gpu_over_cpu=timings['gpu']['middle160ms_wall_s']/timings['cpu']['middle160ms_wall_s'],
        meaning='One paired 200ms continuation at an actual quiet high-M checkpoint. Backend timing only, not new biological evidence; timing excludes first20ms GPU initialization in middle metric.'))

if __name__=='__main__':main()
