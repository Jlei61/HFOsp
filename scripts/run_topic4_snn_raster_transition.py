#!/usr/bin/env python3
"""Continuous SNN raster assay of prescribed inhibition loss and restoration."""
from validate_topic4_fixed_rate_base import ROOT, setup, write, read, make_external_drive, spatial_cell_index
from src.topic4_raster_protocol_engine import simulate_kick
from src.topic4_serial_spike_scatter import scatter
import numpy as np
from pathlib import Path
import time,resource,argparse,copy

OUT=ROOT/'results/topic4_sef_hfo/snn_raster_inhibition_transition_v1'


def gain(t):
    return float(np.interp(t,[0,1000,2000,3500,4500,6500],[1,1,.5,.5,1,1]))


class CurrentClamp:
    """Prescribed E-only Z clamp; no activity-dependent Z or M equation."""
    def __init__(self,ne,dt,all_targets=False):self.ne=ne;self.dt=dt;self.step_index=0;self.all_targets=all_targets
    def apply_currents(self,ie,ii,labels=None,rec=None):
        if self.all_targets:return ie-gain(self.step_index*self.dt)*ii
        result=ie-ii;result[:self.ne]=ie[:self.ne]-gain(self.step_index*self.dt)*ii[:self.ne];return result
    def threshold(self,v):return v
    def step(self,spk,labels,dt):self.step_index+=1


def qa():
    start=time.time();s,tr,fr,ident=setup(9108401);s.params.T=150.;s.params.sigma_n=0.;seed=9108401
    forced=np.zeros(s.n_e+len(s.positions_i),bool) if hasattr(s,'positions_i') else np.zeros(s.net['NE']+s.net['NI'],bool)
    forced[:s.n_e]=np.arange(s.n_e)%20==0
    kw=dict(KICK_BOOST=0.,V_th_per_neuron=s.vtheta,forced_spike_mask=forced,forced_spike_ms=50.,dump_i_spikes=True)
    # Original engine vs unchanged derived path vs JIT path, using identical external RNG.
    from src.topic4_rate_validation_engine import simulate_kick as original
    s.net['rng']=np.random.default_rng(seed);a=original(s.params,s.net,**kw)
    s.net['rng']=np.random.default_rng(seed);b=simulate_kick(s.params,s.net,fast_scatter=True,**kw)
    identity={key:bool(np.array_equal(a[key],b[key])) for key in ['E_spk_bool','I_spk_bool','rate_E','rate_I']};assert all(identity.values()),identity
    # Constant jump multiplier must agree with applying it to the frozen GABA graph.
    s.net['rng']=np.random.default_rng(seed);c=simulate_kick(s.params,s.net,fast_scatter=True,gaba_jump_scale_fn=lambda t:.6,**kw)
    for matrix in s.net['gaba_by_delay']:matrix.data*=.6
    s.net.pop('gaba_flat',None);s.net.pop('ampa_flat',None)
    s.net['rng']=np.random.default_rng(seed);d=original(s.params,s.net,**kw)
    scaled={key:bool(np.array_equal(c[key],d[key])) for key in identity};assert all(scaled.values()),scaled
    # Direct repeated-target scatter parity at multiple gain values.
    rng=np.random.default_rng(19);ptr=np.arange(0,501,10,dtype=np.int64);dst=rng.integers(0,30,500,dtype=np.int64);delay=rng.integers(1,10,500,dtype=np.int32);w=rng.random(500);src=np.array([0,1,8,18,42],np.int64)
    for g in [1.,.5,.731]:
        aa=np.zeros((11,30));bb=aa.copy();idx=np.concatenate([np.arange(ptr[k],ptr[k+1]) for k in src]);np.add.at(aa,((7+delay[idx])%11,dst[idx]),w[idx]*g);scatter(bb,src,ptr,dst,delay,w,7,g);assert np.array_equal(aa,bb)
    write(OUT/'engine_qa.json',{'status':'PASS','original_vs_fast_spikes':identity,'constant_gain_vs_scaled_native_graph':scaled,'repeated_target_scatter':'bitwise equal at 3 gains','forced_E_spikes':int(forced.sum()),'seconds':time.time()-start})
    print('QA PASS',time.time()-start,flush=True)


def run(arm,seed=9108401):
    assert read(OUT/'engine_qa.json')['status']=='PASS'
    start=time.time();s,tr,fr,ident=setup(seed);p=s.params;p.T=6500.;ne,ni=s.net['NE'],s.net['NI'];dt=p.dt;nsteps=round(p.T/dt);frames=round(p.T)
    protocol_gain=(lambda tm:1.) if arm=='base' else gain
    ou=arm in ('jump_ou','z_current_e_ou');native_sigma=p.sigma_n
    if not ou:p.sigma_n=0.
    drive=make_external_drive(s,tr['spatial_ou'],seed) if ou else None
    cells=spatial_cell_index(s.positions_e,n_grid=20,sheet_l_mm=p.L);counts=np.bincount(cells,minlength=400)
    centers=np.array([[3.0639104904,18.7198569555],[12.8235273182,15.0909054692]])
    dist=np.linalg.norm(s.positions_e[:,None]-centers[None],axis=2);groups=np.full(ne,2);groups[dist[:,0]<1.75]=0;groups[(dist[:,1]<1.75)&(dist[:,1]<dist[:,0])]=1
    rng=np.random.default_rng(682);sample=[];labels=[]
    axis=np.array([np.cos(np.deg2rad(-31.84)),np.sin(np.deg2rad(-31.84))])
    for group,number in [(0,60),(1,60),(2,120)]:
        ids=rng.choice(np.flatnonzero(groups==group),number,replace=False);ids=ids[np.argsort(s.positions_e[ids]@axis)];sample.extend(ids.tolist());labels.extend([group]*number)
    sample.extend((ne+rng.choice(ni,60,replace=False)).tolist());labels.extend([3]*60);sample=np.array(sample)
    raster=np.empty((nsteps,len(sample)),bool);field=np.zeros((frames,400),np.uint16);curr=np.zeros((frames,4));regions=np.zeros((frames,3),np.uint32)
    last_progress=[-1]
    def observe_spike(tm,spk):
        step=round(tm/dt);frame=step//round(1/dt);raster[step]=spk[sample];ids=np.flatnonzero(spk[:ne]);field[frame]+=np.bincount(cells[ids],minlength=400).astype(np.uint16);regions[frame]+=np.bincount(groups[ids],minlength=3).astype(np.uint32)
        if step%1000==0:
            write(OUT/'progress'/f'{arm}.json',{'status':'RUNNING','time_ms':tm,'duration_ms':p.T,'q':protocol_gain(tm),'elapsed_s':time.time()-start})
    def observe_current(tm,ie,ii,v):
        step=round(tm/dt)
        if step%round(1/dt)==0:curr[step//round(1/dt)]=[ie[:ne].mean(),ii[:ne].mean(),ie[ne:].mean(),ii[ne:].mean()]
    s.net['rng']=np.random.default_rng(seed)
    current_arm=arm.startswith('z_current')
    result=simulate_kick(p,s.net,KICK_BOOST=0.,V_th_per_neuron=s.vtheta,slow=CurrentClamp(ne,dt,arm=='z_current_all') if current_arm else None,
        external_e_rate_drive=drive,early_stop_runaway=False,gaba_jump_scale_fn=None if current_arm or arm=='base' else gain,
        spike_observer=observe_spike,current_observer=observe_current,record_dense_spikes=False,fast_scatter=True,verbose=True)
    field_rate=field/counts[None,:]*1000;re=result['rate_E'];ri=result['rate_I'];e_counts=re.reshape(frames,-1).sum(1)*ne*dt/1000
    assert np.allclose(field.sum(1),e_counts,atol=1e-7);assert np.array_equal(field.sum(1),regions.sum(1))
    folder=OUT/'runs';folder.mkdir(exist_ok=True);name=f'{arm}_seed{seed}'
    np.savez_compressed(folder/f'{name}.npz',sample_spikes=raster,sample_ids=sample,sample_groups=np.array(labels),sample_positions_e=s.positions_e[sample[:240]],
        rate_e_hz=re,rate_i_hz=ri,field_e_count_1ms=field,cell_e_counts=counts,region_spikes_1ms=regions,region_counts=np.bincount(groups,minlength=3),
        currents_1ms=curr,q_1ms=np.array([protocol_gain(i) for i in range(frames)]),dt_ms=dt,positions_e=s.positions_e,cell_e=cells)
    row={'status':'COMPLETE','arm':arm,'seed':seed,'duration_ms':p.T,'dt_ms':dt,'frozen_identity':ident,'threshold_field':'unchanged frozen dual core','tau_gaba_ms':p.tau_d_GABA,
        'input':{'global_ou':bool(ou),'sigma_n':p.sigma_n,'native_sigma_n':native_sigma,'spatial_ou':tr['spatial_ou'] if ou else 'off','neuron_poisson':'retained'},
        'control':'No parameter intervention; q=1 throughout' if arm=='base' else ('Presynaptic all-GABA jump multiplier at emission' if not current_arm else ('Prescribed postsynaptic all-target Z multiplying accumulated GABA current' if arm=='z_current_all' else 'Prescribed postsynaptic E-only Z multiplying accumulated GABA current; I cells unchanged')),
        'no_state_reset':True,'Z_M_dynamics':False,'raster_sampling':'Fixed preselected 60 core-A E, 60 core-B E, 120 surround E, 60 I; stratified, not population-proportional. Rates use all 40000 neurons.',
        'spatial_counts_match_global':True,'seconds':time.time()-start,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2}
    write(folder/f'{name}.json',row);write(OUT/'progress'/f'{arm}.json',{'status':'COMPLETE','time_ms':p.T,'duration_ms':p.T,'elapsed_s':row['seconds']});print(name,row['seconds'],flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--qa',action='store_true');a.add_argument('--arm',choices=['jump','jump_ou','z_current_e','z_current_e_ou','z_current_all','base'],default='jump');a.add_argument('--seed',type=int,default=9108401);args=a.parse_args()
    if args.qa:qa()
    else:run(args.arm,args.seed)
