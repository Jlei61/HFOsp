#!/usr/bin/env python3
"""Match E-only rate dynamics to native inputs and prescribed/observed/endogenous Z."""
from topic4_e_only_z_rate import EOnlySystem,native_gaba_variance_factor
from validate_topic4_fixed_rate_base import ROOT,read,write
from scipy.special import ndtr
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import time
import os

OUT=ROOT/'results/topic4_sef_hfo/z_transition_bifurcation_audit_v1'
REFERENCE=ROOT/'results/topic4_sef_hfo/autonomous_z_manual_restore_v1'


def run(job):
    started=time.time();s=EOnlySystem();m=s.m;n=s.n;dt=s.dt;D=s.cfg['max_delay_steps']
    source=np.load(OUT/'external_input.npz');expected=source['expected_rate_per_ms'];count_e=source['count_e'];count_i=source['count_i']
    sampled=source['poisson_count'] if job['input']=='realized' else None
    assert np.array_equal(count_e,m.count_e) and np.array_equal(count_i,m.count_i)
    native=np.load(REFERENCE/'trajectory.npz');ce10=source['cell_e'];ce20=native['cell_e']
    mapping=np.bincount(ce10*400+ce20,minlength=n*400).reshape(n,400)/count_e[:,None]
    native_z=native['z_field_10ms']@mapping.T
    proto=read(REFERENCE/'protocol.json');tau_z=proto['tau_z_ms'];threshold=proto['I_th_EI']
    restore=read(REFERENCE/'run.json')['restore_start_ms'];z0_restore=None
    re=np.zeros(n);ri=np.zeros(n);he=np.zeros((D,n));hi=he.copy();g=np.zeros((6,n));c=g.copy();z=np.ones(n)
    rise=np.array([s.ra,s.rg,s.ra,s.rg,s.ra,s.ra])[:,None];decay=np.array([s.ta,s.tau,s.ta,s.tau,s.ta,s.ta])[:,None]
    te,ti=m.tau_mem_e_ms,m.tau_mem_i_ms;membrane=np.array([te,te,ti,ti,te,ti])[:,None]
    ar=np.exp(-dt/rise);ad=np.exp(-dt/decay);B=dt*membrane/rise;variance_factor=native_gaba_variance_factor(s)
    steps=round(job['duration_ms']/dt);stride=round(1/dt);frames=steps//stride
    fields=np.empty((frames,2,n),np.float32);zframes=np.empty((frames,n),np.float32);targets=np.empty_like(zframes)
    currents=np.empty((frames,6,n),np.float32)
    snapshot_times=job.get('phase_snapshots_ms',[])
    snapshot_steps={round(t/dt):t for t in snapshot_times};snapshots={};micro=[]
    for step in range(steps):
        tm=step*dt;nu_e=expected[step,0].astype(float);nu_i=expected[step,1].astype(float)
        if sampled is None:drive_e=nu_e;drive_i=nu_i
        else:
            drive_e=sampled[step,0]/count_e/dt;drive_i=sampled[step,1]/count_i/dt
        if job['Z']=='uniform_cycle':z[:]=np.interp(tm,[0,1000,2000,3500,4500,6500],[1,1,.5,.5,1,1])
        elif job['Z']=='native_replay':
            index=min(step//100,len(native_z)-1);alpha=(step%100)/100
            z=(1-alpha)*native_z[index]+alpha*native_z[min(index+1,len(native_z)-1)]
        elif job['Z'].startswith('autonomous') and tm>=restore:
            if z0_restore is None:z0_restore=z.copy()
            z=z0_restore+min(1.,(tm-restore)/1000)*(1-z0_restore)
        drive=np.array([s.ops['ee']@he.ravel(),s.ops['ei']@hi.ravel(),s.ops['ie']@he.ravel(),s.ops['ii']@hi.ravel(),m.j_ext_e_mv*drive_e,m.j_ext_i_mv*drive_i])
        g=ar*g+B*drive;c=g+(c-g)*ad
        ex=np.r_[te*(m.v_ee@re+m.j_ext_e_mv**2*nu_e),ti*(m.v_ie@re+m.j_ext_i_mv**2*nu_i)]
        raw_inh_e=m.v_ei@ri
        inh=np.r_[te*z*z*raw_inh_e,ti*(m.v_ii@ri)]
        mu=np.r_[c[0]-z*c[1]+c[4],c[2]-c[3]+c[5]]
        r=np.r_[re,ri];nr=r+dt/s.tr*(s.phi(mu,ex,inh)-r)
        if step in snapshot_steps:
            prefix=f't{snapshot_steps[step]:g}'
            for key,value in [('r',r),('next_r',nr),('current',c),('z',z),('expected_e',nu_e),('expected_i',nu_i)]:
                snapshots[prefix+'_'+key]=value.copy()
        if snapshot_times and any(abs(tm-t)<=50 for t in snapshot_times):
            micro.append(np.r_[tm,r])
        if job['Z']=='autonomous_gaussian':
            sd=np.sqrt(np.maximum(variance_factor*raw_inh_e,1e-12));target=ndtr((threshold-c[1])/sd)
        else:target=(c[1]<threshold).astype(float)
        if (step+1)%stride==0:
            k=step//stride;fields[k]=nr.reshape(2,n)*1000;zframes[k]=z;targets[k]=target;currents[k]=c
        if job['Z'].startswith('autonomous') and tm<restore:z+=dt/tau_z*(target-z)
        he[1:]=he[:-1].copy();hi[1:]=hi[:-1].copy();he[0]=re;hi[0]=ri;re,ri=nr[:n],nr[n:]
        if step%10000==0:write(OUT/'rate_progress'/f"{job['name']}.json",{'status':'RUNNING','time_ms':tm,'elapsed_s':time.time()-started})
    assert np.isfinite(fields).all() and np.isfinite(zframes).all()
    folder=OUT/'rate';folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/f"{job['name']}.npz",fields_hz=fields,z=zframes,z_target=targets,current=currents,
                        count_e=count_e,count_i=count_i,frame_ms=1.,final_gating=g,final_current=c,history_e=he,history_i=hi)
    if snapshots:
        np.savez_compressed(folder/f"{job['name']}_phase_slices.npz",**snapshots,microtrajectory=np.array(micro),times_ms=np.array(snapshot_times),dt_ms=dt)
    e=np.average(fields[:,0],axis=1,weights=count_e);i=np.average(fields[:,1],axis=1,weights=count_i)
    def stats(lo,hi):
        y=e[round(lo):round(hi)].reshape(-1,5).mean(1)
        return {'window_ms':[lo,hi],'E_mean_hz':float(y.mean()),'E_peak_5ms_hz':float(y.max()),'quiet_fraction':float(np.mean(y<1)),'cv':float(y.std()/max(y.mean(),1e-12))}
    row=dict(job,status='COMPLETE',seconds=time.time()-started,
             baseline=stats(500,1000),final=stats(job['duration_ms']-1000,job['duration_ms']),
             entire_peak_E_hz=float(e.max()),minimum_weighted_Z=float(np.min(np.average(zframes,axis=1,weights=count_e))),
             reduction='100 E/I cells, 8 E threshold quadrature groups, native delay bins, E-only postsynaptic mean and variance scaling',
             Z_averaging='Gaussian Pr(I_GABA<threshold) under independent-spike current variance' if job['Z']=='autonomous_gaussian' else job['Z'],
             noise_scope='Native expected OU-modulated cell input rates; neuronal Poisson represented by transfer variance' if sampled is None else 'Native projected realized external Poisson counts drive filters; transfer variance retained as an explicit finite-size approximation sensitivity')
    row['high_window']=stats(2000,3500) if job['Z']=='uniform_cycle' else stats(10180,10680)
    write(folder/f"{job['name']}.json",row);write(OUT/'rate_progress'/f"{job['name']}.json",row)
    return row


def main():
    jobs=[]
    for z,inputs in [('fixed',['expected']),('uniform_cycle',['expected','realized']),
                     ('native_replay',['expected','realized']),('autonomous_gaussian',['expected','realized']),
                     ('autonomous_mean_indicator',['expected'])]:
        for inp in inputs:jobs.append({'name':f'{z}_{inp}','Z':z,'input':inp,'duration_ms':6500. if z=='uniform_cycle' else 13680.})
    write(OUT/'rate_validation_protocol.json',{'jobs':jobs,'max_workers':2,
          'scope':'Capability gate before interpreting E-only reduced bifurcations; old global-gain results do not satisfy it',
          'Z_replay':'Neuron-count weighted native 20x20 Z field reduced to 10x10; linear interpolation between 10-ms records',
          'autonomous_Z':'Original tau_z and threshold; manual refill at native 10.68 s for same external protocol, not a fitted onset',
          'closure_caveats':['Averaging H(I<th) is not H(mean I<th); Gaussian and mean-indicator are explicit approximation contrasts',
                             'Within-cell Z-current and Z-threshold correlations are lost in cell-mean closure',
                             'Realized-drive replay is a finite-size sensitivity, not a new fitted stochastic mechanism']})
    while True:
        file=OUT/'external_input_status.json'
        status=read(file) if file.exists() else {}
        if status.get('status','').startswith('COMPLETE'):break
        if status.get('status')=='FAILED':raise RuntimeError(status)
        os.kill(read(OUT/'external_input_process.json')['pid'],0);time.sleep(10)
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run,j) for j in jobs]
        for f in as_completed(futures):
            try:rows.append(f.result())
            except Exception:
                for pending in futures:pending.cancel()
                raise
            write(OUT/'rate_batch_status.json',{'status':'RUNNING','completed':len(rows),'total':len(jobs),'rows':rows})
    write(OUT/'rate_batch_status.json',{'status':'COMPLETE','completed':len(rows),'total':len(jobs),'rows':rows})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(OUT/'rate_batch_status.json',{'status':'FAILED','error':repr(exc)})
        raise
