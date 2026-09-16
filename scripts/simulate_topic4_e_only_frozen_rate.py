#!/usr/bin/env python3
"""Noise-off delayed spatial-rate trajectories at fixed observed Z profiles."""
from analyze_topic4_e_only_bifurcation import Family, OUT, read, write
import numpy as np
import time


def run(job):
    f=Family('native_path');s=f.s;m=s.m;n=s.n;D=s.cfg['max_delay_steps'];dt=s.dt
    q=job['q'];z=f.field(q);duration=job['duration_ms'];start=time.time()
    if job['initial']=='zero':r=np.zeros(200);g=np.zeros((6,n));c=g.copy()
    else:
        rr,err,ok=f.solve(q,np.full(200,.45 if job['initial']=='high' else .00005));assert ok,(q,err)
        r=rr.copy()
        # All filters and histories match the equilibrium before a 1% E-rate perturbation.
        e,i=r[:n],r[n:];nu=m.nu_ext_per_ms
        drive=np.array([m.w_ee@e,m.w_ei@i,m.w_ie@e,m.w_ii@i,np.full(n,m.j_ext_e_mv*nu),np.full(n,m.j_ext_i_mv*nu)])
        rises=np.array([s.ra,s.rg,s.ra,s.rg,s.ra,s.ra])[:,None]
        mem=np.array([m.tau_mem_e_ms]*2+[m.tau_mem_i_ms]*2+[m.tau_mem_e_ms,m.tau_mem_i_ms])[:,None]
        g=dt*mem/rises*drive/(1-np.exp(-dt/rises));c=g.copy()
    re,ri=r[:n].copy(),r[n:].copy();he=np.broadcast_to(re,(D,n)).copy();hi=np.broadcast_to(ri,(D,n)).copy()
    if job['initial']!='zero':re*=1.01
    rise=np.array([s.ra,s.rg,s.ra,s.rg,s.ra,s.ra])[:,None];decay=np.array([s.ta,s.tau,s.ta,s.tau,s.ta,s.ta])[:,None]
    te,ti=m.tau_mem_e_ms,m.tau_mem_i_ms;mem=np.array([te,te,ti,ti,te,ti])[:,None]
    ar=np.exp(-dt/rise);ad=np.exp(-dt/decay);B=dt*mem/rise;nu=m.nu_ext_per_ms
    steps=round(duration/dt);fields=np.empty((round(duration),2,n),np.float32)
    for step in range(steps):
        drive=np.array([s.ops['ee']@he.ravel(),s.ops['ei']@hi.ravel(),s.ops['ie']@he.ravel(),s.ops['ii']@hi.ravel(),np.full(n,m.j_ext_e_mv*nu),np.full(n,m.j_ext_i_mv*nu)])
        g=ar*g+B*drive;c=g+(c-g)*ad
        ex=np.r_[te*(m.v_ee@re+m.j_ext_e_mv**2*nu),ti*(m.v_ie@re+m.j_ext_i_mv**2*nu)]
        inh=np.r_[te*z*z*(m.v_ei@ri),ti*(m.v_ii@ri)]
        mu=np.r_[c[0]-z*c[1]+c[4],c[2]-c[3]+c[5]]
        r=np.r_[re,ri];nr=r+dt/s.tr*(s.phi(mu,ex,inh)-r)
        if (step+1)%10==0:fields[step//10]=nr.reshape(2,n)*1000
        he[1:]=he[:-1].copy();hi[1:]=hi[:-1].copy();he[0]=re;hi[0]=ri;re,ri=nr[:n],nr[n:]
        if step%10000==0:write(OUT/'fixed_rate_progress'/f"{job['name']}.json",{'status':'RUNNING','time_ms':step*dt})
    folder=OUT/'fixed_rate';folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/f"{job['name']}.npz",fields_hz=fields,z=z,count_e=m.count_e,count_i=m.count_i,frame_ms=1.)
    e=np.average(fields[:,0],axis=1,weights=m.count_e);late=e[len(e)//2:]
    row=dict(job,status='COMPLETE',mean_E_hz=float(late.mean()),min_E_hz=float(late.min()),max_E_hz=float(late.max()),std_E_hz=float(late.std()),seconds=time.time()-start,
             scope='Deterministic rate model, OU deviations removed, private-input variance retained in transfer function; not native SNN.')
    write(folder/f"{job['name']}.json",row);return row


def main():
    while read(OUT/'phase_replay_status.json')['status']=='RUNNING':time.sleep(10)
    jobs=[{'q':q,'initial':init,'duration_ms':duration,'name':f'q{q:g}_{init}'}
          for q,init,duration in [(.9,'low',3000.),(.867,'low',8000.),(.85,'zero',3000.),
                                 (.7774485019935391,'zero',3000.),(.6644786387930526,'high',3000.)]]
    rows=[];write(OUT/'fixed_rate_status.json',{'status':'RUNNING','completed':0,'total':len(jobs)})
    for job in jobs:
        rows.append(run(job));write(OUT/'fixed_rate_status.json',{'status':'RUNNING','completed':len(rows),'total':len(jobs),'rows':rows})
    write(OUT/'fixed_rate_status.json',{'status':'COMPLETE','completed':len(rows),'rows':rows})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(OUT/'fixed_rate_status.json',{'status':'FAILED','error':repr(exc)});raise
