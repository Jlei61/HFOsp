#!/usr/bin/env python3
"""Same frozen-Z/history protocol in the existing delayed E-only rate closure."""
from topic4_spatial_boundary_common import OUT, OLD, REFERENCE, read, write, checkpoint_path
from topic4_e_only_z_rate import EOnlySystem
from checkpoint import load
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import numpy as np
import time
import argparse


class Stepper:
    def __init__(self, grid=10, system=None):
        self.s = s = EOnlySystem(grid=grid) if system is None else system
        self.m = m = s.m; self.n = n = s.n
        self.r = np.zeros(2*n); self.g = np.zeros((6,n)); self.c = self.g.copy()
        self.he = np.zeros((s.cfg['max_delay_steps'],n)); self.hi = self.he.copy()
        rise = np.array([s.ra,s.rg,s.ra,s.rg,s.ra,s.ra])[:,None]
        decay = np.array([s.ta,s.tau,s.ta,s.tau,s.ta,s.ta])[:,None]
        te,ti = m.tau_mem_e_ms,m.tau_mem_i_ms
        self.ar = np.exp(-s.dt/rise); self.ad = np.exp(-s.dt/decay)
        self.B = s.dt*np.array([te,te,ti,ti,te,ti])[:,None]/rise

    def step(self,z,nu_e,nu_i):
        s,m,n = self.s,self.m,self.n; re,ri = self.r[:n],self.r[n:]
        te,ti = m.tau_mem_e_ms,m.tau_mem_i_ms
        drive = np.array([s.ops['ee']@self.he.ravel(),s.ops['ei']@self.hi.ravel(),
            s.ops['ie']@self.he.ravel(),s.ops['ii']@self.hi.ravel(),m.j_ext_e_mv*nu_e,m.j_ext_i_mv*nu_i])
        self.g = self.ar*self.g+self.B*drive; self.c = self.g+(self.c-self.g)*self.ad
        ex = np.r_[te*(m.v_ee@re+m.j_ext_e_mv**2*nu_e),ti*(m.v_ie@re+m.j_ext_i_mv**2*nu_i)]
        inh = np.r_[te*z*z*(m.v_ei@ri),ti*(m.v_ii@ri)]
        c = self.c; mu = np.r_[c[0]-z*c[1]+c[4],c[2]-c[3]+c[5]]
        nr = self.r+s.dt/s.tr*(s.phi(mu,ex,inh)-self.r)
        self.he[1:] = self.he[:-1].copy(); self.hi[1:] = self.hi[:-1].copy()
        self.he[0] = re; self.hi[0] = ri; self.r = nr
        return nr

    def save(self,path,**extra):
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,r=self.r,g=self.g,c=self.c,he=self.he,hi=self.hi,**extra)

    def restore(self,path):
        a = np.load(path)
        for key in ('r','g','c','he','hi'): setattr(self,key,a[key].copy())


def replay():
    start=time.time(); st=Stepper(); s=st.s; m=st.m; n=st.n
    source=np.load(OLD/'external_input.npz'); expected=source['expected_rate_per_ms']; cells=source['cell_e']
    native=np.load(REFERENCE/'trajectory.npz'); cell20=native['cell_e']
    mapping=np.bincount(cells*400+cell20,minlength=n*400).reshape(n,400)/m.count_e[:,None]
    z_native=native['z_field_10ms']@mapping.T
    previous=np.load(OLD/'rate/native_replay_expected.npz')['fields_hz']
    fields=np.empty((9400,2,n),np.float32); errors=[]
    for step in range(94000):
        if step in (80000,94000): st.save(OUT/'rate_checkpoints'/f't{step//10}ms.npz',absolute_time_ms=step*s.dt)
        index=step//100; alpha=(step%100)/100
        z=(1-alpha)*z_native[index]+alpha*z_native[index+1]
        r=st.step(z,expected[step,0].astype(float),expected[step,1].astype(float))
        if (step+1)%10==0:
            k=step//10;fields[k]=r.reshape(2,n)*1000
            assert np.array_equal(fields[k],previous[k]), ('rate replay mismatch',step,float(np.max(abs(fields[k]-previous[k]))))
        if (step+1)%10000==0:write(OUT/'rate_replay_status.json',{'status':'RUNNING','time_ms':(step+1)*s.dt,'bitwise_fields':True,'elapsed_s':time.time()-start})
    st.save(OUT/'rate_checkpoints/t9400ms.npz',absolute_time_ms=9400.)
    write(OUT/'rate_replay_status.json',{'status':'COMPLETE','bitwise_fields':True,'elapsed_s':time.time()-start,
        'interpretation':'Rate histories arise from prescribed native Z and native expected input; they are not projected exact microscopic SNN states.'})


def run(job):
    start=time.time(); st=Stepper(); s,m,n=st.s,st.m,st.n; name=job['name']
    st.restore(OUT/'rate_checkpoints'/f"t{job['history_ms']}ms.npz")
    source=np.load(OLD/'external_input.npz'); expected=source['expected_rate_per_ms']; cells=source['cell_e']
    z_neuron=load(checkpoint_path(job['z_profile_ms']))['slow']['z'][:len(cells)]
    z=np.bincount(cells,weights=z_neuron,minlength=n)/m.count_e
    steps=round(job['duration_ms']/s.dt); frames=steps//10
    fields=np.empty((frames,2,n),np.float32); currents=np.empty((frames,6,n),np.float32)
    for step in range(steps):
        index=94000+step
        r=st.step(z,expected[index,0].astype(float),expected[index,1].astype(float))
        if (step+1)%10==0:
            k=step//10; fields[k]=r.reshape(2,n)*1000;currents[k]=st.c
        if (step+1)%10000==0:write(OUT/'rate_progress'/f'{name}.json',{'status':'RUNNING','time_ms':(step+1)*s.dt,'elapsed_s':time.time()-start})
    folder=OUT/'rate';folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/f'{name}.npz',fields_hz=fields,current=currents,z=z,
        count_e=m.count_e,count_i=m.count_i,frame_ms=1.,start_ms=9400.)
    st.save(OUT/'rate_endpoints'/f'{name}.npz',z=z,absolute_time_ms=9400+job['duration_ms'])
    row=dict(job,status='COMPLETE',seconds=time.time()-start,mean_Z=float(np.average(z,weights=m.count_e)),
        late_E_mean_hz=float(np.average(fields[-1000:,0],axis=1,weights=m.count_e).mean()),
        scope='Native expected OU input; microscopic Poisson variability remains in transfer variance. Same Z field and paired histories at 10x10 resolution.')
    write(folder/f'{name}.json',row);return row


def main():
    replay()
    while not (OUT/'replay_status.json').exists() or read(OUT/'replay_status.json')['status']!='COMPLETE':
        if (OUT/'replay_status.json').exists() and read(OUT/'replay_status.json')['status']=='FAILED':raise RuntimeError('Native replay failed')
        time.sleep(10)
    jobs=[{'name':f'z{z}_history{h}','z_profile_ms':z,'history_ms':h,'duration_ms':2000}
          for z in (9400,8000,8400,8800,9200) for h in (9400,8000)]
    write(OUT/'rate_jobs.json',jobs); rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        iterator=iter(jobs);pending={pool.submit(run,next(iterator)) for _ in range(2)}
        while pending:
            done,pending=wait(pending,return_when=FIRST_COMPLETED)
            for future in done: rows.append(future.result())
            write(OUT/'rate_batch_status.json',{'status':'RUNNING','completed':len(rows),'total':len(jobs),'rows':rows})
            for _ in done:
                job=next(iterator,None)
                if job is not None:pending.add(pool.submit(run,job))
    write(OUT/'rate_batch_status.json',{'status':'COMPLETE','completed':len(rows),'total':len(jobs),'rows':rows})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(OUT/'rate_batch_status.json',{'status':'FAILED','error':repr(exc)});raise
