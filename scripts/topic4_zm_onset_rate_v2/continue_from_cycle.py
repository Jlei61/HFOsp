"""Independent time-domain outcome when crossing a verified periodic fold.

The initial rates, all filters, M and the complete delay history come from
one periodic BVP. Only the held spatial Z parameter is changed. CUDA graph
replay exploits the exact integer history-ring period, not a model reduction.
"""
from periodic_zm import *
from run_conditions import summary
from math import lcm,ceil


def initial_from_orbit(s,path,dt,device):
    z=np.load(path);r=z['r'];T=float(z['T']);s.set_D(float(z['D']))
    o=ZMPeriodic(s,len(r),device);cp=o.cp;kernels=o.kernels(T,float(z['D']))
    rf=cp.fft.rfft(cp.asarray(r),axis=0)/len(r);lam=2j*np.pi*cp.arange(o.K)[:,None]/T
    a,b,qa,qb=[(op@rf.ravel()).reshape(o.K,s.P) for op in kernels[0]]
    target=rf/kernels[-2];xa=target/(1+lam*cp.asarray(s.tf));xb=target/(1+lam*cp.asarray(s.ts))
    qav=cp.asarray(s.tm)*s.area[0]*a/(1+lam*s.rise[0]);iav=qav/(1+lam*s.decay[0])
    qgv=cp.asarray(s.tm)*s.area[1]*b/(1+lam*s.rise[1]);igv=qgv/(1+lam*s.decay[1])
    va=cp.asarray(s.tm)*s.area[0]**2*qa/(1+lam*s.tau[0]/2)
    vg=cp.asarray(s.tm)*s.area[1]**2*qb/(1+lam*s.tau[1]/2)
    m=.5*cp.asarray(s.E)*rf/(1+lam*1000)
    factor=np.full(o.K,2.);factor[0]=factor[-1]=1.
    coeff=cp.stack([xa,xb,qav,iav,qgv,igv,va,vg,m]).get()
    state=np.vstack([np.sum(coeff.real*factor[None,:,None],axis=1),s.Z.copy()])
    depth=s.prep['max_delay_steps']*round(.1/dt)+1;lags=np.arange(depth)
    vals=(np.exp(-2j*np.pi*lags[:,None]*dt*np.arange(o.K)[None,:]/T)@(rf.get()*factor[:,None])).real
    hist=np.empty_like(vals);hist[(-lags)%depth]=vals
    assert np.max(abs(hist[0]-s.output(state)))<1e-12
    return state,hist


class RecordedGraph:
    def __init__(self,e):
        self.e=e;cp=e.cp;self.sample=round(1/e.dt);self.steps=lcm(e.depth,self.sample)
        self.bins=self.steps//self.sample;self.rates=cp.zeros((self.bins,e.s.P));self.m=cp.zeros_like(self.rates);self.acc=cp.zeros(e.s.P)
        code=f'#define P {e.s.P}\n'+r'''
extern "C" __global__ void record(const double* rate,const double* y,double* acc,double* rates,double* m,int slot,int sample){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=P)return;
 acc[g]+=rate[g]/sample;
 if(slot>=0){rates[slot*P+g]=acc[g]*1000.;m[slot*P+g]=y[8*P+g];acc[g]=0.;}
}
'''
        self.record=cp.RawKernel(code,'record',options=('--fmad=false',))
        start=e.y.copy();history=e.history.copy();tick=e.tick
        rate=e.step();self.record(((e.s.P+127)//128,),(128,),(rate,e.y,self.acc,self.rates,self.m,np.int32(0),np.int32(self.sample)))
        e.y[:]=start;e.history[:]=history;e.tick=tick;self.acc.fill(0)
        self.stream=cp.cuda.Stream(non_blocking=True);cp.cuda.get_current_stream().synchronize()
        with self.stream:
            self.stream.begin_capture()
            for i in range(self.steps):
                rate=e.step();slot=(i+1)//self.sample-1 if (i+1)%self.sample==0 else -1
                self.record(((e.s.P+127)//128,),(128,),(rate,e.y,self.acc,self.rates,self.m,np.int32(slot),np.int32(self.sample)))
            self.graph=self.stream.end_capture()
        e.tick=tick

    def advance(self):
        with self.stream:self.graph.launch(stream=self.stream)
        self.stream.synchronize();self.e.tick+=self.steps
        return self.rates.get(),self.m.get()


def run(s,D,state,hist,duration,dt,device,check=False,tag=''):
    s.set_D(D);y=state.copy();y[9]=s.Z;e=ZMIntegrator(s,dt=dt,initial=y,history=hist,device=device)
    g=RecordedGraph(e);cp=e.cp;out=PERIODIC_OUT/f'crossing_runs{tag}'/f'D{D:.9f}';out.mkdir(parents=True,exist_ok=True)
    if check:
        g.advance();g.advance();yg=e.y.get();hg=e.history.get()
        ref=ZMIntegrator(s,dt=dt,initial=y,history=hist,device=device)
        for i in range(2*g.steps):ref.step()
        q=dict(state_max_difference=float(abs(yg-ref.y.get()).max()),history_max_difference=float(abs(hg-ref.history.get()).max()),
               steps=2*g.steps,graph_block_steps=g.steps,history_depth=e.depth,dt_ms=dt)
        assert q['state_max_difference']==0 and q['history_max_difference']==0,q
        write(PERIODIC_OUT/f'graph_equation_check{tag}.json',dict(status='PASS',**q));e.y.set(y);e.history.set(hist);e.tick=0
        cp.cuda.get_current_stream().synchronize()
        del ref;cp.get_default_memory_pool().free_all_blocks()
    rates=[];ms=[];start=time.time()
    for j in range(ceil(duration/g.bins)):
        r,m=g.advance();rates.append(r);ms.append(m)
        if j%max(1,1000//g.bins)==0:print('CROSSING',D,'ms',(j+1)*g.bins,'seconds',round(time.time()-start,1),flush=True)
    r=np.concatenate(rates);m=np.concatenate(ms);reg=s.geo['group_region'];sizes=s.sizes
    global_rate=r[:,s.E]@s.mean_weights
    region=np.array([np.average(r[:,s.E&(reg==k)],axis=1,weights=sizes[s.E&(reg==k)]) for k in range(3)]).T
    field=np.zeros((len(r),s.grid*s.grid));cells=s.geo['group_cell'];den=np.bincount(cells[s.E],weights=sizes[s.E],minlength=s.grid*s.grid)
    for k in np.flatnonzero(s.E):field[:,cells[k]]+=r[:,k]*sizes[k]
    field/=np.maximum(den,1)
    np.savez_compressed(out/'trajectory.npz',time_ms=np.arange(len(r))+1,group_rate_hz=r.astype('float32'),regional_rates_hz=region,
        global_E_hz=global_rate,field_E_hz=field.astype('float32'),M_current=m.astype('float32'),Z=s.Z,
        final_state=e.y.get(),final_history=e.history.get(),final_tick=e.tick)
    write(out/'result.json',dict(D=D,J_EE_core=1.,Z='held',M='dynamic',dt_ms=dt,duration_ms=len(r),analysis_start_ms=1000,
        dynamics=summary(np.c_[global_rate,region][1000:]),initial_condition='Same BVP orbit phase, nine states and full delay history; only Z changed',
        status='COMPLETE',seconds=time.time()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--D',type=float,nargs='+',default=[.18862,.1887,.19]);p.add_argument('--duration',type=int,default=6000)
    p.add_argument('--dt',type=float,default=.1);p.add_argument('--device',type=int,default=0);p.add_argument('--tag',default='');a=p.parse_args()
    s=ZMSpatialRate();path=PERIODIC_OUT/'orbits/burstUp_0039_N512.npz';state,hist=initial_from_orbit(s,path,a.dt,a.device)
    import cupy as cp
    cp.get_default_memory_pool().free_all_blocks()
    for j,D in enumerate(a.D):run(s,D,state,hist,a.duration,a.dt,a.device,j==0,a.tag)
