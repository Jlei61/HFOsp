"""Measure native LIF transfer and interval variation for a two-state rate unit.

Independent native single-cell simulations are offline calibration only. The
network candidate has a rate and one auxiliary state, not these sampled cells.
"""
import argparse
import os
import time
from common import *
import cupy as cp

CODE = r'''
#include <curand_kernel.h>
extern "C" __global__ void size(int* n){n[0]=sizeof(curandStatePhilox4_32_10_t);}
extern "C" __global__ void init(curandStatePhilox4_32_10_t* states,int N,unsigned long long seed){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<N)curand_init(seed,i,0,&states[i]);}
extern "C" __global__ void run(
 curandStatePhilox4_32_10_t* states,double* q,double* syn,double* voltage,int* ref,
 unsigned int* count,unsigned int* interval_count,double* isum,double* isum2,int* last,
 const double* theta,const double* input,const double* tau,const double* jump,const int* nr,
 int replicates,int C,int steps,int offset,int measure,double ar,double ad,double lam,
 unsigned int* trace,int trace_start){
 int idx=blockIdx.x*blockDim.x+threadIdx.x;if(idx>=replicates*C)return;
 int c=idx/replicates;auto rng=states[idx];double qq=q[idx],ii=syn[idx],v=voltage[idx];
 int rr=ref[idx],ll=last[idx];unsigned int n=count[idx],ni=interval_count[idx];double s=isum[idx],s2=isum2[idx];
 double decay=exp(-0.1/tau[c]);
 for(int t=0;t<steps;t++){
  unsigned int K=curand_poisson(&rng,lam);qq=ar*qq+jump[c]*K;ii=ad*ii+(1.-ad)*qq;
  rr=max(0,rr-1);bool spike=false;
  if(rr==0){double u=ii+input[c];v=u+(v-u)*decay;
   if(v>=theta[c]){v=11.;rr=nr[c];spike=true;}}
  if(spike && measure){n++;int now=offset+t;
   if(ll>=0){double isi=(now-ll)*.1;s+=isi;s2+=isi*isi;ni++;}ll=now;
   if(trace_start>=0)atomicAdd(trace+((now-trace_start)/10)*C+c,1);}
 }
 q[idx]=qq;syn[idx]=ii;voltage[idx]=v;ref[idx]=rr;states[idx]=rng;
 count[idx]=n;interval_count[idx]=ni;isum[idx]=s;isum2[idx]=s2;last[idx]=ll;
}
'''


def main(a):
    folder = OUT / 'local_response' / a.label
    folder.mkdir(parents=True, exist_ok=False)
    prep=read(GRID/'prepared.json'); p=prep['params']
    currents=np.array([-100,-40,-20,-10,-5,-2,0,1,2,3,4,5,6,7,8,9,10,12,15,20,30,50,80,120,200,400,800,1600.])
    thresholds=np.array([11.25,12.,13.,14.,15.,16.,17.,18.])
    pairs=[(0,x) for x in thresholds]+[(1,18.)]
    pop=np.repeat([x[0] for x in pairs],len(currents))
    theta=np.repeat([x[1] for x in pairs],len(currents)); u=np.tile(currents,len(pairs))
    C=len(theta); N=C*a.neurons
    cfg=dict(status='RUNNING',pid=os.getpid(),neurons_per_condition=a.neurons,
        seed=a.seed,burn_ms=a.burn,duration_ms=a.duration,theta=theta,input_mv=u,population=pop,
        native_source=str(GRID),role='Offline local response calibration; not a particle network candidate',
        statistical_unit='Independent private-Poisson realization per condition; ISIs within a neuron are dependent',
        shared_input='Constant prescribed recurrent current, original colored private Poisson input',
        protocol='Fixed input/threshold grid selected before outcomes; no network event or D fitting')
    write(folder/'config.json',cfg);write(folder/'status.json',cfg)
    cp.cuda.Device(a.device).use();module=cp.RawModule(code=CODE,options=('--fmad=false','-I/usr/local/cuda/include'),name_expressions=['size','init','run'])
    size=cp.zeros(1,dtype=cp.int32);module.get_function('size')((1,),(1,),(size,))
    rng=cp.empty(N*int(size.get()[0]),dtype=cp.uint8);blocks=((N+127)//128,)
    module.get_function('init')(blocks,(128,),(rng,np.int32(N),np.uint64(a.seed)))
    tau=np.where(pop==0,p['tau_m_E'],p['tau_m_I'])
    jump=tau/p['tau_r_AMPA']*np.where(pop==0,p['J_ext_E'],p['J_ext_I'])
    refstep=np.where(pop==0,round(p['tau_ref_E']/DT),round(p['tau_ref_I']/DT)).astype('int32')
    arrays=[cp.zeros(N),cp.zeros(N),cp.full(N,11.),cp.zeros(N,dtype=cp.int32),
            cp.zeros(N,dtype=cp.uint32),cp.zeros(N,dtype=cp.uint32),cp.zeros(N),cp.zeros(N),cp.full(N,-1,dtype=cp.int32)]
    pars=[cp.asarray(theta),cp.asarray(u),cp.asarray(tau),cp.asarray(jump),cp.asarray(refstep)]
    ar=float(np.exp(-DT/p['tau_r_AMPA']));ad=float(np.exp(-DT/p['tau_d_AMPA']));lam=float(prep['nu_ext_per_ms']*DT)
    dummy=cp.zeros(1,dtype=cp.uint32);started=time.time()
    for phase,duration,measure in [('burn',a.burn,0),('sample',a.duration,1)]:
        for offset in range(0,round(duration/DT),1000):
            steps=min(1000,round(duration/DT)-offset)
            module.get_function('run')(blocks,(128,),(rng,*arrays,*pars,np.int32(a.neurons),np.int32(C),np.int32(steps),np.int32(offset),np.int32(measure),ar,ad,lam,dummy,np.int32(-1)))
            cp.cuda.get_current_stream().synchronize()
            status=dict(status='RUNNING',pid=os.getpid(),phase=phase,completed_ms=(offset+steps)*DT,wall_s=time.time()-started)
            write(folder/'status.json',status)
            if offset%5000==0:print(status,flush=True)
    count=cp.asnumpy(arrays[4]).reshape(C,a.neurons);ni=cp.asnumpy(arrays[5]).reshape(C,a.neurons)
    sums=cp.asnumpy(arrays[6]).reshape(C,a.neurons);sums2=cp.asnumpy(arrays[7]).reshape(C,a.neurons)
    rates=count/(a.duration/1000);mean=rates.mean(1);se=rates.std(1,ddof=1)/np.sqrt(a.neurons)
    intervals=ni.sum(1);mu=sums.sum(1)/np.maximum(intervals,1)
    cv=np.sqrt(np.maximum(sums2.sum(1)/np.maximum(intervals,1)-mu**2,0))/np.maximum(mu,1e-12)
    # CV is available only when an interval was observed; never silently replace missing data by zero.
    cv[intervals==0]=np.nan
    np.savez_compressed(folder/'table.npz',theta=theta,population=pop,input_mv=u,rate_hz=mean,
        standard_error_hz=se,cv=cv,interval_count=intervals,neuron_counts=count,
        thresholds_e=thresholds,currents_mv=currents,final_voltage=cp.asnumpy(arrays[2]).reshape(C,a.neurons))
    write(folder/'status.json',dict(status='COMPLETE',wall_s=time.time()-started,conditions=C,
        missing_cv_conditions=int(np.isnan(cv).sum()),minimum_rate_hz=float(mean.min()),maximum_rate_hz=float(mean.max())))
    print('COMPLETE',folder,time.time()-started,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--label',default='native_colored_grid_v1')
    ap.add_argument('--neurons',type=int,default=512);ap.add_argument('--burn',type=float,default=500)
    ap.add_argument('--duration',type=float,default=1500);ap.add_argument('--seed',type=int,default=230901)
    ap.add_argument('--device',type=int,default=0);main(ap.parse_args())
