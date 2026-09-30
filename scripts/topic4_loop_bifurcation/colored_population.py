"""Distribution-retaining colored LIF particles; no fitted rate response.

Particles are quadrature samples of a population density, NOT new network
neurons. This file implements only the local Gaussian-current Markov process.
Synaptic fluctuations remain in raw physical current units when g changes.
Native graph coupling and finite-network correlations are not validated here.
"""
from pathlib import Path
import sys
import numpy as np
from campaign import REPO

sys.path.insert(0, str(REPO/'scripts/topic4_zm_onset_rate_v3'))
import lif_mc

CODE = r'''
#include <curand_kernel.h>
extern "C" __global__ void rng_bytes(int* out){out[0]=sizeof(curandStatePhilox4_32_10_t);}
extern "C" __global__ void init_rng(unsigned char* memory,int n,unsigned long long seed,int R,int crn){
 int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=n)return;
 curandStatePhilox4_32_10_t* states=(curandStatePhilox4_32_10_t*)memory;
 curand_init(seed,crn?id%R:id,0,&states[id]);
}
// State layout: V, q_A, I_A, q_G, I_G. Coefficients are for unit raw
// diffusion strengths. drive=[mu_raw, vE_raw, vI_raw, g] at each endpoint.
__device__ bool cell_step(double* s,int& ref,const double* p,const double* d,
                         double nx,double ny,double nz,double nw,double dt){
 double ea=sqrt(d[1]),ei=sqrt(d[2]);
 s[2]=p[7]*s[1]+p[8]*s[2]+ea*(p[12]*nx+p[13]*ny);
 s[1]=p[6]*s[1]+ea*p[11]*nx;
 s[4]=p[9]*s[3]+p[10]*s[4]+ei*(p[15]*nz+p[16]*nw);
 s[3]=p[17]*s[3]+ei*p[14]*nz;
 ref=max(0,ref-1);bool spike=false;
 if(ref==0){
  double h=1.+d[3],vinf=(d[0]+s[2]-s[4])/h,a=exp(-dt*h/p[22]);
  s[0]=vinf+(s[0]-vinf)*a;
  if(s[0]>=p[1]){s[0]=p[21];ref=(int)p[19];spike=true;}
 }else s[0]=p[21];
 return spike;
}
extern "C" __global__ void prescribed(const double* pars,const double* drive,
 double* state,int* refractory,unsigned char* memory,int* counts,
 int P,int R,int steps,int offset,int bins,int perbin,double dt){
 int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=P*R)return;
 int g=id/R;double s[5];for(int j=0;j<5;j++)s[j]=state[id*5+j];int ref=refractory[id];
 curandStatePhilox4_32_10_t* states=(curandStatePhilox4_32_10_t*)memory;
 curandStatePhilox4_32_10_t rng=states[id];
 for(int tick=0;tick<steps;tick++){
  float4 n=curand_normal4(&rng);int absolute=offset+tick;
  bool spike=cell_step(s,ref,pars+g*24,drive+((long long)absolute*P+g)*4,n.x,n.y,n.z,n.w,dt);
  if(spike)counts[(long long)id*bins+absolute/perbin]++;
 }
 for(int j=0;j<5;j++)state[id*5+j]=s[j];refractory[id]=ref;states[id]=rng;
}
// Deliberately small explicit-noise interface for independent CPU/GPU parity.
extern "C" __global__ void supplied_noise(const double* pars,const double* drive,
 const double* noise,double* state,int* refractory,unsigned char* spikes,
 int P,int R,int steps,double dt){
 int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=P*R)return;
 int g=id/R;double s[5];for(int j=0;j<5;j++)s[j]=state[id*5+j];int ref=refractory[id];
 for(int tick=0;tick<steps;tick++){
  const double* n=noise+((long long)tick*P*R+id)*4;
  spikes[(long long)tick*P*R+id]=cell_step(s,ref,pars+g*24,drive+((long long)tick*P+g)*4,n[0],n[1],n[2],n[3],dt);
 }
 for(int j=0;j<5;j++)state[id*5+j]=s[j];refractory[id]=ref;
}
'''


def parameters(theta, population, dt=.1):
    """Existing physical current covariance, unit raw diffusion, native reset."""
    rows=[]
    for th,pop in zip(theta,population):
        p=lif_mc.condition(0.,th,1.,1.,pop,dt=dt)
        p[22]=lif_mc.PARAMS['tau_m_'+pop]
        rows.append(p)
    return np.asarray(rows)


def cpu_supplied(pars,drive,noise,initial,ref_initial,dt=.1):
    """Independent NumPy update in raw current units, used only for parity."""
    P,R=initial.shape[:2];s=initial.copy();ref=ref_initial.copy()
    fired=np.empty((len(drive),P,R),bool)
    def col(j):return pars[:,j,None]
    for tick,d in enumerate(drive):
        old=s.copy();ea=np.sqrt(d[:,1,None]);ei=np.sqrt(d[:,2,None]);n=noise[tick]
        s[:,:,1]=col(6)*old[:,:,1]+ea*col(11)*n[:,:,0]
        s[:,:,2]=col(7)*old[:,:,1]+col(8)*old[:,:,2]+ea*(col(12)*n[:,:,0]+col(13)*n[:,:,1])
        s[:,:,3]=col(17)*old[:,:,3]+ei*col(14)*n[:,:,2]
        s[:,:,4]=col(9)*old[:,:,3]+col(10)*old[:,:,4]+ei*(col(15)*n[:,:,2]+col(16)*n[:,:,3])
        ref=np.maximum(ref-1,0);free=ref==0
        h=1+d[:,3,None];vinf=(d[:,0,None]+s[:,:,2]-s[:,:,4])/h
        candidate=vinf+(old[:,:,0]-vinf)*np.exp(-dt*h/col(22))
        sp=free&(candidate>=col(1));s[:,:,0]=np.where(free&~sp,candidate,col(21))
        ref=np.where(sp,col(19).astype(int),ref);fired[tick]=sp
    return s,ref,fired


class LocalDensityParticles:
    def __init__(self,pars,replicas,drive,seed=1,device=1,dt=.1,bin_ms=10.,crn=False,initial=None,ref_initial=None):
        import cupy as cp
        cp.cuda.Device(device).use();self.cp=cp;self.P=len(pars);self.R=replicas;self.dt=dt
        self.module=cp.RawModule(code=CODE,options=('--fmad=false',),name_expressions=['rng_bytes','init_rng','prescribed','supplied_noise'])
        self.pars=cp.asarray(pars,dtype=cp.float64);self.drive=cp.asarray(drive,dtype=cp.float64)
        assert self.drive.shape[1:]==(self.P,4)
        assert np.isfinite(drive).all() and (drive[:,:,1:3]>=0).all() and (drive[:,:,3]>=0).all()
        self.steps=len(drive);self.perbin=round(bin_ms/dt);assert abs(self.perbin*dt-bin_ms)<1e-10
        self.bins=(self.steps+self.perbin-1)//self.perbin;self.n=self.P*self.R
        if initial is None:
            initial=np.zeros((self.P,self.R,5));initial[:,:,0]=pars[:,21,None]
        if ref_initial is None:ref_initial=np.zeros((self.P,self.R),np.int32)
        self.state=cp.asarray(initial,dtype=cp.float64);self.ref=cp.asarray(ref_initial,dtype=cp.int32)
        size=cp.zeros(1,dtype=cp.int32);self.module.get_function('rng_bytes')((1,),(1,),(size,))
        self.rng_bytes=int(size.get()[0]);self.rng=cp.empty(self.n*self.rng_bytes,dtype=cp.uint8)
        self.module.get_function('init_rng')(((self.n+127)//128,),(128,),(self.rng,np.int32(self.n),np.uint64(seed),np.int32(self.R),np.int32(crn)))
        self.counts=cp.zeros((self.P,self.R,self.bins),dtype=cp.int32);self.tick=0

    def advance(self,steps):
        assert 0<=self.tick+steps<=self.steps
        self.module.get_function('prescribed')(((self.n+127)//128,),(128,),(
            self.pars,self.drive,self.state,self.ref,self.rng,self.counts,
            np.int32(self.P),np.int32(self.R),np.int32(steps),np.int32(self.tick),
            np.int32(self.bins),np.int32(self.perbin),self.dt))
        self.tick+=steps

    def finish(self):
        self.advance(self.steps-self.tick);return self.counts.get()
