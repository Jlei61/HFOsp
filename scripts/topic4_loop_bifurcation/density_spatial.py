"""Spatial Gaussian population-density particles with native local updates.

The remaining closure is explicit: grouped delayed independent Gaussian
arrivals, and projected recorded external drive. Particle sampling is numerical
noise, not finite-native-network noise. No learned transfer/hazard is used.
"""
import numpy as np
from scipy import sparse
from campaign import ROOT,REPO,NATIVE,read

OPERATORS=REPO/'results/topic4_sef_hfo/interictal_brunel_spatial_bifurcation_20260917/operators/g20'
DRIVE=REPO/'results/topic4_sef_hfo/fig5_zm_rate_v3_20260918/native_reference/seed9108401_external_drive.npz'
DT=.1
EG=-17.662847938268442

CODE=r'''
#include <curand_kernel.h>
extern "C" __global__ void rng_bytes(int* out){out[0]=sizeof(curandStatePhilox4_32_10_t);}
extern "C" __global__ void init_rng(unsigned char* mem,int n,unsigned long long seed,int R){
 int id=blockIdx.x*blockDim.x+threadIdx.x;if(id<n){unsigned long long stream=((unsigned long long)(id/R)<<32)+(id%R);curand_init(seed,stream,0,&((curandStatePhilox4_32_10_t*)mem)[id]);}
}
extern "C" __global__ void delayed(const int* pa,const int* ca,const double* wa,const double* va,
 const int* pb,const int* cb,const double* wb,const double* vb,const double* hist,double* out,const int* clock,int depth,int P){
 int g=blockIdx.x,lane=threadIdx.x,tick=clock[0];double a=0.,b=0.,q=0.,v=0.;
 for(int i=pa[g]+lane;i<pa[g+1];i+=blockDim.x){int d=ca[i]/P+1,slot=(tick-d)%depth;if(slot<0)slot+=depth;double r=hist[slot*P+ca[i]%P];a+=wa[i]*r;q+=va[i]*r;}
 for(int i=pb[g]+lane;i<pb[g+1];i+=blockDim.x){int d=cb[i]/P+1,slot=(tick-d)%depth;if(slot<0)slot+=depth;double r=hist[slot*P+cb[i]%P];b+=wb[i]*r;v+=vb[i]*r;}
 __shared__ double buf[4][128];buf[0][lane]=a;buf[1][lane]=b;buf[2][lane]=q;buf[3][lane]=v;__syncthreads();
 for(int k=64;k>0;k/=2){if(lane<k)for(int j=0;j<4;j++)buf[j][lane]+=buf[j][lane+k];__syncthreads();}
 if(lane==0)for(int j=0;j<4;j++)out[j*P+g]=buf[j][0];
}
// Per-group pars: tm,ref_steps,theta,E,Jext,size. synaptic coefficients:
// exp(-dt/trA),exp(-dt/tdA),exp(-dt/trI),exp(-dt/tdI),trA,trI,Vreset.
// Each sample retains V,sA,IA,sI,II,M,Z,K in native units.
__device__ bool native_cell(double* x,int& ref,const double* p,const double* c,
 const double* ar,double nu,const double* global,double gain,double nx,double ni){
 double tm=p[0],E=p[3],dt=.1,rawG=E*gain*global[1],q=fmin(1.,fmax(0.,(global[0]-200.)/300.));
 double ma=ar[0]+p[4]*nu,va=ar[2]+p[4]*p[4]*nu;
 x[1]=x[1]*c[0]+tm/c[4]*(ma*dt+sqrt(fmax(va*dt,0.))*nx);
 x[3]=x[3]*c[2]+tm/c[5]*(ar[1]*dt+sqrt(fmax(ar[3]*dt,0.))*ni);
 x[2]=x[1]+(x[2]-x[1])*c[1];x[4]=x[3]+(x[4]-x[3])*c[3];
 double gg=E*x[6]*rawG,kk=E*x[7],den=1.+gg+kk;
 double vinf=(x[2]-x[6]*x[4]-.0005*x[5]+gg*(-17.662847938268442)+kk*(-30.))/den;
 ref=max(ref-1,0);bool spike=false;
 if(ref==0){x[0]=vinf+(x[0]-vinf)*exp(-dt*den/tm);if(x[0]>=p[2]){spike=true;x[0]=c[6];ref=(int)p[1];}}
 else x[0]=c[6];
 if(E>.5){
  double eligible=x[4]+(18.+17.662847938268442)*rawG<95.19851312666987?1.:0.;
  x[6]+=(eligible-x[6])*dt/5000.;x[5]*=(1.-dt/1000.);if(spike)x[5]+=1.;
  double tau=global[0]<=5.?5000.:500.;x[7]*=exp(-dt/tau);
  if(gain>0. && spike)x[7]+=.16*q;
 }
 return spike;
}
extern "C" __global__ void particles(double* state,int* refs,unsigned char* memory,
 const double* pars,const double* constants,const double* arr,const double* drive,
 const int* clock,const double* global,double gain,int* spikes,int P,int R,int ndrive){
 int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=P*R)return;int g=id/R;
 curandStatePhilox4_32_10_t* states=(curandStatePhilox4_32_10_t*)memory;
 curandStatePhilox4_32_10_t rng=states[id];float4 n=curand_normal4(&rng);states[id]=rng;
 double ar[4];for(int j=0;j<4;j++)ar[j]=arr[j*P+g];
 int index=min(clock[0]/10,ndrive-1),ref=refs[id];double x[8];for(int j=0;j<8;j++)x[j]=state[id*8+j];
 spikes[id]=native_cell(x,ref,pars+6*g,constants,ar,drive[(long long)index*P+g],global,gain,n.x,n.z);
 refs[id]=ref;for(int j=0;j<8;j++)state[id*8+j]=x[j];
}
extern "C" __global__ void supplied(double* state,int* refs,const double* normals,const double* pars,
 const double* constants,const double* arr,const double* drive,const double* global,double gain,int* spikes,int P,int R){
 int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=P*R)return;int g=id/R;double x[8],ar[4];int ref=refs[id];
 for(int j=0;j<8;j++)x[j]=state[id*8+j];for(int j=0;j<4;j++)ar[j]=arr[j*P+g];
 spikes[id]=native_cell(x,ref,pars+g*6,constants,ar,drive[g],global,gain,normals[id*2],normals[id*2+1]);
 refs[id]=ref;for(int j=0;j<8;j++)state[id*8+j]=x[j];
}
extern "C" __global__ void collect(const double* state,const int* spikes,const double* pars,
 double* history,double* rates,double* accumulator,double* output,const int* clock,int P,int R,int depth){
 int g=blockIdx.x,lane=threadIdx.x,tick=clock[0];double sum[8]={0,0,0,0,0,0,0,0};
 for(int k=lane;k<R;k+=128){int id=g*R+k;const double* x=state+id*8;
  sum[0]+=spikes[id];sum[1]+=x[6];sum[2]+=x[5];sum[3]+=x[7];sum[4]+=x[2];sum[5]+=x[6]*x[4];sum[6]+=x[0];sum[7]+=fabs(x[2])+fabs(x[6]*x[4]);}
 __shared__ double buf[8][128];for(int j=0;j<8;j++)buf[j][lane]=sum[j];__syncthreads();
 for(int k=64;k>0;k/=2){if(lane<k)for(int j=0;j<8;j++)buf[j][lane]+=buf[j][lane+k];__syncthreads();}
 if(lane==0){double r=buf[0][0]/R/.1;history[(long long)(tick%depth)*P+g]=r;rates[g]=r;
  accumulator[g]+=r*.1;
  if((tick+1)%10==0){int row=((tick+1)/10-1)%10;output[(row*8)*P+g]=accumulator[g]*1000.;accumulator[g]=0.;
   for(int j=1;j<8;j++)output[(row*8+j)*P+g]=buf[j][0]/R;}}
}
extern "C" __global__ void global_step(const double* rates,const double* pars,double* global,int* clock,int P){
 int lane=threadIdx.x;double n=0.;for(int g=lane;g<P;g+=128)if(pars[6*g+3]>.5)n+=rates[g]*pars[6*g+5];
 __shared__ double buf[128];buf[lane]=n;__syncthreads();for(int k=64;k>0;k/=2){if(lane<k)buf[lane]+=buf[lane+k];__syncthreads();}
 if(lane==0){double q=fmin(1.,fmax(0.,(global[0]-200.)/300.)),a=exp(-.1/500.);
  global[1]=a*global[1]+(1-a)*q;global[0]=global[0]*exp(-.1/15.)+buf[0]/32000.*.1*1000./15.;clock[0]++;}
}
'''


class DensityNetwork:
    def __init__(self,replicas=512,seed=927611,device=1,duration_ms=3000,gain=0.):
        import cupy as cp
        cp.cuda.Device(device).use();self.cp=cp;self.R=replicas;self.seed=seed;self.gain=gain
        self.prep=read(OPERATORS/'prepared.json');self.p=p=self.prep['params']
        assert self.prep['graph_identity']==read(NATIVE/'protocol.json')['identity']
        self.geo=dict(np.load(OPERATORS/'geometry.npz'));g=self.geo
        self.P=P=len(g['group_size']);self.E=g['population']==0;self.sizes=g['group_size'];self.depth=self.prep['max_delay_steps']+1
        assert self.sizes[self.E].sum()==32000 and self.sizes[~self.E].sum()==8000
        self.pars_cpu=np.c_[np.where(self.E,p['tau_m_E'],p['tau_m_I']),np.where(self.E,p['tau_ref_E'],p['tau_ref_I'])/DT,
            g['threshold_mv'],self.E,np.where(self.E,p['J_ext_E'],p['J_ext_I']),self.sizes]
        self.constants_cpu=np.array([np.exp(-DT/p[n]) for n in ['tau_r_AMPA','tau_d_AMPA','tau_r_GABA','tau_d_GABA']]+[p['tau_r_AMPA'],p['tau_r_GABA'],p['V_reset']])
        self.pars=cp.asarray(self.pars_cpu);self.constants=cp.asarray(self.constants_cpu)
        self.ops=[];self.ops_cpu=[]
        for kind in ['ampa','gaba']:
            a=sparse.load_npz(OPERATORS/f'mean_{kind}.npz').tocsr();q=sparse.load_npz(OPERATORS/f'variance_{kind}.npz').tocsr()
            assert np.array_equal(a.indptr,q.indptr) and np.array_equal(a.indices,q.indices)
            self.ops_cpu.append((a,q));self.ops.extend([cp.asarray(a.indptr,dtype=cp.int32),cp.asarray(a.indices,dtype=cp.int32),cp.asarray(a.data),cp.asarray(q.data)])
        with np.load(DRIVE) as z:
            n=round(duration_ms);self.drive_cpu=np.empty((n,P));cell=g['group_cell']
            self.drive_cpu[:,self.E]=z['drive_mean'][:n,cell[self.E]];self.drive_cpu[:,~self.E]=z['glob'][:n,None]
        self.drive=cp.asarray(self.drive_cpu);self.ndrive=n
        self.module=cp.RawModule(code=CODE,options=('--fmad=false',),name_expressions=['rng_bytes','init_rng','delayed','particles','supplied','collect','global_step'])
        self.k={n:self.module.get_function(n) for n in ['init_rng','delayed','particles','supplied','collect','global_step']}
        size=cp.zeros(1,dtype=cp.int32);self.module.get_function('rng_bytes')((1,),(1,),(size,));self.rng_size=int(size.get()[0])
        self.rng=cp.empty(P*replicas*self.rng_size,dtype=cp.uint8);self.state=cp.zeros((P,replicas,8));self.ref=cp.zeros((P,replicas),dtype=cp.int32)
        self.spikes=cp.zeros((P,replicas),dtype=cp.int32);self.history=cp.zeros((self.depth,P));self.arr=cp.zeros((4,P));self.rate=cp.zeros(P)
        self.clock=cp.zeros(1,dtype=cp.int32);self.global_state=cp.zeros(2);self.accumulator=cp.zeros(P);self.output=cp.zeros((10,8,P));self.reset()

    def reset(self):
        for x in [self.state,self.ref,self.spikes,self.history,self.arr,self.rate,self.clock,self.global_state,self.accumulator,self.output]:x.fill(0)
        self.state[:,:,0]=self.p['V_reset'];self.state[:,:,6]=1.
        self.k['init_rng'](((self.P*self.R+127)//128,),(128,),(self.rng,np.int32(self.P*self.R),np.uint64(self.seed),np.int32(self.R)))

    def step(self):
        cp=self.cp;P=np.int32(self.P);R=np.int32(self.R);depth=np.int32(self.depth)
        self.k['delayed']((self.P,),(128,),(*self.ops,self.history,self.arr,self.clock,depth,P))
        self.k['particles'](((self.P*self.R+127)//128,),(128,),(self.state,self.ref,self.rng,self.pars,self.constants,self.arr,self.drive,self.clock,self.global_state,float(self.gain),self.spikes,P,R,np.int32(self.ndrive)))
        self.k['collect']((self.P,),(128,),(self.state,self.spikes,self.pars,self.history,self.rate,self.accumulator,self.output,self.clock,P,R,depth))
        self.k['global_step']((1,),(128,),(self.rate,self.pars,self.global_state,self.clock,P))

    def graph(self):
        self.step();self.cp.cuda.get_current_stream().synchronize();self.reset();self.cp.cuda.get_current_stream().synchronize()
        self.stream=self.cp.cuda.Stream(non_blocking=True)
        with self.stream:
            self.stream.begin_capture()
            for _ in range(100):self.step()
            self.graph_object=self.stream.end_capture()

    def chunk(self):
        self.graph_object.launch(self.stream);self.stream.synchronize();return self.output.get()


def cpu_cell(state,ref,pars,constants,arr,drive,global_state,gain,normal):
    """One-step NumPy mirror in native units for implementation checking."""
    x=state.copy();p=pars[:,None,:];nu=drive[:,None];ar=arr[:,:,None];dt=DT;tm=p[:,:,0];E=p[:,:,3]
    G=E*gain*global_state[1];q=np.clip((global_state[0]-200)/300,0,1)
    ma=ar[0]+p[:,:,4]*nu;va=ar[2]+p[:,:,4]**2*nu
    x[:,:,1]=x[:,:,1]*constants[0]+tm/constants[4]*(ma*dt+np.sqrt(va*dt)*normal[:,:,0])
    x[:,:,3]=x[:,:,3]*constants[2]+tm/constants[5]*(ar[1]*dt+np.sqrt(ar[3]*dt)*normal[:,:,1])
    x[:,:,2]=x[:,:,1]+(x[:,:,2]-x[:,:,1])*constants[1]
    x[:,:,4]=x[:,:,3]+(x[:,:,4]-x[:,:,3])*constants[3]
    gg=E*x[:,:,6]*G;kk=E*x[:,:,7];den=1+gg+kk
    vinf=(x[:,:,2]-x[:,:,6]*x[:,:,4]-.0005*x[:,:,5]+gg*EG-kk*30)/den
    refs=np.maximum(ref-1,0);v=vinf+(x[:,:,0]-vinf)*np.exp(-dt*den/tm)
    sp=(refs==0)&(v>=p[:,:,2]);x[:,:,0]=np.where((refs==0)&~sp,v,constants[6]);refs=np.where(sp,p[:,:,1].astype(int),refs)
    eligible=(x[:,:,4]+(18-EG)*G<95.19851312666987)
    x[:,:,6]+=E*(eligible-x[:,:,6])*dt/5000.
    x[:,:,5]=np.where(E>0,x[:,:,5]*(1-dt/1000.)+sp,x[:,:,5])
    if gain>0:x[:,:,7]=np.where(E>0,x[:,:,7]*np.exp(-dt/(5000. if global_state[0]<=5. else 500.))+.16*q*sp,x[:,:,7])
    else:x[:,:,7]=np.where(E>0,x[:,:,7]*np.exp(-dt/(5000. if global_state[0]<=5. else 500.)),x[:,:,7])
    return x,refs,sp
