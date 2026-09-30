"""Local stationary colored-current assays; no physical-time network iteration."""
import numpy as np

CODE=r'''
#include <curand_kernel.h>
extern "C" __global__ void sample(const double* ie,const double* ii,const double* p,
 const double* cfg,const int* extra,unsigned char* flags,double* stats,int B,int R,
 int N,int burn,unsigned long long seed,int first_cell,int replay){
 int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=B*R)return;
 int cell=id/R,rep=id%R,Q=B*R;const double* a=p+cell*13;
 double v=a[9],m=a[11];int ref=(int)a[10];bool excitatory=a[12]>0.5;
 double es=a[7]*a[8]/(1-cfg[0]),ei=es;
 curandStatePhilox4_32_10_t rng;
 curand_init(seed,((unsigned long long)(first_cell+cell))*R+rep,0,&rng);
 int warm=burn+extra[id],count1=0,count2=0;
 double im=0,sm=0,si=0,sj=0,si2=0,sj2=0,sij=0,sz=0,ni=0,nj=0;
 for(int t=-warm;t<N;t++){
  int row=(t+warm)%N;
  double e=ie[(long long)row*Q+id],i=ii[(long long)row*Q+id];
  if(!replay){
   es*=cfg[0];es+=a[7]*curand_poisson(&rng,a[8]);ei=es+(ei-es)*cfg[1];e+=ei;
  }
  if(t==0)im=m;
  double value=e-a[3]*i-cfg[3]*m;
  if(excitatory)value+=a[4]*(cfg[5]-cfg[4]);
  double g=excitatory*a[3]*cfg[2]+a[4];
  double inf=(value+g*cfg[4])/(1+g);
  ref=max(ref-1,0);bool sp=false;
  if(ref==0){v=inf+(v-inf)*a[0];if(v>=a[1]){v=cfg[6];ref=(int)a[2];sp=true;}}
  else v=cfg[6];
  if(t>=0){
   flags[(long long)t*Q+id]=sp;
   if(t<N/2)count1+=sp;else count2+=sp;
   sm+=m;si+=e;sj+=i;si2+=e*e;sj2+=i*i;sij+=e*i;
   sz+=i+(18-cfg[4])*cfg[2]<cfg[8];ni+=e<0;nj+=i<0;
  }
  if(excitatory){m-=cfg[7]*m;m=fmax(m,0.);if(sp)m+=1.;}
 }
 double* o=stats+id*16;
 o[0]=count1;o[1]=count2;o[2]=im;o[3]=m;o[4]=sm/N;
 o[5]=si/N;o[6]=sj/N;o[7]=si2/N;o[8]=sj2/N;o[9]=sij/N;
 o[10]=sz/N;o[11]=ni/N;o[12]=nj/N;o[13]=v;o[14]=ref;o[15]=ei;
}
'''


def kernel(cp):
    return cp.RawKernel(CODE,'sample',options=('--fmad=false',))


def make_parameters(raw, cells, rate, G, params, replay=False):
    cells=np.asarray(cells);E=cells<32000
    Z,K=raw['Z'][cells],raw['K'][cells]
    h=1+E*Z*G+K
    p=np.zeros((len(cells),13))
    p[:,0]=np.exp(-.1/raw['tm'][cells])**h
    p[:,1]=raw['theta'][cells];p[:,2]=raw['ref_steps'][cells]
    p[:,3]=Z;p[:,4]=K
    p[:,7]=raw['jump_external'][cells];p[:,8]=raw['nu_per_ms'][cells]*.1
    p[:,9]=params['V_reset'];p[:,11]=np.where(E,rate,0.)
    p[:,12]=E
    if replay:
        p[:,9]=raw['initial_V'][cells];p[:,10]=raw['initial_ref'][cells];p[:,11]=raw['initial_M'][cells]
    cfg=np.array([np.exp(-.1/params['tau_r_AMPA']),np.exp(-.1/params['tau_d_AMPA']),
        G,.0005,-17.662847938268442,-30.,params['V_reset'],.1/1000.,95.19851312666987])
    return p,cfg


def run(cp, fn, ie, ii, p, cfg, R, burn, extra, seed, first_cell, replay=False):
    N,B=ie.shape[0],len(p);Q=B*R
    assert ie.shape==ii.shape==(N,Q)
    flags=cp.empty((N,Q),dtype='u1');stats=cp.empty((Q,16),dtype='f8')
    fn(((Q+127)//128,),(128,), (ie,ii,cp.asarray(p),cp.asarray(cfg),cp.asarray(extra,dtype='i4'),
        flags,stats,np.int32(B),np.int32(R),np.int32(N),np.int32(burn),np.uint64(seed),np.int32(first_cell),np.int32(replay)))
    return flags,stats.reshape(B,R,16)
