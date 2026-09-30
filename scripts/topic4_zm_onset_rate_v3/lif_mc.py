"""Colored-noise LIF Monte-Carlo transfer assay with the native engine discretisation.

One thread = one replicate (independent noise path) of one condition. Synaptic
AMPA/GABA rise+decay pairs are driven by Gaussian increments whose exact
discrete covariance reproduces the continuous diffusion moments
(Var[s]=tau_m v/(2 tau_r), Var[I]=tau_m v/(2(tau_r+tau_d))). Membrane, reset,
refractory follow src/topic4_raster_protocol_engine.membrane_step ordering.
Common random numbers: replicate k of every condition uses stream (seed,k).
Modes: static count; paired +/- sinusoidal modulation of mean (channel 0),
AMPA variance (1) or GABA variance (2), demodulated as in the v2 assay.
"""
from common_v3 import *
import cupy as cp

CODE=r'''
#include <curand_kernel.h>
// pars layout per condition (24 doubles):
// 0 mu,1 theta,2 ve,3 vi,4 amp,5 omega(rad/step),6 ar_A,7 b_A,8 ad_A,9 b_G,10 ad_G,
// 11 C11_A,12 C21_A,13 C22_A,14 C11_G,15 C21_G,16 C22_G,17 ar_G,18 decay_V,19 ref_steps,20 channel,21 vreset,22,23 unused
extern "C" __global__ void assay(const double* pars,double* output,int R,int P,int steps,int burn,unsigned long long seed,int crn){
 int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=P*R)return;int g=id/R,k=id-g*R;const double* p=pars+g*24;
 curandStatePhilox4_32_10_t rng;curand_init(seed,crn?(unsigned long long)k:(unsigned long long)id,0,&rng);
 double vr=p[21];
 double qa[2]={0,0},ia[2]={0,0},qg[2]={0,0},ig[2]={0,0},v[2]={vr,vr};int ref[2]={0,0};
 double sp=0,sm=0,ns[2]={0,0},co=1,si=0,cd=cos(p[5]),sd=sin(p[5]);
 int channel=(int)p[20];bool modulated=p[4]!=0.;
 for(int t=-burn;t<steps;t++){
  float4 n=curand_normal4(&rng);
  double na=p[11]*n.x,nb=p[12]*n.x+p[13]*n.y,nc=p[14]*n.z,nd=p[15]*n.z+p[16]*n.w;
  double cc=co*cd-si*sd;si=si*cd+co*sd;co=cc;
  double osc=(t>=0&&modulated)?p[4]*(p[5]==0.?1.:si):0.;bool fired[2]={false,false};
  int nside=modulated?2:1;
  for(int side=0;side<nside;side++){
   double o=side==0?osc:-osc;
   double af=1.,gf=1.,mo=0.;
   if(channel==0)mo=o;else if(channel==1)af=sqrt(1+o);else gf=sqrt(1+o);
   // synaptic: decay-then-jump for s, then I <- s + (I-s)*ad  (engine order)
   ia[side]=p[7]*qa[side]+p[8]*ia[side]+af*nb;qa[side]=p[6]*qa[side]+af*na;
   ig[side]=p[9]*qg[side]+p[10]*ig[side]+gf*nd;qg[side]=p[17]*qg[side]+gf*nc;
   double cur=p[0]+mo+ia[side]-ig[side];
   ref[side]=max(0,ref[side]-1);
   if(ref[side]==0){v[side]=p[18]*v[side]+(1-p[18])*cur;if(v[side]>=p[1]){v[side]=vr;ref[side]=(int)p[19];fired[side]=true;}}
   else v[side]=vr;
  }
  if(t>=0){
   if(modulated){double d=(double)fired[0]-(double)fired[1];sp+=d*(p[5]==0.?.5:si);sm+=d*(p[5]==0.?0.:co);}
   ns[0]+=fired[0];ns[1]+=fired[1];}
 }
 output[id*4]=sp;output[id*4+1]=sm;output[id*4+2]=ns[0];output[id*4+3]=ns[1];
}
'''
_kernel=None
def kernel():
    global _kernel
    if _kernel is None:_kernel=cp.RawKernel(CODE,'assay',options=('--fmad=false',))
    return _kernel

PARAMS=read(OPERATORS/'g20/prepared.json')['params']
DT=PARAMS['dt']
def synaptic_block(tm,var,tr,td,dt=DT):
    ar,ad=np.exp(-dt/tr),np.exp(-dt/td);b=tr/(tr-td)*(ar-ad)
    S=tm*var/2*np.array([[1/tr,1/(tr+td)],[1/(tr+td),1/(tr+td)]])
    A=np.array([[ar,0],[b,ad]]);Q=S-A@S@A.T
    C=np.linalg.cholesky(Q) if var>0 else np.zeros((2,2))
    return ar,b,ad,C

def condition(mu,theta,ve,vi,pop,amplitude=0.,freq_hz=0.,channel=0,dt=DT):
    """Parameter vector for one LIF condition; pop 'E' or 'I'."""
    p=PARAMS;tm=p['tau_m_E'] if pop=='E' else p['tau_m_I'];ref=p['tau_ref_E'] if pop=='E' else p['tau_ref_I']
    q=np.zeros(24);q[:6]=[mu,theta,ve,vi,amplitude,2*np.pi*freq_hz/1000*dt]
    ar,b,ad,C=synaptic_block(tm,ve,p['tau_r_AMPA'],p['tau_d_AMPA'],dt);q[6:9]=[ar,b,ad];q[11:14]=[C[0,0],C[1,0],C[1,1]]
    ar,b,ad,C=synaptic_block(tm,vi,p['tau_r_GABA'],p['tau_d_GABA'],dt);q[9:11]=[b,ad];q[14:17]=[C[0,0],C[1,0],C[1,1]];q[17]=ar
    q[18]=np.exp(-dt/tm);q[19]=round(ref/dt);q[20]=channel;q[21]=p['V_reset']
    return q

def run(pars,R,duration_ms,burn_ms,seed,crn=True,dt=DT,device=0,batch=None):
    """Return observed (P,R,4) array: [sin-sum, cos-sum, spikes side0, spikes side1]."""
    cp.cuda.Device(device).use();pars=np.asarray(pars,dtype=np.float64);P=len(pars)
    steps=int(round(duration_ms/dt));burn=int(round(burn_ms/dt))
    out=np.empty((P,R,4));batch=P if batch is None else batch
    for start in range(0,P,batch):
        sub=pars[start:start+batch];n=len(sub)*R;o=cp.zeros((n,4))
        kernel()(((n+127)//128,),(128,),(cp.asarray(sub),o,np.int32(R),np.int32(len(sub)),np.int32(steps),np.int32(burn),np.uint64(seed),np.int32(crn)))
        out[start:start+len(sub)]=o.get().reshape(len(sub),R,4)
    return out

def rates_hz(observed,duration_ms):
    return (observed[...,2]+observed[...,3]).mean(axis=-1)/(2 if observed[...,3].any() else 1)/duration_ms*1000

if __name__=='__main__':
    # Reproduce the 8 assayed v2 workpoints (static rate + zero-frequency gains) and benchmark throughput.
    rows=[];pars=[]
    for lab in ['local_response','local_response_additional_mode_groups']:
        d=read(OLDV2/lab/'result.json');seen=set()
        for r in d['rows']:
            key=(r['state'],r['group'])
            if key in seen:continue
            seen.add(key);meas=[q for q in d['rows'] if q['state']==r['state'] and q['group']==r['group']]
            gE=[q['measured'] for q in meas if q['channel']=='variance_E' and q['frequency_hz']==0][0]
            gI=[q['measured'] for q in meas if q['channel']=='variance_I' and q['frequency_hz']==0][0]
            rows.append(dict(state=r['state'],group=r['group'],pop=r['population'],measured_hz=np.mean([q['measured_rate_hz'] for q in meas]),
                             measured_gain_vE=gE[0] if isinstance(gE,list) else gE,measured_gain_vI=gI[0] if isinstance(gI,list) else gI))
            pars.append(condition(r['mu_mv'],r['threshold_mv'],r['variance_E'],r['variance_I'],r['population']))
            pars.append(condition(r['mu_mv'],r['threshold_mv'],r['variance_E'],r['variance_I'],r['population'],amplitude=.05,channel=1))
            pars.append(condition(r['mu_mv'],r['threshold_mv'],r['variance_E'],r['variance_I'],r['population'],amplitude=.05,channel=2))
    R=2048;T=4000;t0=time.time();obs=run(pars,R,T,500,20260918,crn=True);el=time.time()-t0
    n_steps=len(pars)*R*45000;print(f'throughput {n_steps/el:.3e} neuron-steps/s ({el:.1f}s, {len(pars)} conditions x {R})')
    for i,row in enumerate(rows):
        base=obs[3*i];rate=base[:,2].mean()/T*1000;sem=base[:,2].std()/np.sqrt(R)/T*1000
        gains=[]
        for c,vkey in [(1,'ve'),(2,'vi')]:
            o=obs[3*i+c];p=pars[3*i+c];amp=p[4]*p[2 if c==1 else 3]
            est=o[:,0]/(T*amp)*1000;gains.append((est.mean(),est.std()/np.sqrt(R)))
        print(f"{row['state']:>10} g{row['group']:4d} {row['pop']} rate {rate:7.2f}±{sem:.2f} (v2 assay {row['measured_hz']:7.2f})  gE {gains[0][0]:+.4f}±{gains[0][1]:.4f} (v2 {row['measured_gain_vE']:+.4f})  gI {gains[1][0]:+.4f}±{gains[1][1]:.4f} (v2 {row['measured_gain_vI']:+.4f})")
    # scaling invariance: theta=18 vs theta=14.5 with scaled mu, variances
    base=condition(17.98,18.,89.139,216.888,'E');sc=(14.5-11)/7.
    scaled=condition(11+(17.98-11)*sc,14.5,89.139*sc*sc,216.888*sc*sc,'E')
    o=run([base,scaled],R,T,500,7,crn=True);print('scaling invariance rates',o[:,:,2].mean(axis=1)/T*1000)
