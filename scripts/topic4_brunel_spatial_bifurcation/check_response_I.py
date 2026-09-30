"""Independent colored-LIF response check at the actual critical workpoints.

Uncoupled neurons are an offline response assay, as in the paper's supplement.
They are not the network model and do not receive recorded future SNN spikes.
Paired +/- sinusoidal drive uses the same Gaussian current realization.
"""
from common import *
from model import SpatialBrunel
from response import susceptibility
import cupy as cp

CODE=r'''
#include <curand_kernel.h>
extern "C" __global__ void response(const double* pars,double* output,int R,int P,int steps,int burn,unsigned long long seed){
 int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=P*R)return;int g=id/R;const double* p=pars+g*20;
 curandStatePhilox4_32_10_t rng;curand_init(seed,id,0,&rng);
 double qa=0,ia=0,qg=0,ig=0,vp=11,vm=11,sp=0,sm=0,dp=0,dm=0,nsp=0,nsm=0;
 int rp=0,rm=0;double co=1,si=0,cd=cos(p[5]*.1),sd=sin(p[5]*.1);
 for(int t=-burn;t<steps;t++){
  float4 n=curand_normal4(&rng);
  double na=p[11]*n.x,nb=p[12]*n.x+p[13]*n.y,nc=p[14]*n.z,nd=p[15]*n.z+p[16]*n.w;
  ia=p[7]*qa+p[8]*ia+nb;qa=p[6]*qa+na;
  ig=p[9]*qg+p[10]*ig+nd;qg=p[17]*qg+nc;
  double cc=co*cd-si*sd;si=si*cd+co*sd;co=cc;
  double oscillation=t>=0?p[4]*(p[5]==0?1.:si):0.;double cur=p[0]+ia-ig;
  rp=max(0,rp-1);rm=max(0,rm-1);bool fp=false,fm=false;
  if(rp==0){vp=p[18]*vp+(1-p[18])*(cur+oscillation);if(vp>=p[1]){vp=11;rp=(int)p[19];fp=true;}}
  else vp=11;
  if(rm==0){vm=p[18]*vm+(1-p[18])*(cur-oscillation);if(vm>=p[1]){vm=11;rm=(int)p[19];fm=true;}}
  else vm=11;
  if(t>=0){double difference=(double)fp-(double)fm;sp+=difference*(p[5]==0?.5:si);sm+=difference*(p[5]==0?0.:co);nsp+=fp;nsm+=fm;}
 }
 output[id*4]=sp;output[id*4+1]=sm;output[id*4+2]=nsp;output[id*4+3]=nsm;
}
'''

def main():
    cp.cuda.Device(0).use();s=SpatialBrunel();dest=OUT/'local_response_check_I';dest.mkdir(exist_ok=True)
    rows=[];pars=[]
    for core in ('A','B'):
        z=np.load(OUT/f'g20/hopf_{core}/critical.npz');r=z['rates'];J=float(z['J']);mode=z['vector']
        mask=~s.E;index=np.flatnonzero(mask)
        mass=s.geo['group_size'][index]*abs(mode[index])**2
        selected=index[np.argsort(-mass)[:3]]
        mu,ve,vi=s.moments(r,J)
        for g in selected:
            for f in [0.,2.,4.,6.,10.,20.,40.]:
                p=np.zeros(20);p[:6]=[mu[g],s.theta[g],ve[g],vi[g],.15,2*np.pi*f/1000]
                for k,var in enumerate([ve[g],vi[g]]):
                    tr,td=s.rise[k],s.decay[k];ar,ad=np.exp(-.1/tr),np.exp(-.1/td);b=tr/(tr-td)*(ar-ad)
                    S=s.tm[g]*var/2*np.array([[1/tr,1/(tr+td)],[1/(tr+td),1/(tr+td)]])
                    A=np.array([[ar,0],[b,ad]]);C=np.linalg.cholesky(S-A@S@A.T)
                    if k==0:p[6:9]=[ar,b,ad];p[11:14]=[C[0,0],C[1,0],C[1,1]]
                    else:p[9:11]=[b,ad];p[14:17]=[C[0,0],C[1,0],C[1,1]];p[17]=ar
                p[18]=np.exp(-.1/s.tm[g]);p[19]=round(s.ref[g]/.1)
                prediction=susceptibility(s,r,J,2j*np.pi*f/1000)[0][g]
                rows.append(dict(core=core,group=int(g),frequency_hz=f,mu_mv=mu[g],variance_E=ve[g],variance_I=vi[g],threshold_mv=s.theta[g],
                    predicted_rate_hz=r[g]*1000,predicted_chi_hz_per_mv=prediction*1000))
                pars.append(p)
    P=len(pars);R=16384;steps=100000;burn=10000;N=P*R
    kernel=cp.RawKernel(CODE,'response',options=('--fmad=false',));out=cp.zeros((N,4));started=time.time()
    print('START',P,'conditions',R,'members','10000ms',flush=True)
    kernel(((N+127)//128,),(128,),(cp.asarray(pars),out,np.int32(R),np.int32(P),np.int32(steps),np.int32(burn),np.uint64(202609177)))
    observed=out.get().reshape(P,R,4)
    for k,row in enumerate(rows):
        d=observed[k];T=steps*.1
        # r+=r0+eps Im[chi exp(iwt)], r-=r0-eps Im[chi exp(iwt)]
        estimates=(d[:,0]+1j*d[:,1])/(T*.15)*1000
        mean=estimates.mean();sem=np.sqrt(np.mean(abs(estimates-mean)**2)/R)
        rate=(d[:,2]+d[:,3]).mean()/2/T*1000
        predicted=complex(*row['predicted_chi_hz_per_mv']) if isinstance(row['predicted_chi_hz_per_mv'],list) else row['predicted_chi_hz_per_mv']
        row.update(observed_rate_hz=float(rate),measured_chi_hz_per_mv=mean,complex_sem=float(sem),relative_response_error=float(abs(mean-predicted)/abs(mean)))
    np.savez_compressed(dest/'paired_demodulation.npz',observed=observed,parameters=np.array(pars))
    write(dest/'result.json',dict(status='COMPLETE',duration_ms=10000,R=R,amplitude_mv=.15,rows=rows,seconds=time.time()-started,
        source='Colored Gaussian diffusion LIF with exact joint synaptic OU update; independent local response assay',
        mean_relative_response_error=float(np.mean([r['relative_response_error'] for r in rows]))))
    print('COMPLETE',time.time()-started,[(r['core'],r['frequency_hz'],r['relative_response_error']) for r in rows],flush=True)

if __name__=='__main__':main()
