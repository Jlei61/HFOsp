"""Independent response to AMPA/GABA input-variance modulation."""
from common import *
from model import SpatialBrunel
from response import susceptibility
import cupy as cp

CODE=r'''
#include <curand_kernel.h>
extern "C" __global__ void response(const double* pars,double* output,int R,int P,int steps,int burn,unsigned long long seed){
 int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=P*R)return;int g=id/R;const double* p=pars+g*21;
 curandStatePhilox4_32_10_t rng;curand_init(seed,id,0,&rng);
 double qa[2]={0,0},ia[2]={0,0},qg[2]={0,0},ig[2]={0,0},v[2]={11,11};int ref[2]={0,0};
 double sp=0,sm=0,ns[2]={0,0},co=1,si=0,cd=cos(p[5]*.1),sd=sin(p[5]*.1);
 for(int t=-burn;t<steps;t++){
  float4 n=curand_normal4(&rng);
  double na=p[11]*n.x,nb=p[12]*n.x+p[13]*n.y,nc=p[14]*n.z,nd=p[15]*n.z+p[16]*n.w;
  double cc=co*cd-si*sd;si=si*cd+co*sd;co=cc;
  double oscillation=t>=0?p[4]*(p[5]==0?1.:si):0.;bool fired[2]={false,false};
  for(int side=0;side<2;side++){
   double factor=sqrt(1+(side==0?oscillation:-oscillation));
   double af=p[20]==0?factor:1.,gf=p[20]==1?factor:1.;
   ia[side]=p[7]*qa[side]+p[8]*ia[side]+af*nb;qa[side]=p[6]*qa[side]+af*na;
   ig[side]=p[9]*qg[side]+p[10]*ig[side]+gf*nd;qg[side]=p[17]*qg[side]+gf*nc;
   double cur=p[0]+ia[side]-ig[side];ref[side]=max(0,ref[side]-1);
   if(ref[side]==0){v[side]=p[18]*v[side]+(1-p[18])*cur;if(v[side]>=p[1]){v[side]=11;ref[side]=(int)p[19];fired[side]=true;}}
   else v[side]=11;
  }
  if(t>=0){double d=(double)fired[0]-(double)fired[1];sp+=d*(p[5]==0?.5:si);sm+=d*(p[5]==0?0.:co);ns[0]+=fired[0];ns[1]+=fired[1];}
 }
 output[id*4]=sp;output[id*4+1]=sm;output[id*4+2]=ns[0];output[id*4+3]=ns[1];
}
'''

def main():
    cp.cuda.Device(0).use();s=SpatialBrunel();dest=OUT/'local_variance_response_check';dest.mkdir(exist_ok=True)
    rows=[];parameters=[]
    for pop,source in [('E','local_response_check_v2'),('I','local_response_check_I')]:
        old=read(OUT/source/'result.json')['rows'];pold=np.load(OUT/source/'paired_demodulation.npz')['parameters']
        for index,row in enumerate(old):
            core=row['core'];g=row['group'];f=row['frequency_hz'];z=np.load(OUT/f'g20/hopf_{core}/critical.npz');r=z['rates'];J=float(z['J'])
            chi=susceptibility(s,r,J,2j*np.pi*f/1000)
            for kind in (0,1):
                p=np.r_[pold[index],kind];p[4]=.1
                prediction=chi[1+kind][g]*2/(2+2j*np.pi*f/1000*s.tau[kind])*1000
                rows.append(dict(population=pop,core=core,group=g,input_variance=['E','I'][kind],frequency_hz=f,
                    variance_modulation_mv2=.1*p[2+kind],predicted_chi_hz_per_mv2=prediction))
                parameters.append(p)
    P=len(rows);R=8192;T=10000;N=P*R;out=cp.zeros((N,4));started=time.time()
    print('START variance',P,R,flush=True)
    kernel=cp.RawKernel(CODE,'response',options=('--fmad=false',))
    kernel(((N+127)//128,),(128,),(cp.asarray(parameters),out,np.int32(R),np.int32(P),np.int32(T*10),np.int32(10000),np.uint64(202609179)))
    d=out.get().reshape(P,R,4)
    for k,row in enumerate(rows):
        e=(d[k,:,0]+1j*d[k,:,1])/(T*row['variance_modulation_mv2'])*1000
        row.update(measured_chi_hz_per_mv2=e.mean(),complex_sem=float(np.sqrt(np.mean(abs(e-e.mean())**2)/R)))
    write(dest/'result.json',dict(status='COMPLETE',duration_ms=T,R=R,relative_amplitude=.1,rows=rows,seconds=time.time()-started))
    print('COMPLETE',time.time()-started,flush=True)

if __name__=='__main__':main()
