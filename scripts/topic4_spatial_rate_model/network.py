"""Spatial rate/auxiliary dynamics with the original graph's delay operators.

All network states are population rates, auxiliary response, synaptic currents,
and Z/M. There is no particle or membrane-potential-density state.
"""
from common import *
from rate_unit import RateTransfer
import cupy as cp
from scipy import sparse

CODE=r'''
#include <cupy/complex.cuh>
#define M_PI 3.1415926535897932384626433832795
extern "C" __global__ void delayed(const int* ptr,const int* index,const double* weights,
 const double* history,double* out,int tick,int depth){
 int row=blockIdx.x;int lane=threadIdx.x;double v=0.;
 for(int e=ptr[row]+lane;e<ptr[row+1];e+=blockDim.x){int d=index[e]/800;int src=index[e]%800;
  int slot=(tick-d)%depth;if(slot<0)slot+=depth;v+=weights[e]*history[slot*800+src];}
 __shared__ double sums[128];sums[lane]=v;__syncthreads();
 for(int d=blockDim.x/2;d>0;d/=2){if(lane<d)sums[lane]+=sums[lane+d];__syncthreads();}
 if(lane==0)out[row]=sums[0];}
extern "C" __global__ void synapses(double* qa,double* ia,double* qg,double* ig,
 const double* e,const double* i,const double* nu,double base_nu,double dt,double rg,double dg){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=800)return;
 double tm=c<400?20.:10.,je=c<400?.455:.85;
 double ra=.7,da=3.5,ea=exp(-dt/ra),ed=exp(-dt/da),eg=exp(-dt/rg),gd=exp(-dt/dg);
 // Preserve the original discrete synaptic DC area. Continuous two-pole
 // dynamics and exact constant-input flow are explicit model choices.
 double A=tm/ra*.1/(1-exp(-.1/ra))*(e[c]+je*(nu[c]-base_nu));
 double G=tm/rg*.1/(1-exp(-.1/rg))*i[c];
 ia[c]=ed*ia[c]+(1-ed)*A+(qa[c]-A)*ra/(ra-da)*(ea-ed);
 ig[c]=gd*ig[c]+(1-gd)*G+(qg[c]-G)*rg/(rg-dg)*(eg-gd);
 qa[c]=ea*qa[c]+(1-ea)*A;qg[c]=eg*qg[c]+(1-eg)*G;}
__device__ double interp(const double* coeff,int row,int k,int NX,double h){
 int at=(row*(NX-1)+k)*4;return ((coeff[at]*h+coeff[at+1])*h+coeff[at+2])*h+coeff[at+3];}
extern "C" __global__ void rates(double* r,double* aux,double* z,double* m,double* output,
 const int* cell,const double* theta,int NE,int P,const double* ia,const double* ig,
 const double* xs,int NX,const double* fs,const double* cvs,const double* parameters,
 double dt,int nonlinear,int dynamic_z,double ith,double zwidth,double variance_a,double variance_b){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=P)return;int c=cell[g];bool E=g<NE;
 double u=ia[c]-(E?z[g]:1.)*ig[c]-(E?.0005*m[g]:0.);
 u=fmin(fmax(u,xs[0]),xs[NX-1]);int k=0;while(k<NX-2 && u>=xs[k+1])k++;double h=u-xs[k];
 int lo=8,hi=8;double mix=0;
 if(E){double th=theta[g];double levels[8]={11.25,12.,13.,14.,15.,16.,17.,18.};
  hi=1;while(hi<7 && th>levels[hi])hi++;lo=hi-1;mix=(th-levels[lo])/(levels[hi]-levels[lo]);}
 double F=(1-mix)*interp(fs,lo,k,NX,h)+mix*interp(fs,hi,k,NX,h);
 double CV=(1-mix)*interp(cvs,lo,k,NX,h)+mix*interp(cvs,hi,k,NX,h);CV=fmin(fmax(CV,0.),1.5);
 double tm=E?20.:10.;int pop=E?0:1;
 double alpha=parameters[2*pop]*F*(CV/.22)*(CV/.22)+parameters[2*pop+1]*exp(-F*tm)/tm;
 double rate=r[g],b=aux[g],value;
 if(!nonlinear){double decay=exp(-dt*alpha),co=cos(dt*2*M_PI*F),si=sin(dt*2*M_PI*F);double x=rate-F;
  rate=F+decay*(co*x+si*b);b=decay*(co*b-si*x);
  value=.5*(rate+hypot(rate,1e-5));
 }else{
  complex<double> w(b,M_PI*rate),eq(-alpha/2,M_PI*F);
  complex<double> ratio=(w-eq)/(w+eq),next=ratio*exp(2.*eq*dt);
  complex<double> nw=eq*(1.+next)/(1.-next),integral=eq*dt-log((1.-next)/(1.-ratio));
  rate=nw.imag()/M_PI;b=nw.real();value=integral.imag()/(M_PI*dt);
 }
 r[g]=rate;aux[g]=b;output[g]=value;
 if(E){m[g]=exp(-dt/1000.)*m[g]+1000.*(1-exp(-dt/1000.))*value;
  if(dynamic_z){double target=.5*erfc((ig[c]-ith)/(sqrt(2.)*zwidth));
   if(variance_a>=0.){double mu=fmax(ig[c],1e-8),s2=log1p(variance_a/mu+variance_b);
    target=.5*erfc((log(mu/ith)-.5*s2)/sqrt(2*s2));}
   z[g]=exp(-dt/5000.)*z[g]+(1-exp(-dt/5000.))*target;}}
}
extern "C" __global__ void restrict_rate(const int* ptr,const int* groups,const double* weights,
 const double* output,double* history,int tick,int depth){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=800)return;double v=0.;
 for(int i=ptr[c];i<ptr[c+1];i++){int g=groups[i];v+=weights[g]*output[g];}
 history[(tick%depth)*800+c]=v;
}
'''


class SpatialRate:
    def __init__(self,kind='linear',device=0,D=None,resource_closure='mean'):
        cp.cuda.Device(device).use();self.kind=kind;self.tick=0
        self.geo=dict(np.load(OUT/'operators/groups.npz'));self.prep=read(OUT/'operators/params.json')
        self.NE=int(self.geo['nE']);self.P=len(self.geo['cell']);self.depth=self.prep['max_delay_steps']+1
        fit=read(OUT/'local_response/dynamic_rate_identification_exactflow_v2/result.json')
        self.parameters=np.array(next(x for x in fit['rows'] if x['model']==kind)['parameters'])
        self.module=cp.RawModule(code=CODE,options=('--fmad=false',),name_expressions=['delayed','synapses','rates','restrict_rate'])
        self.ops=[]
        for name in ('ampa','gaba'):
            op=sparse.load_npz(OUT/f'operators/{name}_delay.npz').tocsr()
            self.ops.append([cp.asarray(op.indptr,dtype=cp.int32),cp.asarray(op.indices,dtype=cp.int32),cp.asarray(op.data)])
        tr=RateTransfer();self.xs=cp.asarray(tr.x)
        self.fs=cp.asarray(tr.f.c.transpose(1,2,0).transpose(1,0,2).copy().reshape(9,len(tr.x)-1,4))
        self.cvs=cp.asarray(tr.c.c.transpose(1,2,0).transpose(1,0,2).copy().reshape(9,len(tr.x)-1,4))
        # scipy coefficient shape is (power, interval, row).
        assert tr.f.c.shape==(4,len(tr.x)-1,9)
        self.cell=cp.asarray(self.geo['cell']);self.theta=cp.asarray(self.geo['theta']);self.weights=cp.asarray(self.geo['weight'])
        counts=np.bincount(self.geo['cell'],minlength=800)
        self.ptr=cp.asarray(np.r_[0,np.cumsum(counts)],dtype=cp.int32)
        self.groups=cp.asarray(np.argsort(self.geo['cell'],kind='stable'),dtype=cp.int32)
        self.pars=cp.asarray(self.parameters)
        self.r=cp.zeros(self.P);self.aux=cp.asarray(np.where(self.geo['population']==0,-self.parameters[1]/40.,-self.parameters[3]/20.)) if kind=='nonlinear' else cp.zeros(self.P)
        self.z=cp.ones(self.NE);self.m=cp.zeros(self.NE);self.output=cp.zeros(self.P)
        self.history=cp.zeros((self.depth,800));self.qa=cp.zeros(800);self.ia=cp.zeros(800);self.qg=cp.zeros(800);self.ig=cp.zeros(800)
        self.arrive=[cp.zeros(800),cp.zeros(800)];self.nu=cp.full(800,self.prep['nu_ext_per_ms']);self.D=D
        if D is not None:
            from scipy.optimize import brentq
            source=SOURCE/'replay/runs/eta0.0005_s9108401/checkpoints/t9420ms.npz'
            with np.load(source) as data:original=data['slow__z'][:32000]
            if D==0:field=np.ones_like(original)
            elif D==1:field=np.zeros_like(original)
            else:
                exponent=brentq(lambda a:np.mean(original**a)-(1-D),0.,1e5)
                field=original**exponent
            group=np.bincount(self.geo['membership_e'],weights=field,minlength=self.NE)/self.geo['count'][:self.NE]
            self.z.set(group)
        self.ith=float(read(ROOT/'results/topic4_sef_hfo/fig5_preentry_event_audit_20260914/protocol.json')['I_th'])
        self.resource_coefficients=(-1.,0.)
        if resource_closure=='lognormal':
            fit=read(OUT/'resource_closure/result.json')
            self.resource_coefficients=(float(fit['variance_a_mv']),float(fit['variance_b']))

    def step(self,nu=None):
        if nu is not None:self.nu.set(nu)
        for op,out in zip(self.ops,self.arrive):
            self.module.get_function('delayed')((800,),(128,),(*op,self.history,out,np.int32(self.tick),np.int32(self.depth)))
        self.module.get_function('synapses')((7,),(128,),(self.qa,self.ia,self.qg,self.ig,*self.arrive,self.nu,
            float(self.prep['nu_ext_per_ms']),DT,1.,float(self.prep['params']['tau_d_GABA'])))
        self.module.get_function('rates')(((self.P+127)//128,),(128,),(self.r,self.aux,self.z,self.m,self.output,
            self.cell,self.theta,np.int32(self.NE),np.int32(self.P),self.ia,self.ig,self.xs,np.int32(len(self.xs)),self.fs,self.cvs,self.pars,
            DT,np.int32(self.kind=='nonlinear'),np.int32(self.D is None),self.ith,5.,*self.resource_coefficients))
        self.module.get_function('restrict_rate')((7,),(128,),(self.ptr,self.groups,self.weights,self.output,self.history,
            np.int32(self.tick),np.int32(self.depth)))
        self.tick+=1
        return self.history[(self.tick-1)%self.depth]
