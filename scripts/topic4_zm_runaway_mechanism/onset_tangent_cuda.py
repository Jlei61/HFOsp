"""Variational flow of the actual fixed-Z, dynamic-M conditional drift.

This tangent state never feeds back into the nominal rate model. All local
response and transport parameters are read from the existing engine. No
stationary or frozen-v3 derivative is substituted for the transient response.
"""
from common import np, read, write, log, OUT
from onset_state_continuation import DEST, build
from fine_rate_frozen_Z_fields import capture, restore
from transfer_spline import CUDA_DEVICE
import argparse


def source(P):
    p=read(OUT/'transient_response_correction/locked.json')['parameters']
    pre=f'#define P {P}\n#define JT_BE {p["E"]["b"]:.17g}\n#define JT_BI {p["I"]["b"]:.17g}\n#define JT_HE {p["E"]["h"]:.17g}\n#define JT_HI {p["I"]["h"]:.17g}\n'
    return pre+CUDA_DEVICE+r'''
extern "C" __global__ void tangent_physical(double* syn,const double* arr,const double* pars,
 const double* coefficient,const double* Z,double* physical){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=P)return;
 double tm=pars[g],aA=pars[4*P+g],aG=pars[5*P+g];
 for(int c=0;c<2;c++){
  double a=coefficient[3*c],b=coefficient[3*c+1],d=coefficient[3*c+2];
  double force=tm*(c==0?aA:aG)*arr[c*P+g],q=syn[2*c*P+g],I=syn[(2*c+1)*P+g];
  syn[2*c*P+g]=a*q+(1.-a)*force;
  syn[(2*c+1)*P+g]=b*q+d*I+(1.-d-b)*force;
 }
 physical[g]=syn[P+g]-Z[g]*syn[3*P+g]-syn[4*P+g];
 physical[P+g]=tm*aA*aA*arr[2*P+g];
 physical[2*P+g]=tm*aG*aG*arr[3*P+g];
}
extern "C" __global__ void tangent_local(double* state,const double* nominal_state,
 const double* physical,const double* nominal_physical,const double* Z,
 const double* theta,const int* pop,const double* refractory,const double* coefficients,
 const double* bank,const double* weights,const double* SE,const double* SI,
 const double* history,const double* nominal_history,double* rate,const int* clock,int depth,double dt){
 int g=blockIdx.x,lane=threadIdx.x,p=pop[g],tick=clock[0];
 const double* w=weights+p*6785;
 __shared__ double f[39],df[39],hidden[64],dhidden[64],hidden2[64],dhidden2[64],baseline,dbaseline;
 if(lane==0){
  double mu=nominal_physical[g],dmu=physical[g],v[2],dv[2];
  for(int c=0;c<2;c++){
   const double* a=coefficients+(p*2+c)*13;
   double x[3]={state[3*c*P+g],state[(3*c+1)*P+g],state[(3*c+2)*P+g]};
   for(int j=0;j<3;j++)state[(3*c+j)*P+g]=a[j*3]*x[0]+a[j*3+1]*x[1]+a[j*3+2]*x[2]+a[9+j]*physical[(c+1)*P+g];
   v[c]=a[12]*nominal_state[(3*c+2)*P+g];dv[c]=a[12]*state[(3*c+2)*P+g];
  }
  v[1]*=Z[g]*Z[g];dv[1]*=Z[g]*Z[g];double scale=theta[g]-11.,u=(mu-11.)/scale;
  f[0]=asinh(u)/3.;df[0]=dmu/(3.*scale*sqrt(1.+u*u));
  f[1]=log1p(v[0]/(scale*scale))/2.;df[1]=dv[0]/(2.*(scale*scale+v[0]));
  f[2]=log1p(v[1]/(scale*scale))/2.;df[2]=dv[1]/(2.*(scale*scale+v[1]));
  for(int c=0;c<3;c++)for(int j=0;j<4;j++){
   const double* a=bank+j*6;int k=6+c*12+j*3;
   double h1=state[k*P+g],h2=state[(k+1)*P+g],h3=state[(k+2)*P+g],u=df[c];
   state[k*P+g]=a[0]*h1+a[3]*u;
   state[(k+1)*P+g]=a[0]*(h2+a[1]*h1)+a[4]*u;
   state[(k+2)*P+g]=a[0]*(h3+a[1]*h2+a[2]*h1)+a[5]*u;
   for(int l=0;l<3;l++){
    f[3+c*12+j*3+l]=nominal_state[(k+l)*P+g]-f[c];
    df[3+c*12+j*3+l]=state[(k+l)*P+g]-u;
   }
  }
  double r,d1,d2,d3;phi_spline(p==0?SE:SI,mu,v[0],v[1],theta[g],&r,&d1,&d2,&d3);
  double dr=d1*dmu+d2*dv[0]+d3*dv[1],maximum=1./refractory[g],a=r/maximum;
  double logden=log1p(pow(a,16.)),capped=a*exp(-logden/16.);
  double probability=1e-8+(1.-2e-8)*capped;
  double q=maximum*probability*.1,available=1.-(refractory[g]/.1-1.)*q,pr=q/available;
  baseline=log(pr)-log1p(-pr);
  double dq=.1*(1.-2e-8)*exp(-17.*logden/16.)*dr;
  dbaseline=dq/(q*(1.-refractory[g]/.1*q));
 }
 __syncthreads();
 double x=w[2496+lane],dx=0.;
 for(int j=0;j<39;j++){x+=w[lane*39+j]*f[j];dx+=w[lane*39+j]*df[j];}
 hidden[lane]=tanh(x);dhidden[lane]=(1.-hidden[lane]*hidden[lane])*dx;
 __syncthreads();
 x=w[6656+lane];dx=0.;
 for(int j=0;j<64;j++){x+=w[2560+lane*64+j]*hidden[j];dx+=w[2560+lane*64+j]*dhidden[j];}
 hidden2[lane]=tanh(x);dhidden2[lane]=(1.-hidden2[lane]*hidden2[lane])*dx;
 __syncthreads();
 if(lane==0){
  double ell=baseline+w[6784],dell=dbaseline;
  for(int j=0;j<64;j++){ell+=w[6720+j]*hidden2[j];dell+=w[6720+j]*dhidden2[j];}
  double H=0.,dH=0.;for(int j=3;j<39;j++){H+=f[j]*f[j];dH+=2.*f[j]*df[j];}H/=36.;dH/=36.;
  double b=p==0?JT_BE:JT_BI,h=p==0?JT_HE:JT_HI;
  ell+=b*H/(H+h);dell+=b*h*dH/((H+h)*(H+h));
  double occupied=0.,doccupied=0.;int nref=(int)llround(refractory[g]/dt);
  for(int j=1;j<nref;j++){
   int slot=(tick-j)%depth;if(slot<0)slot+=depth;
   occupied+=nominal_history[(long long)slot*P+g]*dt;
   doccupied+=history[(long long)slot*P+g]*dt;
  }
  double l=ell+log(dt/.1),pr=l>=0?1./(1.+exp(-l)):exp(l)/(1.+exp(l));
  rate[g]=(-doccupied*pr+(1.-occupied)*pr*(1.-pr)*dell)/dt;
 }
}
extern "C" __global__ void tangent_finish(double* syn,const double* rate,const double* pars,
 const double* constants,double* history,const int* clock,int depth,double dt){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=P)return;
 double em=exp(-dt/constants[6]);
 syn[4*P+g]=em*syn[4*P+g]+(1.-em)*.5*pars[3*P+g]*rate[g];
 history[(long long)(clock[0]%depth)*P+g]=rate[g];
}
'''


class Tangent:
    def __init__(self,e):
        assert getattr(e,'numerical_method','old_endpoint')=='old_endpoint', 'The endpoint-update tangent must not be used with another time integrator; validate its matching variational equations first.'
        self.e=e;cp=e.cp;P=e.s.P
        assert not e.noise and not e.transport.drive_on
        assert np.all(e.transport.pars[19].get()==0) and np.all(e.transport.pars[20].get()==1)
        self.syn=cp.zeros((5,P));self.local=cp.zeros_like(e.local.state)
        self.history=cp.zeros_like(e.local.history);self.physical=cp.zeros_like(e.local.physical)
        self.arr=cp.zeros_like(e.transport.arr);self.rate=cp.zeros_like(e.local.rate)
        names=['tangent_physical','tangent_local','tangent_finish']
        self.module=cp.RawModule(code=source(P),options=('--fmad=false',),name_expressions=names)
        self.k={n:self.module.get_function(n) for n in names}

    def reset(self):
        for a in [self.syn,self.local,self.history,self.physical,self.arr,self.rate]:a.fill(0)

    def step(self):
        e=self.e;t=e.transport;l=e.local;n=(e.s.P+127)//128
        e.k['delayed']((e.s.P,),(128,),(*t.ops,self.history,self.arr,l.clock,np.int32(t.depth),np.int32(t.factor)))
        self.k['tangent_physical']((n,),(128,),(self.syn,self.arr,t.pars,e.coefficients,e.syn[5],self.physical))
        e.step()
        self.k['tangent_local']((e.s.P,),(64,),(self.local,l.state,self.physical,l.physical,e.syn[5],
            l.theta,l.pop,l.refractory,l.coefficients,l.bank,l.network,l.SE,l.SI,self.history,l.history,
            self.rate,l.clock,np.int32(t.depth),e.dt))
        self.k['tangent_finish']((n,),(128,),(self.syn,self.rate,t.pars,t.consts,self.history,l.clock,np.int32(t.depth),e.dt))

    def graph(self,ms=10):
        # Compile without retaining a warm-up mutation.
        state=capture(self.e);self.step();self.e.cp.cuda.get_current_stream().synchronize()
        restore(self.e,state);self.reset();self.e.cp.cuda.get_current_stream().synchronize()
        stream=self.e.cp.cuda.Stream(non_blocking=True)
        with stream:
            stream.begin_capture()
            for _ in range(round(ms/self.e.dt)):self.step()
            graph=stream.end_capture()
        self.stream=stream;self.graph_object=graph

    def chunk(self):
        self.graph_object.launch(self.stream);self.stream.synchronize()


def check(device):
    folder=DEST/'tangent_implementation';folder.mkdir(exist_ok=True)
    assert not (folder/'result.json').exists()
    e=build(device);base={k:v for k,v in np.load(DEST/'lower_endpoint/final_state.npz').items()}
    restore(e,base);t=Tangent(e);t.graph()
    restore(e,base);nominal=e.chunk();terminal=capture(e)
    restore(e,base);t.reset();t.chunk();actual=capture(e)
    assert all(np.array_equal(v,actual[k]) for k,v in terminal.items())
    assert np.array_equal(nominal,actual['output'])
    rng=np.random.default_rng(92316)
    direction=rng.normal(size=(5,e.s.P))*np.maximum(abs(base['syn'][:5]),1e-4)
    direction[4,~e.s.E]=0
    restore(e,base);t.reset();t.syn[:]=e.cp.asarray(direction)
    e.cp.cuda.get_current_stream().synchronize();t.chunk()
    tangent=dict(syn=t.syn.get(),local=t.local.get(),history=t.history.get())
    rows=[]
    for eps in [1e-4,5e-5,2.5e-5]:
        states=[]
        for sign in [-1,1]:
            restore(e,base);e.syn[:5]+=sign*eps*e.cp.asarray(direction)
            e.cp.cuda.get_current_stream().synchronize();e.chunk();states.append(capture(e))
        result={}
        for key,v in tangent.items():
            fd=(states[1][key]-states[0][key])/(2*eps)
            if key=='syn':fd=fd[:5]
            result[key]=float(np.linalg.norm(fd-v)/max(np.linalg.norm(v),1e-12))
        rows.append(dict(epsilon=eps,relative_errors=result))
        log('ONSET TANGENT FINITE DIFFERENCE',eps,result)
    passed=max(rows[-1]['relative_errors'].values())<1e-4
    write(folder/'result.json',dict(status='PASS' if passed else 'FAIL',
        exact_nominal10ms_preserved=True,dt_ms=e.dt,rows=rows,
        scope='Variational implementation only; no Floquet spectrum or bifurcation type. Initial direction perturbs synaptic currents and M; all transported/history tangent states are produced by the actual flow.'))
    assert passed


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);a=p.parse_args();check(a.device)
