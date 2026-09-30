"""Reuse exact local Jacobians along one fixed nominal delayed trajectory.

All cached numbers are float64 derivatives of the existing response at each
actual time step. The full spatial delayed variational state is retained.
This is a derivative implementation, never a different physical rate model.
"""
from common import OUT,np,read
from onset_tangent_cuda import Tangent,source
from fine_rate_frozen_Z_fields import capture,restore


def cache_source(P):
    return source(P)+r'''
extern "C" __global__ void cache_response(const double* state,const double* physical,
 const double* Z,const double* theta,const int* pop,const double* refractory,
 const double* coefficients,const double* weights,const double* SE,const double* SI,
 const double* history,const int* clock,int depth,double dt,int start,double* cache){
 int g=blockIdx.x,lane=threadIdx.x,p=pop[g],tick=clock[0];
 const double* w=weights+p*6785;
 double* c=cache+(long long)(tick-start-1)*44*P;
 __shared__ double f[39],h1[64],h2[64],b2[64],b1[64],fg[39],basegrad[3],baseline;
 if(lane==0){
  double mu=physical[g],v[2];
  for(int j=0;j<2;j++)v[j]=coefficients[(p*2+j)*13+12]*state[(3*j+2)*P+g];
  v[1]*=Z[g]*Z[g];double scale=theta[g]-11.,u=(mu-11.)/scale;
  f[0]=asinh(u)/3.;f[1]=log1p(v[0]/(scale*scale))/2.;f[2]=log1p(v[1]/(scale*scale))/2.;
  c[g]=1./(3.*scale*sqrt(1.+u*u));
  c[P+g]=1./(2.*(scale*scale+v[0]));c[2*P+g]=1./(2.*(scale*scale+v[1]));
  for(int j=0;j<36;j++)f[3+j]=state[(6+j)*P+g]-f[j/12];
  double r,d1,d2,d3;phi_spline(p==0?SE:SI,mu,v[0],v[1],theta[g],&r,&d1,&d2,&d3);
  double maximum=1./refractory[g],a=r/maximum,logden=log1p(pow(a,16.));
  double capped=a*exp(-logden/16.),probability=1e-8+(1.-2e-8)*capped;
  double q=maximum*probability*.1,available=1.-(refractory[g]/.1-1.)*q,pr=q/available;
  baseline=log(pr)-log1p(-pr);
  double factor=.1*(1.-2e-8)*exp(-17.*logden/16.)/(q*(1.-refractory[g]/.1*q));
  basegrad[0]=factor*d1;basegrad[1]=factor*d2;basegrad[2]=factor*d3;
 }
 __syncthreads();
 double a=w[2496+lane];for(int j=0;j<39;j++)a+=w[lane*39+j]*f[j];h1[lane]=tanh(a);
 __syncthreads();
 a=w[6656+lane];for(int j=0;j<64;j++)a+=w[2560+lane*64+j]*h1[j];h2[lane]=tanh(a);
 b2[lane]=w[6720+lane]*(1.-h2[lane]*h2[lane]);
 __syncthreads();
 double v=0.;for(int j=0;j<64;j++)v+=b2[j]*w[2560+j*64+lane];
 b1[lane]=(1.-h1[lane]*h1[lane])*v;
 __syncthreads();
 if(lane<39){v=0.;for(int j=0;j<64;j++)v+=b1[j]*w[j*39+lane];fg[lane]=v;}
 __syncthreads();
 if(lane==0){
  double ell=baseline+w[6784];for(int j=0;j<64;j++)ell+=w[6720+j]*h2[j];
  double H=0.;for(int j=3;j<39;j++)H+=f[j]*f[j];H/=36.;
  double b=p==0?JT_BE:JT_BI,h=p==0?JT_HE:JT_HI;
  ell+=b*H/(H+h);
  for(int j=3;j<39;j++)fg[j]+=b*h*2.*f[j]/(36.*(H+h)*(H+h));
  for(int channel=0;channel<3;channel++){
   double grad=fg[channel];
   for(int j=0;j<12;j++)grad-=fg[3+channel*12+j];
   c[(3+channel)*P+g]=basegrad[channel]+grad*c[channel*P+g];
  }
  for(int j=0;j<36;j++)c[(6+j)*P+g]=fg[3+j];
  double occupied=0.;int nref=(int)llround(refractory[g]/dt);
  for(int j=1;j<nref;j++){int slot=(tick-j)%depth;if(slot<0)slot+=depth;occupied+=history[(long long)slot*P+g]*dt;}
  double l=ell+log(dt/.1),pr=l>=0?1./(1.+exp(-l)):exp(l)/(1.+exp(l));
  c[42*P+g]=-pr/dt;c[43*P+g]=(1.-occupied)*pr*(1.-pr)/dt;
 }
}
extern "C" __global__ void tangent_cached_local(double* state,const double* physical,
 const double* Z,const int* pop,const double* refractory,const double* coefficients,
 const double* bank,const double* history,double* rate,const int* clock,int depth,
 double dt,int start,const double* cache){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=P)return;
 int p=pop[g],tick=clock[0];const double* c=cache+(long long)(tick-start-1)*44*P;
 double dv[2];
 for(int channel=0;channel<2;channel++){
  const double* a=coefficients+(p*2+channel)*13;
  double x[3]={state[3*channel*P+g],state[(3*channel+1)*P+g],state[(3*channel+2)*P+g]};
  for(int j=0;j<3;j++)state[(3*channel+j)*P+g]=a[j*3]*x[0]+a[j*3+1]*x[1]+a[j*3+2]*x[2]+a[9+j]*physical[(channel+1)*P+g];
  dv[channel]=a[12]*state[(3*channel+2)*P+g];
 }
 dv[1]*=Z[g]*Z[g];
 double df[3]={c[g]*physical[g],c[P+g]*dv[0],c[2*P+g]*dv[1]};
 for(int channel=0;channel<3;channel++)for(int j=0;j<4;j++){
  const double* a=bank+j*6;int k=6+channel*12+j*3;
  double h1=state[k*P+g],h2=state[(k+1)*P+g],h3=state[(k+2)*P+g],u=df[channel];
  state[k*P+g]=a[0]*h1+a[3]*u;
  state[(k+1)*P+g]=a[0]*(h2+a[1]*h1)+a[4]*u;
  state[(k+2)*P+g]=a[0]*(h3+a[1]*h2+a[2]*h1)+a[5]*u;
 }
 double dell=c[3*P+g]*physical[g]+c[4*P+g]*dv[0]+c[5*P+g]*dv[1];
 for(int j=0;j<36;j++)dell+=c[(6+j)*P+g]*state[(6+j)*P+g];
 double occupied=0.;int nref=(int)llround(refractory[g]/dt);
 for(int j=1;j<nref;j++){int slot=(tick-j)%depth;if(slot<0)slot+=depth;occupied+=history[(long long)slot*P+g]*dt;}
 rate[g]=c[42*P+g]*occupied+c[43*P+g]*dell;
}
'''


class CachedTangent(Tangent):
    def __init__(self,e,steps,cache_buffer=None):
        super().__init__(e);cp=e.cp;l=e.local;t=e.transport
        self.base=capture(e);self.start=int(self.base['clock'][0]);self.steps=steps
        required=steps*44*e.s.P*8;free,total=cp.cuda.runtime.memGetInfo()
        if cache_buffer is None:
            assert required<.8*free,('Insufficient free memory for exact derivative cache',required,free)
            self.cache=cp.empty((steps,44,e.s.P),dtype='f8')
        else:
            assert cache_buffer.shape[0]>=steps and cache_buffer.shape[1:]==(44,e.s.P) and cache_buffer.dtype==cp.float64
            self.cache=cache_buffer[:steps]
        names=['cache_response','tangent_cached_local']
        self.cache_module=cp.RawModule(code=cache_source(e.s.P),options=('--fmad=false',),name_expressions=names)
        self.cache_kernel=self.cache_module.get_function(names[0]);self.cached_local=self.cache_module.get_function(names[1])
        def nominal():
            e.step()
            self.cache_kernel((e.s.P,),(64,),(l.state,l.physical,e.syn[5],l.theta,l.pop,l.refractory,
                l.coefficients,l.network,l.SE,l.SI,l.history,l.clock,np.int32(t.depth),e.dt,np.int32(self.start),self.cache))
        nominal();cp.cuda.get_current_stream().synchronize();restore(e,self.base)
        n=round(10/e.dt);whole,tail=divmod(steps,n)
        stream=cp.cuda.Stream(non_blocking=True)
        with stream:
            stream.begin_capture()
            for _ in range(n):nominal()
            graph=stream.end_capture()
        for _ in range(whole):graph.launch(stream);stream.synchronize()
        for _ in range(tail):nominal()
        cp.cuda.get_current_stream().synchronize()
        self.nominal_terminal=capture(e);restore(e,self.base)
        self.cache_bytes=required

    def step(self):
        e=self.e;t=e.transport;l=e.local;n=(e.s.P+127)//128
        e.k['delayed']((e.s.P,),(128,),(*t.ops,self.history,self.arr,l.clock,np.int32(t.depth),np.int32(t.factor)))
        self.k['tangent_physical']((n,),(128,),(self.syn,self.arr,t.pars,e.coefficients,e.syn[5],self.physical))
        l.advance((1,),(1,),(l.clock,))
        self.cached_local((n,),(128,),(self.local,self.physical,e.syn[5],l.pop,l.refractory,
            l.coefficients,l.bank,self.history,self.rate,l.clock,np.int32(t.depth),e.dt,np.int32(self.start),self.cache))
        self.k['tangent_finish']((n,),(128,),(self.syn,self.rate,t.pars,t.consts,self.history,l.clock,np.int32(t.depth),e.dt))


class CachedCubicSectionDerivative:
    def __init__(self,A,x,T,slope):
        from onset_cubic_section import weights
        self.A=A;self.x=x.copy();self.T=T;self.slope=slope.copy();self.normal=A.normal
        self.den=float(self.normal@self.slope);assert self.den>0
        e=A.e;restore(e,A.state(x));n=int(np.floor(T/e.dt));self.alpha=T/e.dt-n
        self.t=CachedTangent(e,n+2);self.t.graph()
        self.whole=(n-1)//round(10/e.dt);tail=(n-1)%round(10/e.dt)
        with self.t.stream:
            self.t.stream.begin_capture()
            for _ in range(tail):self.t.step()
            self.tail=self.t.stream.end_capture()
        self.calls=0

    def __call__(self,v):
        from onset_cubic_section import weights
        A=self.A;e=A.e;t=self.t;c=A.c
        restore(e,A.state(self.x));c.set_tangent(t,v)
        for _ in range(self.whole):t.chunk()
        self.tail.launch(t.stream);t.stream.synchronize();points=[c.tangent(t)]
        for _ in range(3):
            t.step();e.cp.cuda.get_current_stream().synchronize();points.append(c.tangent(t))
        fixed=sum(a*q for a,q in zip(weights(self.alpha),points))
        self.last_return_time_derivative=-float(self.normal@fixed)/self.den;self.calls+=1
        return fixed+self.slope*self.last_return_time_derivative
