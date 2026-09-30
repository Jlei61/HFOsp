"""Exact float64 response cache for the matching midpoint variational flow.

The cache evaluates the locked response on actual midpoint states. It retains
the released refractory flux derivative as a separate coefficient. The old
endpoint-update coefficients are not used for this integrator.
"""
from common import np
from onset_cached_tangent import cache_source as endpoint_cache_source
from onset_midpoint_tangent import MidpointTangent
from fine_rate_frozen_Z_fields import capture, restore


def cache_source(P):
    code=endpoint_cache_source(P)
    code=code[:code.index('extern "C" __global__ void tangent_cached_local')]
    code=code.replace(')*44*P',')*45*P')
    begin=code.index('  double occupied=0.;int nref=',code.index('void cache_response'))
    code=code[:begin]+r'''
  double occupied=0.;int nref=(int)llround(refractory[g]/dt);
  for(int j=1;j<=nref;j++){
   int slot=(tick-j)%depth;if(slot<0)slot+=depth;
   occupied+=history[(long long)slot*P+g]*dt;
  }
  int oldslot=(tick-nref)%depth;if(oldslot<0)oldslot+=depth;
  double release=history[(long long)oldslot*P+g];
  double l=ell+log(dt/.1),p,fraction,sensitivity;
  if(l>40.){p=1.;double k=exp(-l);fraction=1.-k;sensitivity=release*k;}
  else{
   double x=exp(l),survival=exp(-x);p=-expm1(-x);
   if(x<1e-4){
    fraction=x*(.5-x/6.+x*x/24.-x*x*x/120.);
    sensitivity=(1.-occupied)*x*survival/dt+release*x*(.5-x/3.+x*x/8.-x*x*x/30.);
   }else{
    double k=p/x;fraction=1.-k;
    sensitivity=(1.-occupied)*x*survival/dt+release*(k-survival);
   }
  }
  c[42*P+g]=-p/dt;c[43*P+g]=sensitivity;c[44*P+g]=fraction;
 }
}
extern "C" __global__ void midpoint_cached_local(double* state,const double* physical,
 const double* Z,const int* pop,const double* refractory,const double* coefficients,
 const double* bank,const double* history,double* rate,const int* clock,int depth,
 double dt,int start,const double* cache,double* midpoint_features){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=P)return;
 int p=pop[g],tick=clock[0];const double* c=cache+(long long)(tick-start-1)*45*P;
 double dv[2];
 for(int channel=0;channel<2;channel++){
  const double* a=coefficients+(p*2+channel)*13;
  double x[3]={state[3*channel*P+g],state[(3*channel+1)*P+g],state[(3*channel+2)*P+g]};
  for(int j=0;j<3;j++)state[(3*channel+j)*P+g]=a[j*3]*x[0]+a[j*3+1]*x[1]+a[j*3+2]*x[2]+a[9+j]*physical[(channel+1)*P+g];
  dv[channel]=a[12]*state[(3*channel+2)*P+g];
 }
 dv[1]*=Z[g]*Z[g];
 double df[3]={c[g]*physical[g],c[P+g]*dv[0],c[2*P+g]*dv[1]};
 for(int channel=0;channel<3;channel++)midpoint_features[channel*P+g]=df[channel];
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
 for(int j=1;j<=nref;j++){
  int slot=(tick-j)%depth;if(slot<0)slot+=depth;
  occupied+=history[(long long)slot*P+g]*dt;
 }
 int oldslot=(tick-nref)%depth;if(oldslot<0)oldslot+=depth;
 double release=history[(long long)oldslot*P+g];
 rate[g]=c[42*P+g]*occupied+c[43*P+g]*dell+c[44*P+g]*release;
}
'''
    return code


class MidpointCachedTangent(MidpointTangent):
    def __init__(self,e,steps,cache_buffer=None):
        super().__init__(e);cp=e.cp;l=e.local;t=e.transport
        self.base=capture(e);self.start=int(self.base['clock'][0]);self.steps=steps
        required=steps*45*e.s.P*8;free,_=cp.cuda.runtime.memGetInfo()
        if cache_buffer is None:
            assert required<.8*free,('Insufficient free memory',required,free)
            self.cache=cp.empty((steps,45,e.s.P),dtype='f8')
        else:
            assert cache_buffer.shape[0]>=steps and cache_buffer.shape[1:]==(45,e.s.P)
            assert cache_buffer.dtype==cp.float64
            self.cache=cache_buffer[:steps]
        names=['cache_response','midpoint_cached_local']
        self.cache_module=cp.RawModule(code=cache_source(e.s.P),options=('--fmad=false',),name_expressions=names)
        self.cache_kernel=self.cache_module.get_function(names[0]);self.cached_local=self.cache_module.get_function(names[1])
        def nominal():
            e.step()
            self.cache_kernel((e.s.P,),(64,),(e.mid_local,e.mid_physical,e.syn[5],l.theta,l.pop,l.refractory,
                e.mid_covariance,l.network,l.SE,l.SI,l.history,l.clock,np.int32(t.depth),e.dt,np.int32(self.start),self.cache))
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
        self.nominal_terminal=capture(e);restore(e,self.base);self.cache_bytes=required

    def step(self):
        e=self.e;t=e.transport;l=e.local;n=(e.s.P+127)//128
        e.k['delayed']((e.s.P,),(128,),(*t.ops,self.history,self.arr,l.clock,np.int32(t.depth),np.int32(t.factor)))
        self.prepare((n,),(128,),(self.syn,self.local,self.history,l.clock,np.int32(t.depth),
            t.pars,t.consts,e.dt,self.mid_syn,self.mid_local))
        self.k['tangent_physical']((n,),(128,),(self.mid_syn,self.arr,t.pars,e.mid_coefficients,e.syn[5],self.mid_physical))
        l.advance((1,),(1,),(l.clock,))
        self.cached_local((n,),(128,),(self.mid_local,self.mid_physical,e.syn[5],l.pop,l.refractory,
            e.mid_covariance,e.mid_bank,self.history,self.rate,l.clock,np.int32(t.depth),e.dt,np.int32(self.start),self.cache,self.mid_features))
        self.k['tangent_physical']((n,),(128,),(self.syn,self.arr,t.pars,e.coefficients,e.syn[5],self.physical))
        self.finish((n,),(128,),(self.local,self.physical,l.pop,l.coefficients,l.bank,self.mid_features,
            self.syn,self.rate,t.pars,t.consts,self.emitted,e.dt,self.history,l.clock,np.int32(t.depth)))
