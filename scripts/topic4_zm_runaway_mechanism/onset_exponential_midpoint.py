"""Parallel numerical integrator of the locked deterministic spatial equations.

Synaptic, covariance and input-memory filters use exponential midpoint drives.
The renewal occupancy is integrated exactly for a constant midpoint hazard and
the delayed release flux in that bin. History entries are bin-average rates.
This module is experimental until its numerical checks pass; it does not
replace the existing engine, refit a response, or certify a bifurcation.
"""
from common import np
from physical_delay_count_rate import PhysicalDelayCountEngine
from transient_response_network import corrected_source
from refractory_rate_cuda import covariance_matrices


def midpoint_local_source(P):
    code = corrected_source(P)
    old = ' const int* clock,int depth,double dt,int streamed_input){'
    assert code.count(old) == 1
    code = code.replace(old, ' const int* clock,int depth,double dt,int streamed_input,double* midpoint_features){')
    old = '  for(int c=0;c<3;c++)for(int j=0;j<4;j++){'
    assert code.count(old) == 1
    code = code.replace(old, '  for(int c=0;c<3;c++)midpoint_features[c*P+g]=f[c];\n'+old)
    begin = code.index('  double occupied=0.;int nref=')
    end = code.index('  rate[g]=r;', begin)
    code = code[:begin]+r'''
  double occupied=0.;int nref=(int)llround(refractory[g]/dt);
  for(int j=1;j<=nref;j++){
   int slot=(tick-j)%depth;if(slot<0)slot+=depth;
   occupied+=own_history[(long long)slot*P+g]*dt;
  }
  int oldslot=(tick-nref)%depth;if(oldslot<0)oldslot+=depth;
  double release=own_history[(long long)oldslot*P+g];
  double l=ell+log(dt/.1),p,release_fraction;
  if(l>40.){p=1.;release_fraction=1.-exp(-l);}
  else{
   double x=exp(l);p=-expm1(-x);
   release_fraction=x<1e-4?x*(.5-x/6.+x*x/24.-x*x*x/120.):1.-p/x;
  }
  double r=(1.-occupied)*p/dt+release*release_fraction;
  if(occupied< -1e-10 || occupied>1.+1e-10 || release<0. || !isfinite(r) || r<0.)r=nan("");
''' + code[end:]
    return code


def auxiliary_source(P):
    return f'#define P {P}\n'+r'''
extern "C" __global__ void midpoint_prepare(const double* syn,const double* local,
 const double* history,const int* clock,int depth,const double* pars,const double* constants,
 double dt,double* mid_syn,double* mid_local){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=P)return;
 for(int j=0;j<6;j++)mid_syn[j*P+g]=syn[j*P+g];
 for(int j=0;j<42;j++)mid_local[j*P+g]=local[j*P+g];
 double r=history[(long long)(clock[0]%depth)*P+g],em=exp(-.5*dt/constants[6]);
 mid_syn[4*P+g]=em*syn[4*P+g]+(1.-em)*.5*pars[3*P+g]*r;
}
extern "C" __global__ void midpoint_finish(double* local,const double* physical,const int* pop,
 const double* coefficient,const double* bank,const double* midpoint_features,
 double* syn,const double* rate,const double* pars,const double* constants,double* emitted,
 double dt){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=P)return;int p=pop[g];
 for(int c=0;c<2;c++){
  const double* a=coefficient+(p*2+c)*13;
  double x[3]={local[(3*c)*P+g],local[(3*c+1)*P+g],local[(3*c+2)*P+g]};
  double raw=physical[(c+1)*P+g];
  for(int j=0;j<3;j++)local[(3*c+j)*P+g]=a[3*j]*x[0]+a[3*j+1]*x[1]+a[3*j+2]*x[2]+a[9+j]*raw;
 }
 for(int c=0;c<3;c++)for(int j=0;j<4;j++){
  const double* a=bank+j*6;int k=6+c*12+j*3;double u=midpoint_features[c*P+g];
  double h1=local[k*P+g],h2=local[(k+1)*P+g],h3=local[(k+2)*P+g];
  local[k*P+g]=a[0]*h1+a[3]*u;
  local[(k+1)*P+g]=a[0]*(h2+a[1]*h1)+a[4]*u;
  local[(k+2)*P+g]=a[0]*(h3+a[1]*h2+a[2]*h1)+a[5]*u;
 }
 double em=exp(-dt/constants[6]),r=rate[g];
 syn[4*P+g]=em*syn[4*P+g]+(1.-em)*.5*pars[3*P+g]*r;
 emitted[g]=r;
}
'''


class ExponentialMidpointEngine(PhysicalDelayCountEngine):
    """Fixed Z / dynamic M / constant input, expected rates only."""
    numerical_method = 'exponential_midpoint_v1'
    def __init__(self, dt=.05, device=0):
        super().__init__(dt=dt, seed=1, device=device, count_sampling=False, constant_input=True)
        cp=self.cp; s=self.s; half=dt/2
        self.transport.pars[19].fill(0); self.transport.pars[20].fill(1)
        self.mid_syn=cp.empty_like(self.syn); self.mid_local=cp.empty_like(self.local.state)
        self.mid_physical=cp.empty_like(self.local.physical); self.mid_features=cp.empty((3,s.P))
        physical=[]
        for tr,td in zip(s.rise,s.decay):
            a=np.exp(-half/tr); d=np.exp(-half/td)
            physical.extend([a,tr/(tr-td)*(a-d),d])
        self.mid_coefficients=cp.asarray(physical)
        covariance=[]
        for pop in 'EI':
            A,b,c,_,_=covariance_matrices(pop,half)
            covariance.extend([np.r_[A[k].ravel(),b[k],c[k]] for k in range(2)])
        self.mid_covariance=cp.asarray(covariance); bank=[]
        for tau in [1.,4.,16.,64.]:
            a=half/tau; decay=np.exp(-a)
            bank.append([decay,a,.5*a*a,1-decay,1-decay*(1+a),1-decay*(1+a+.5*a*a)])
        self.mid_bank=cp.asarray(bank)
        self.mid_response_module=cp.RawModule(code=midpoint_local_source(s.P),options=('--fmad=false',),name_expressions=['local_rate'])
        self.mid_response=self.mid_response_module.get_function('local_rate')
        self.mid_aux_module=cp.RawModule(code=auxiliary_source(s.P),options=('--fmad=false',),name_expressions=['midpoint_prepare','midpoint_finish'])
        self.mid_prepare=self.mid_aux_module.get_function('midpoint_prepare')
        self.mid_finish=self.mid_aux_module.get_function('midpoint_finish')
        assert self.local.history.data.ptr==self.transport.history.data.ptr

    def step(self):
        t=self.transport; l=self.local; n=(self.s.P+127)//128
        self.arrivals()
        self.mid_prepare((n,),(128,),(self.syn,l.state,l.history,l.clock,np.int32(t.depth),
            t.pars,t.consts,self.dt,self.mid_syn,self.mid_local))
        self.k['physical_step']((n,),(128,),(self.mid_syn,t.arr,t.pars,self.mid_coefficients,
            t.drive,np.int32(0),np.int32(t.n_drive),l.clock,self.dt,self.mid_physical))
        self.mid_response((self.s.P,),(64,),(self.mid_local,self.mid_physical,self.syn[5],l.theta,
            l.pop,l.refractory,self.mid_covariance,self.mid_bank,l.network,l.SE,l.SI,l.history,l.rate,
            l.clock,np.int32(t.depth),self.dt,np.int32(0),self.mid_features))
        self.k['physical_step']((n,),(128,),(self.syn,t.arr,t.pars,self.coefficients,
            t.drive,np.int32(0),np.int32(t.n_drive),l.clock,self.dt,l.physical))
        self.mid_finish((n,),(128,),(l.state,l.physical,l.pop,l.coefficients,l.bank,self.mid_features,
            self.syn,l.rate,t.pars,t.consts,self.emitted,self.dt))
        l.advance((1,),(1,),(l.clock,))
        self.k['record']((n,),(128,),(self.emitted,l.rate,self.accumulator,self.output,l.clock,
            self.dt,np.int32(round(1/self.dt)),np.int32(10)))
