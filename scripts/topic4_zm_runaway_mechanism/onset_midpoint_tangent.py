"""Matching variational equations for the experimental midpoint integrator.

No old endpoint-update Jacobian is reused as the derivative of the new flow.
Only unchanged filter/readout derivative algebra is shared. This class must
pass full nonlinear directional differences before a continuation uses it.
"""
from common import np
from onset_tangent_cuda import source as original_source, Tangent as EndpointTangent
from onset_exponential_midpoint import auxiliary_source


def source(P):
    code=original_source(P)
    old=' const double* history,const double* nominal_history,double* rate,const int* clock,int depth,double dt){'
    assert code.count(old)==1
    code=code.replace(old,old[:-2]+',double* midpoint_features){')
    old='  for(int c=0;c<3;c++)for(int j=0;j<4;j++){'
    assert code.count(old)==1
    code=code.replace(old,'  for(int c=0;c<3;c++)midpoint_features[c*P+g]=df[c];\n'+old)
    begin=code.index('  double occupied=0.,doccupied=0.;int nref=')
    end=code.index('\n }\n}\nextern "C" __global__ void tangent_finish',begin)
    code=code[:begin]+r'''
  double occupied=0.,doccupied=0.;int nref=(int)llround(refractory[g]/dt);
  for(int j=1;j<=nref;j++){
   int slot=(tick-j)%depth;if(slot<0)slot+=depth;
   occupied+=nominal_history[(long long)slot*P+g]*dt;
   doccupied+=history[(long long)slot*P+g]*dt;
  }
  int oldslot=(tick-nref)%depth;if(oldslot<0)oldslot+=depth;
  double release=nominal_history[(long long)oldslot*P+g],drelease=history[(long long)oldslot*P+g];
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
  rate[g]=-doccupied*p/dt+drelease*fraction+sensitivity*dell;
''' + code[end:]
    return code


def auxiliary(P):
    code=auxiliary_source(P)
    code=code.replace('for(int j=0;j<6;j++)','for(int j=0;j<5;j++)')
    old=' double dt){';assert code.count(old)==1
    code=code.replace(old,' double dt,double* history,const int* clock,int depth){')
    old=' emitted[g]=r;';assert code.count(old)==1
    return code.replace(old,old+'history[(long long)(clock[0]%depth)*P+g]=r;')


class MidpointTangent(EndpointTangent):
    def __init__(self,e):
        assert getattr(e,'numerical_method',None)=='exponential_midpoint_v1'
        assert not e.noise and not e.transport.drive_on
        assert np.all(e.transport.pars[19].get()==0) and np.all(e.transport.pars[20].get()==1)
        self.e=e;cp=e.cp;P=e.s.P
        self.syn=cp.zeros((5,P));self.local=cp.zeros_like(e.local.state)
        self.history=cp.zeros_like(e.local.history);self.physical=cp.zeros_like(e.local.physical)
        self.arr=cp.zeros_like(e.transport.arr);self.rate=cp.zeros_like(e.local.rate)
        self.mid_syn=cp.empty_like(self.syn);self.mid_local=cp.empty_like(self.local)
        self.mid_physical=cp.empty_like(self.physical);self.mid_features=cp.empty((3,P));self.emitted=cp.empty(P)
        self.module=cp.RawModule(code=source(P),options=('--fmad=false',),name_expressions=['tangent_physical','tangent_local'])
        self.k={n:self.module.get_function(n) for n in ['tangent_physical','tangent_local']}
        self.aux_module=cp.RawModule(code=auxiliary(P),options=('--fmad=false',),name_expressions=['midpoint_prepare','midpoint_finish'])
        self.prepare=self.aux_module.get_function('midpoint_prepare');self.finish=self.aux_module.get_function('midpoint_finish')

    def step(self):
        e=self.e;t=e.transport;l=e.local;n=(e.s.P+127)//128
        e.k['delayed']((e.s.P,),(128,),(*t.ops,self.history,self.arr,l.clock,np.int32(t.depth),np.int32(t.factor)))
        self.prepare((n,),(128,),(self.syn,self.local,self.history,l.clock,np.int32(t.depth),
            t.pars,t.consts,e.dt,self.mid_syn,self.mid_local))
        self.k['tangent_physical']((n,),(128,),(self.mid_syn,self.arr,t.pars,e.mid_coefficients,e.syn[5],self.mid_physical))
        e.step()
        self.k['tangent_local']((e.s.P,),(64,),(self.mid_local,e.mid_local,self.mid_physical,e.mid_physical,e.syn[5],
            l.theta,l.pop,l.refractory,e.mid_covariance,e.mid_bank,l.network,l.SE,l.SI,
            self.history,l.history,self.rate,l.clock,np.int32(t.depth),e.dt,self.mid_features))
        self.k['tangent_physical']((n,),(128,),(self.syn,self.arr,t.pars,e.coefficients,e.syn[5],self.physical))
        self.finish((n,),(128,),(self.local,self.physical,l.pop,l.coefficients,l.bank,self.mid_features,
            self.syn,self.rate,t.pars,t.consts,self.emitted,e.dt,self.history,l.clock,np.int32(t.depth)))
