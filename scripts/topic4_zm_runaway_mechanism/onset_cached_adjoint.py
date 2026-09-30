"""Exact transpose of the cached full spatial delayed tangent map.

This is a derivative operator for the unchanged conditional rate flow. No
adjoint state feeds into the nominal model. The forward local derivatives
and all original delayed graph weights are reused, not fitted or averaged.
"""
from common import np


def source(P):
    return f'#define P {P}\n'+r'''
extern "C" __global__ void reverse_local(double* syn,double* state,double* history,
 double* arr,const double* Z,const int* pop,const double* refractory,
 const double* coefficients,const double* bank,const double* pars,
 const double* syn_coeff,const double* constants,const double* cache,
 const int* clock,int start,int depth,double dt){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=P)return;
 int tick=clock[0],p=pop[g],slot=tick%depth;
 const double* c=cache+(long long)(tick-start-1)*44*P;
 double em=exp(-dt/constants[6]);
 double br=history[(long long)slot*P+g]+(1.-em)*.5*pars[3*P+g]*syn[4*P+g];
 history[(long long)slot*P+g]=0.;
 double bm=em*syn[4*P+g],bell=c[43*P+g]*br,bocc=c[42*P+g]*br;
 int nr=(int)llround(refractory[g]/dt);
 for(int j=1;j<nr;j++){
  int index=(tick-j)%depth;if(index<0)index+=depth;
  history[(long long)index*P+g]+=dt*bocc;
 }
 double bp=c[3*P+g]*bell,bv[2]={c[4*P+g]*bell,c[5*P+g]*bell};
 double bdf[3]={0.,0.,0.};
 for(int ch=0;ch<3;ch++)for(int j=0;j<4;j++){
  int k=6+ch*12+j*3;const double* a=bank+j*6;
  double b1=state[k*P+g]+c[(6+ch*12+j*3)*P+g]*bell;
  double b2=state[(k+1)*P+g]+c[(7+ch*12+j*3)*P+g]*bell;
  double b3=state[(k+2)*P+g]+c[(8+ch*12+j*3)*P+g]*bell;
  bdf[ch]+=a[3]*b1+a[4]*b2+a[5]*b3;
  state[k*P+g]=a[0]*(b1+a[1]*b2+a[2]*b3);
  state[(k+1)*P+g]=a[0]*(b2+a[1]*b3);
  state[(k+2)*P+g]=a[0]*b3;
 }
 bp+=c[g]*bdf[0];bv[0]+=c[P+g]*bdf[1];bv[1]+=c[2*P+g]*bdf[2];
 double bvar[2];
 for(int ch=0;ch<2;ch++){
  const double* a=coefficients+(p*2+ch)*13;
  double b[3]={state[3*ch*P+g],state[(3*ch+1)*P+g],state[(3*ch+2)*P+g]};
  b[2]+=a[12]*bv[ch]*(ch==1?Z[g]*Z[g]:1.);
  bvar[ch]=a[9]*b[0]+a[10]*b[1]+a[11]*b[2];
  for(int j=0;j<3;j++)state[(3*ch+j)*P+g]=a[j]*b[0]+a[3+j]*b[1]+a[6+j]*b[2];
 }
 syn[P+g]+=bp;syn[3*P+g]-=Z[g]*bp;syn[4*P+g]=bm-bp;
 double tm=pars[g],areaA=pars[4*P+g],areaG=pars[5*P+g];
 arr[2*P+g]=tm*areaA*areaA*bvar[0];arr[3*P+g]=tm*areaG*areaG*bvar[1];
 for(int ch=0;ch<2;ch++){
  double a=syn_coeff[3*ch],b=syn_coeff[3*ch+1],d=syn_coeff[3*ch+2];
  double bq=syn[2*ch*P+g],bi=syn[(2*ch+1)*P+g];
  arr[ch*P+g]=tm*(ch==0?areaA:areaG)*((1.-a)*bq+(1.-d-b)*bi);
  syn[2*ch*P+g]=a*bq+b*bi;syn[(2*ch+1)*P+g]=d*bi;
 }
}
extern "C" __global__ void reverse_delayed(const int* pa,const int* ca,const double* wa,const double* va,
 const int* pb,const int* cb,const double* wb,const double* vb,double* history,const double* arr,
 const int* clock,int depth,int factor){
 int g=blockIdx.x,lane=threadIdx.x,tick=clock[0];
 for(int i=pa[g]+lane;i<pa[g+1];i+=blockDim.x){
  int delay=(ca[i]/P+1)*factor,slot=(tick-delay)%depth;if(slot<0)slot+=depth;
  atomicAdd(history+(long long)slot*P+ca[i]%P,wa[i]*arr[g]+va[i]*arr[2*P+g]);
 }
 for(int i=pb[g]+lane;i<pb[g+1];i+=blockDim.x){
  int delay=(cb[i]/P+1)*factor,slot=(tick-delay)%depth;if(slot<0)slot+=depth;
  atomicAdd(history+(long long)slot*P+cb[i]%P,wb[i]*arr[P+g]+vb[i]*arr[3*P+g]);
 }
}
extern "C" __global__ void decrement_clock(int* clock){if(threadIdx.x==0 && blockIdx.x==0)clock[0]--;}
'''


class CachedAdjoint:
    def __init__(self,forward):
        self.forward=forward;self.e=e=forward.e;cp=e.cp
        assert getattr(e,'numerical_method','old_endpoint')=='old_endpoint', 'Endpoint adjoint is not the derivative of the midpoint integrator'
        self.syn=cp.zeros_like(forward.syn);self.local=cp.zeros_like(forward.local)
        self.history=cp.zeros_like(forward.history);self.arr=cp.zeros_like(forward.arr)
        self.clock=cp.asarray(np.array([forward.start+forward.steps],dtype=np.int32))
        names=['reverse_local','reverse_delayed','decrement_clock']
        self.module=cp.RawModule(code=source(e.s.P),options=('--fmad=false',),name_expressions=names)
        self.k={name:self.module.get_function(name) for name in names}

    def set_covector(self,c,v,tick):
        a=v.reshape(-1,c.P)*c.weight/c.scale
        self.syn[:]=self.e.cp.asarray(a[:5]);self.local[:]=self.e.cp.asarray(a[5:47])
        history=np.empty_like(a[47:]);history[(tick-np.arange(c.depth))%c.depth]=a[47:]
        self.history[:]=self.e.cp.asarray(history);self.clock[0]=tick
        self.e.cp.cuda.get_current_stream().synchronize()

    def covector(self,c):
        tick=int(self.clock.get()[0]);history=self.history.get()
        a=np.concatenate([self.syn.get(),self.local.get(),history[(tick-np.arange(c.depth))%c.depth]])
        a[4,~self.e.s.E]=0.
        return (a*c.scale/c.weight).ravel()

    def add_covector(self,c,v):
        tick=int(self.clock.get()[0]);a=v.reshape(-1,c.P)*c.weight/c.scale
        self.syn+=self.e.cp.asarray(a[:5]);self.local+=self.e.cp.asarray(a[5:47])
        history=np.empty_like(a[47:]);history[(tick-np.arange(c.depth))%c.depth]=a[47:]
        self.history+=self.e.cp.asarray(history);self.e.cp.cuda.get_current_stream().synchronize()

    def step(self):
        e=self.e;l=e.local;t=e.transport;n=(e.s.P+127)//128
        self.k['reverse_local']((n,),(128,),(self.syn,self.local,self.history,self.arr,e.syn[5],l.pop,l.refractory,
            l.coefficients,l.bank,t.pars,e.coefficients,t.consts,self.forward.cache,
            self.clock,np.int32(self.forward.start),np.int32(t.depth),e.dt))
        self.k['reverse_delayed']((e.s.P,),(128,),(*t.ops,self.history,self.arr,self.clock,np.int32(t.depth),np.int32(t.factor)))
        self.k['decrement_clock']((1,),(1,),(self.clock,))

    def graph(self,ms=10):
        cp=self.e.cp
        # Compile on zero adjoint arrays, then restore its independent clock.
        tick=int(self.clock.get()[0]);self.step();cp.cuda.get_current_stream().synchronize();self.clock[0]=tick
        self.stream=cp.cuda.Stream(non_blocking=True)
        with self.stream:
            self.stream.begin_capture()
            for _ in range(round(ms/self.e.dt)):self.step()
            self.graph_object=self.stream.end_capture()

    def chunk(self):
        self.graph_object.launch(self.stream);self.stream.synchronize()


class CachedCubicSectionAdjoint:
    """Transpose includes cubic endpoint injection and return-time derivative."""
    def __init__(self,derivative):
        self.J=derivative;self.A=derivative.A
        self.reverse=CachedAdjoint(derivative.t);self.reverse.graph()
        n=int(np.floor(derivative.T/self.A.e.dt));self.n=n
        self.whole,self.tail=divmod(n-1,round(10/self.A.e.dt))

    def __call__(self,w):
        from onset_cubic_section import weights
        J=self.J;A=self.A;c=A.c;e=A.e;reverse=self.reverse
        # DP=(I-flow_slope*normal.T/section_speed)*D(Phi_T).
        q=w-J.normal*float(J.slope@w)/J.den
        alpha=weights(J.alpha)
        reverse.set_covector(c,alpha[3]*q,c.tick+self.n+2)
        for j in [2,1,0]:
            reverse.step();e.cp.cuda.get_current_stream().synchronize()
            reverse.add_covector(c,alpha[j]*q)
        for _ in range(self.whole):reverse.chunk()
        for _ in range(self.tail):reverse.step()
        e.cp.cuda.get_current_stream().synchronize()
        assert int(reverse.clock.get()[0])==c.tick
        return reverse.covector(c)
