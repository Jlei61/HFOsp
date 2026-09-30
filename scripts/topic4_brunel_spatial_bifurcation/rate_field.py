"""Autonomous nonlinear spatial rate DDE sharing SpatialBrunel equilibria.

This is an explicit NEW dynamic closure, not the old Eq36 susceptibility.
Two positive relaxation filters approximate the calibrated local mean response.
Their constants are frozen at J=.94. All spectra and trajectories below use
this same closure. No spikes, voltage particles, or native trajectory forcing.
"""
from common import *
from model import SpatialBrunel
from scipy.optimize import least_squares

RATE_OUT=ROOT/'results/topic4_sef_hfo/interictal_spatial_rate_only_20260917'


def calibrate():
    from response import susceptibility
    s=SpatialBrunel(response='calibrated_full');r,ok,_=s.solve(.94);assert ok
    gains=s.gains(s.moments(r,.94));f=np.array([2.,4.,6.,10.,20.,40.,80.])
    response=np.array([susceptibility(s,r,.94,2j*np.pi*x/1000) for x in f])
    train=np.isin(f,[2,4,10,20,80]);rows=[]
    for pop in [0,1]:
        mask=(s.geo['population']==pop)&((s.geo['group_region']<2) if pop==0 else True)
        weights=s.geo['group_size'][mask]*gains[0][mask]**2;weights/=weights.sum()
        target=response[:,0,mask]/gains[0][mask]
        def h(x):
            z=2j*np.pi*f[:,None]/1000
            return x[0]/(1+z*x[1])+(1-x[0])/(1+z*x[2])
        def fun(x):
            d=(h(x)-target)[train]*np.sqrt(weights)
            return np.r_[d.real.ravel(),d.imag.ravel()]
        fit=least_squares(fun,[.4,1,15],bounds=([0,.5,.5],[1,100,100]),xtol=1e-12,ftol=1e-12,gtol=1e-12)
        x=fit.x
        if x[1]>x[2]:x=np.array([1-x[0],x[2],x[1]])
        error=np.sqrt(np.sum(weights*abs(h(x)-target)**2,axis=1))
        rows.append(dict(population='EI'[pop],mixture_weight=x[0],tau_fast_ms=x[1],tau_slow_ms=x[2],
            frequencies_hz=f,frequency_shape_rms=error,heldout_shape_rms=float(np.sqrt(np.mean(error[~train]**2))),
            calibration_groups=np.flatnonzero(mask),parameter_source='calibrated local linear mean-input susceptibility at J=.94'))
    write(RATE_OUT/'closure.json',dict(rows=rows,J_calibration=.94,fit_frequencies_hz=f[train],heldout_frequencies_hz=f[~train],
        model='positive two-filter nonlinear spatial rate DDE',dynamic_equivalence='NOT_VALIDATED',
        limits=['Mean response fit only; the same rate filters also act on variance perturbations.',
                'Filter constants frozen across J; no fit to burst shape or propagation.',
                'Fixed points coincide with the earlier closure; temporal eigenvalues need recomputation.']))
    print('calibration',[(q['population'],q['mixture_weight'],q['tau_fast_ms'],q['tau_slow_ms'],q['heldout_shape_rms']) for q in rows],flush=True)


class RateField(SpatialBrunel):
    def __init__(self,grid=20):
        super().__init__(grid=grid)
        fit=read(RATE_OUT/'closure.json')['rows'];p=self.geo['population']
        self.alpha=np.array([q['mixture_weight'] for q in fit])[p]
        self.tf=np.array([q['tau_fast_ms'] for q in fit])[p]
        self.ts=np.array([q['tau_slow_ms'] for q in fit])[p]
        self.private_mu=self.tm*self.area[0]*self.jext*self.nu
        self.private_ve=self.tm*self.area[0]**2*self.jext**2*self.nu

    def output(self,y):return self.alpha*y[0]+(1-self.alpha)*y[1]

    def equilibrium_state(self,r,J):
        a,b,qa,qb=self.matrices(J)
        ma=self.tm*self.area[0]*(a@r);mg=self.tm*self.area[1]*(b@r)
        va=self.tm*self.area[0]**2*(qa@r);vg=self.tm*self.area[1]**2*(qb@r)
        return np.array([r,r,ma,ma,mg,mg,va,vg,.5*self.E*r])

    def rhs(self,y,arrivals):
        xf,xs,qa,ia,qg,ig,va,vg,m=y;a,b,aa,bb=arrivals
        r=self.output(y);mu=ia-self.Z*ig-m+self.private_mu
        target=self.phi(mu,va+self.private_ve,vg)
        return np.array([(target-xf)/self.tf,(target-xs)/self.ts,
            (self.tm*self.area[0]*a-qa)/self.rise[0],(qa-ia)/self.decay[0],
            (self.tm*self.area[1]*b-qg)/self.rise[1],(qg-ig)/self.decay[1],
            (self.tm*self.area[0]**2*aa-va)/(self.tau[0]/2),
            (self.tm*self.area[1]**2*bb-vg)/(self.tau[1]/2),(.5*self.E*r-m)/1000])

    def filter_response(self,lam):return self.alpha/(1+lam*self.tf)+(1-self.alpha)/(1+lam*self.ts)

    def characteristic(self,r,J,lam):
        a,b,qa,qb=self.matrices(J,lam);gm,ge,gi=self.gains(self.moments(r,J))
        ha=1/((1+lam*self.rise[0])*(1+lam*self.decay[0]))
        hg=1/((1+lam*self.rise[1])*(1+lam*self.decay[1]))
        hqa=1/(1+lam*self.tau[0]/2);hqg=1/(1+lam*self.tau[1]/2)
        K=sparse.diags(gm*self.tm)@(self.area[0]*ha*a-sparse.diags(self.Z*self.area[1]*hg)@b)
        K+=sparse.diags(ge*self.tm*self.area[0]**2*hqa)@qa+sparse.diags(gi*self.tm*(self.Z*self.area[1])**2*hqg)@qb
        return sparse.diags(1/self.filter_response(lam)+.5*self.E*gm/(1+1000*lam))-K

    def eigenstate(self,r,J,lam,v):
        a,b,qa,qb=self.matrices(J,lam)
        qa1=self.tm*self.area[0]*(a@v)/(1+lam*self.rise[0]);ia=qa1/(1+lam*self.decay[0])
        qg1=self.tm*self.area[1]*(b@v)/(1+lam*self.rise[1]);ig=qg1/(1+lam*self.decay[1])
        va=self.tm*self.area[0]**2*(qa@v)/(1+lam*self.tau[0]/2)
        vg=self.tm*self.area[1]**2*(qb@v)/(1+lam*self.tau[1]/2)
        drive=v/self.filter_response(lam)
        return np.array([drive/(1+lam*self.tf),drive/(1+lam*self.ts),qa1,ia,qg1,ig,va,vg,.5*self.E*v/(1+1000*lam)])


def cuda_code(s):
    nodes=','.join(format(x,'.17g') for x in s.nodes);weights=','.join(format(x,'.17g') for x in s.weights)
    return f'#define P {s.P}\n#define NQ {len(s.nodes)}\n__device__ __constant__ double X[NQ]={{{nodes}}};\n__device__ __constant__ double W[NQ]={{{weights}}};\n'+r'''
extern "C" __global__ void delayed(const int* pa,const int* ca,const double* wa,const double* va,
 const int* pb,const int* cb,const double* wb,const double* vb,const double* hist,double* out,int tick,int depth,int delay_factor){
 int g=blockIdx.x,lane=threadIdx.x;double a=0.,b=0.,q=0.,v=0.;
 for(int i=pa[g]+lane;i<pa[g+1];i+=blockDim.x){int d=(ca[i]/P+1)*delay_factor,slot=(tick-d)%depth;if(slot<0)slot+=depth;
  double r=hist[slot*P+ca[i]%P];a+=wa[i]*r;q+=va[i]*r;}
 for(int i=pb[g]+lane;i<pb[g+1];i+=blockDim.x){int d=(cb[i]/P+1)*delay_factor,slot=(tick-d)%depth;if(slot<0)slot+=depth;
  double r=hist[slot*P+cb[i]%P];b+=wb[i]*r;v+=vb[i]*r;}
 __shared__ double buf[4][128];buf[0][lane]=a;buf[1][lane]=b;buf[2][lane]=q;buf[3][lane]=v;__syncthreads();
 for(int k=64;k>0;k/=2){if(lane<k)for(int j=0;j<4;j++)buf[j][lane]+=buf[j][lane+k];__syncthreads();}
 if(lane==0)for(int j=0;j<4;j++)out[j*P+g]=buf[j][0];
}
__device__ double ex_erfc(double x){
 if(x<24.)return exp(x*x)*erfc(x);
 double z=1/(x*x);return (1-.5*z+.75*z*z-1.875*z*z*z+6.5625*z*z*z*z)/(sqrt(3.141592653589793)*x);
}
extern "C" __global__ void rhs(const double* y,const double* arr,const double* pars,double* out){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=P)return;
 double tm=pars[g],ref=pars[P+g],th=pars[2*P+g],alpha=pars[3*P+g],tf=pars[4*P+g],ts=pars[5*P+g];
 double pm=pars[6*P+g],pv=pars[7*P+g],E=pars[8*P+g],aa=pars[9*P+g],ag=pars[10*P+g],dg=pars[11*P+g];
 double r=alpha*y[g]+(1-alpha)*y[P+g];
 double mu=y[3*P+g]-y[5*P+g]-y[8*P+g]+pm;
 double ve=y[6*P+g]+pv,vi=y[7*P+g],var=fmax(ve+vi,1e-12),sig=sqrt(var);
 double teff=var/fmax(ve/4.2+vi/(1+dg),1e-12),shift=1.0325*sqrt(teff/tm);
 double lo=(11.-mu)/sig+shift,hi=(th-mu)/sig+shift,target=0.;
 if(hi<26.){double sum=0.;for(int k=0;k<NQ;k++){double x=(hi+lo)/2+(hi-lo)/2*X[k];sum+=W[k]*ex_erfc(-x);}
  target=1/(ref+tm*sqrt(3.141592653589793)*(hi-lo)/2*sum);}
 out[g]=(target-y[g])/tf;out[P+g]=(target-y[P+g])/ts;
 out[2*P+g]=(tm*aa*arr[g]-y[2*P+g])/.7;out[3*P+g]=(y[2*P+g]-y[3*P+g])/3.5;
 out[4*P+g]=tm*ag*arr[P+g]-y[4*P+g];out[5*P+g]=(y[4*P+g]-y[5*P+g])/dg;
 out[6*P+g]=(tm*aa*aa*arr[2*P+g]-y[6*P+g])/2.1;
 out[7*P+g]=(tm*ag*ag*arr[3*P+g]-y[7*P+g])/((1+dg)/2);
 out[8*P+g]=(.5*E*r-y[8*P+g])/1000.;
}
extern "C" __global__ void predictor(const double* y,const double* f,double* pred,double dt){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<9*P)pred[i]=y[i]+dt*f[i];}
extern "C" __global__ void finish(double* y,const double* f,const double* f2,const double* pars,double* hist,double dt,int tick,int depth){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=P)return;
 for(int j=0;j<9;j++){int i=j*P+g;y[i]+=.5*dt*(f[i]+f2[i]);}
 hist[(tick%depth)*P+g]=pars[3*P+g]*y[g]+(1-pars[3*P+g])*y[P+g];
}
'''


class RateIntegrator:
    def __init__(self,s,J,dt=.1,initial=None,history=None,device=0):
        import cupy as cp
        self.cp=cp;cp.cuda.Device(device).use();self.s=s;self.J=J;self.dt=dt;self.tick=0
        self.factor=round(.1/dt);assert abs(self.factor*dt-.1)<1e-12
        self.depth=s.prep['max_delay_steps']*self.factor+1
        self.y=cp.asarray(np.zeros((9,s.P)) if initial is None else initial)
        self.history=cp.asarray(np.broadcast_to(s.output(self.y.get()),(self.depth,s.P)).copy() if history is None else history)
        self.arr=cp.zeros((4,s.P));self.f=cp.zeros_like(self.y);self.f2=cp.zeros_like(self.y);self.pred=cp.zeros_like(self.y)
        self.ops=[]
        for kind in ['ampa','gaba']:
            a=sparse.load_npz(s.folder/f'mean_{kind}.npz').tocsr();q=sparse.load_npz(s.folder/f'variance_{kind}.npz').tocsr()
            assert np.array_equal(a.indices,q.indices) and np.array_equal(a.indptr,q.indptr)
            if kind=='ampa':
                rr=np.repeat(np.arange(s.P),np.diff(a.indptr));cc=a.indices%s.P;reg=s.geo['group_region']
                mask=s.E[rr]&s.E[cc]&(reg[rr]<2)&(reg[rr]==reg[cc]);a.data[mask]*=J;q.data[mask]*=J*J
            self.ops.extend([cp.asarray(a.indptr,dtype=cp.int32),cp.asarray(a.indices,dtype=cp.int32),cp.asarray(a.data),cp.asarray(q.data)])
        self.pars=cp.asarray(np.array([s.tm,s.ref,s.theta,s.alpha,s.tf,s.ts,s.private_mu,s.private_ve,s.E,
            np.full(s.P,s.area[0]),np.full(s.P,s.area[1]),np.full(s.P,s.decay[1])]))
        self.module=cp.RawModule(code=cuda_code(s),options=('--fmad=false',),name_expressions=['delayed','rhs','predictor','finish'])
        self.k={n:self.module.get_function(n) for n in ['delayed','rhs','predictor','finish']}

    def arrivals(self,tick):
        self.k['delayed']((self.s.P,),(128,),(*self.ops,self.history,self.arr,np.int32(tick),np.int32(self.depth),np.int32(self.factor)))

    def step(self):
        n=(self.s.P+127)//128
        self.arrivals(self.tick);self.k['rhs']((n,),(128,),(self.y,self.arr,self.pars,self.f))
        self.k['predictor'](((9*self.s.P+127)//128,),(128,),(self.y,self.f,self.pred,self.dt))
        self.arrivals(self.tick+1);self.k['rhs']((n,),(128,),(self.pred,self.arr,self.pars,self.f2))
        self.tick+=1
        self.k['finish']((n,),(128,),(self.y,self.f,self.f2,self.pars,self.history,self.dt,np.int32(self.tick),np.int32(self.depth)))
        return self.history[self.tick%self.depth]


if __name__=='__main__':calibrate()
