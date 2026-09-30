"""Conditional variational flow with a constant spatial-Z parameter direction.

This is for branch-family consistency, not a Floquet eigensolver. Z remains a
held parameter; its tangent is retained to supply -I_G*dZ and 2Z*v_G*dZ.
The existing homogeneous conditional Floquet operator is not changed.
"""
from preinterpolated_rk4 import PreinterpolatedRK4
from cached_monodromy import CACHED,NS
from common import np,log,time
from host_array_storage import allocate_host_array


def forcing_code(P):
    code=CACHED[CACHED.index('extern "C" __global__ void cached_rhs'):]
    code=code.replace('void cached_rhs(const double* gains,','void forced_rhs(const double* gains,const double* factors,')
    code=code.replace('1LL*(*offset+local)*8*P','1LL*local*8*P')
    old='double dmu=dia-z*dig-dm,dvI=z*z*dvg;'
    new='const double* b=factors+1LL*local*2*P+g; double dz=dy[11*P+g]; double dmu=dia-z*dig-dm-b[0]*dz,dvI=z*z*dvg+b[P]*dz;'
    assert code.count(old)==1
    return f'#define P {P}\n'+code.replace(old,new)


class ParameterForcedRK4(PreinterpolatedRK4):
    def __init__(self,s,o,sol,*args,**kw):
        assert kw.get('host_gain_cache') is True
        sampler=o.sample_state;self.forcing_ready=False
        super().__init__(s,o,sol,*args,**kw)
        cp=self.cp;p=s.P
        self.factors=cp.empty((2*self.block+1,2,p))
        self.host_factors,self.factor_storage=allocate_host_array((2*self.n+1,2,p),'auto')
        from spectral_grid_sampler import SpectralGridSampler
        exact_grid=isinstance(sampler,SpectralGridSampler)
        if exact_grid:
            assert sampler.n==2*self.n and sampler.period==self.T
        for lo in range(0,2*self.n+1,512):
            hi=min(lo+512,2*self.n+1)
            # These are exactly the RK4 half-step rows. A view avoids copying
            # all 14 state components to read just the three forcing inputs.
            states=sampler.states[lo:hi] if exact_grid else sampler(np.arange(lo,hi)*self.dt/2)
            self.host_factors[lo:hi,0]=states[:,7]
            self.host_factors[lo:hi,1]=2*states[:,11]*states[:,9]
        mod=cp.RawModule(code=forcing_code(p),options=('--fmad=false',),name_expressions=['forced_rhs'])
        self.forced_rhs=mod.get_function('forced_rhs');self.forcing_ready=True
        self.factors.set(self.host_factors[:len(self.factors)])
        with self.stream:self.step(0)
        self.stream.synchronize();self.graphs={}
        for length in sorted(set([self.block,self.n%self.block])-{0}):
            with self.stream:
                self.stream.begin_capture()
                for i in range(length):self.step(i)
                self.k['advance']((1,),(1,),(self.offset,np.int32(length)))
                self.graphs[length]=self.stream.end_capture()

    def step(self,i):
        if not self.forcing_ready:return super().step(i)
        p=self.s.P;npop=(p+127)//128;nstate=(NS*p+127)//128
        def rhs(stage,state,f):
            self.forced_rhs((npop,),(128,),
                (self.gains,self.factors,self.Z,self.offset,np.int32(2*i+stage),state,self.arr,
                 self.pars,self.consts,f,self.dr2))
        def pred(f,dt):self.k['tpredictor']((nstate,),(128,),(self.y,f,self.pred,dt))
        self.arrivals(i,0.);rhs(0,self.y,self.f);pred(self.f,.5*self.dt)
        self.arrivals(i,.5);rhs(1,self.pred,self.f2);pred(self.f2,.5*self.dt)
        rhs(1,self.pred,self.f3);pred(self.f3,self.dt)
        self.arrivals(i,1.);rhs(2,self.pred,self.f4)
        self.k['rkfinish']((nstate,),(128,),(self.y,self.f,self.f2,self.f3,self.f4,self.dt))
        rhs(2,self.y,self.f4);self.store(self.dr2,self.hist,self.offset,np.int32(i+1),np.int32(self.depth),size=p)

    def matvec(self,x):
        cp=self.cp;p=self.s.P;start=time.time()
        with self.stream:
            self.offset.fill(0);x=cp.asarray(x)
            self.y[:]=x[:NS*p].reshape(NS,p)
            self.hist[self.order]=x[NS*p:].reshape(self.Dd,p)
            # The original tangent kernel already includes the Z parameter
            # forcing while its held-Z derivative is exactly zero.
            self.k['tangent_rhs'](((p+127)//128,),(128,),
                (self.orbit[0],self.y,self.arr,self.pars,self.consts,
                 self.SE,self.SI,self.WE,self.WI,self.f,self.dr))
            cp.copyto(self.hist[0],self.dr)
            for lo in range(0,self.n,self.block):
                length=min(self.block,self.n-lo)
                self.gains[:2*length+1].set(self.host_gains[2*lo:2*(lo+length)+1],stream=self.stream)
                self.factors[:2*length+1].set(self.host_factors[2*lo:2*(lo+length)+1],stream=self.stream)
                self.graphs[length].launch(stream=self.stream)
            result=cp.concatenate([self.y.ravel(),self.hist[self.finalorder].ravel()])
        self.stream.synchronize();self.calls+=1
        result=result.get();old=cp.asnumpy(x)
        assert np.array_equal(result[11*p:12*p],old[11*p:12*p])
        log('PARAMETER FORCED MAP',self.calls,'seconds',round(time.time()-start,1))
        return result


def local_check(device=0):
    """Cached forcing versus full analytic kernel and independent nonlinear FD."""
    from common import model,OUT,BASE,write
    from rk4_monodromy import model_device_arrays,CUDA_DEVICE,CUDA_RESP,TANGENT
    import cupy as cp
    cp.cuda.Device(device).use();s=model();p=s.P
    y=np.load(BASE/'runs/A4_det_meandrive/checkpoints/t8000ms.npz')['state']
    rng=np.random.default_rng(919802);dy=rng.normal(size=y.shape)*np.maximum(abs(y),1)*1e-3
    dy[11]=rng.normal(size=p)*1e-3
    arr=np.zeros((4,p));factors=np.array([y[7],2*y[11]*y[9]])
    pars,consts,SE,SI,WE,WI=model_device_arrays(s,cp,dynamic_z=False)
    consts=cp.concatenate([consts,cp.asarray([1.])])
    code=f'#define P {p}\n'+CUDA_DEVICE+CUDA_RESP+CACHED+TANGENT[TANGENT.index('extern "C" __global__ void tangent_rhs'):TANGENT.index('extern "C" __global__ void tfinish')]
    module=cp.RawModule(code=code,options=('--fmad=false',),name_expressions=['cache_gains','tangent_rhs'])
    gain=cp.empty((1,8,p));out=cp.empty_like(cp.asarray(y));dr=cp.empty(p)
    module.get_function('cache_gains')(((p+127)//128,),(128,),(cp.asarray(y),pars,consts,SE,SI,WE,WI,gain,np.int32(1)))
    module.get_function('tangent_rhs')(((p+127)//128,),(128,),
       (cp.asarray(y),cp.asarray(dy),cp.asarray(arr),pars,consts,SE,SI,WE,WI,out,dr))
    ref=out.get();ref_rate=dr.get()
    kernel=cp.RawModule(code=forcing_code(p),options=('--fmad=false',),name_expressions=['forced_rhs']).get_function('forced_rhs')
    kernel(((p+127)//128,),(128,),
       (gain,cp.asarray(factors),cp.asarray(y[11]),cp.zeros(1,dtype=cp.int32),np.int32(0),
        cp.asarray(dy),cp.asarray(arr),pars,consts,out,dr))
    err=float(abs(out.get()-ref).max());rerr=float(abs(dr.get()-ref_rate).max())
    assert err<1e-10 and rerr<1e-11,(err,rerr)
    rows=[]
    for eps in [1e-3,5e-4]:
        plus,rplus=s.rhs(y+eps*dy,arr,dynamic_z=False)
        minus,rminus=s.rhs(y-eps*dy,arr,dynamic_z=False)
        error=float(np.linalg.norm((plus-minus)/(2*eps)-ref)/np.linalg.norm(ref))
        re=float(np.linalg.norm((rplus-rminus)/(2*eps)-ref_rate)/max(np.linalg.norm(ref_rate),1e-20))
        assert error<1e-6 and re<1e-5,(error,re)
        rows.append(dict(epsilon=eps,RHS_relative=error,rate_relative=re))
    q=dict(status='LOCAL_FORCED_VARIATION_PASS',cached_full_max_error=err,cached_full_rate_error=rerr,
           nonlinear_finite_difference=rows,scope='Constant Z-parameter tangent is retained; Z dynamics remain held. Not a Floquet or cycle validation.')
    write(OUT/'parameter_forced_monodromy_check.json',q);log('FORCED VARIATION CHECK',q)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0);local_check(p.parse_args().device)
