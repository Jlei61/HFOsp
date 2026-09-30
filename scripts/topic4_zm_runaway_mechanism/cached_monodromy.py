"""Cache the exact local rate derivative along a fixed orbit.

Only evaluation scheduling changes: the same spline/response gradients need
not be recalculated for every Krylov vector at every time step. Delay operators,
Heun stages and endpoint history remain unchanged.
"""
from chunk_monodromy import *


CACHED=r'''
extern "C" __global__ void cache_gains(const double* orbit,const double* pars,const double* consts,
 const double* SE,const double* SI,const double* WE,const double* WI,double* gains,int nt){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nt*P)return;int t=i/P,g=i%P;
 const double* y=orbit+1LL*t*14*P;double th=pars[2*P+g],E=pars[3*P+g];
 double mus=y[P+g],vEf=y[2*P+g],vIf=y[3*P+g],z=y[11*P+g];
 double mu=y[5*P+g]-z*y[7*P+g]-y[10*P+g]+pars[6*P+g];
 double vE=y[8*P+g]+pars[7*P+g],vI=z*z*y[9*P+g],vEv=y[12*P+g],vIv=y[13*P+g];
 double w[5],gr[15];
 if(consts[9]>.5){resp_weights(E>.5?WE:WI,mu,vE,vI,th,w,gr);}
 else{for(int k=0;k<5;k++)w[k]=pars[(14+k)*P+g];for(int k=0;k<15;k++)gr[k]=0.;}
 double me=w[0]*mu+(1-w[0])*mus+w[3]*(vE-vEf)+w[4]*(vI-vIf);
 double ve=w[1]*vE+(1-w[1])*vEv,vi=w[2]*vI+(1-w[2])*vIv;
 double r,pm,pE,pI;phi_spline(E>.5?SE:SI,me,fmax(ve,0.),fmax(vi,0.),th,&r,&pm,&pE,&pI);
 pE*=ve>0?1.:0.;pI*=vi>0?1.:0.;for(int k=0;k<15;k++)gr[k]*=consts[10];
 double* out=gains+1LL*t*8*P+g;
 for(int c=0;c<3;c++){
  double direct=c==0?w[0]:(c==1?w[3]:w[4]);
  out[c*P]=pm*(direct+(mu-mus)*gr[c]+(vE-vEf)*gr[9+c]+(vI-vIf)*gr[12+c])
     +pE*((c==1?w[1]:0.)+(vE-vEv)*gr[3+c])
     +pI*((c==2?w[2]:0.)+(vI-vIv)*gr[6+c]);
 }
 out[3*P]=pm*(1-w[0]);out[4*P]=-pm*w[3];out[5*P]=-pm*w[4];
 out[6*P]=pE*(1-w[1]);out[7*P]=pI*(1-w[2]);
}
extern "C" __global__ void cached_rhs(const double* gains,const double* Z,const int* offset,int local,
 const double* dy,const double* arr,const double* pars,const double* consts,double* out,double* drate){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=P)return;
 const double* c=gains+(1LL*(*offset+local)*8*P)+g;
 double z=Z[g],tm=pars[g],E=pars[3*P+g],aa=pars[4*P+g],ag=pars[5*P+g];
 double dmuf=dy[g],dmus=dy[P+g],dvEf=dy[2*P+g],dvIf=dy[3*P+g];
 double dqa=dy[4*P+g],dia=dy[5*P+g],dqg=dy[6*P+g],dig=dy[7*P+g];
 double dva=dy[8*P+g],dvg=dy[9*P+g],dm=dy[10*P+g],dvEv=dy[12*P+g],dvIv=dy[13*P+g];
 double dmu=dia-z*dig-dm,dvI=z*z*dvg;
 double dr=c[0]*dmu+c[P]*dva+c[2*P]*dvI+c[3*P]*dmus+c[4*P]*dvEf+c[5*P]*dvIf+c[6*P]*dvEv+c[7*P]*dvIv;
 out[g]=(dmu-dmuf)/pars[8*P+g];out[P+g]=(dmu-dmus)/pars[9*P+g];
 out[2*P+g]=(dva-dvEf)/pars[10*P+g];out[3*P+g]=(dvI-dvIf)/pars[11*P+g];
 out[12*P+g]=(dva-dvEv)/pars[12*P+g];out[13*P+g]=(dvI-dvIv)/pars[13*P+g];
 out[4*P+g]=(tm*aa*arr[g]-dqa)/consts[0];out[5*P+g]=(dqa-dia)/consts[1];
 out[6*P+g]=(tm*ag*arr[P+g]-dqg)/consts[2];out[7*P+g]=(dqg-dig)/consts[3];
 out[8*P+g]=(tm*aa*aa*arr[2*P+g]-dva)/consts[4];out[9*P+g]=(tm*ag*ag*arr[3*P+g]-dvg)/consts[5];
 out[10*P+g]=(.5*E*dr-dm)/consts[6];out[11*P+g]=0.;drate[g]=dr;
}
'''


class CachedMonodromy(ChunkEndpointMonodromy):
    def __init__(self,*args,**kwargs):
        o=args[1];sampler=getattr(o,'sample_state',None)
        if sampler is None:super().__init__(*args,**kwargs)
        else:
            # Parent graphs are superseded below by the cached graphs. Their
            # capture only accesses the first block, so store that small prefix.
            import chunk_monodromy
            previous=chunk_monodromy.orbit_states
            def prefix(oo,sol,n):
                times=np.arange(min(n+1,kwargs.get('block',128)+1))*sol['T']/n
                return sampler(times),None
            chunk_monodromy.orbit_states=prefix
            try:super().__init__(*args,**kwargs)
            finally:chunk_monodromy.orbit_states=previous
        assert not self.dynamic_z
        cp=self.cp;p=self.s.P
        mod=cp.RawModule(code=f'#define P {p}\n'+CUDA_DEVICE+CUDA_RESP+CACHED,
                         options=('--fmad=false',),name_expressions=['cache_gains','cached_rhs'])
        self.gains=cp.empty((self.n+1,8,p));self.Z=cp.asarray(self.s.Z)
        cache=mod.get_function('cache_gains');self.cached_rhs=mod.get_function('cached_rhs')
        if sampler is None:
            cache((((self.n+1)*p+127)//128,),(128,),
                  (self.orbit,self.pars,self.consts,self.SE,self.SI,self.WE,self.WI,self.gains,np.int32(self.n+1)))
        else:
            for lo in range(0,self.n+1,512):
                hi=min(lo+512,self.n+1);states=cp.asarray(sampler(np.arange(lo,hi)*self.dt))
                cache((((hi-lo)*p+127)//128,),(128,),
                      (states,self.pars,self.consts,self.SE,self.SI,self.WE,self.WI,self.gains[lo:hi],np.int32(hi-lo)))
            del states
        cp.cuda.get_current_stream().synchronize()
        with self.stream:self.cached_step(0)
        self.stream.synchronize();self.graphs={}
        for length in sorted(set([self.block,self.n%self.block])-{0}):
            with self.stream:
                self.stream.begin_capture()
                for i in range(length):self.cached_step(i)
                self.k['advance']((1,),(1,),(self.offset,np.int32(length)))
                self.graphs[length]=self.stream.end_capture()
        self.cp.get_default_memory_pool().free_all_blocks()

    def cached_step(self,i):
        p=self.s.P;n=(p+127)//128
        def rhs(stage,state,f,dr):
            self.cached_rhs((n,),(128,),(self.gains,self.Z,self.offset,np.int32(stage),state,self.arr,self.pars,self.consts,f,dr))
        self.k['delayed_linear']((p,),(128,),(*self.ops,self.hist,self.arr,self.offset,np.int32(i),np.int32(self.depth),self.dt))
        rhs(i,self.y,self.f,self.dr)
        self.k['tpredictor'](((NS*p+127)//128,),(128,),(self.y,self.f,self.pred,self.dt))
        self.k['delayed_linear']((p,),(128,),(*self.ops,self.hist,self.arr,self.offset,np.int32(i+1),np.int32(self.depth),self.dt))
        rhs(i+1,self.pred,self.f2,self.dr2)
        self.k['tfinish']((n,),(128,),(self.y,self.f,self.f2,self.dr,self.dr2,self.hist,self.dt,self.offset,np.int32(i+1),np.int32(self.depth)))
        rhs(i+1,self.y,self.f2,self.dr2);self.store(self.dr2,self.hist,self.offset,np.int32(i+1),np.int32(self.depth),size=p)

    def release_full_orbit(self):
        # The inherited initial-rate evaluation only needs orbit[0]. Call after
        # constructing/checking the phase vector, before repeated matvecs.
        self.orbit=self.orbit[:1].copy();self.cp.get_default_memory_pool().free_all_blocks()


if __name__=='__main__':
    from native_path import *
    from streaming_periodic import StreamPeriodic
    import argparse,gc
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);a=p.parse_args()
    s=model();attach_native_path(s);z=np.load(OUT/'periodic/seed_N512.npz');sol=dict(r=z['r'],T=float(z['T']),D=float(z['D']))
    o=StreamPeriodic(s,512,a.device);o.cache_mean_operators=False
    m=ChunkEndpointMonodromy(s,o,sol,dtmax=.1,device=a.device)
    x=np.random.default_rng(195).normal(size=m.dim);x[11*s.P:12*s.P]=0.
    tic=time.time();expected=m.matvec(x);old_seconds=time.time()-tic
    del m;gc.collect();o.cp.get_default_memory_pool().free_all_blocks()
    m=CachedMonodromy(s,o,sol,dtmax=.1,device=a.device);m.release_full_orbit()
    tic=time.time();actual=m.matvec(x);new_seconds=time.time()-tic
    rel=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected));err=float(abs(actual-expected).max())
    assert rel<1e-11,(rel,err)
    write(OUT/'cached_monodromy_parity.json',dict(status='PASS',relative_error=rel,max_error=err,
          original_seconds=old_seconds,cached_seconds=new_seconds,
          model_change=False,method='Identical endpoint Heun/delay flow; precomputed local output derivatives'))
    log('CACHED MONODROMY PARITY',rel,err,old_seconds,new_seconds)
