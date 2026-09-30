"""Matrix-free monodromy (Floquet) of the v3 periodic orbits, all 12 states plus delay history.

The orbit comes from the periodic BVP (periodic_v3). Full orbit states on the fine grid are
reconstructed in the harmonic domain (LTI states from r) and Fourier-resampled. The linearised
Heun integrator (dt = T/ceil(T/dtmax), linearly interpolated physical delays) evaluates all
gains (spline derivatives, response-constant gradients) on the fly from the stored orbit state.
Required checks: autonomous phase multiplier near +1; step-size refinement.
"""
from periodic_v3 import *
from response_tables import CUDA_RESP
from scipy.sparse.linalg import LinearOperator,eigs

def orbit_states(o,sol,n):
    """(n+1,12,P) full states along the orbit at t_i = i T/n (periodic, last = first), from r via the LTI filters."""
    cp=o.cp;s=o.s;N=len(sol['r']);K=N//2+1;T=sol['T'];s.set_D(sol['D']);Z=s.Z
    r=cp.asarray(sol['r']);ops,(ha,hg,hva,hvg,hm),(fs,fE,fI,fvE,fvI),lam=o.kernels(T);rf=cp.fft.rfft(r,axis=0);v=rf.ravel()
    a,b,qa,qb=[(x@v).reshape(K,s.P) for x in ops];tm=o.gp[0];E=o.gp[2];tf=cp.asarray(s.poles[0])
    qa_h=tm*s.area[0]*a/(1+lam*s.rise[0]);ia_h=qa_h/(1+lam*s.decay[0]);qg_h=tm*s.area[1]*b/(1+lam*s.rise[1]);ig_h=qg_h/(1+lam*s.decay[1])
    va_h=tm*s.area[0]**2*qa/(1+lam*s.tau[0]/2);vg_h=tm*s.area[1]**2*qb/(1+lam*s.tau[1]/2);m_h=.5*E*rf/(1+lam*TAU_M)
    muh=tm*(s.area[0]*ha*a-cp.asarray(Z)*s.area[1]*hg*b)-.5*E*hm*rf;vEh=tm*s.area[0]**2*hva*qa;vIh=cp.asarray(Z)**2*tm*s.area[1]**2*hvg*qb
    muf_h=muh/(1+lam*tf[None,:]);mus_h=muh*fs;vEf_h=vEh*fE;vIf_h=vIh*fI;vEv_h=vEh*fvE;vIv_h=vIh*fvI
    st=cp.fft.irfft(cp.stack([muf_h,mus_h,vEf_h,vIf_h,qa_h,ia_h,qg_h,ig_h,va_h,vg_h,m_h,vEv_h,vIv_h]),n=N,axis=1).get()
    st[0]+=s.private_mu;st[1]+=s.private_mu;st[2]+=s.private_ve;st[11]+=s.private_ve
    Y=np.concatenate([st[:11],np.broadcast_to(Z,(1,N,s.P)),st[11:]],axis=0).transpose(1,0,2)   # N,14,P
    full=resample(Y,n,axis=0);full=np.concatenate([full,full[:1]],axis=0);full[:,11]=Z
    rr=resample(sol['r'],n,axis=0);rr=np.concatenate([rr,rr[:1]],axis=0)
    return full,rr

TANGENT=r'''
extern "C" __global__ void delayed_linear(const int* pa,const int* ca,const double* wa,const double* va,
 const int* pb,const int* cb,const double* wb,const double* vb,const double* hist,double* out,int tick,int depth,double dt){
 int g=blockIdx.x,lane=threadIdx.x;double a=0.,b=0.,q=0.,v=0.;
 for(int i=pa[g]+lane;i<pa[g+1];i+=blockDim.x){double d=(ca[i]/P+1)*.1/dt;int lag=(int)floor(d);double f=d-lag;
  int s0=(tick-lag)%depth;if(s0<0)s0+=depth;int s1=(s0+depth-1)%depth;
  double r=(1-f)*hist[s0*P+ca[i]%P]+f*hist[s1*P+ca[i]%P];a+=wa[i]*r;q+=va[i]*r;}
 for(int i=pb[g]+lane;i<pb[g+1];i+=blockDim.x){double d=(cb[i]/P+1)*.1/dt;int lag=(int)floor(d);double f=d-lag;
  int s0=(tick-lag)%depth;if(s0<0)s0+=depth;int s1=(s0+depth-1)%depth;
  double r=(1-f)*hist[s0*P+cb[i]%P]+f*hist[s1*P+cb[i]%P];b+=wb[i]*r;v+=vb[i]*r;}
 __shared__ double buf[4][128];buf[0][lane]=a;buf[1][lane]=b;buf[2][lane]=q;buf[3][lane]=v;__syncthreads();
 for(int k=64;k>0;k/=2){if(lane<k)for(int j=0;j<4;j++)buf[j][lane]+=buf[j][lane+k];__syncthreads();}
 if(lane==0)for(int j=0;j<4;j++)out[j*P+g]=buf[j][0];
}
// orbit: (12,P) state at this stage; dy: tangent (12,P); arr: delayed tangent arrivals
extern "C" __global__ void tangent_rhs(const double* orbit,const double* dy,const double* arr,const double* pars,const double* consts,const double* SE,const double* SI,const double* WE,const double* WI,double* out,double* drate){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=P)return;
 double tm=pars[g],th=pars[2*P+g],E=pars[3*P+g],aa=pars[4*P+g],ag=pars[5*P+g],pm=pars[6*P+g],pv=pars[7*P+g];
 double tf=pars[8*P+g],ts=pars[9*P+g],tE=pars[10*P+g],tI=pars[11*P+g],tvE=pars[12*P+g],tvI=pars[13*P+g],dynZ=pars[19*P+g];
 double mus=orbit[P+g],vEf=orbit[2*P+g],vIf=orbit[3*P+g],ig=orbit[7*P+g],va=orbit[8*P+g],vg=orbit[9*P+g],m=orbit[10*P+g],z=orbit[11*P+g],ia=orbit[5*P+g],vEv=orbit[12*P+g],vIv=orbit[13*P+g];
 double mu=ia-z*ig-m+pm,vE=va+pv,vI=z*z*vg;
 double w[5],gr[15];
 if(consts[9]>.5){resp_weights(E>.5?WE:WI,mu,vE,vI,th,w,gr);}else{for(int p=0;p<5;p++)w[p]=pars[(14+p)*P+g];for(int k=0;k<15;k++)gr[k]=0.;}
 double mueff=w[0]*mu+(1-w[0])*mus+w[3]*(vE-vEf)+w[4]*(vI-vIf);double vEe=w[1]*vE+(1-w[1])*vEv,vIe=w[2]*vI+(1-w[2])*vIv;double onE=vEe>0?1.:0.,onI=vIe>0?1.:0.;
 double r,pmu,pvE,pvI;phi_spline(E>0.5?SE:SI,mueff,fmax(vEe,0.),fmax(vIe,0.),th,&r,&pmu,&pvE,&pvI);pvE*=onE;pvI*=onI;
 double dmuf=dy[g],dmus=dy[P+g],dvEf=dy[2*P+g],dvIf=dy[3*P+g],dqa=dy[4*P+g],dia=dy[5*P+g],dqg=dy[6*P+g],dig=dy[7*P+g],dva=dy[8*P+g],dvg=dy[9*P+g],dm=dy[10*P+g],dz=dy[11*P+g],dvEv=dy[12*P+g],dvIv=dy[13*P+g];
 double dmu=dia-z*dig-dm-ig*dz,dvE=dva,dvI=z*z*dvg+2*z*vg*dz;
 double dw[5];double ug=consts[10];for(int p=0;p<5;p++)dw[p]=ug*(gr[3*p]*dmu+gr[3*p+1]*dvE+gr[3*p+2]*dvI);
 double dmueff=w[0]*dmu+(1-w[0])*dmus+(mu-mus)*dw[0]+w[3]*(dvE-dvEf)+(vE-vEf)*dw[3]+w[4]*(dvI-dvIf)+(vI-vIf)*dw[4];
 double dvEe=w[1]*dvE+(1-w[1])*dvEv+(vE-vEv)*dw[1],dvIe=w[2]*dvI+(1-w[2])*dvIv+(vI-vIv)*dw[2];
 double dr=pmu*dmueff+pvE*dvEe+pvI*dvIe;
 out[g]=(dmu-dmuf)/tf;out[P+g]=(dmu-dmus)/ts;out[2*P+g]=(dvE-dvEf)/tE;out[3*P+g]=(dvI-dvIf)/tI;out[12*P+g]=(dvE-dvEv)/tvE;out[13*P+g]=(dvI-dvIv)/tvI;
 out[4*P+g]=(tm*aa*arr[g]-dqa)/consts[0];out[5*P+g]=(dqa-dia)/consts[1];
 out[6*P+g]=(tm*ag*arr[P+g]-dqg)/consts[2];out[7*P+g]=(dqg-dig)/consts[3];
 out[8*P+g]=(tm*aa*aa*arr[2*P+g]-dva)/consts[4];out[9*P+g]=(tm*ag*ag*arr[3*P+g]-dvg)/consts[5];
 out[10*P+g]=(.5*E*dr-dm)/consts[6];
 double sd=sqrt(fmax(tm*vg/(2*(consts[2]+consts[3])),1e-20));double u=(consts[8]-ig)/sd;double pdf=exp(-.5*u*u)/2.5066282746310002;
 double dzinf=-pdf/sd*dig-pdf*u/(2*fmax(vg,1e-20))*dvg;
 out[11*P+g]=dynZ*E*(dzinf-dz)/consts[7];
 drate[g]=dr;
}
extern "C" __global__ void tpredictor(const double* y,const double* f,double* pred,double dt){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<14*P)pred[i]=y[i]+dt*f[i];}
extern "C" __global__ void tfinish(double* y,const double* f,const double* f2,const double* dr,const double* dr2,double* hist,double dt,int tick,int depth){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=P)return;for(int j=0;j<14;j++){int i=j*P+g;y[i]+=.5*dt*(f[i]+f2[i]);}
 hist[(tick%depth)*P+g]=.5*(dr[g]+dr2[g]);}
'''

class Monodromy:
    def __init__(self,s,o,sol,dtmax=.1,dynamic_z=False,device=0,use_gradients=True):
        import cupy as cp
        self.cp=cp;self.s=s;self.dynamic_z=dynamic_z;cp.cuda.Device(device).use();self.T=sol['T'];self.n=int(np.ceil(self.T/dtmax));self.dt=self.T/self.n
        self.Dd=int(np.ceil(s.delays[-1]/self.dt))+1;self.depth=self.Dd+1;self.dim=(NS+self.Dd)*s.P
        full,rr=orbit_states(o,sol,self.n);self.orbit=cp.asarray(np.ascontiguousarray(full));self.rr=rr
        self.ops=[]
        for kind in ['ampa','gaba']:
            a=sparse.load_npz(s.folder/f'mean_{kind}.npz').tocsr();q=sparse.load_npz(s.folder/f'variance_{kind}.npz').tocsr()
            self.ops.extend([cp.asarray(a.indptr,dtype=cp.int32),cp.asarray(a.indices,dtype=cp.int32),cp.asarray(a.data),cp.asarray(q.data)])
        self.pars,self.consts,self.SE,self.SI,self.WE,self.WI=model_device_arrays(s,cp,dynamic_z=dynamic_z)
        self.consts=cp.concatenate([self.consts,cp.asarray([1. if use_gradients else 0.])])
        names=['delayed_linear','tangent_rhs','tpredictor','tfinish'];mod=cp.RawModule(code=f'#define P {s.P}\n'+CUDA_DEVICE+CUDA_RESP+TANGENT,options=('--fmad=false',),name_expressions=names)
        self.k={n:mod.get_function(n) for n in names};self.y=cp.zeros((NS,s.P));self.hist=cp.zeros((self.depth,s.P))
        self.arr=cp.zeros((4,s.P));self.f=cp.zeros_like(self.y);self.f2=cp.zeros_like(self.y);self.pred=cp.zeros_like(self.y);self.dr=cp.zeros(s.P);self.dr2=cp.zeros(s.P)
        self.order=cp.asarray((-np.arange(1,self.Dd+1))%self.depth);self.finalorder=cp.asarray((self.n-np.arange(1,self.Dd+1))%self.depth)
        self.stream=cp.cuda.Stream(non_blocking=True);cp.cuda.get_current_stream().synchronize()
        with self.stream:self.step(0)
        self.stream.synchronize()
        with self.stream:
            self.stream.begin_capture()
            for i in range(self.n):self.step(i)
            self.graph=self.stream.end_capture()
        self.calls=0;self.start=time.time();cp.get_default_memory_pool().free_all_blocks()
    def step(self,i):
        p=self.s.P;n=(p+127)//128
        self.k['delayed_linear']((p,),(128,),(*self.ops,self.hist,self.arr,np.int32(i),np.int32(self.depth),self.dt))
        self.k['tangent_rhs']((n,),(128,),(self.orbit[i],self.y,self.arr,self.pars,self.consts,self.SE,self.SI,self.WE,self.WI,self.f,self.dr))
        self.k['tpredictor'](((NS*p+127)//128,),(128,),(self.y,self.f,self.pred,self.dt))
        self.k['delayed_linear']((p,),(128,),(*self.ops,self.hist,self.arr,np.int32(i+1),np.int32(self.depth),self.dt))
        self.k['tangent_rhs']((n,),(128,),(self.orbit[i+1],self.pred,self.arr,self.pars,self.consts,self.SE,self.SI,self.WE,self.WI,self.f2,self.dr2))
        self.k['tfinish']((n,),(128,),(self.y,self.f,self.f2,self.dr,self.dr2,self.hist,self.dt,np.int32(i+1),np.int32(self.depth)))
    def matvec(self,x):
        cp=self.cp;p=self.s.P
        with self.stream:
            x=cp.asarray(x);self.y[:]=x[:NS*p].reshape(NS,p);self.hist[self.order]=x[NS*p:].reshape(self.Dd,p)
            if not self.dynamic_z:self.y[11]=0.   # frozen-Z conditional system: the z directions are not part of the tangent space (they would give trivial unit multipliers)
            # slot 0 holds the tangent rate at t=0 (from the tangent state): approximate by the linearised rate of the initial tangent
            self.hist[0]=self.hist[self.order[0]]
            self.graph.launch(stream=self.stream);result=cp.concatenate([self.y.ravel(),self.hist[self.finalorder].ravel()])
        self.stream.synchronize();self.calls+=1
        if self.calls%10==0:log('MONODROMY',self.calls,'sec',round(time.time()-self.start,1))
        return result.get()
    def phase_vector(self,full):
        """Time derivative of the orbit (spectral) as tangent state + rate history: multiplier +1 check."""
        n=self.n;Y=full[:-1];k=np.fft.rfftfreq(n,d=1/n);lam=2j*np.pi*k/self.T
        dY=np.fft.irfft(np.fft.rfft(Y,axis=0)*lam[:,None,None],n=n,axis=0);dr=np.fft.irfft(np.fft.rfft(self.rr[:-1],axis=0)*lam[:,None],n=n,axis=0)
        hist=np.array([dr[(-j)%n] for j in range(1,self.Dd+1)]);dY0=dY[0].copy()
        if not self.dynamic_z:dY0[11]=0.
        return np.r_[dY0.ravel(),hist.ravel()]

def analyse(s,o,sol,dtmax=.1,nev=6,dynamic_z=False,device=0):
    m=Monodromy(s,o,sol,dtmax,dynamic_z,device);full,_=orbit_states(o,sol,m.n)
    A=LinearOperator((m.dim,m.dim),matvec=m.matvec,dtype=np.float64)
    ev,vec=eigs(A,k=nev,which='LM',tol=1e-8,maxiter=3000)
    ph=m.phase_vector(full);Mph=m.matvec(ph);phase_defect=float(np.linalg.norm(Mph-ph)/np.linalg.norm(ph))
    order=np.argsort(-abs(ev));ev=ev[order];vec=vec[:,order]
    res=[float(np.linalg.norm(m.matvec(vec[:,i].real)-(ev[i]*vec[:,i]).real)/np.linalg.norm(vec[:,i].real)) if abs(ev[i].imag)<1e-12 else None for i in range(len(ev))]
    return dict(multipliers=[[float(e.real),float(e.imag)] for e in ev],residuals=res,phase_defect=phase_defect,dt_ms=m.dt,n_steps=m.n,dynamic_z=dynamic_z,matvecs=m.calls),vec,m

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('orbit');p.add_argument('--dt',type=float,default=.1);p.add_argument('--nev',type=int,default=6);p.add_argument('--dynamic-z',action='store_true');p.add_argument('--device',type=int,default=0)
    a=p.parse_args();s=load_model(a.device);z=np.load(a.orbit);sol={k:z[k] for k in z.files};sol['T']=float(sol['T']);sol['D']=float(sol['D']);sol['residual']=float(sol['residual'])
    o=PeriodicV3(s,len(sol['r']),a.device);row,vec,m=analyse(s,o,sol,a.dt,a.nev,a.dynamic_z,a.device)
    # classify: exclude the multiplier closest to +1 as the autonomous phase (only if within 1e-2), report the rest
    mults=np.array([complex(*e) for e in row['multipliers']]);ph=np.argmin(abs(mults-1));others=np.delete(mults,ph)
    row.update(phase_index=int(ph),phase_multiplier=[float(mults[ph].real),float(mults[ph].imag)],max_other_modulus=float(max(abs(others))) if len(others) else None,
        stability='UNSTABLE' if len(others) and max(abs(others))>1+1e-3 else ('STABLE_SAMPLED' if len(others) and max(abs(others))<1-1e-3 else 'UNRESOLVED_NEAR_UNIT'),orbit=a.orbit)
    out=PERIODIC_OUT/'floquet';out.mkdir(exist_ok=True,parents=True);name=Path(a.orbit).stem+f'_dt{a.dt}'+('_dynZ' if a.dynamic_z else '')
    np.savez_compressed(out/(name+'.npz'),multipliers=mults,vectors=vec);write(out/(name+'.json'),row);log('FLOQUET',row)
