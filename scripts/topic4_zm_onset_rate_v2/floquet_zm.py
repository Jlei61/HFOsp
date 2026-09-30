"""Matrix-free monodromy of all rate, filter, adaptation and delay states.

The orbit is obtained independently by a periodic BVP. Linearized propagation
uses Heun and linearly interpolated physical delays at dt=T/ceil(T/dtmax).
An autonomous phase multiplier near +1 and step refinement are required checks.
"""
from periodic_zm import *
from scipy.sparse.linalg import LinearOperator,eigs


class Monodromy:
    def __init__(self,s,path,dtmax=.1,device=0):
        import cupy as cp
        self.cp=cp;self.s=s;cp.cuda.Device(device).use();z=np.load(path);self.path=Path(path)
        assert float(z['residual'])<2e-8
        self.r=z['r'];self.T=float(z['T']);self.J=1.;self.depletion=float(z['D']);s.set_D(self.depletion);self.n=int(np.ceil(self.T/dtmax));self.dt=self.T/self.n
        self.D=int(np.ceil(s.delays[-1]/self.dt))+1;self.depth=self.D+1;self.dim=(9+self.D)*s.P
        orbit=ZMPeriodic(s,len(self.r),device);kernels=orbit.kernels(self.T,self.depletion)
        mom=orbit.moments(cp.asarray(self.r),kernels)+orbit.private[:,None,:]
        # Fourier-resample moments; compute the transfer derivative at each
        # integration stage, not by interpolating a derivative with sharp peaks.
        full=resample(mom.get(),self.n,axis=1);full=np.concatenate([full,full[:,:1]],axis=1);mom=cp.asarray(full)
        def phi(m):
            out=cp.empty((self.n+1,s.P));orbit.phik((((self.n+1)*s.P+127)//128,),(128,),
                (cp.ascontiguousarray(m),orbit.pars,out,np.int32(self.n+1)));return out
        gains=[]
        for k in range(3):
            step=1e-5*cp.maximum(cp.abs(mom[k]),1.);hi=mom.copy();lo=mom.copy();hi[k]+=step;lo[k]-=step
            gains.append((phi(hi)-phi(lo))/(2*step))
        self.gains=cp.ascontiguousarray(cp.stack(gains,axis=1))
        del full,mom,gains,hi,lo
        self.pars=cp.vstack([orbit.pars,cp.asarray(s.Z)]);del orbit,kernels
        # Native delay-edge arrays are identical to RateIntegrator.
        self.ops=[]
        for kind in ['ampa','gaba']:
            a=sparse.load_npz(s.folder/f'mean_{kind}.npz').tocsr();q=sparse.load_npz(s.folder/f'variance_{kind}.npz').tocsr()
            if kind=='ampa':
                rr=np.repeat(np.arange(s.P),np.diff(a.indptr));cc=a.indices%s.P;reg=s.geo['group_region']
                mask=s.E[rr]&s.E[cc]&(reg[rr]<2)&(reg[rr]==reg[cc]);a.data[mask]*=self.J;q.data[mask]*=self.J**2
            self.ops.extend([cp.asarray(a.indptr,dtype=cp.int32),cp.asarray(a.indices,dtype=cp.int32),cp.asarray(a.data),cp.asarray(q.data)])
        code=cuda_code(s)+r'''
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
extern "C" __global__ void tangent_rhs(const double* y,const double* arr,const double* pars,const double* gains,double* out,int tick){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=P)return;
 double tm=pars[g],alpha=pars[3*P+g],tf=pars[4*P+g],ts=pars[5*P+g],E=pars[8*P+g];
 double aa=pars[9*P+g],ag=pars[10*P+g],dg=pars[11*P+g],r=alpha*y[g]+(1-alpha)*y[P+g];
 double z=pars[12*P+g];
 double target=gains[tick*3*P+g]*(y[3*P+g]-z*y[5*P+g]-y[8*P+g])+gains[tick*3*P+P+g]*y[6*P+g]+gains[tick*3*P+2*P+g]*z*z*y[7*P+g];
 out[g]=(target-y[g])/tf;out[P+g]=(target-y[P+g])/ts;
 out[2*P+g]=(tm*aa*arr[g]-y[2*P+g])/.7;out[3*P+g]=(y[2*P+g]-y[3*P+g])/3.5;
 out[4*P+g]=tm*ag*arr[P+g]-y[4*P+g];out[5*P+g]=(y[4*P+g]-y[5*P+g])/dg;
 out[6*P+g]=(tm*aa*aa*arr[2*P+g]-y[6*P+g])/2.1;
 out[7*P+g]=(tm*ag*ag*arr[3*P+g]-y[7*P+g])/((1+dg)/2);
 out[8*P+g]=(.5*E*r-y[8*P+g])/1000.;
}
'''
        names=['delayed_linear','tangent_rhs','predictor','finish'];mod=cp.RawModule(code=code,options=('--fmad=false',),name_expressions=names)
        self.k={n:mod.get_function(n) for n in names};self.y=cp.zeros((9,s.P));self.hist=cp.zeros((self.depth,s.P))
        self.arr=cp.zeros((4,s.P));self.f=cp.zeros_like(self.y);self.f2=cp.zeros_like(self.y);self.pred=cp.zeros_like(self.y)
        self.order=cp.asarray((-np.arange(1,self.D+1))%self.depth);self.finalorder=cp.asarray((self.n-np.arange(1,self.D+1))%self.depth)
        self.stream=cp.cuda.Stream(non_blocking=True);cp.cuda.get_current_stream().synchronize()
        with self.stream:
            self.step(0)  # compile before graph capture
        self.stream.synchronize()
        with self.stream:
            self.stream.begin_capture()
            for i in range(self.n):self.step(i)
            self.graph=self.stream.end_capture()
        self.calls=0;self.start=time.time()
        # Fourier operators are only needed to initialize the gains. Do not
        # retain their freed device blocks during the long monodromy solve.
        cp.get_default_memory_pool().free_all_blocks()

    def step(self,i):
        p=self.s.P;n=(p+127)//128
        self.k['delayed_linear']((p,),(128,),(*self.ops,self.hist,self.arr,np.int32(i),np.int32(self.depth),self.dt))
        self.k['tangent_rhs']((n,),(128,),(self.y,self.arr,self.pars,self.gains,self.f,np.int32(i)))
        self.k['predictor'](((9*p+127)//128,),(128,),(self.y,self.f,self.pred,self.dt))
        self.k['delayed_linear']((p,),(128,),(*self.ops,self.hist,self.arr,np.int32(i+1),np.int32(self.depth),self.dt))
        self.k['tangent_rhs']((n,),(128,),(self.pred,self.arr,self.pars,self.gains,self.f2,np.int32(i+1)))
        self.k['finish']((n,),(128,),(self.y,self.f,self.f2,self.pars,self.hist,self.dt,np.int32(i+1),np.int32(self.depth)))

    def matvec(self,x):
        cp=self.cp;p=self.s.P
        with self.stream:
            x=cp.asarray(x);self.y[:]=x[:9*p].reshape(9,p);self.hist[self.order]=x[9*p:].reshape(self.D,p)
            self.hist[0]=self.pars[3]*self.y[0]+(1-self.pars[3])*self.y[1]
            self.graph.launch(stream=self.stream)
            result=cp.concatenate([self.y.ravel(),self.hist[self.finalorder].ravel()])
        self.stream.synchronize();self.calls+=1
        if self.calls%10==0:print('MONODROMY',self.calls,'sec',round(time.time()-self.start,1),flush=True)
        return result.get()

    def phase_vector(self):
        s=self.s;N=len(self.r);cf=np.fft.rfft(self.r,axis=0)/N
        local=np.zeros((9,s.P),complex);history=np.zeros((self.D,s.P),complex)
        for k,v in enumerate(cf):
            if k==0:continue
            lam=2j*np.pi*k/self.T;a,b,qa,qb=s.matrices(self.J,lam)
            target=v/s.filter_response(lam)
            xa=target/(1+lam*s.tf);xb=target/(1+lam*s.ts)
            qav=s.tm*s.area[0]*(a@v)/(1+lam*s.rise[0]);iav=qav/(1+lam*s.decay[0])
            qgv=s.tm*s.area[1]*(b@v)/(1+lam*s.rise[1]);igv=qgv/(1+lam*s.decay[1])
            va=s.tm*s.area[0]**2*(qa@v)/(1+lam*s.tau[0]/2);vg=s.tm*s.area[1]**2*(qb@v)/(1+lam*s.tau[1]/2)
            m=.5*s.E*v/(1+lam*1000);factor=1 if k==N//2 else 2
            local+=factor*lam*np.array([xa,xb,qav,iav,qgv,igv,va,vg,m])
            history+=factor*lam*np.exp(-lam*np.arange(1,self.D+1)[:,None]*self.dt)*v
        return np.r_[local.real.ravel(),history.real.ravel()]


def compute(path,dt=.1,nev=6,device=0,filtered=False):
    s=ZMSpatialRate();m=Monodromy(s,path,dt,device);op=LinearOperator((m.dim,m.dim),matvec=m.matvec,dtype=float)
    phase=m.phase_vector();phase_defect=float(np.linalg.norm(m.matvec(phase)-phase)/np.linalg.norm(phase))
    print('PHASE TANGENT DEFECT',phase_defect,flush=True)
    rho=np.exp(-m.T/1000)
    def polynomial(x):
        ax=m.matvec(x);return m.matvec(ax)-rho*ax
    target=LinearOperator((m.dim,m.dim),matvec=polynomial,dtype=float) if filtered else op
    transformed,vec=eigs(target,k=nev,which='LM',ncv=max(8,2*nev+2) if filtered else max(18,2*nev+2),tol=2e-7,maxiter=120,v0=np.random.default_rng(7301).normal(size=m.dim))
    if filtered:
        vals=np.array([np.vdot(vec[:,j],m.matvec(vec[:,j].real)+1j*m.matvec(vec[:,j].imag))/np.vdot(vec[:,j],vec[:,j]) for j in range(nev)])
    else:vals=transformed
    idx=np.argsort(abs(vals))[::-1];vals=vals[idx];vec=vec[:,idx];res=[]
    for i,v in enumerate(vals):
        av=m.matvec(vec[:,i].real)+1j*m.matvec(vec[:,i].imag)
        res.append(float(np.linalg.norm(av-v*vec[:,i])/np.linalg.norm(vec[:,i])))
    row=dict(orbit=str(path),D=m.depletion,J_EE_core=m.J,T_ms=m.T,dt_ms=m.dt,history_dimension=m.dim,
        multipliers=vals,residuals=res,phase_multiplier_error=float(min(abs(vals-1))),
        phase_tangent_relative_defect=phase_defect,
        raw_multipliers_above_1p001=int(np.sum(abs(vals)>1.001)),requested_multipliers=nev,
        method='Full variational DDE monodromy, Arnoldi largest modulus',seconds=time.time()-m.start)
    if filtered:row.update(polynomial_filter_rho=rho,transformed_eigenvalues=transformed[idx],
        filter_coverage_threshold=1-rho,smallest_returned_transformed_modulus=float(min(abs(transformed))),
        filter_note='K=M(M-rho I). Every |mu|>=1 has |mu(mu-rho)|>=1-rho. Recover mu using M and verify residuals; coverage requires reaching below this transformed threshold.')
    overlaps=abs(vec.conj().T@phase)/(np.linalg.norm(vec,axis=0)*np.linalg.norm(phase))
    neutral=int(np.argmin(abs(vals-1)))
    if nev==1 or abs(vals[neutral]-1)>.01 or overlaps[neutral]<.995:neutral=None
    nontrivial=np.delete(vals,neutral) if neutral is not None else vals
    row.update(phase_eigenvector_overlaps=overlaps,identified_neutral_index=neutral,
        nontrivial_multipliers=nontrivial,largest_computed_nontrivial_modulus=float(max(abs(nontrivial))),
        stability_note='Exclude the autonomous phase multiplier; compare phase error under time-step refinement before interpreting crossings near +1.')
    dest=PERIODIC_OUT/'floquet';dest.mkdir(exist_ok=True)
    name=f'{Path(path).stem}_dt{dt:g}'+('' if filtered else '_raw');write(dest/f'{name}.json',row)
    np.savez_compressed(dest/f'{name}.npz',multipliers=vals,local_vectors=vec[:9*s.P],rate_history_vectors=vec[9*s.P:],dt=m.dt)
    print('FLOQUET',row,flush=True);return row


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('--dt',type=float,default=.1);p.add_argument('--nev',type=int,default=6);p.add_argument('--device',type=int,default=0);p.add_argument('--filtered',action='store_true')
    a=p.parse_args();compute(a.path,a.dt,a.nev,a.device,a.filtered)
