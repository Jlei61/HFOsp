"""Fourier boundary-value continuation of the full spatial rate DDE.

Only linear synaptic/rate/adaptation states are eliminated algebraically.
Every original spatial group and every delay survives at every harmonic.
This is a periodic BVP, not extrema collected from a time simulation.
"""
from rate_hopf_normal_form import PERIODIC_OUT
from rate_field import *
import argparse
from scipy.signal import resample,find_peaks
from scipy.interpolate import CubicSpline


class Periodic:
    def __init__(self,s,N,device=0):
        import cupy as cp
        from cupyx.scipy import sparse as cs
        self.cp=cp;self.cs=cs;cp.cuda.Device(device).use();self.s=s;self.N=N;self.K=N//2+1
        self.raw=[]
        for row,col,mask,d in s.raw:
            counts=np.bincount(row,minlength=s.P);ind=np.r_[0,np.cumsum(counts)]
            index=(col[None,:]+np.arange(self.K)[:,None]*s.P).ravel().astype(np.int32)
            ptr=np.r_[np.concatenate([ind[:-1]+k*len(col) for k in range(self.K)]),self.K*len(col)].astype(np.int32)
            self.raw.append((cs.csr_matrix(d),cp.asarray(mask),cp.asarray(index),cp.asarray(ptr)))
        self.gpars=[cp.asarray(x) for x in (s.tm,s.ref,s.theta,s.alpha,s.tf,s.ts,s.E)]
        self.private=cp.asarray(np.array([s.private_mu,s.private_ve,np.zeros(s.P)]))
        self.code=cp.RawModule(code=cuda_code(s)+r'''
extern "C" __global__ void phi_periodic(const double* mom,const double* pars,double* out,int N){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N*P)return;int g=i%P;
 double tm=pars[g],ref=pars[P+g],th=pars[2*P+g],dg=pars[11*P+g];
 double mu=mom[i],ve=mom[N*P+i],vi=mom[2*N*P+i],var=fmax(ve+vi,1e-12),sig=sqrt(var);
 double teff=var/fmax(ve/4.2+vi/(1+dg),1e-12),shift=1.0325*sqrt(teff/tm);
 double lo=(11.-mu)/sig+shift,hi=(th-mu)/sig+shift,target=0.;
 if(hi<26.){double sum=0.;for(int k=0;k<NQ;k++){double x=(hi+lo)/2+(hi-lo)/2*X[k];sum+=W[k]*ex_erfc(-x);}
 target=1/(ref+tm*sqrt(3.141592653589793)*(hi-lo)/2*sum);}
 out[i]=target;
}
''',options=('--fmad=false',),name_expressions=['phi_periodic'])
        self.phik=self.code.get_function('phi_periodic')
        self.pars=cp.asarray(np.array([s.tm,s.ref,s.theta,s.alpha,s.tf,s.ts,s.private_mu,s.private_ve,s.E,
            np.full(s.P,s.area[0]),np.full(s.P,s.area[1]),np.full(s.P,s.decay[1])]))
        self.cache_key=None

    def kernels(self,T,J):
        if self.cache_key==(T,J):return self.cache
        self.cache=None;self.cache_key=None
        cp=self.cp;s=self.s;lam=2j*np.pi*cp.arange(self.K)[:,None]/T;phase=cp.exp(-cp.asarray(s.delays)[:,None]*lam[:,0])
        phasep=phase*(cp.asarray(s.delays)[:,None]*lam[:,0]);ops=[];opst=[];opsj=[]
        for k,(d,mask,index,ptr) in enumerate(self.raw):
            vals=(d@phase).T.copy();vt=(d@phasep).T.copy();scale=cp.where(mask,J**(1 if k==0 else 2),1.) if k in (0,2) else 1.
            make=lambda x:self.cs.csr_matrix((x.ravel(),index,ptr),shape=(self.K*s.P,self.K*s.P))
            ops.append(make(vals*scale));opst.append(make(vt*scale))
            opsj.append(make(vals*mask*(1 if k==0 else 2*J)) if k in (0,2) else None)
        ha=1/((1+lam*s.rise[0])*(1+lam*s.decay[0]));hg=1/((1+lam*s.rise[1])*(1+lam*s.decay[1]))
        hva=1/(1+lam*s.tau[0]/2);hvg=1/(1+lam*s.tau[1]/2);hm=1/(1+lam*1000)
        hat=ha*(lam*s.rise[0]/(1+lam*s.rise[0])+lam*s.decay[0]/(1+lam*s.decay[0]))
        hgt=hg*(lam*s.rise[1]/(1+lam*s.rise[1])+lam*s.decay[1]/(1+lam*s.decay[1]))
        hvat=hva**2*lam*s.tau[0]/2;hvgt=hvg**2*lam*s.tau[1]/2;hmt=hm**2*lam*1000
        tm,ref,th,alpha,tf,ts,E=self.gpars
        H=alpha/(1+lam*tf)+(1-alpha)/(1+lam*ts)
        Ht=alpha*lam*tf/(1+lam*tf)**2+(1-alpha)*lam*ts/(1+lam*ts)**2
        self.cache_key=(T,J);self.cache=(ops,opst,opsj,(ha,hg,hva,hvg,hm),(hat,hgt,hvat,hvgt,hmt),H,Ht)
        return self.cache

    def moments(self,r,kernels,mode='normal'):
        cp=self.cp;s=self.s;ops,opst,opsj,fil,ft,H,Ht=kernels
        rf=cp.fft.rfft(r,axis=0);v=rf.ravel();tm,ref,th,alpha,tf,ts,E=self.gpars
        def apply(operators,filters,adapt=True):
            a,b,qa,qb=[(o@v).reshape(self.K,s.P) if o is not None else cp.zeros_like(rf) for o in operators]
            ha,hg,hva,hvg,hm=filters
            return cp.stack([tm*(s.area[0]*ha*a-s.area[1]*hg*b)-(.5*E*hm*rf if adapt else 0),
                tm*s.area[0]**2*hva*qa,tm*s.area[1]**2*hvg*qb])
        if mode=='normal':out=apply(ops,fil)
        elif mode=='T':out=apply(opst,fil,False)+apply(ops,ft)
        elif mode=='J':out=apply(opsj,fil,False)
        return cp.fft.irfft(out,n=self.N,axis=1)

    def phi(self,mom):
        cp=self.cp;out=cp.empty((self.N,self.s.P));self.phik(((self.N*self.s.P+127)//128,),(128,),
            (cp.ascontiguousarray(mom),self.pars,out,np.int32(self.N)));return out

    def filt(self,r,H):return self.cp.fft.irfft(self.cp.fft.rfft(r,axis=0)*H,n=self.N,axis=0)

    def evaluate(self,y,reference,phase,J,amplitude=None,derivative=False,arc=None):
        from cupyx.scipy.sparse.linalg import LinearOperator
        cp=self.cp;s=self.s;extra=2 if amplitude is not None or arc is not None else 1
        r=y[:-extra].reshape(self.N,s.P)*.001;T=float(cp.exp(y[-extra]));J=float(y[-1]*.001) if extra==2 else J
        kernels=self.kernels(T,J);H,Ht=kernels[-2:];mom=self.moments(r,kernels)+self.private[:,None,:]
        phi=self.phi(mom);F=(r-self.filt(phi,H))/.001
        constraints=[cp.sum((r-reference)*phase)/.001]
        if amplitude is not None:
            q,target=amplitude;projection=cp.vdot(q,cp.fft.rfft(r,axis=0)[1]/self.N)/cp.vdot(q,q)
            constraints.append(projection.real-target)
        if arc is not None:
            yp,tan,weight=arc;constraints.append(cp.sum((y-yp)*tan*weight**2))
        residual=cp.concatenate([F.ravel(),cp.asarray(constraints)])
        if not derivative:return residual
        gains=[]
        for k in range(3):
            step=1e-5*cp.maximum(cp.abs(mom[k]),1.);hi=mom.copy();lo=mom.copy();hi[k]+=step;lo[k]-=step
            gains.append((self.phi(hi)-self.phi(lo))/(2*step))
        gains=cp.stack(gains)
        colT=-(self.filt(cp.sum(gains*self.moments(r,kernels,'T'),axis=0),H)+self.filt(phi,Ht))/.001
        colJ=-self.filt(cp.sum(gains*self.moments(r,kernels,'J'),axis=0),H) if extra==2 else None
        def matvec(dy):
            dr=dy[:-extra].reshape(self.N,s.P)*.001
            dphi=cp.sum(gains*self.moments(dr,kernels),axis=0)
            out=(dr-self.filt(dphi,H))/.001+colT*dy[-extra]
            if extra==2:out+=colJ*dy[-1]
            cc=[cp.sum(dr*phase)/.001]
            if amplitude is not None:cc.append((cp.vdot(q,cp.fft.rfft(dr,axis=0)[1]/self.N)/cp.vdot(q,q)).real)
            if arc is not None:cc.append(cp.sum(dy*tan*weight**2))
            return cp.concatenate([out.ravel(),cp.asarray(cc)])
        return residual,LinearOperator((len(y),len(y)),matvec=matvec,dtype=np.float64),dict(moments=mom,gains=gains)

    def solve(self,r,T,J,amplitude=None,maxiter=18,tol=2e-8,arc=None):
        from cupyx.scipy.sparse.linalg import gmres
        cp=self.cp;r=cp.asarray(r);reference=r.copy();df=2j*np.pi*cp.arange(self.K)[:,None]*cp.fft.rfft(r,axis=0)
        dr=cp.fft.irfft(df,n=self.N,axis=0);phase=dr/cp.sum(dr*dr)*.001
        if amplitude is not None:amplitude=(cp.asarray(amplitude[0]),amplitude[1])
        if arc is not None:arc=tuple(cp.asarray(x) for x in arc)
        extra=2 if amplitude is not None or arc is not None else 1
        y=cp.concatenate([(r/.001).ravel(),cp.asarray([np.log(T),J/.001] if extra==2 else [np.log(T)])]);history=[]
        start=time.time()
        for it in range(maxiter):
            F,A,data=self.evaluate(y,reference,phase,J,amplitude,True,arc);err=float(cp.max(cp.abs(F)));history.append(err)
            print('BVP',self.N,it,'J',float(y[-1]*.001) if extra==2 else J,'T',float(cp.exp(y[-extra])),'err',err,'s',round(time.time()-start,1),flush=True)
            if err<tol:break
            dy,info=gmres(A,-F,tol=min(.02,max(1e-7,err*.02)),atol=1e-11,
                restart=120 if self.N>=512 else 70,maxiter=1200 if self.N>=512 else 420)
            print('linear',info,float(cp.linalg.norm(A@dy+F)/cp.linalg.norm(F)),flush=True)
            del A,data
            cp.get_default_memory_pool().free_all_blocks()
            for back in range(12):
                yy=y+dy*2.**-back
                if abs(float(yy[-extra]-y[-extra]))>.5:continue
                ff=self.evaluate(yy,reference,phase,J,amplitude,arc=arc)
                if float(cp.linalg.norm(ff))<float(cp.linalg.norm(F)):y=yy;break
            else:break
        error=float(cp.max(cp.abs(self.evaluate(y,reference,phase,J,amplitude,arc=arc))))
        return y[:-extra].reshape(self.N,self.s.P).get()*.001,float(cp.exp(y[-extra])),float(y[-1]*.001) if extra==2 else J,error,history


def save_orbit(s,r,T,J,error,history,name):
    dest=PERIODIC_OUT/'orbits';dest.mkdir(exist_ok=True,parents=True)
    path=dest/f'{name}.npz';np.savez_compressed(path,r=r,T=T,J=J,residual=error,history=history)
    rr=np.array([s.regional_rates(x) for x in r]);row=dict(path=str(path),N=len(r),J_EE_core=J,T_ms=T,
        residual_hz=error,mean_rates_hz=rr.mean(0),min_rates_hz=rr.min(0),max_rates_hz=rr.max(0),
        status='CONVERGED' if error<2e-8 else 'NOT_CONVERGED',method='Full-space Fourier periodic boundary-value solve')
    write(path.with_suffix('.json'),row);print('SAVED',row,flush=True);return path


def main():
    p=argparse.ArgumentParser();p.add_argument('--core',choices=list('AB'));p.add_argument('--amplitudes',type=float,nargs='+',default=[.1,.2,.4,.8,1.2])
    p.add_argument('--N',type=int,default=32);p.add_argument('--J',type=float);p.add_argument('--from-orbit');p.add_argument('--device',type=int,default=0)
    p.add_argument('--label',default='branch');p.add_argument('--trajectory');p.add_argument('--period-bursts',type=int)
    args=p.parse_args();s=RateField();o=Periodic(s,args.N,args.device)
    if args.core:
        z=np.load(PERIODIC_OUT/f'normal_form_{args.core}.npz');q=z['q'];J0=float(z['J']);r0=z['r'];w=float(z['w'])
        a0=None
        if args.from_orbit:
            start=np.load(args.from_orbit);r=resample(start['r'],args.N,axis=0);T=float(start['T']);J=float(start['J'])
            a0=float((np.vdot(q,np.fft.rfft(r,axis=0)[1]/args.N)/np.vdot(q,q)).real)
        for amp in args.amplitudes:
            if a0 is None:
                nf=read(PERIODIC_OUT/f'normal_form_{args.core}.json');J=J0+nf['J_shift_per_amplitude_squared']*amp**2
                T=2*np.pi/(w+nf['omega_shift_per_amplitude_squared']*amp**2);eq,ok,_=s.solve(J,r0);assert ok
                phase=np.exp(2j*np.pi*np.arange(args.N)/args.N)[:,None]
                r=eq+2*amp*np.real(phase*q)+amp**2*(z['h11'].real+np.real(phase**2*z['h20']))
            else:r=r.mean(0)+(r-r.mean(0))*amp/a0
            r,T,J,error,history=o.solve(r,T,J,amplitude=(q,amp));save_orbit(s,r,T,J,error,history,f'H{args.core}_a{amp:.5f}_N{args.N}')
            if error>2e-8:break
            a0=amp
    elif args.from_orbit:
        z=np.load(args.from_orbit);r=resample(z['r'],args.N,axis=0);T=float(z['T']);J=float(z['J']) if args.J is None else args.J
        r,T,J,error,history=o.solve(r,T,J);save_orbit(s,r,T,J,error,history,f'{args.label}_J{J:.9f}_N{args.N}')
    else:
        J=args.J;label='long' if J==.942 else 'main';z=np.load(args.trajectory or RATE_OUT/f'runs/{label}/J{J:.7f}/trajectory.npz')
        pk=find_peaks(z['regional_rates_hz'][:,0],height=20,distance=100)[0];cycles=args.period_bursts or (2 if J==.942 else 1);a,b=pk[-cycles-1],pk[-1]
        T=float(b-a);r=CubicSpline(np.arange(a-5,b+6)-a,z['group_rate_hz'][a-5:b+6]/1000)(np.arange(args.N)*T/args.N)
        r,T,J,error,history=o.solve(r,T,J);save_orbit(s,r,T,J,error,history,f'{args.label}_J{J:.9f}_N{args.N}')


if __name__=='__main__':main()
