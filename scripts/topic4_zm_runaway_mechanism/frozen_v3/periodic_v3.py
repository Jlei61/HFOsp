"""Fourier collocation periodic BVP for the v3 model, rate-only unknowns (full space, Newton-GMRES on GPU).

All filters are LTI (fixed poles) and are eliminated in the harmonic domain together with the
physical delay operators; only the mixing weights (alpha, eta_E, eta_I) depend on the instantaneous
moments. Unknowns: r (N x P, Hz), log T, [D/1e-3]. Residual F = (r - phi(mu_eff, vE_eff, vI_eff))/1e-3 with
  mu_eff = a mu + (1-a) mu_s + eta_E (vE - vE_f) + eta_I (vI - vI_f),  vc_eff = a_c vc + (1-a_c) vc_v,
  (a, a_E, a_I, eta_E, eta_I) = tables(mu, vE, vI); mu_s, vc_f, vc_v LTI filters of the moments
plus a phase condition and optionally an amplitude/arclength condition. Z held (conditional), M dynamic.
"""
from dynamics_v3 import *
from response_tables import CUDA_RESP
from scipy.signal import resample,find_peaks
from scipy.interpolate import CubicSpline
import argparse
PERIODIC_OUT=DEST/'periodic'
RS=1e-3

class PeriodicV3:
    def __init__(self,s,N,device=0):
        import cupy as cp
        from cupyx.scipy import sparse as cs
        self.cp=cp;self.cs=cs;cp.cuda.Device(device).use();self.s=s;self.N=N;self.K=N//2+1
        self.raw=[]
        for row,col,d in s.raw:
            counts=np.bincount(row,minlength=s.P);ind=np.r_[0,np.cumsum(counts)]
            index=(col[None,:]+np.arange(self.K)[:,None]*s.P).ravel().astype(np.int32)
            ptr=np.r_[np.concatenate([ind[:-1]+k*len(col) for k in range(self.K)]),self.K*len(col)].astype(np.int32)
            self.raw.append((cs.csr_matrix(d),cp.asarray(index),cp.asarray(ptr)))
        self.pars,self.consts,self.SE,self.SI,self.WE,self.WI=model_device_arrays(s,cp)
        self.gp=cp.asarray(np.array([s.tm,s.theta,s.E.astype(float),s.private_mu,s.private_ve]))
        self.code=cp.RawModule(code=f'#define P {s.P}\n'+CUDA_DEVICE+CUDA_RESP+r'''
// per (time, group): phi and derivatives at mu_eff, weights and their gradients at the instantaneous moments
extern "C" __global__ void phi_batch(const double* mu,const double* ve,const double* vi,const double* mus,const double* vEf,const double* vIf,const double* vEv,const double* vIv,
 const double* pars,const double* consts,const double* SE,const double* SI,const double* WE,const double* WI,double* out,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int g=i%P;double th=pars[2*P+g];int E=pars[3*P+g]>.5;
 double w[5],gr[15];
 if(consts[9]>.5){resp_weights(E?WE:WI,mu[i],ve[i],vi[i],th,w,gr);}else{for(int p=0;p<5;p++)w[p]=pars[(14+p)*P+g];for(int k=0;k<15;k++)gr[k]=0.;}
 double mueff=w[0]*mu[i]+(1-w[0])*mus[i]+w[3]*(ve[i]-vEf[i])+w[4]*(vi[i]-vIf[i]);
 double vEe=w[1]*ve[i]+(1-w[1])*vEv[i],vIe=w[2]*vi[i]+(1-w[2])*vIv[i];double onE=vEe>0?1.:0.,onI=vIe>0?1.:0.;
 double r,a,b,c;phi_spline(E?SE:SI,mueff,fmax(vEe,0.),fmax(vIe,0.),th,&r,&a,&b,&c);
 out[i]=r;out[n+i]=a;out[2*n+i]=b*onE;out[3*n+i]=c*onI;out[4*n+i]=mueff;
 for(int k=0;k<5;k++)out[(5+k)*n+i]=w[k];for(int k=0;k<15;k++)out[(10+k)*n+i]=gr[k];}
''',options=('--fmad=false',),name_expressions=['phi_batch']);self.phik=self.code.get_function('phi_batch')
        self.cache_key=None
    def kernels(self,T):
        if self.cache_key==T:return self.cache
        cp=self.cp;s=self.s;lam=2j*np.pi*cp.arange(self.K)[:,None]/T;phase=cp.exp(-cp.asarray(s.delays)[:,None]*lam[:,0]);ops=[]
        for d,index,ptr in self.raw:
            vals=(d@phase).T.copy();ops.append(self.cs.csr_matrix((vals.ravel(),index,ptr),shape=(self.K*s.P,self.K*s.P)))
        ha=1/((1+lam*s.rise[0])*(1+lam*s.decay[0]));hg=1/((1+lam*s.rise[1])*(1+lam*s.decay[1]))
        hva=1/(1+lam*s.tau[0]/2);hvg=1/(1+lam*s.tau[1]/2);hm=1/(1+lam*TAU_M)
        tf,ts,tE,tI,tvE,tvI=[cp.asarray(x) for x in s.poles];fs=1/(1+lam*ts[None,:]);fE=1/(1+lam*tE[None,:]);fI=1/(1+lam*tI[None,:]);fvE=1/(1+lam*tvE[None,:]);fvI=1/(1+lam*tvI[None,:])
        self.cache_key=T;self.cache=(ops,(ha,hg,hva,hvg,hm),(fs,fE,fI,fvE,fvI),lam);return self.cache
    def inputs(self,r,T,Z):
        """From r (N,P): mu, vE, vI (instantaneous), mu_s, vE_f, vI_f, vE_v, vI_v (LTI filtered), each (N,P)."""
        cp=self.cp;s=self.s;ops,(ha,hg,hva,hvg,hm),(fs,fE,fI,fvE,fvI),lam=self.kernels(T);rf=cp.fft.rfft(r,axis=0);v=rf.ravel()
        a,b,qa,qb=[(o@v).reshape(self.K,s.P) for o in ops];tm=self.gp[0];E=self.gp[2];z=cp.asarray(Z)
        muh=tm*(s.area[0]*ha*a-z*s.area[1]*hg*b)-.5*E*hm*rf;vEh=tm*s.area[0]**2*hva*qa;vIh=z*z*tm*s.area[1]**2*hvg*qb
        stack=cp.stack([muh,vEh,vIh,muh*fs,vEh*fE,vIh*fI,vEh*fvE,vIh*fvI]);out=cp.fft.irfft(stack,n=self.N,axis=1)
        out[0]+=self.gp[3];out[1]+=self.gp[4];out[3]+=self.gp[3];out[4]+=self.gp[4];out[6]+=self.gp[4];return out
    def phi(self,inp):
        cp=self.cp;n=self.N*self.s.P;out=cp.empty((25,n));args=[cp.ascontiguousarray(x.ravel()) for x in inp]
        self.phik(((n+127)//128,),(128,),(*args,self.pars,self.consts,self.SE,self.SI,self.WE,self.WI,out,np.int32(n)));return out.reshape(25,self.N,self.s.P)
    def residual(self,r,T,Z,details=False):
        inp=self.inputs(r,T,Z);ph=self.phi(inp);F=(r-ph[0])/RS
        return (F,inp,ph) if details else F
    def evaluate(self,y,reference,phase,D,amplitude=None,arc=None,derivative=False):
        from cupyx.scipy.sparse.linalg import LinearOperator
        cp=self.cp;s=self.s;n=self.N*s.P;extra=2 if amplitude is not None or arc is not None else 1
        r=y[:n].reshape(self.N,s.P)*RS;T=float(cp.exp(y[-extra]));Dv=float(y[-1]*1e-3) if extra==2 else D
        s.set_D(Dv);Z=s.Z.copy();F,inp,ph=self.residual(r,T,Z,True)
        constraints=[cp.sum((r-reference)*phase)/RS]
        if amplitude is not None:
            q,target=amplitude;projection=cp.vdot(q,cp.fft.rfft(r,axis=0)[1]/self.N)/cp.vdot(q,q);constraints.append(projection.real-target)
        if arc is not None:
            yp,tan,weight=arc;constraints.append(cp.sum((y-yp)*tan*weight**2))
        res=cp.concatenate([F.ravel(),cp.asarray(constraints)])
        if not derivative:return res
        h=1e-6;colT=(self.residual(r,T*np.exp(h),Z)-self.residual(r,T*np.exp(-h),Z)).ravel()/(2*h)
        colD=None
        if extra==2:
            hD=1e-6;s.set_D(Dv+hD);Zp=s.Z.copy();s.set_D(Dv-hD);Zm=s.Z.copy();s.set_D(Dv);colD=(self.residual(r,T,Zp)-self.residual(r,T,Zm)).ravel()/(2*hD)*1e-3
        pmu,pvE,pvI=ph[1],ph[2],ph[3];w=ph[5:10];gr=ph[10:25].reshape(5,3,self.N,s.P);mu,vE,vI,mus,vEf,vIf,vEv,vIv=inp
        def matvec(dy):
            dr=dy[:n].reshape(self.N,s.P)*RS;dinp=self.inputs(dr,T,Z);dinp[0]-=self.gp[3];dinp[1]-=self.gp[4];dinp[3]-=self.gp[3];dinp[4]-=self.gp[4];dinp[6]-=self.gp[4]
            dmu,dvE,dvI,dmus,dvEf,dvIf,dvEv,dvIv=dinp;dm=cp.stack([dmu,dvE,dvI]);dw=cp.einsum('pcnj,cnj->pnj',gr,dm)
            dmueff=w[0]*dmu+(1-w[0])*dmus+(mu-mus)*dw[0]+w[3]*(dvE-dvEf)+(vE-vEf)*dw[3]+w[4]*(dvI-dvIf)+(vI-vIf)*dw[4]
            dvEe=w[1]*dvE+(1-w[1])*dvEv+(vE-vEv)*dw[1];dvIe=w[2]*dvI+(1-w[2])*dvIv+(vI-vIv)*dw[2]
            dphi=pmu*dmueff+pvE*dvEe+pvI*dvIe
            out=((dr-dphi)/RS).ravel()+colT*dy[-extra]
            if extra==2:out=out+colD*dy[-1]
            cc=[cp.sum(dr*phase)/RS]
            if amplitude is not None:cc.append((cp.vdot(q,cp.fft.rfft(dr,axis=0)[1]/self.N)/cp.vdot(q,q)).real)
            if arc is not None:cc.append(cp.sum(dy*tan*weight**2))
            return cp.concatenate([out,cp.asarray(cc)])
        return res,LinearOperator((len(y),len(y)),matvec=matvec,dtype=np.float64),dict(inputs=inp,phi=ph)
    def solve(self,r,T,D,amplitude=None,arc=None,maxiter=20,tol=2e-8,restart=120,maxit_lin=1500):
        from cupyx.scipy.sparse.linalg import gmres
        cp=self.cp;s=self.s;r=cp.asarray(r);reference=r.copy();df=2j*np.pi*cp.arange(self.K)[:,None]*cp.fft.rfft(r,axis=0);dr=cp.fft.irfft(df,n=self.N,axis=0);phase=dr/cp.sum(dr*dr)*RS
        extra=2 if amplitude is not None or arc is not None else 1
        if amplitude is not None:amplitude=(cp.asarray(amplitude[0]),amplitude[1])
        if arc is not None:arc=tuple(cp.asarray(x) for x in arc)
        y=cp.concatenate([(r/RS).ravel(),cp.asarray([np.log(T),D/1e-3] if extra==2 else [np.log(T)])]);history=[];start=time.time()
        for it in range(maxiter):
            F,A,_=self.evaluate(y,reference,phase,D,amplitude,arc,derivative=True);err=float(cp.max(cp.abs(F)));history.append(err)
            log('BVP',self.N,it,'D %.9f'%(float(y[-1]*1e-3) if extra==2 else D),'T %.6f'%float(cp.exp(y[-extra])),'err %.3e'%err,'s',round(time.time()-start,1))
            if err<tol:break
            dy,info=gmres(A,-F,tol=min(.02,max(1e-7,err*.02)),atol=1e-11,restart=restart,maxiter=maxit_lin)
            lin=float(cp.linalg.norm(A@dy+F)/cp.linalg.norm(F));log('  linear',info,'%.3e'%lin)
            del A;cp.get_default_memory_pool().free_all_blocks()
            for back in range(12):
                yy=y+dy*2.**-back
                if abs(float(yy[-extra]-y[-extra]))>.5:continue
                if extra==2 and not 0<=float(yy[-1]*1e-3)<=1:continue
                ff=self.evaluate(yy,reference,phase,D,amplitude,arc)
                if float(cp.linalg.norm(ff))<float(cp.linalg.norm(F)):y=yy;break
            else:break
        err=float(cp.max(cp.abs(self.evaluate(y,reference,phase,D,amplitude,arc))))
        n=self.N*s.P;rr=y[:n].reshape(self.N,s.P).get()*RS;Dout=float(y[-1]*1e-3) if extra==2 else D
        return dict(r=rr,T=float(cp.exp(y[-extra])),D=Dout,residual=err,history=history,y=y.get())

def save_orbit(s,sol,name,extra=None):
    dest=PERIODIC_OUT/'orbits';dest.mkdir(exist_ok=True,parents=True);path=dest/f'{name}.npz'
    np.savez_compressed(path,r=sol['r'],T=sol['T'],D=sol['D'],residual=sol['residual'],history=sol['history'])
    r=sol['r'];rr=np.array([s.regional_rates(x) for x in r]);g=r[:,s.E]@s.mean_weights*1000
    row=dict(path=str(path),N=len(r),D=sol['D'],T_ms=sol['T'],global_mean_hz=float(g.mean()),global_min_hz=float(g.min()),global_max_hz=float(g.max()),
        smallest_group_rate_hz=float(r.min()*1000),regional_mean_hz=rr.mean(0).tolist(),regional_max_hz=rr.max(0).tolist(),residual=sol['residual'],
        status='CONVERGED' if sol['residual']<2e-8 else 'NOT_CONVERGED',stability='NOT_COMPUTED',dynamic_M=True,frozen_Z=True,response=getattr(s.resp,'source','PLACEHOLDER'))
    if extra:row.update(extra)
    write(path.with_suffix('.json'),row);log('SAVED',name,'D %.9f T %.4f mean %.3f Hz res %.2e'%(sol['D'],sol['T'],row['global_mean_hz'],sol['residual']));return path

def seed_from_trajectory(traj,N,region=0,height=20,distance=100,bursts=1):
    z=np.load(traj);x=z['regional_rates_hz'][:,region];pk=find_peaks(x,height=height,distance=distance)[0]
    assert len(pk)>=bursts+1,'not enough bursts'
    left,right=pk[-bursts-1],pk[-1];T=float(right-left)
    r=CubicSpline(np.arange(left-5,right+6)-left,z['group_rate_hz'][left-5:right+6]/1000)(np.arange(N)*T/N);return r,T

def load_model(device=0):
    resp=ResponseParams(DEST/'response_closure/closure.json') if (DEST/'response_closure/closure.json').exists() else ResponseParams()
    return DynamicModel(resp=resp,quiet=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--N',type=int,default=256);p.add_argument('--D',type=float,required=True);p.add_argument('--trajectory');p.add_argument('--from-orbit')
    p.add_argument('--label',default='cycle');p.add_argument('--device',type=int,default=0);p.add_argument('--bursts',type=int,default=1);p.add_argument('--distance',type=int,default=100);p.add_argument('--region',type=int,default=0)
    a=p.parse_args();s=load_model(a.device);o=PeriodicV3(s,a.N,a.device)
    if a.from_orbit:
        z=np.load(a.from_orbit);r=resample(z['r'],a.N,axis=0);T=float(z['T'])
    else:r,T=seed_from_trajectory(a.trajectory,a.N,region=a.region,distance=a.distance,bursts=a.bursts)
    sol=o.solve(r,T,a.D);save_orbit(s,sol,f'{a.label}_D{sol["D"]:.9f}_N{a.N}')
