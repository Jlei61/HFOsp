"""Variational return spectra of physical-D invariant periodic waveforms.

The 0.1-ms map is differentiated analytically, including its delay registers,
mean/variance filters and M. Noninteger periods use polynomial interpolation
of the final state, checked by changing interpolation order and by the neutral
phase vector. These are numerical Floquet estimates, not integer-step cycles.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time
import numpy as np
import torch
from scipy import sparse
from scipy.signal import resample
from scipy.sparse.linalg import LinearOperator,eigs
from topic4_fig5_D_physical_model import Orbit,OUT
from topic4_fig5_z_frozen_v1 import filters
from topic4_fig5_z_frequency_parallel import install
from topic4_fig5_D_gpu_map import GPUMap

def fourier_values(x,times,T):
    N=len(x);f=np.fft.rfft(x,axis=0)/N;k=np.arange(len(f))
    f[1:]*=2
    if N%2==0:f[-1]*=.5
    return (np.exp(2j*np.pi*np.asarray(times)[:,None]*k[None,:]/T)@f).real

class ReturnMap(GPUMap):
    def __init__(self,o,r,T,device=0,degree=3):
        self.o=o;self.T=T;self.degree=degree;m=o.m
        m.vops={k:sparse.load_npz(m.folder/f'vdelay_{k}.npz').tocsr() for k in m.ops}
        m.vops['ei']*=o.eq.q_ie**2
        super().__init__(o.eq,[m.state_dict()],device=device,compile_step=False)
        self.shapes=[(m.n*m.K,),(m.n,),(m.n*m.K,),(4,m.n),(4,m.n),(4,3,m.n),(m.D,m.n),(m.D,m.n)]
        self.sizes=[int(np.prod(s)) for s in self.shapes];self.cuts=np.cumsum([0]+self.sizes);self.dimension=sum(self.sizes)
        self.z=self.tensor(o.z[None]);self.z2=self.tensor(o.z2[None])
        self.Kstep=int(np.floor(T/m.dt));self.nodes=np.arange(self.Kstep-degree//2,self.Kstep-degree//2+degree+1)
        x=T/m.dt;weights=[]
        for k in self.nodes:weights.append(np.prod([(x-j)/(k-j) for j in self.nodes if j!=k]))
        self.interp=dict(zip(self.nodes,weights));steps=self.nodes[-1]
        H,le,li,lm=o.kernels(T);vals=o.inputs(r[:,:o.U],r[:,o.U:],H,lm)
        vals[0]+=m.te*m.gaA*m.je*m.nu_sig;vals[1]+=m.te*m.je**2*m.nu_sig
        vals[3]+=m.ti*m.gaA*m.ji*m.nu_sig;vals[4]+=m.ti*m.ji**2*m.nu_sig
        ts=np.arange(steps)*m.dt;vv=[fourier_values(v,ts,T) for v in vals]
        # Reuse the analytic transfer derivatives with the actual number of
        # time samples, without rebuilding any Fourier kernels.
        Nold=o.N;thE=o.thE;thI=o.thI;o.N=steps;o.thE=np.tile(m.theta_u,steps);o.thI=np.full(steps*m.n,m.theta_i)
        ge,gi=o.gains(vv);o.N=Nold;o.thE=thE;o.thI=thI
        self.ge=self.tensor(np.stack(ge,axis=1));self.gi=self.tensor(np.stack(gi,axis=1))
        self.variational_update=torch.compile(self._linear_update,fullgraph=True)
        self.calls=0;self.start=time.time()

    def _linear_update(self,r,ri,M,g,c,y,products,ge,gi):
        m=self.m;means=products[:,0].permute(2,0,1);seconds=products[:,1].permute(2,0,1)
        gn=self.a*g+self.amp*means;cn=gn+(c-gn)*self.b
        yn=self.coeff*(y+seconds[:,:,None,:]);v=(yn[:,:,0]+yn[:,:,1]-2*yn[:,:,2])/self.norm
        mu=cn[:,0].repeat_interleave(m.K,dim=1)-self.z*cn[:,1].repeat_interleave(m.K,dim=1)-m.eta_M*M
        ex=m.te*v[:,0].repeat_interleave(m.K,dim=1);inh=m.te*self.z2*v[:,1].repeat_interleave(m.K,dim=1)
        pe=ge[0]*mu+ge[1]*ex+ge[2]*inh
        pi=gi[0]*(cn[:,2]-cn[:,3])+gi[1]*m.ti*v[:,2]+gi[2]*m.ti*v[:,3]
        rn=r+self.rate_step_e*(pe-r);rin=ri+self.rate_step_i*(pi-ri);Mn=(1-m.dt/m.tau_M)*M+m.dt*rn
        return rn,rin,Mn,gn,cn,yn

    def apply(self,x):
        arr=self.tensor(x)
        r,ri,M,g,c,y,hE,hI=[arr[a:b].reshape((1,)+shape) for a,b,shape in zip(self.cuts[:-1],self.cuts[1:],self.shapes)]
        ans=torch.zeros_like(arr)
        for step in range(self.nodes[-1]):
            re=(r*self.w).reshape(1,self.m.n,self.m.K).sum(2);oldri=ri
            histories=torch.cat([hE.reshape(1,-1),hI.reshape(1,-1)],dim=1).T
            products=torch.sparse.mm(self.matrix,histories).reshape(4,2,self.m.n,1)
            r,ri,M,g,c,y=self.variational_update(r,ri,M,g,c,y,products,self.ge[step],self.gi[step])
            hE=torch.cat([re[:,None,:],hE[:,:-1]],dim=1);hI=torch.cat([oldri[:,None,:],hI[:,:-1]],dim=1)
            if step+1 in self.interp:
                ans+=self.interp[step+1]*torch.cat([a.flatten() for a in [r,ri,M,g,c,y,hE,hI]])
        self.calls+=1
        if self.calls%5==0:print('RETURN',self.calls,'wall',time.time()-self.start,flush=True)
        return ans.cpu().numpy()

    def phase_vector(self,r):
        o=self.o;m=self.m;N=len(r);H,le,li,lm=o.kernels(self.T);om=2*np.pi*o.k/self.T;q=np.exp(-1j*om*m.dt)
        # Coefficients of d/dtheta of the rate waveform, in numpy rfft units.
        fr=np.fft.rfft(r,axis=0)*(1j*o.k[:,None]);fu,fi=fr[:,:o.U],fr[:,o.U:]
        fe=(fu*m.w_u).reshape(len(q),m.n,m.K).sum(2)
        def zero(f):
            ff=f.copy();ff[1:]*=2
            if N%2==0:ff[-1]*=.5
            return ff.sum(0).real/N
        rc=zero(fu);ic=zero(fi);mc=zero(fu*lm[:,None]);gs=[];cs=[];ys=[]
        for key in ('ee','ei','ie','ii'):
            ex=key[-1]=='e';targetE=key[0]=='e';tm=m.te if targetE else m.ti
            b=m.adA if ex else m.adG;a=m.arA if ex else m.arG
            src=fe if ex else fi
            cnew=tm*np.einsum('kij,kj->ki',H[key],src)
            cs.append(zero(cnew*q[:,None]));gs.append(zero(cnew*(q*(1-b*q)/(1-b))[:,None]))
            _,_,var,_=filters(m,1j*om,ex);raw=np.einsum('kij,kj->ki',H['v'+key],src)/var[:,None]
            ys.append(np.array([zero(raw*(q*co/(1-co*q))[:,None]) for co in [a*a,b*b,a*b]]))
        ts=-(np.arange(m.D)+1)*m.dt
        dmacro=np.fft.irfft(np.c_[fe,fi],n=N,axis=0)
        history=fourier_values(dmacro,ts,self.T)
        return np.concatenate([a.ravel() for a in [rc,ic,mc,np.array(gs),np.array(cs),np.array(ys),history[:,:m.n],history[:,m.n:]]])

def run(q,core,amp,degree,N,device,nev=3):
    install(4);file=OUT/f'q{q:g}_cycles_{core}_N64/amp{amp:g}_N64.npz'
    if not file.exists():file=OUT/f'q{q:g}_cycles_{core}_N32/amp{amp:g}_N32.npz'
    a=np.load(file);r=resample(a['r'],N,axis=0);T=float(a['T']);D=float(a['s'])
    o=Orbit(D,N,q);res=float(abs(o.evaluate_fixed(r,T)).max()*1000)
    M=ReturnMap(o,r,T,device,degree);phase=M.phase_vector(r);phase/=np.linalg.norm(phase)
    pv=M.apply(phase);phase_error=float(np.linalg.norm(pv-phase));print('PHASE RETURN ERROR',phase_error,'DENSE DEFECT',res,flush=True)
    op=LinearOperator((M.dimension,M.dimension),matvec=M.apply,dtype=float)
    rng=np.random.default_rng(615);vals,vec=eigs(op,k=nev,which='LM',ncv=12,tol=3e-7,maxiter=50,v0=rng.normal(size=M.dimension))
    order=np.argsort(-abs(vals));vals=vals[order];vec=vec[:,order];errs=[]
    for z,v in zip(vals,vec.T):
        out=M.apply(v.real)+1j*M.apply(v.imag) if abs(v.imag).max()>1e-14 else M.apply(v.real)
        errs.append(float(np.linalg.norm(out-z*v)/np.linalg.norm(v)))
    info=dict(q_ie=q,core=core,amplitude_hz=amp,D=D,period_ms=T,phase_nodes=N,interpolation_degree=degree,
      orbit_defect_hz=res,phase_return_error=phase_error,multipliers=[[z.real,z.imag] for z in vals],moduli=abs(vals).tolist(),eigen_residuals=errs,
      method='Full delay/filter/M variational map; interpolated noninteger-period return',source=str(file),wall_s=time.time()-M.start,calls=M.calls)
    path=OUT/f'q{q:g}_floquet_{core}_amp{amp:g}_degree{degree}.json';path.write_text(json.dumps(info,indent=2)+'\n');print('FLOQUET',info,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--q',type=float,default=1.25);p.add_argument('--core',default='a');p.add_argument('--amp',type=float,default=3.2);p.add_argument('--degree',type=int,default=3);p.add_argument('--N',type=int,default=64);p.add_argument('--gpu',type=int,default=0);p.add_argument('--nev',type=int,default=3);a=p.parse_args();run(a.q,a.core,a.amp,a.degree,a.N,a.gpu,a.nev)
