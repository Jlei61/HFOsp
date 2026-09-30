"""Real Floquet exponent by a full-space periodic variational BVP.

Use delta r=exp(growth*t)u(t), with real periodic u. The phase-orthogonal
normalization covector excludes the trivial phase eigenfunction without
removing it from the physical operator. All spatial groups and delays remain.
Independent propagation and mesh checks are required before classification.
"""
from rate_periodic import *
from scipy.sparse.linalg import LinearOperator as HostOperator,gmres as host_gmres


class RealFloquet:
    def __init__(self,s,path,N,device=0,stream_harmonics=False):
        z=np.load(path);self.s=s;self.path=str(path);self.N=N
        self.T=float(z['T']);self.J=float(z['J']);self.o=Periodic(s,N,device)
        o=self.o;o.low_memory=True;cp=o.cp;self.cp=cp
        self.stream_harmonics=stream_harmonics;o.stream_harmonics=stream_harmonics
        o.harmonic_chunk_size=64
        self.r=cp.asarray(resample(z['r'],N,axis=0))
        mom=o.moments(self.r,o.kernels(self.T,self.J))+o.private[:,None,:]
        gains=[]
        for k in range(3):
            step=1e-5*cp.maximum(cp.abs(mom[k]),1.);hi=mom.copy();lo=mom.copy()
            hi[k]+=step;lo[k]-=step;gains.append((o.phi(hi)-o.phi(lo))/(2*step))
        self.gains=cp.stack(gains)
        self.phase=cp.fft.irfft(cp.fft.rfft(self.r,axis=0)*(2j*np.pi*cp.arange(o.K)[:,None]/self.T),n=N,axis=0)
        o.cache=None;o.cache_key=None;self.last=None;self.cache=None
        del mom,gains,hi,lo;cp.get_default_memory_pool().free_all_blocks()

    def kernels(self,growth):
        if self.last==growth:return self.cache
        from cupyx.scipy.sparse.linalg import LinearOperator
        self.cache=None;cp=self.cp;o=self.o;s=self.s;K=o.K
        z=growth+2j*np.pi*cp.arange(K)[:,None]/self.T
        phase=cp.exp(-cp.asarray(s.delays)[:,None]*z[:,0])
        phasep=-cp.asarray(s.delays)[:,None]*phase;ops=[];der=[]
        for k,(d,mask,index,ptr) in enumerate(o.raw):
            scale=cp.where(mask,self.J**(1 if k==0 else 2),1.) if k in (0,2) else 1.
            shape=(K*s.P,K*s.P)
            def streamed(ph,d=d,scale=scale,index=index,ptr=ptr):
                def action(v):
                    result=cp.empty_like(v);edges=d.shape[0]
                    for first in range(0,K,64):
                        last=min(K,first+64);data=(d@ph[:,first:last]).T.copy()*scale
                        ii=index[first*edges:last*edges]-first*s.P
                        pp=ptr[first*s.P:last*s.P+1]-first*edges
                        matrix=o.cs.csr_matrix((data.ravel(),ii,pp),shape=((last-first)*s.P,)*2)
                        result[first*s.P:last*s.P]=matrix@v[first*s.P:last*s.P]
                    return result
                return LinearOperator(shape,matvec=action,dtype=complex)
            ops.append(streamed(phase) if self.stream_harmonics else
                o.cs.csr_matrix((((d@phase).T.copy()*scale).ravel(),index,ptr),shape=shape))
            der.append(streamed(phasep))
        ha=1/((1+z*s.rise[0])*(1+z*s.decay[0]));hg=1/((1+z*s.rise[1])*(1+z*s.decay[1]))
        hva=1/(1+z*s.tau[0]/2);hvg=1/(1+z*s.tau[1]/2);hm=1/(1+z*1000)
        hp=-ha*(s.rise[0]/(1+z*s.rise[0])+s.decay[0]/(1+z*s.decay[0]))
        gp=-hg*(s.rise[1]/(1+z*s.rise[1])+s.decay[1]/(1+z*s.decay[1]))
        tm,ref,th,alpha,tf,ts,E=o.gpars
        H=alpha/(1+z*tf)+(1-alpha)/(1+z*ts)
        Hd=-alpha*tf/(1+z*tf)**2-(1-alpha)*ts/(1+z*ts)**2
        # The legacy moments() derivative slot is algebraic: here it
        # contains growth derivatives, rather than derivatives of log T.
        self.cache=(ops,der,None,(ha,hg,hva,hvg,hm),
            (hp,gp,-hva*hva*s.tau[0]/2,-hvg*hvg*s.tau[1]/2,-hm*hm*1000),H,Hd)
        self.last=growth;return self.cache

    def apply(self,u,k):
        return u-self.o.filt(self.cp.sum(self.gains*self.o.moments(u,k),axis=0),k[-2])

    def derivative(self,u,k):
        o=self.o;cp=self.cp
        return -o.filt(cp.sum(self.gains*o.moments(u,k),axis=0),k[-1])-o.filt(cp.sum(self.gains*o.moments(u,k,'T'),axis=0),k[-2])

    def checks(self,u):
        cp=self.cp;o=self.o;u=cp.asarray(u)
        k=self.kernels(0.);v=self.apply(u,k)
        original=o.kernels(self.T,self.J)
        reference=u-o.filt(cp.sum(self.gains*o.moments(u,original),axis=0),original[-2])
        same=float(cp.linalg.norm(v-reference)/cp.linalg.norm(u));del original
        o.cache=None;o.cache_key=None
        growth=-4e-5;k=self.kernels(growth);analytic=self.derivative(u,k);del k
        errors=[]
        for h in [1e-6,5e-7]:
            hi=self.apply(u,self.kernels(growth+h));lo=self.apply(u,self.kernels(growth-h))
            errors.append(float(cp.linalg.norm((hi-lo)/(2*h)-analytic)/cp.linalg.norm(analytic)))
        assert same<1e-11 and errors[-1]<1e-5 and errors[0]>3*errors[1],(same,errors)
        return dict(zero_growth_operator_difference=same,growth_derivative_relative_errors=errors)

    def refine(self,u,growth):
        cp=self.cp;u=cp.asarray(u);phase=self.phase
        q=u-phase*(cp.sum(phase*u)/cp.sum(phase*phase));q/=cp.linalg.norm(q)
        overlap=float(abs(cp.sum(q*phase))/cp.linalg.norm(phase));assert overlap<1e-10
        u/=cp.sum(q*u);history=[];dim=u.size;scale=.001
        for it in range(16):
            k=self.kernels(growth);f=self.apply(u,k);constraint=cp.sum(q*u)-1
            error=float(cp.linalg.norm(f)/cp.linalg.norm(u));history.append(error)
            print('REAL FLOQUET',it,growth,np.exp(growth*self.T),error,flush=True)
            if error<2e-9 and abs(float(constraint))<1e-8:break
            dl=self.derivative(u,k)
            def mv(x):
                xx=cp.asarray(x);du=xx[:-1].reshape(u.shape)
                return cp.r_[(self.apply(du,k)+scale*dl*xx[-1]).ravel(),cp.sum(q*du)].get()
            op=HostOperator((dim+1,dim+1),matvec=mv,dtype=float)
            rhs=np.r_[-f.get().ravel(),-float(constraint)];norm=np.linalg.norm(rhs)
            count=[0]
            def callback(value):
                count[0]+=1
                if count[0]%100==0:print('REAL KRYLOV',count[0],value,flush=True)
            dy,info=host_gmres(op,rhs/norm,rtol=1e-8,atol=0.,restart=160,maxiter=1800,
                callback=callback,callback_type='legacy');dy*=norm
            residual=np.linalg.norm(op@dy-rhs)/norm
            print('REAL LINEAR',info,residual,flush=True)
            if residual>1e-5:break
            u+=cp.asarray(dy[:-1].reshape(u.shape));growth+=scale*dy[-1]
            del op,dl,k;cp.get_default_memory_pool().free_all_blocks()
        err=float(cp.linalg.norm(self.apply(u,self.kernels(growth)))/cp.linalg.norm(u))
        return growth,u.get(),dict(residual=err,history=history,
            normalization_error=abs(float(cp.sum(q*u))-1),covector_phase_overlap=overlap,
            mode_phase_overlap=float(abs(cp.sum(phase*u))/(cp.linalg.norm(phase)*cp.linalg.norm(u))))


def main():
    p=argparse.ArgumentParser();p.add_argument('orbit');p.add_argument('--seed',required=True)
    p.add_argument('--N',type=int,default=1024);p.add_argument('--growth',type=float,default=-4e-5)
    p.add_argument('--device',type=int,default=0);p.add_argument('--label',required=True)
    p.add_argument('--antiperiodic-seed',action='store_true')
    p.add_argument('--stream-harmonics',action='store_true');a=p.parse_args()
    z=np.load(a.seed);u=z['u']
    if a.antiperiodic_seed:u=np.r_[u,-u]
    u=resample(u.real,a.N,axis=0);f=RealFloquet(RateField(),a.orbit,a.N,a.device,a.stream_harmonics)
    checks=f.checks(u);growth,u,info=f.refine(u,a.growth)
    row=dict(status='EIGENPAIR_CONVERGED_CHECKS_PENDING' if info['residual']<2e-9 else 'NOT_CONVERGED',
        label=a.label,orbit=a.orbit,N=a.N,J_EE_core=f.J,T_ms=f.T,
        streamed_harmonic_actions=a.stream_harmonics,
        lambda_per_ms=growth,multiplier=float(np.exp(growth*f.T)),implementation_checks=checks,**info,
        scope='One real Floquet eigenpair from a periodic BVP. No complete spectrum or stability classification; independently check the mode in full state/history and refine temporal resolution.')
    write(PERIODIC_OUT/f'{a.label}_N{a.N}.json',row)
    save_periodic_array(PERIODIC_OUT/f'{a.label}_mode_N{a.N}.npz',u=u,lam=growth,J=f.J,T=f.T)
    print('REAL FLOQUET RESULT',row,flush=True)


if __name__=='__main__':main()
