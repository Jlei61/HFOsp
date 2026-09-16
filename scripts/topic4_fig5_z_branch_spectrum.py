#!/usr/bin/env python3
"""Matrix-free Chebyshev spectrum of the continuous-time v1 delay equations.

Keeps 3200 E rate and adaptation units, 400 I units, all realized delay bins,
and both poles of each synaptic filter. Discrete DC filter areas are retained.
No stability inference is made from the stationary Jacobian alone.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import json,time,argparse
import numpy as np
from scipy.sparse.linalg import LinearOperator,eigs,ArpackNoConvergence
from scipy.linalg import lu_factor,lu_solve
from topic4_fig5_z_bifurcation_preview import Equilibrium,OUT
from topic4_fig5_z_branch_dynamics import accelerate


class Spectrum:
    def __init__(self,eq,N):
        self.eq=eq;self.m=m=eq.m;self.N=N;n=m.n
        theta=(np.cos(np.arange(N+1)*np.pi/N)-1)*(m.D*m.dt)/2
        bw=(-1.)**np.arange(N+1);bw[[0,-1]]*=.5
        dx=theta[:,None]-theta[None,:];np.fill_diagonal(dx,1.)
        self.D=(bw[None,:]/bw[:,None])/dx;np.fill_diagonal(self.D,0.);np.fill_diagonal(self.D,-self.D.sum(1))
        interp=np.empty((m.D,N+1))
        for k in range(m.D):
            dist=-(k+1)*m.dt-theta;j=np.argmin(abs(dist))
            if abs(dist[j])<1e-10:interp[k]=0.;interp[k,j]=1.
            else:
                a=bw/dist;interp[k]=a/a.sum()
        # One matrix product per E/I target, with histories ordered (node, source).
        self.W=np.zeros((2*n,(N+1)*2*n))
        for key,ti,si in (('ee',0,0),('ei',0,1),('ie',1,0),('ii',1,1)):
            c=m.ops[key].tocoo();d=c.col//n;src=c.col%n;index=c.row*n+src
            for k in range(N+1):
                w=np.bincount(index,weights=c.data*interp[d,k],minlength=n*n).reshape(n,n)
                self.W[ti*n:(ti+1)*n,k*2*n+si*n:k*2*n+(si+1)*n]=w*(m.gaA if si==0 else -m.gaG)
        self.dim=2*n*m.K+n+2*n+(N+1)*2*n

    def operator(self,r,s):
        eq=self.eq;m=self.m;n=m.n;K=m.K;U=n*K
        eq.evaluate(r,s);z,z2=eq.z(s);mu=eq.last['mu'];ex=eq.last['ex'];inh=eq.last['inh']
        h=2e-4
        u=(m.phi_e(mu+h,ex,inh)-m.phi_e(mu-h,ex,inh))/(2*h)
        v=(m.phi_e(mu,ex+h,inh)-m.phi_e(mu,ex-h,inh))/(2*h)
        w=(m.phi_e(mu,ex,inh+h)-m.phi_e(mu,ex,inh-h))/(2*h)
        re,ri=r[:n]/1000,r[n:]/1000
        mui=m.ti*(m.gaA*(m.w_ie@re+m.ji*m.nu_sig)-m.gaG*(m.w_ii@ri))
        ei=m.ti*(m.v_ie@re+m.ji**2*m.nu_sig);ii=m.ti*(m.v_ii@ri)
        ui=(m.phi_i(mui+h,ei,ii)-m.phi_i(mui-h,ei,ii))/(2*h)
        vi=(m.phi_i(mui,ei+h,ii)-m.phi_i(mui,ei-h,ii))/(2*h)
        wi=(m.phi_i(mui,ei,ii+h)-m.phi_i(mui,ei,ii-h))/(2*h)
        self.gains=(u,v,w,ui,vi,wi,z,z2)
        # Split source E/I contributions so target-side Z acts only on inhibition.
        WE=self.W.copy();WI=self.W.copy()
        for k in range(self.N+1):
            WE[:,k*2*n+n:(k+1)*2*n]=0
            WI[:,k*2*n:k*2*n+n]=0
        def apply(x):
            ru=x[:U];ri=x[U:U+n];mm=x[U+n:2*U+n]
            gg=x[2*U+n:2*U+3*n];cc=x[2*U+3*n:].reshape(self.N+1,2*n)
            rr=(ru*m.w_u).reshape(n,K).sum(1)
            ae=WE@cc.ravel();ai=WI@cc.ravel()
            dmu=m.te*(np.repeat(ae[:n],K)+z*np.repeat(ai[:n],K))-m.eta_M*mm
            dru=(-ru+u*dmu+v*np.repeat(m.te*(m.v_ee@rr),K)+w*z2*np.repeat(m.te*(m.v_ei@ri),K))/5.
            dri=(-ri+ui*m.ti*(ae[n:]+ai[n:])+vi*m.ti*(m.v_ie@rr)+wi*m.ti*(m.v_ii@ri))/2.5
            dmm=ru-mm/m.tau_M
            dgg=(np.r_[rr,ri]-gg)/np.r_[np.full(n,m.ra),np.full(n,m.rg)]
            dcc=self.D@cc
            dcc[0]=(gg-cc[0])/np.r_[np.full(n,m.ta),np.full(n,m.tg)]
            return np.r_[dru,dri,dmm,dgg,dcc.ravel()]*1000
        return LinearOperator((self.dim,self.dim),matvec=apply,dtype=float)

    def inverse(self,sigma):
        m=self.m;n=m.n;K=m.K;U=n*K;N=self.N;lam=sigma/1000
        u,v,w,ui,vi,wi,z,z2=self.gains
        tm=lam+1/m.tau_M
        rise=np.r_[np.full(n,m.ra),np.full(n,m.rg)]
        decay=np.r_[np.full(n,m.ta),np.full(n,m.tg)]
        luD=lu_factor(lam*np.eye(N)-self.D[1:,1:])
        e=np.r_[1.,lu_solve(luD,self.D[1:,0])]
        filt=1/((1+lam*rise)*(1+lam*decay))
        W=np.einsum('ink,n->ik',self.W.reshape(2*n,N+1,2*n),e)*filt
        den=1+lam*5+u*m.eta_M/tm
        Ae=np.c_[u[:,None]*np.repeat(m.te*W[:n,:n],K,axis=0)+v[:,None]*np.repeat(m.te*m.v_ee,K,axis=0),
                 (u*z)[:,None]*np.repeat(m.te*W[:n,n:],K,axis=0)+(w*z2)[:,None]*np.repeat(m.te*m.v_ei,K,axis=0)]/den[:,None]
        Ai=(ui[:,None]*m.ti*W[n:]+np.c_[vi[:,None]*m.ti*m.v_ie,wi[:,None]*m.ti*m.v_ii])/(1+lam*2.5)
        Ac=(Ae*m.w_u[:,None]).reshape(n,K,2*n).sum(1)
        lu=lu_factor(np.eye(2*n)-np.r_[Ac,Ai])
        WE=self.W.copy();WI=self.W.copy()
        for k in range(N+1):
            WE[:,k*2*n+n:(k+1)*2*n]=0;WI[:,k*2*n:k*2*n+n]=0
        def apply(bb):
            b=np.asarray(bb,dtype=np.result_type(sigma,float))/1000
            br=b[:U];bi=b[U:U+n];bm=b[U+n:2*U+n]
            bg=b[2*U+n:2*U+3*n];bc=b[2*U+3*n:].reshape(N+1,2*n)
            gconst=bg/(lam+1/rise)
            c0=gconst/(1+lam*decay)+bc[0]/(lam+1/decay)
            cc=np.empty_like(bc);cc[0]=c0;cc[1:]=lu_solve(luD,bc[1:])+e[1:,None]*c0
            ce=WE@cc.ravel();ci=WI@cc.ravel()
            be=(5*br-u*m.eta_M*bm/tm+u*m.te*(np.repeat(ce[:n],K)+z*np.repeat(ci[:n],K)))/den
            bI=(2.5*bi+ui*m.ti*(ce[n:]+ci[n:]))/(1+lam*2.5)
            rhs=np.r_[(be*m.w_u).reshape(n,K).sum(1),bI]
            rm=lu_solve(lu,rhs)
            ru=Ae@rm+be;ri=Ai@rm+bI;mm=(ru+bm)/tm
            gg=rm/(1+lam*rise)+gconst
            cc+=e[:,None]*(filt*rm)
            return np.r_[ru,ri,mm,gg,cc.ravel()]
        return LinearOperator((self.dim,self.dim),matvec=apply,dtype=np.result_type(sigma,float))

    def leading(self,r,s):
        op=self.operator(r,s);t=time.time();rng=np.random.default_rng(15)
        inv=self.inverse(3.)
        trial=rng.normal(size=self.dim);sol=inv@trial
        inverse_error=float(np.linalg.norm(3*sol-op@sol-trial)/np.linalg.norm(trial))
        assert inverse_error<1e-7,inverse_error
        try:
            # ARPACK expects (G - sigma I)^-1, the negative of our resolvent.
            op_inv=LinearOperator(op.shape,matvec=lambda x:-(inv@x),dtype=float)
            vals,vec=eigs(op,k=2,sigma=3.,OPinv=op_inv,which='LM',ncv=18,tol=1e-5,maxiter=60,v0=rng.normal(size=self.dim))
            ok=True
        except ArpackNoConvergence as exc:
            vals,vec=exc.eigenvalues,exc.eigenvectors;ok=False
        order=np.argsort(vals.real)[::-1];vals=vals[order];vec=vec[:,order]
        errors=[float(np.linalg.norm(op@vec[:,i]-vals[i]*vec[:,i])) for i in range(len(vals))]
        return dict(s=float(s),mean_e_hz=float(np.average(r[:self.m.n],weights=self.m.count_e)),N=self.N,
                    eigenvalues_per_s=[[float(x.real),float(x.imag)] for x in vals],residuals=errors,
                    arpack_converged=ok,inverse_check=inverse_error,search_shift_per_s=3.,seconds=time.time()-t)


def main():
    p=argparse.ArgumentParser();p.add_argument('--branch',default='extended_low_equilibria');p.add_argument('--indices',default='0,5,10,20,40,60,89');p.add_argument('--N',type=int,default=10);a=p.parse_args()
    eq=Equilibrium();accelerate(eq.m);spec=Spectrum(eq,a.N);data=np.load(OUT/f'{a.branch}.npz');rows=[]
    dest=OUT/f'{a.branch}_spectrum_N{a.N}.json'
    for i in map(int,a.indices.split(',')):
        row=spec.leading(data['r_hz'][i],data['s'][i]);row['index']=i;rows.append(row)
        dest.write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)


if __name__=='__main__':main()
