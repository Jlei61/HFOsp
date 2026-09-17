"""Refine a periodic-orbit fold and its left/right bordered null vectors."""
from common import *
from periodic import Orbit
from scipy.signal import resample
from scipy.sparse.linalg import LinearOperator,gmres
from scipy.optimize import brentq
import numpy as np,argparse

def metric(a,b,N):return a[:-2]@b[:-2]/N+a[-2:]@b[-2:]

class Chart:
    def __init__(self,s,z0,tan,N,gscale):
        self.s=s;self.N=N;self.gscale=gscale;self.z0=z0;self.normal=tan.copy();self.normal[:-2]/=N
        self.ref=z0[:-2].reshape(N,6)*.01
        der=np.fft.irfft(2j*np.pi*np.arange(N//2+1)[:,None]*np.fft.rfft(self.ref,axis=0),n=N,axis=0)
        self.phase=der/np.sum(der*der)*.01
    def evaluate(self,z,coordinate=0,jac=True):
        N=self.N;s=self.s;g=z[-1]*self.gscale;o=Orbit(s,g,N)
        if not jac:return np.r_[o.evaluate(z[:-1],self.ref,self.phase),self.normal@(z-self.z0)-coordinate]
        F,J,m=o.evaluate(z[:-1],self.ref,self.phase,True);r=z[:-2].reshape(N,6)*.01;T=np.exp(z[-2]);H,Hp,L,Lp=o.kernels(T);u,v,h=m['gains']
        Hg=np.zeros_like(H);Qg=np.zeros_like(o.Q)
        for i in (0,1):Hg[:,i,i]=H[:,i,i]/g;Qg[i,i]=2*o.Q[i,i]/g
        dg=-o.filt(u*o.mean(r,Hg)+v*((r[:,:3]@Qg[:,:3].T)*s.tm),L)/.01
        b=np.r_[dg.ravel(),0]*self.gscale
        e=np.zeros(len(z)-1);e[-1]=1;col=J@e
        def adjoint(w):
            a=w[:-1].reshape(N,6);q=o.filt(a,np.conj(L))
            hadj=np.fft.irfft(np.einsum('kij,ki->kj',np.conj(H),np.fft.rfft(u*q,axis=0)),n=N,axis=0)
            out=a-hadj;out[:,:3]-=(s.tm*v*q)@o.Q[:,:3];out[:,3:]-=(s.tm*h*q)@o.Q[:,3:]
            out+=self.phase*w[-1]
            return np.r_[out.ravel(),col@w]
        def mv(dz):return np.r_[J@dz[:-1]+b*dz[-1],self.normal@dz]
        def rmv(w):return np.r_[adjoint(w[:-1]),b@w[:-1]]+self.normal*w[-1]
        B=LinearOperator((len(z),len(z)),matvec=mv,rmatvec=rmv)
        return np.r_[F,self.normal@(z-self.z0)-coordinate],B,J,b
    def solve(self,z,coordinate):
        z=z.copy();hist=[]
        for k in range(15):
            F,B,J,b=self.evaluate(z,coordinate);err=float(abs(F).max());hist.append(err)
            if err<2e-11:break
            dz,info=gmres(B,-F,rtol=min(1e-7,max(1e-11,err*.005)),atol=1e-13,restart=150,maxiter=20)
            alpha=1
            for _ in range(14):
                trial=z+alpha*dz
                if np.linalg.norm(self.evaluate(trial,coordinate,False))<np.linalg.norm(F):break
                alpha*=.5
            else:raise RuntimeError(('chart line search',coordinate,hist))
            z=trial
        else:raise RuntimeError(('chart no convergence',coordinate,hist))
        F,B,J,b=self.evaluate(z,coordinate);rhs=np.zeros(len(z));rhs[-1]=1
        tan,info=gmres(B,rhs,rtol=2e-10,atol=1e-12,restart=150,maxiter=25)
        if info:raise RuntimeError(('chart tangent',info))
        tan/=np.sqrt(metric(tan,tan,self.N))
        return z,tan,float(abs(F).max()),B,J,b

def refine(left,right,name,N=2048):
    gscale=.01
    def get(path):
        a=np.load(path);return np.r_[(resample(a['r'],N,axis=0)/.01).ravel(),np.log(float(a['T'])),float(a['g'])/gscale]
    a=get(left);b=get(right);tan=b-a;tan/=np.sqrt(metric(tan,tan,N));span=metric(b-a,tan,N);chart=Chart(System(),a,tan,N,gscale);cache={}
    def at(x):
        if x not in cache:
            guess=a+(b-a)*(x/span);cache[x]=chart.solve(guess,x)
            z,t,err,*_=cache[x];print('FOLD_REFINE',name,'s',x,'g',z[-1]*gscale,'dJ',t[-1]*gscale,'res',err,flush=True)
        return cache[x]
    def target(x):return at(x)[1][-1]*gscale
    sf=brentq(target,0,span,xtol=2e-9,rtol=1e-12);z,t,err,B,J,bg=at(sf)
    # At a simple fold B remains invertible, while the fixed-parameter J is singular.
    rhs=np.zeros(len(z));rhs[-1]=1
    adj,info=gmres(B.T,rhs,rtol=2e-9,atol=1e-11,restart=150,maxiter=25)
    if info:raise RuntimeError(('adjoint',info))
    leftvec=adj[:-1];leftvec/=np.linalg.norm(leftvec);rightvec=t[:-1]
    rng=np.random.default_rng(45);v=rng.normal(size=len(z));w=rng.normal(size=len(z));duality=abs(w@(B@v)-(B.T@w)@v)/max(abs(w@(B@v)),1)
    assert duality<1e-8
    F0=chart.evaluate(z,sf,False)[:-1];seconds=[]
    for eps in (.003,.001):
        delta=np.r_[rightvec,0]*eps
        sec=(chart.evaluate(z+delta,sf,False)[:-1]-2*F0+chart.evaluate(z-delta,sf,False)[:-1])/eps**2
        seconds.append(dict(step=eps,left_Fvv=float(leftvec@sec)))
    lFg=float(leftvec@bg/gscale);curvature=-seconds[-1]['left_Fvv']/lFg
    q1=at(sf-.01)[0];q2=at(sf+.01)[0];observed=(q1[-1]-2*z[-1]+q2[-1])*gscale/.01**2
    row=dict(name=name,g=float(z[-1]*gscale),T_ms=float(np.exp(z[-2])),N=N,residual=err,tangent_g=float(t[-1]*gscale),
        fixed_parameter_null_residual=float(abs(J@rightvec).max()),adjoint_last=float(adj[-1]),adjoint_duality_error=float(duality),
        left_Fg=lFg,second_derivatives=seconds,predicted_g_curvature=float(curvature),measured_g_curvature=float(observed),
        left_source=str(left),right_source=str(right),mean_hz=(z[:-2].reshape(N,6).mean(0)*10).tolist())
    dest=OUT/'folds';dest.mkdir(exist_ok=True)
    path=dest/f'{name}_N{N}.npz';np.savez_compressed(path,r=z[:-2].reshape(N,6)*.01,T=np.exp(z[-2]),g=z[-1]*gscale,N=N,residual=err,tangent=t,right_null=rightvec,left_null=leftvec,gscale=gscale)
    row['source']=str(path);(dest/f'{name}_N{N}.json').write_text(json.dumps(row,indent=2)+'\n');print('FOLD',json.dumps(row),flush=True)
    return row

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--left',required=True);ap.add_argument('--right',required=True);ap.add_argument('--name',required=True);ap.add_argument('--N',type=int,default=2048);a=ap.parse_args();refine(a.left,a.right,a.name,a.N)
