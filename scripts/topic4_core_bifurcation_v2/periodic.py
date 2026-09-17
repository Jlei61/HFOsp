"""Fourier collocation of true periodic delay-rate solutions with free period.

Both synaptic poles and every physical delay are evaluated at each harmonic.
Newton corrections solve a bordered system with an explicit phase condition.
"""
from model import System,OUT
from dynamics import transfer
import numpy as np,json,argparse
from scipy.signal import find_peaks
from scipy.sparse.linalg import LinearOperator,gmres
from scipy.interpolate import CubicSpline
from numba import njit

@njit(cache=True)
def batch_phi(mu,ve,vi,tm,ref,threshold,tw,xx,qw,reset):
    out=np.empty_like(mu)
    for k in range(len(mu)):out[k]=transfer(mu[k],ve[k],vi[k],tm,ref,threshold,tw,xx,qw,reset)
    return out

class Orbit:
    def __init__(self,s,g,N):
        self.s=s;self.g=g;self.N=N;self.k=np.arange(N//2+1);self.W,self.Q=s.weights(g)
    def phi(self,mu,ve,vi):
        s=self.s
        return batch_phi(np.ascontiguousarray(mu),np.ascontiguousarray(ve),np.ascontiguousarray(vi),s.tm,s.ref,s.threshold,s.tw,s.x,s.qw,s.p['V_reset'])
    def kernels(self,T):
        s=self.s;z=2j*np.pi*self.k/T
        E=np.exp(-z[:,None]*s.delay[None,:]);W=np.einsum('kd,dij->kij',E,self.W)
        Wp=np.einsum('kd,dij->kij',E*z[:,None]*s.delay[None,:],self.W)
        filt=(s.area*s.sign)[None,:]/((1+z[:,None]*s.rise)*(1+z[:,None]*s.decay))
        H=s.tm[None,:,None]*W*filt[:,None,:]
        Hp=s.tm[None,:,None]*Wp*filt[:,None,:]+H*(z[:,None]*s.rise/(1+z[:,None]*s.rise)+z[:,None]*s.decay/(1+z[:,None]*s.decay))[:,None,:]
        L=1/(1+z[:,None]*s.tr);Lp=z[:,None]*s.tr*L*L
        return H,Hp,L,Lp
    def mean(self,r,H):return np.fft.irfft(np.einsum('kij,kj->ki',H,np.fft.rfft(r,axis=0)),n=self.N,axis=0)
    def filt(self,r,L):return np.fft.irfft(np.fft.rfft(r,axis=0)*L,n=self.N,axis=0)
    def evaluate(self,y,ref,phase,derivative=False):
        s=self.s;r=y[:-1].reshape(self.N,6)*.01;T=np.exp(y[-1]);H,Hp,L,Lp=self.kernels(T)
        mu=s.ext_mu+self.mean(r,H);ve=s.ext_var+(r[:,:3]@self.Q[:,:3].T)*s.tm;vi=(r[:,3:]@self.Q[:,3:].T)*s.tm
        phi=self.phi(mu,ve,vi)
        residual=np.r_[((r-self.filt(phi,L))/.01).ravel(),np.sum((r-ref)*phase)/.01]
        if not derivative:return residual
        gains=[]
        vals=[mu,ve,vi]
        for j in range(3):
            step=1e-5*np.maximum(abs(vals[j]),1.);hi=vals.copy();lo=vals.copy();hi[j]=vals[j]+step;lo[j]=vals[j]-step
            gains.append((self.phi(*hi)-self.phi(*lo))/(2*step))
        u,v,h=gains
        colp=-(self.filt(u*self.mean(r,Hp),L)+self.filt(phi,Lp))/.01
        def matvec(dy):
            dr=dy[:-1].reshape(self.N,6)*.01
            dmu=self.mean(dr,H);dve=(dr[:,:3]@self.Q[:,:3].T)*s.tm;dvi=(dr[:,3:]@self.Q[:,3:].T)*s.tm
            jj=(dr-self.filt(u*dmu+v*dve+h*dvi,L))/.01+colp*dy[-1]
            return np.r_[jj.ravel(),np.sum(dr*phase)/.01]
        return residual,LinearOperator((len(y),len(y)),matvec=matvec),dict(mu=mu,ve=ve,vi=vi,gains=np.array(gains))
    def solve(self,r,T,maxiter=15):
        ref=r.copy();dr=np.fft.irfft(2j*np.pi*self.k[:,None]*np.fft.rfft(r,axis=0),n=self.N,axis=0)
        phase=dr/np.sum(dr*dr)*.01
        y=np.r_[(r/.01).ravel(),np.log(T)];history=[]
        for it in range(maxiter):
            F,J,m=self.evaluate(y,ref,phase,True);err=float(abs(F).max());history.append(err)
            print('NEWTON',self.g,self.N,it,'T',np.exp(y[-1]),'residual',err,flush=True)
            if err<1e-9:break
            dy,info=gmres(J,-F,rtol=min(1e-5,max(1e-9,err*.01)),atol=1e-12,restart=120,maxiter=12)
            if info:print('GMRES',info,float(np.linalg.norm(J@dy+F)),flush=True)
            alpha=1.
            for back in range(14):
                yy=y+alpha*dy
                if np.isfinite(yy).all() and abs(yy[-1]-y[-1])<1 and np.linalg.norm(self.evaluate(yy,ref,phase))<np.linalg.norm(F):break
                alpha*=.5
            else:break
            y=yy
        F=self.evaluate(y,ref,phase)
        return y[:-1].reshape(self.N,6)*.01,float(np.exp(y[-1])),float(abs(F).max()),history

def initial(g,N):
    z=np.load(OUT/'dynamics'/f'g{g:g}.npz');r=z['r'];dt=float(z['dt'])
    pk,_=find_peaks(r[:,0],height=.005,distance=round(50/dt));a,b=pk[-2:];T=(b-a)*dt
    t=(np.arange(a-5,b+6)-a)*dt
    return CubicSpline(t,r[a-5:b+6])(np.arange(N)*T/N),T

def main():
    p=argparse.ArgumentParser();p.add_argument('--g',type=float,default=1.15);p.add_argument('--N',type=int,default=512);p.add_argument('--from-orbit');a=p.parse_args()
    s=System();o=Orbit(s,a.g,a.N)
    if a.from_orbit:
        from scipy.signal import resample
        z=np.load(a.from_orbit);r=resample(z['r'],a.N,axis=0);T=float(z['T'])
    else:r,T=initial(a.g,a.N)
    r,T,err,history=o.solve(r,T)
    dest=OUT/'periodic';dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/f'g{a.g:.8f}_N{a.N}.npz',r=r,T=T,g=a.g,residual=err,history=history)
    print('ORBIT RESULT',a.g,T,err,r.min(0)*1000,r.max(0)*1000,flush=True)

if __name__=='__main__':main()
