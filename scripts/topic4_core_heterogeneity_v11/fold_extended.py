"""Direct augmented cycle-fold corrector for moving two-parameter folds."""
from periodic_boundaries import *
from scipy.optimize import newton_krylov,NoConvergence


def correct(h,z,t,N):
    n=len(z)-1;s=System(h);ref=z[:-2].reshape(N,6)*.01
    dr=np.fft.irfft(2j*np.pi*np.arange(N//2+1)[:,None]*np.fft.rfft(ref,axis=0),n=N,axis=0)
    phase=dr/np.sum(dr*dr)*.01
    v=t[:-1].copy();v/=np.sqrt(v[:-1]@v[:-1]/N+v[-1]**2)
    def fn(x):
        y=x[:n];g=x[n]*.01;v=x[n+1:]
        o=Orbit(s,g,N);F,J,m=o.evaluate(y,ref,phase,True)
        return np.r_[F,J@v,(v[:-1]@v[:-1]/N+v[-1]**2-1)]
    x=np.r_[z,v]
    answer=newton_krylov(fn,x,method='lgmres',rdiff=1e-5,f_tol=1e-8,maxiter=35,
                         inner_maxiter=100,line_search='armijo',verbose=True)
    ff=fn(answer);q=answer[:n+1];tt=np.r_[answer[n+1:],0.]
    row=dict(h=h,g=float(q[-1]*.01),T_ms=float(np.exp(q[-2])),N=N,
             orbit_residual=float(abs(ff[:n]).max()),null_residual=float(abs(ff[n:-1]).max()),
             critical_residual=float(abs(ff[-1])),criterion='Augmented periodic fold F=0, Jv=0')
    return q,tt,row


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--h',type=float,default=.999);a=ap.parse_args()
    N=512;z,t=load_seed(seeds()['LP1'],N)
    q,tt,row=correct(a.h,z,t,N);print('RESULT',json.dumps(row),flush=True)
