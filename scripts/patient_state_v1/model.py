"""Exact irregular OU prior, Laplace marginal likelihood and Gaussian assumed-density filter.

Laplace and assumed-density filtering are approximations, checked independently by particles.
The fitted latent state is in log-odds units. No seizure outcomes enter fitting.
"""
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from scipy.linalg import solveh_banded, cholesky_banded
from numpy.polynomial.hermite import hermgauss
from numba import njit


def prior(dt,reset,tau,sd):
    a=np.exp(-dt/tau);a[reset]=0
    q=sd*sd*(-np.expm1(-2*dt/tau));q[reset]=sd*sd
    diag=1/q
    diag[:-1]+=a[1:]**2/q[1:]
    off=-a[1:]/q[1:]
    return a,q,diag,off,-np.log(q).sum()


def qmul(s,diag,off):
    z=diag*s;z[:-1]+=off*s[1:];z[1:]+=off*s[:-1]
    return z


def laplace(theta,data,cyclic=False,return_state=False):
    nc=3 if cyclic else 1
    beta=np.asarray(theta[:nc]);tau,sd=np.exp(theta[nc:nc+2])
    dt,reset,y,n=data['dt'],data['reset'],data['y'],data['n']
    eta0=data['x'][:,:nc]@beta
    a,q,diag,off,ldq=prior(dt,reset,tau,sd)
    s=np.zeros(len(y))
    def objective(s):
        eta=eta0+s
        return np.sum(n*np.logaddexp(0,eta)-y*eta)+.5*np.dot(s,qmul(s,diag,off))
    value=objective(s); converged=False
    for iteration in range(60):
        p=expit(eta0+s);w=n*p*(1-p)
        grad=qmul(s,diag,off)+n*p-y
        band=np.zeros((2,len(y)));band[0]=diag+w;band[1,:-1]=off
        delta=solveh_banded(band,grad,lower=True,check_finite=False)
        step=1.
        for _ in range(25):
            candidate=s-step*delta;v=objective(candidate)
            if v<=value+1e-8:break
            step*=.5
        change=abs(value-v);s=candidate;value=v
        if np.max(np.abs(step*delta))<1e-6 or (change<1e-8 and np.max(abs(delta))<1e-3):
            converged=True;break
    p=expit(eta0+s);band[0]=diag+n*p*(1-p)
    chol=cholesky_banded(band,lower=True,check_finite=False)
    ldh=2*np.log(chol[0]).sum()
    ll=-value+.5*(ldq-ldh)
    if return_state:
        # Selected inverse diagonal via bidiagonal Cholesky recursion.
        var=np.empty(len(y));var[-1]=1/chol[0,-1]**2
        for i in range(len(y)-2,-1,-1):var[i]=(1+chol[1,i]**2*var[i+1])/chol[0,i]**2
        return dict(loglik=float(ll),mode=s,variance=var,converged=converged,iterations=iteration+1)
    if not converged:return float(ll)-1e3
    return float(ll)


def fit(data,model='ou',initial=None,maxiter=150):
    cyclic='cycle' in model;nc=3 if cyclic else 1
    p=np.clip(data['y'].sum()/data['n'].sum(),1e-4,1-1e-4)
    beta=np.r_[np.log(p/(1-p)),np.zeros(nc-1)]
    if model in ('constant','cycle'):
        x=data['x'][:,:nc];y=data['y'];n=data['n']
        def fun(b):
            eta=x@b
            return np.sum(n*np.logaddexp(0,eta)-y*eta),x.T@(n*expit(eta)-y)
        opt=minimize(fun,beta,jac=True,method='L-BFGS-B',bounds=[(-8,8)]*nc)
        return dict(model=model,theta=opt.x.tolist(),loglik=float(-opt.fun),success=bool(opt.success),message=str(opt.message),nfev=int(opt.nfev))
    start=np.r_[beta,np.log([1.,.6])] if initial is None else np.asarray(initial,float)
    bounds=[(-8,8)]*nc+[(np.log(1/60),np.log(24)),(np.log(.01),np.log(5))]
    def fun(t):
        try:return -laplace(t,data,cyclic)
        except (ValueError,np.linalg.LinAlgError,FloatingPointError):return 1e20
    opt=minimize(fun,start,method='L-BFGS-B',bounds=bounds,options={'maxiter':maxiter,'ftol':1e-10,'gtol':1e-4,'maxls':25,'eps':2e-5})
    out=dict(model=model,theta=opt.x.tolist(),loglik=float(-opt.fun),success=bool(opt.success),message=str(opt.message),nfev=int(opt.nfev),
             tau_hours=float(np.exp(opt.x[nc])),stationary_sd=float(np.exp(opt.x[nc+1])),initial=start.tolist(),method='Laplace_exact_irregular_OU_prior')
    return out


@njit(cache=True)
def adf_kernel(dt,reset,y,n,baseline,tau,sd,nodes,weights):
    length=len(y);means=np.zeros(length);vars=np.zeros(length);predict=np.zeros(length);ll=np.zeros(length)
    m=0.;v=sd*sd
    for i in range(length):
        a=np.exp(-dt[i]/tau)
        if reset[i]:m=0.;v=sd*sd
        else:m=a*m;v=a*a*v+sd*sd*(-np.expm1(-2*dt[i]/tau))
        z=m+np.sqrt(2*v)*nodes
        norm=0.;s1=0.;s2=0.;pp=0.
        for j in range(len(nodes)):
            eta=baseline[i]+z[j]
            p=1/(1+np.exp(-eta));pp+=weights[j]*p
            lp=y[i]*eta-n[i]*(max(eta,0)+np.log1p(np.exp(-abs(eta))))
            w=weights[j]*np.exp(lp)
            norm+=w;s1+=w*z[j];s2+=w*z[j]*z[j]
        norm=max(norm,1e-300);m=s1/norm;v=max(s2/norm-m*m,1e-12)
        means[i]=m;vars[i]=v;predict[i]=pp;ll[i]=np.log(norm)
    return means,vars,predict,ll


def filter_adf(theta,data,cyclic=False,order=32):
    nc=3 if cyclic else 1
    tau,sd=np.exp(theta[nc:nc+2]);baseline=data['x'][:,:nc]@theta[:nc]
    nodes,weights=hermgauss(order);weights/=np.sqrt(np.pi)
    m,v,p,ll=adf_kernel(data['dt'],data['reset'],data['y'],data['n'],baseline,tau,sd,nodes,weights)
    return dict(mean=m,variance=v,predict_tb=p,loglik_terms=ll,loglik=float(ll.sum()),method='Gaussian_assumed_density_GH_filter')


def slice_data(data,start=0,end=None):
    d={k:np.array(v[start:end],copy=True) for k,v in data.items() if np.ndim(v)>0}
    d['reset'][0]=True;d['dt'][0]=0
    return d


def simulate(data,b,tau,sd,seed,cyclic=(0.,0.)):
    rng=np.random.default_rng(seed);s=0.;state=np.zeros(len(data['y']))
    for i in range(len(state)):
        if data['reset'][i]:s=rng.normal()*sd
        else:
            a=np.exp(-data['dt'][i]/tau)
            s=a*s+sd*np.sqrt(-np.expm1(-2*data['dt'][i]/tau))*rng.normal()
        state[i]=s
    probability=expit(b+state+data['x'][:,1:]@np.array(cyclic))
    return rng.binomial(data['n'],probability),state
