"""Candidate continuous population-rate response with absolute refractoriness.

r(t)=rho(t)*[1-integral_{t-tref}^t r(s)ds], rho=exp(logit)/dt0.
The implicit Euler rule atdt0 is exactly r_k=a_k*sigmoid(logit_k)/dt0,
where a_k excludes the preceding nref-1 step fluxes. The reciprocal refractory
time bounds stationary rate; it does not cap instantaneous population flux.

This local candidate is not a promoted spatial model. Colored-LIF conditional
current/voltage history is approximated by a calibrated finite input history;
an absolute refractory factor alone is not an exact colored-LIF reduction.
"""
from common import OUT,np,read,write,log
from nonlinear_rate_response import Baseline,TAUS,SCALE,normalized_input,normalized_jacobian
from lif_mc import condition,PARAMS
from numba import njit
from scipy.linalg import expm
import torch
from torch import nn

DEST=OUT/'refractory_rate_response'
DT0=.1


def covariance_matrices(pop,dt):
    """Exact unconditional current covariance update from native OU matrices."""
    p=condition(0.,18.,1.,1.,pop,dt=dt);tm=PARAMS['tau_m_'+pop]
    matrices=[];forcing=[];readouts=[];generators=[];noise=[]
    for kind,ar,b,ad,q0,q1,q2 in [('AMPA',p[6],p[7],p[8],p[11],p[12],p[13]),
                                  ('GABA',p[17],p[9],p[10],p[14],p[15],p[16])]:
        tr=PARAMS['tau_r_'+kind];td=PARAMS['tau_d_'+kind]
        matrices.append([[ar*ar,0,0],[ar*b,ar*ad,0],[b*b,2*b*ad,ad*ad]])
        forcing.append([q0*q0,q0*q1,q1*q1+q2*q2]);readouts.append(2*(tr+td)/tm)
        generators.append([[-2/tr,0,0],[1/td,-1/tr-1/td,0],[0,2/td,-2/td]])
        noise.append([tm/tr**2,0,0])
    return np.array(matrices),np.array(forcing),np.array(readouts),np.array(generators),np.array(noise)


@njit
def _features(wave,T,dt,burn,steps,theta,matrices,forcing,readouts,include_burn):
    cov=np.zeros((2,3));h=np.zeros((3,4,3));out=np.empty((steps+burn if include_burn else steps,39))
    W=wave.shape[1];sc=theta-11.;scale=np.array([3.,2.,2.]);taus=np.array([1.,4.,16.,64.])
    for k in range(-burn,steps):
        phase=((k+1)*dt/T)%1;pos=phase*W;lo=int(np.floor(pos))%W;hi=(lo+1)%W;a=pos-np.floor(pos)
        x=(1-a)*wave[:,lo]+a*wave[:,hi]
        for c in range(2):cov[c]=matrices[c]@cov[c]+forcing[c]*x[c+1]
        u=np.array([np.arcsinh((x[0]-11)/sc),np.log1p(readouts[0]*cov[0,2]/sc**2),np.log1p(readouts[1]*cov[1,2]/sc**2)])/scale
        for c in range(3):
            for j in range(4):
                b=dt/taus[j];e=np.exp(-b);h1,h2,h3=h[c,j]
                h[c,j,0]=e*h1+(1-e)*u[c]
                h[c,j,1]=e*(h2+b*h1)+(1-e*(1+b))*u[c]
                h[c,j,2]=e*(h3+b*h2+.5*b*b*h1)+(1-e*(1+b+.5*b*b))*u[c]
        if k>=0 or include_burn:
            index=k+burn if include_burn else k
            out[index,:3]=u
            for c in range(3):
                for j in range(4):
                    for l in range(3):out[index,3+12*c+3*j+l]=h[c,j,l]-u[c]
    return out


def features(wave,T,dt,burn,steps,pop,theta=18.,include_burn=False):
    a,b,c,_,_=covariance_matrices(pop,dt)
    return _features(wave,T,dt,burn,steps,theta,a,b,c,include_burn)


class BaseLogit:
    def __init__(self,pop):
        self.pop=pop;self.base=Baseline(pop);self.ref=PARAMS['tau_ref_'+pop]
        self.maximum=1000/self.ref;self.nref=round(self.ref/DT0)

    def evaluate(self,physical,theta=18.,derivatives=False):
        b,g=self.base.evaluate(physical,theta,True)
        probability=1/(1+np.exp(-b));r=self.maximum*probability
        dr=r[...,None]*(1-probability[...,None])*g
        q=r*DT0/1000;available=1-(self.nref-1)*q;p=q/available
        ell=np.log(p)-np.log1p(-p)
        gradient=dr*(DT0/(1000*available**2*p*(1-p)))[...,None]
        return (ell,gradient) if derivatives else ell


def transfer_factors(pop,frequencies,dt=DT0,continuous=False):
    """Input covariance, history and refractory-integral frequency responses."""
    f=np.asarray(frequencies);lam=2j*np.pi*f/1000;ref=PARAMS['tau_ref_'+pop]
    bank=np.empty((len(f),12),complex);A,b,c,G,Q=covariance_matrices(pop,dt)
    cov=np.ones((len(f),3),complex)
    for i,l in enumerate(lam):
        z=np.exp(-l*dt)
        for j,tau in enumerate(TAUS):
            if continuous:bank[i,3*j:3*j+3]=(1+l*tau)**(-np.arange(1,4))-1
            else:
                v=dt/tau;e=np.exp(-v)
                L=e*np.array([[1,0,0],[v,1,0],[.5*v*v,v,1.]])
                B=np.array([1-e,1-e*(1+v),1-e*(1+v+.5*v*v)])
                bank[i,3*j:3*j+3]=np.linalg.solve(np.eye(3)-L*z,B)-1
        for ch in range(2):
            state=np.linalg.solve(l*np.eye(3)-G[ch],Q[ch]) if continuous else np.linalg.solve(np.eye(3)-A[ch]*z,b[ch])
            cov[i,ch+1]=c[ch]*state[2]
    if continuous:
        K=np.full(len(lam),ref,dtype=complex);nz=lam!=0;K[nz]=-np.expm1(-lam[nz]*ref)/lam[nz]
    else:K=dt*np.exp(-lam[:,None]*dt*np.arange(round(ref/dt))[None,:]).sum(1)
    return bank,cov,K


class RefractoryReadout(nn.Module):
    def __init__(self,pop):
        super().__init__();self.pop=pop;self.ref=PARAMS['tau_ref_'+pop];self.maximum=1000/self.ref
        self.network=nn.Sequential(nn.Linear(39,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)

    def logits(self,features,base_logits):
        return base_logits+self.network(features).squeeze(-1)

    def stationary(self,features,base_logits):
        return self.maximum*torch.sigmoid(self.logits(features,base_logits)+np.log(self.ref/DT0))

    def linear_response(self,features,base_logits,base_gradient,input_gradient,channel,bank,cov,K,create_graph=False):
        f=features.detach().requires_grad_(True);correction=self.network(f).squeeze(-1)
        grad=torch.autograd.grad(correction.sum(),f,create_graph=create_graph)[0]
        ell=base_logits+correction;p=torch.sigmoid(ell)
        rate=self.maximum*torch.sigmoid(ell+np.log(self.ref/DT0))
        index=torch.arange(len(f),device=f.device)
        grad_in=base_gradient[index,channel]+grad[index,channel]*input_gradient[index,channel]
        weights=grad[:,3:].reshape(-1,3,12)[index,channel]
        response_logit=(grad_in+(weights*bank).sum(1)*input_gradient[index,channel])*cov[index,channel]
        return rate*(1-p)*response_logit/((1-p)+p*K/DT0)


@njit
def implicit_flux(logits,dt,ref,initial_rate_hz=0.):
    """Implicit Euler of the continuous refractory integral equation.

    Input includes burn, from initially available mass1 and zero spike history.
    A returned rate is our own prediction; no measured firing history enters.
    """
    nref=int(round(ref/dt));history=np.full(nref,initial_rate_hz*dt/1000);occupied=initial_rate_hz*ref/1000;r=np.empty(len(logits));minimum=1-occupied
    for k,ell in enumerate(logits):
        slot=k%nref;occupied-=history[slot]
        available=1-occupied
        if available < -1e-10 or available > 1+1e-10:raise ValueError('Invalid refractory population mass')
        l=ell+np.log(dt/DT0)
        p=1/(1+np.exp(-l)) if l>=0 else np.exp(l)/(1+np.exp(l))
        fired=available*p;history[slot]=fired;occupied+=fired
        r[k]=1000*fired/dt;minimum=min(minimum,1-occupied)
    return r,minimum
