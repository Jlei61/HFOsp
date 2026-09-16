"""Continuous-time latent core: exact linear OU substep, nonlinear residual
split, and the Gaussian evidence update.

The linear part is integrated exactly (matrix exponential plus the Van Loan
covariance integral), so a long recording gap inherits the continuous-time
stationary behaviour instead of an Euler drift. The nonlinear residual is a
midpoint step whose analytic Jacobian propagates the covariance. The evidence
update is a learned approximate Gaussian filter, not an exact Bayesian
posterior over the event process.
"""
from __future__ import annotations
import math
import torch
from torch import nn

BLOCK=8
N_BLOCK=3
LATENT=BLOCK*N_BLOCK
RANK=6
TAU_MIN_HOURS=2./60.
TAU_MAX_HOURS=48.
SIGMA_FLOOR=1e-4
SIGMA_CEIL=1e3
MAX_STEP_HOURS=1./60.


def stationary_covariance(A,sigma):
    """Solve ``A P + P A^T + diag(sigma^2) = 0`` for the OU stationary covariance.

    ``A=-D+Omega`` with antisymmetric Omega is negative definite in its
    symmetric part, so every eigenvalue has negative real part and P exists.
    """
    d=A.shape[-1]
    # The 576x576 Kronecker system is badly conditioned once Omega grows, so it is
    # solved in FP64 regardless of the network dtype; float32 goes singular.
    if not torch.isfinite(A).all() or not torch.isfinite(sigma).all():
        raise FloatingPointError(
            'non-finite dynamics before the Lyapunov solve: '
            f'A finite={bool(torch.isfinite(A).all())} max|A|={float(A.abs().max()) if torch.isfinite(A).any() else float("nan")} '
            f'sigma finite={bool(torch.isfinite(sigma).all())} '
            f'max sigma={float(sigma[torch.isfinite(sigma)].max()) if torch.isfinite(sigma).any() else float("nan")}')
    A64=A.double()
    eye=torch.eye(d,dtype=A64.dtype,device=A.device)
    M=torch.kron(A64,eye)+torch.kron(eye,A64)
    rhs=-torch.diag((sigma*sigma).double()).reshape(-1)
    P=torch.linalg.solve(M,rhs).reshape(d,d)
    P=0.5*(P+P.transpose(-1,-2))
    if not torch.isfinite(P).all():
        raise FloatingPointError('stationary covariance solve produced non-finite values')
    return P.to(A.dtype)


def ou_transition(A,sigma,dt,P_inf=None):
    """Exact ``F=exp(A dt)`` and ``Q=P_inf-F P_inf F^T``.

    The Van Loan block form is numerically unusable here: its top-left block is
    ``exp(-A dt)``, which for a 5-minute decay and a multi-hour gap reaches
    ``e^{44}`` and destroys Q through cancellation. The stationary-covariance
    identity is exact for a stable OU at every dt.
    ``sigma**2 * dt`` remains illegal once A rotates or decays.
    """
    dt=torch.as_tensor(dt,dtype=A.dtype,device=A.device).reshape(-1)
    if P_inf is None:P_inf=stationary_covariance(A,sigma)
    F=torch.matrix_exp(A*dt.reshape(-1,1,1))
    Q=P_inf-F@P_inf@F.transpose(-1,-2)
    return F,0.5*(Q+Q.transpose(-1,-2))


def van_loan(A,sigma,dt):
    """Block-exponential reference for short steps; kept for cross-checking."""
    d=A.shape[-1]
    dt=torch.as_tensor(dt,dtype=A.dtype,device=A.device).reshape(-1)
    S=torch.diag_embed(sigma*sigma)
    top=torch.cat((-A,S),dim=-1)
    bot=torch.cat((torch.zeros_like(A),A.transpose(-1,-2)),dim=-1)
    M=torch.cat((top,bot),dim=-2)*dt.reshape(-1,1,1)
    E=torch.matrix_exp(M)
    F=E[...,d:,d:].transpose(-1,-2)
    Q=F@E[...,:d,d:]
    return F,0.5*(Q+Q.transpose(-1,-2))


def cholesky_psd(P,name='covariance'):
    """FP64 Cholesky with a bounded, recorded jitter ladder."""
    P=0.5*(P+P.transpose(-1,-2))
    scale=torch.clamp(P.diagonal(dim1=-2,dim2=-1).mean(-1),min=1e-12)
    eye=torch.eye(P.shape[-1],dtype=P.dtype,device=P.device)
    for k,rel in enumerate((0.,1e-12,1e-9,1e-6)):
        L,info=torch.linalg.cholesky_ex(P+rel*scale.reshape(-1,1,1)*eye)
        if int(info.max())==0:return L,rel
    raise FloatingPointError(f'{name} needed more than 1e-6 relative jitter to factor')


class Dynamics(nn.Module):
    """``dz=(Az+r(z))dt+diag(sigma)dW`` with A=-D+Omega on three 8-dim blocks.

    ``coupled`` opens cross-block coefficients; ``nonlinear`` adds the
    within-block low-rank residual. The two switches move together in the first
    round by design, so their difference cannot attribute an effect to either.
    """

    def __init__(self,coupled=False,nonlinear=False,tau_init=(5./60.,1.,8.),generator=None):
        super().__init__()
        self.coupled=bool(coupled);self.nonlinear=bool(nonlinear)
        tau=torch.tensor([t for t in tau_init for _ in range(BLOCK)],dtype=torch.float32)
        self.log_tau=nn.Parameter(torch.log(tau))
        n_off=LATENT*(LATENT-1)//2
        idx=torch.triu_indices(LATENT,LATENT,offset=1)
        self.register_buffer('omega_i',idx[0]);self.register_buffer('omega_j',idx[1])
        block=(idx[0]//BLOCK)==(idx[1]//BLOCK)
        self.register_buffer('omega_within',block)
        w=torch.zeros(n_off)
        if generator is not None:w.normal_(0.,0.01,generator=generator)
        self.omega=nn.Parameter(w)
        self.log_sigma=nn.Parameter(torch.full((LATENT,),math.log(0.1)))
        if nonlinear:
            U=torch.zeros(N_BLOCK,BLOCK,RANK);V=torch.zeros(N_BLOCK,RANK,BLOCK)
            if generator is not None:
                U.normal_(0.,0.01,generator=generator);V.normal_(0.,1./math.sqrt(BLOCK),generator=generator)
            self.U=nn.Parameter(U);self.V=nn.Parameter(V);self.b=nn.Parameter(torch.zeros(N_BLOCK,RANK))

    def tau_hours(self):
        return torch.clamp(torch.exp(self.log_tau),TAU_MIN_HOURS,TAU_MAX_HOURS)

    def A(self):
        w=self.omega if self.coupled else self.omega*self.omega_within
        M=torch.zeros(LATENT,LATENT,dtype=w.dtype,device=w.device)
        M=M.index_put((self.omega_i,self.omega_j),w)
        M=M-M.transpose(0,1)
        return M-torch.diag(1./self.tau_hours())

    def sigma(self):
        # Clamp in log space: clamp(exp(x), min=...) leaves +inf untouched, so an
        # overflowing log_sigma would reach the Lyapunov solve as a non-finite value.
        return torch.exp(torch.clamp(self.log_sigma,math.log(SIGMA_FLOOR),math.log(SIGMA_CEIL)))

    def residual(self,z):
        if not self.nonlinear:return torch.zeros_like(z)
        shape=z.shape;zb=z.reshape(-1,N_BLOCK,BLOCK)
        h=torch.tanh(torch.einsum('arb,nab->nar',self.V,zb)+self.b)
        return torch.einsum('abr,nar->nab',self.U,h).reshape(shape)

    def residual_jacobian(self,z):
        n=z.reshape(-1,LATENT).shape[0]
        J=torch.zeros(n,LATENT,LATENT,dtype=z.dtype,device=z.device)
        if not self.nonlinear:return J
        zb=z.reshape(-1,N_BLOCK,BLOCK)
        h=torch.tanh(torch.einsum('arb,nab->nar',self.V,zb)+self.b)
        g=1.-h*h
        Jb=torch.einsum('abr,nar,arc->nabc',self.U,g,self.V)
        for a in range(N_BLOCK):
            J[:,a*BLOCK:(a+1)*BLOCK,a*BLOCK:(a+1)*BLOCK]=Jb[:,a]
        return J


def _p_inf(dyn,A,sigma,cache):
    if cache is None:return stationary_covariance(A,sigma)
    if 'p_inf' not in cache:cache['p_inf']=stationary_covariance(A,sigma)
    return cache['p_inf']


def _substeps(dt_hours,max_step=MAX_STEP_HOURS):
    n=max(1,int(math.ceil(float(dt_hours)/max_step-1e-12)))
    return n,float(dt_hours)/n


def midpoint_map(dyn,z,h):
    y=z+0.5*h*dyn.residual(z)
    return z+h*dyn.residual(y)


def midpoint_jacobian(dyn,z,h):
    eye=torch.eye(LATENT,dtype=z.dtype,device=z.device)
    if not dyn.nonlinear:return eye.expand(z.reshape(-1,LATENT).shape[0],LATENT,LATENT)
    Jz=dyn.residual_jacobian(z)
    y=z+0.5*h*dyn.residual(z)
    Jy=dyn.residual_jacobian(y)
    return eye+h*Jy@(eye+0.5*h*Jz)


def propagate_moments(dyn,m,P,dt_hours,max_step=MAX_STEP_HOURS,cache=None):
    """Strang split: exact OU half step, nonlinear midpoint, exact OU half step."""
    if float(dt_hours)<=0:return m,P
    n,h=_substeps(dt_hours,max_step)
    key=('mom',round(h,12),n)
    if cache is not None and key in cache:F,Q=cache[key]
    else:
        A=dyn.A().to(m.dtype);sigma=dyn.sigma().to(m.dtype)
        F,Q=ou_transition(A,sigma,h/2.,_p_inf(dyn,A,sigma,cache))
        F=F[0];Q=Q[0]
        if cache is not None:cache[key]=(F,Q)
    Ft=F.transpose(0,1)
    for _ in range(n):
        m=m@Ft;P=F@P@Ft+Q
        if dyn.nonlinear:
            J=midpoint_jacobian(dyn,m,h)
            m=midpoint_map(dyn,m,h)
            P=J@P@J.transpose(-1,-2)
        m=m@Ft;P=F@P@Ft+Q
    return m,P


def propagate_samples(dyn,z,dt_hours,generator=None,max_step=MAX_STEP_HOURS,cache=None,noise=None):
    """Same split applied to sampled paths; noise uses the exact OU covariance."""
    if float(dt_hours)<=0:return z
    n,h=_substeps(dt_hours,max_step)
    key=('sam',round(h,12))
    if cache is not None and key in cache:F,L=cache[key]
    else:
        A=dyn.A().to(z.dtype);sigma=dyn.sigma().to(z.dtype)
        F,Q=ou_transition(A,sigma,h/2.,_p_inf(dyn,A,sigma,cache))
        L,_=cholesky_psd(Q,'OU half-step Q')
        F=F[0];L=L[0]
        if cache is not None:cache[key]=(F,L)
    Ft=F.transpose(0,1);Lt=L.transpose(0,1)
    for _ in range(n):
        for half in range(2):
            e=(noise.pop(0) if noise else torch.randn(z.shape,dtype=z.dtype,device=z.device,generator=generator))
            z=z@Ft+e@Lt
            if half==0 and dyn.nonlinear:z=midpoint_map(dyn,z,h)
    return z


def evidence_update(m,P,a,R):
    """Joseph-form Gaussian correction of the full predictive moments.

    R is learned evidence noise, never the process noise already inside P.
    """
    S=P+R
    L,jitter=cholesky_psd(S,'innovation covariance')
    K=torch.cholesky_solve(P.transpose(-1,-2),L).transpose(-1,-2)
    m_plus=m+torch.einsum('nij,nj->ni',K,a-m)
    eye=torch.eye(P.shape[-1],dtype=P.dtype,device=P.device)
    IK=eye-K
    P_plus=IK@P@IK.transpose(-1,-2)+K@R@K.transpose(-1,-2)
    P_plus=0.5*(P_plus+P_plus.transpose(-1,-2))
    innov=a-m
    alpha=torch.cholesky_solve(innov.unsqueeze(-1),L).squeeze(-1)
    logdet=2.*torch.log(torch.diagonal(L,dim1=-2,dim2=-1)).sum(-1)
    loglik=-0.5*((innov*alpha).sum(-1)+logdet+P.shape[-1]*math.log(2*math.pi))
    return m_plus,P_plus,loglik,jitter
