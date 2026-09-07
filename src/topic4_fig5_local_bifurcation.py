"""Local continuation of the two GIF bases, with actual checkpoint Z fields.

This is a deterministic diffusion LIF reduction, not the finite spiking SNN.
Keep equilibrium fold evidence separate from delay-dependent stability.
"""
from dataclasses import replace
import numpy as np
from scipy import sparse, linalg
from scipy.optimize import root
from src.topic4_patient_zm_meanfield import _transfer_derivatives
from src.topic4_dual_core_spatial_z import (
    spatial_z_moments, spatial_z_residual, spatial_z_jacobian)
from src.topic4_dual_core_spatial_z_delay import delayed_step_matrix


class Family:
    def __init__(self, model, z_moments, eta, parameter, operators=None):
        self.base = model
        self.z_moments = z_moments
        self.eta = eta
        self.parameter = parameter
        self.operators = operators

    def at(self, p):
        m = self.base
        eta, tau, dose = self.eta, 500., 1.
        ops = self.operators
        if self.parameter == 'z_loss':
            dose = p
        elif self.parameter in ('ee', 'e_to_i', 'i_to_e'):
            # Matrix names use TARGET,SOURCE: E->I is w_ie, not w_ei.
            pathway = {'ee': 'ee', 'e_to_i': 'ie', 'i_to_e': 'ei'}[self.parameter]
            m = replace(m, **{'w_'+pathway: getattr(m, 'w_'+pathway)*p,
                              'v_'+pathway: getattr(m, 'v_'+pathway)*p*p})
            if ops is not None:
                key = 'w_'+pathway+'_history'
                ops = replace(ops, **{key: getattr(ops, key)*p})
        elif self.parameter == 'eta_m':
            eta *= p
        elif self.parameter == 'tau_m':
            tau *= p
        elif self.parameter == 'tau_m_matched':
            tau *= p
            eta /= p  # same stationary eta*tau, different recovery dynamics
        elif self.parameter == 'tau_gaba':
            m = replace(m, tau_gaba_ms=m.tau_gaba_ms*p)
        else:
            raise ValueError(self.parameter)
        a, b, aa, ab, bb = self.z_moments
        z = a+dose*b
        z2 = aa+2*dose*ab+dose*dose*bb
        if np.any(z < 0) or np.any(z > 1):
            raise ValueError('nonphysical Z path')
        return m, z, z2, eta, tau, ops

    def f(self, x, p):
        m,z,z2,eta,tau,_ = self.at(p)
        return spatial_z_residual(m,x,z_field=z,z_second_moment=z2,
                                  eta_m=eta,tau_m_slow_ms=tau)

    def jac(self, x, p):
        m,z,z2,eta,tau,_ = self.at(p)
        return spatial_z_jacobian(m,x,z_field=z,z_second_moment=z2,
                                  eta_m=eta,tau_m_slow_ms=tau)

    def dp(self, x, p):
        h=1e-5
        return (self.f(x,p+h)-self.f(x,p-h))/(2*h)

    def physical(self, x):
        n=self.base.n_cells
        cap=np.r_[np.full(n,1/self.base.tau_ref_e_ms),
                  np.full(n,1/self.base.tau_ref_i_ms)]
        return bool(np.all(np.isfinite(x)) and np.all(x >= -1e-9)
                    and np.all(x <= cap+1e-9))

    def solve(self, p, x):
        fit=root(lambda q:self.f(q,p),x,jac=lambda q:self.jac(q,p),
                 options={'xtol':1e-9,'maxfev':2000})
        err=float(np.max(np.abs(self.f(fit.x,p))))
        return fit.x, bool(err<1e-8 and self.physical(fit.x)), err


def arc_continue(family, first, second, bounds, max_steps=450, step=.008):
    """Scaled predictor-corrector; preserves traversal order through folds."""
    x0,p0=first; x,p=second; n=len(x)
    def norm(v): return np.sqrt(np.mean(v[:-1]**2)+v[-1]**2)
    t=np.r_[x-x0,p-p0];t/=norm(t)
    rows=[(x0.copy(),p0,float(t[-1])),(x.copy(),p,float(t[-1]))]
    reason='STEP_LIMIT'
    for k in range(max_steps):
        current=np.r_[x,p]
        accepted=False
        for trial in range(7):
            ds=step/2**trial; pred=current+ds*t; q=pred.copy()
            for iteration in range(16):
                if not bounds[0]-.02<=q[-1]<=bounds[1]+.02: break
                try:
                    f=family.f(q[:-1],q[-1]);j=family.jac(q[:-1],q[-1]);dp=family.dp(q[:-1],q[-1])
                except (ValueError,FloatingPointError): break
                metric=np.r_[t[:-1]/n,t[-1]]
                residual=np.r_[f,metric@(q-pred)]
                if np.max(np.abs(residual))<2e-10:
                    accepted=family.physical(q[:-1]);break
                aug=np.block([[j,dp[:,None]],[metric[None,:]]])
                try: delta=linalg.solve(aug,-residual)
                except linalg.LinAlgError: break
                if np.any(~np.isfinite(delta)):break
                q+=delta
            if accepted:break
        if not accepted:
            reason='CORRECTOR_FAILED';break
        x,p=q[:-1],float(q[-1])
        j=family.jac(x,p);dp=family.dp(x,p)
        aug=np.block([[j,dp[:,None]],[np.r_[t[:-1]/n,t[-1]][None,:]]])
        new=linalg.solve(aug,np.r_[np.zeros(n),1.]);new/=norm(new)
        if np.mean(new[:-1]*t[:-1])+new[-1]*t[-1]<0:new=-new
        t=new;rows.append((x.copy(),p,float(t[-1])))
        if not bounds[0]<p<bounds[1]:reason='PARAMETER_BOUND';break
    return rows,reason


def polish_fold(family, x, p):
    """Augmented F=0, Jv=0, |v|=1 plus generic-fold evidence."""
    j=family.jac(x,p)
    _,_,vh=linalg.svd(j);v=vh[-1]
    n=len(x)
    def equations(q):
        r,s,w=q[:n],q[n],q[n+1:]
        return np.r_[family.f(r,s),family.jac(r,s)@w,np.dot(w,w)-1.]
    fit=root(equations,np.r_[x,p,v],options={'xtol':1e-8,'maxfev':3500})
    q=fit.x;r,s,v=q[:n],float(q[n]),q[n+1:]
    err=float(np.max(np.abs(equations(q))))
    if err>1e-7 or not family.physical(r):
        return {'confirmed':False,'residual':err,'message':str(fit.message)},None
    j=family.jac(r,s);u,sv,vh=linalg.svd(j);w=u[:,-1];v=vh[-1]
    trans=float(w@family.dp(r,s))
    coeff=[]
    for h in (1e-4,5e-5):
        coeff.append(float(w@(family.f(r+h*v,s)-2*family.f(r,s)+family.f(r-h*v,s))/h**2))
    vals=linalg.eigvals(j);idx=np.argsort(np.abs(vals))
    generic=bool(sv[-2]>1e-5 and abs(trans)>1e-6 and abs(coeff[1])>1e-5
                 and abs(coeff[0]-coeff[1])<.05*max(abs(coeff[1]),1e-12))
    record=dict(confirmed=generic,parameter=s,residual=err,
        singular_values_smallest=sv[-3:].tolist(),transversality=trans,
        quadratic_coefficients=coeff,zero_eigenvalue=[float(vals[idx[0]].real),float(vals[idx[0]].imag)],
        next_eigenvalue_modulus=float(abs(vals[idx[1]])),
        population_hz=float(np.average(r[:family.base.n_cells],weights=family.base.count_e)*1000))
    return record,r


def corrected_delay_matrix(family, x, p):
    """Native-dt reduction tangent, evaluated at the self-consistent M input.

    Historical delayed_step_matrix evaluated transfer derivatives at eta=0
    even when adding the M block. Rebuild those rows at mu-eta*tau*rE.
    Keep its delay/synaptic update convention and variance feedback intact.
    """
    m,z,z2,eta,tau,ops=family.at(p)
    mat=delayed_step_matrix(m,ops,x,z_field=z,z_second_moment=z2,
                            eta_m=eta,tau_m_slow_ms=tau).tolil()
    n=m.n_cells;dt=ops.dt_ms;te=m.tau_mem_e_ms
    mu,se,mi,si=spatial_z_moments(m,x[:n],x[n:],z_field=z,z_second_moment=z2,
                                  eta_m=eta,tau_m_slow_ms=tau)
    pme,pse,_,_=_transfer_derivatives(m,mu,se,mi,si)
    eye=np.eye(n)
    mat[:n,:n]=(1-dt/te)*eye+dt*(pse/(2*se))[:,None]*m.v_ee
    mat[:n,n:2*n]=dt*(pse*z2/(2*se))[:,None]*m.v_ei
    mat[:n,2*n:3*n]=np.diag(dt/te*pme)
    mat[:n,3*n:4*n]=np.diag(-dt/te*z*pme)
    if eta>0:mat[:n,-n:]=np.diag(-dt/te*eta*pme)
    return mat.tocsr()
