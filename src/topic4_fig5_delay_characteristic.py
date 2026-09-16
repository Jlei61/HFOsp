"""Eliminate delay histories and filters for a small characteristic matrix.

The matrix is equivalent to the native Euler delay map away from filter
poles; it also supports dt->0 and smaller dt at the same physical delays.
"""
import numpy as np
from scipy import linalg
from scipy.optimize import root
from src.topic4_dual_core_spatial_z import spatial_z_moments
from src.topic4_patient_zm_meanfield import _transfer_derivatives


def positive_majorant_certificate(family,rates,parameter):
    """Sufficient contraction bound for the native delay map, not an eigenvalue.

    For |mu|>=r above all isolated filter poles, take absolute values of
    rate/filter feedback and bound each delay by r**(-d). A nonnegative
    loop gain with spectral radius <1 excludes every such multiplier.
    """
    m,z,z2,eta,tau,ops=family.at(parameter);n=m.n_cells;dt=ops.dt_ms
    max_tau=max(m.tau_mem_e_ms,m.tau_mem_i_ms,m.tau_ampa_ms,m.tau_gaba_ms,tau if eta else 0.)
    growth=-.5/max_tau;r=np.exp(growth*dt);a=(r-1)/dt
    mu,se,mi,si=spatial_z_moments(m,rates[:n],rates[n:],z_field=z,z_second_moment=z2,eta_m=eta,tau_m_slow_ms=tau)
    pe,ve,pi,vi=[abs(x) for x in _transfer_derivatives(m,mu,se,mi,si)]
    k={}
    for key in ('ee','ei','ie','ii'):
        coo=getattr(ops,'w_'+key+'_history').tocoo();steps=coo.col//n+1
        k[key]=np.bincount(coo.row*n+coo.col%n,weights=abs(coo.data)*r**(-steps),minlength=n*n).reshape(n,n)
    ae=a+1/m.tau_mem_e_ms;ai=a+1/m.tau_mem_i_ms
    ee=(ve/(2*se*ae))[:,None]*m.v_ee+(pe/(ae*(1+a*m.tau_ampa_ms)))[:,None]*k['ee']
    if eta:ee+=np.diag(eta*pe/(m.tau_mem_e_ms*ae*(a+1/tau)))
    ei=(ve*z2/(2*se*ae))[:,None]*m.v_ei+(z*pe/(ae*(1+a*m.tau_gaba_ms)))[:,None]*k['ei']
    ie=(vi/(2*si*ai))[:,None]*m.v_ie+(pi/(ai*(1+a*m.tau_ampa_ms)))[:,None]*k['ie']
    ii=(vi/(2*si*ai))[:,None]*m.v_ii+(pi/(ai*(1+a*m.tau_gaba_ms)))[:,None]*k['ii']
    loop=np.block([[ee,ei],[ie,ii]])
    radius=float(max(abs(linalg.eigvals(loop))))
    return dict(certified=bool(radius<1 and np.all(loop>=0)),loop_spectral_radius=radius,
                growth_upper_bound_per_ms=growth,multiplier_bound=r,
                method='Nonnegative absolute-feedback majorant, including all physical delay bins and dynamic M. Sufficient bound, not a leading eigenvalue estimate.')


class DelayCharacteristic:
    def __init__(self, family, rates):
        self.family=family;self.rates=np.asarray(rates);self.kernels={}
        n=family.base.n_cells
        for name in ('ee','ei','ie','ii'):
            m=getattr(family.operators,'w_'+name+'_history').tocoo()
            self.kernels[name]=(m.row*n+m.col%n,m.col//n+1,m.data)

    def matrix(self, exponent, parameter, dt_ms=.1):
        m,z,z2,eta,tau,ops=self.family.at(parameter);n=m.n_cells
        # Only tau_GABA is varied by the current runner; reject pathway mutation
        # because those would need correspondingly rescaled delay kernels.
        if self.family.parameter!='tau_gaba':
            raise ValueError('current characteristic family is tau_gaba only')
        mu,se,mi,si=spatial_z_moments(m,self.rates[:n],self.rates[n:],z_field=z,z_second_moment=z2,
                                      eta_m=eta,tau_m_slow_ms=tau)
        pe,ve,pi,vi=_transfer_derivatives(m,mu,se,mi,si)
        a=exponent if dt_ms==0 else np.expm1(exponent*dt_ms)/dt_ms
        k={}
        for name,(flat,steps,weights) in self.kernels.items():
            w=weights*np.exp(-exponent*steps*ops.dt_ms)
            k[name]=(np.bincount(flat,weights=w.real,minlength=n*n)+
                     1j*np.bincount(flat,weights=w.imag,minlength=n*n)).reshape(n,n)
        ee=np.diag(np.full(n,a+1/m.tau_mem_e_ms,dtype=complex))
        ee-= (ve/(2*se))[:,None]*m.v_ee
        ee-= (pe/(1+a*m.tau_ampa_ms))[:,None]*k['ee']
        if eta>0:ee+=np.diag(eta*pe/m.tau_mem_e_ms/(a+1/tau))
        ei=-(ve*z2/(2*se))[:,None]*m.v_ei+(z*pe/(1+a*m.tau_gaba_ms))[:,None]*k['ei']
        ie=-(vi/(2*si))[:,None]*m.v_ie-(pi/(1+a*m.tau_ampa_ms))[:,None]*k['ie']
        ii=np.diag(np.full(n,a+1/m.tau_mem_i_ms,dtype=complex))
        ii-= (vi/(2*si))[:,None]*m.v_ii
        ii+= (pi/(1+a*m.tau_gaba_ms))[:,None]*k['ii']
        return np.block([[ee,ei],[ie,ii]])

    def small_eigenvalue(self,exponent,parameter,dt_ms=.1):
        vals=linalg.eigvals(self.matrix(exponent,parameter,dt_ms))
        return vals[np.argmin(abs(vals))]

    def crossing(self,initial_frequency_hz,initial_scale,dt_ms=.1):
        def residual(q):
            freq,scale=q
            if scale<=0:return np.array([1e2+abs(scale),freq])
            v=self.small_eigenvalue(2j*np.pi*freq/1000,scale,dt_ms)
            return np.array([v.real,v.imag])
        fit=root(residual,[initial_frequency_hz,initial_scale],options={'xtol':1e-8})
        error=float(np.max(abs(residual(fit.x))))
        return dict(frequency_hz=float(fit.x[0]),scale=float(fit.x[1]),residual=error,
            converged=bool(error<1e-8 and fit.x[1]>0 and fit.x[0]>1.),dt_ms=dt_ms)

    def follow_mode(self,parameter,initial_frequency_hz,dt_ms=.1,initial_growth_per_ms=0.):
        def residual(q):
            growth,freq=q
            v=self.small_eigenvalue(growth+2j*np.pi*freq/1000,parameter,dt_ms)
            return np.array([v.real,v.imag])
        fit=root(residual,[initial_growth_per_ms,initial_frequency_hz],options={'xtol':1e-9})
        error=float(np.max(abs(residual(fit.x))))
        return dict(growth_per_ms=float(fit.x[0]),frequency_hz=float(fit.x[1]),
                    residual=error,converged=bool(error<1e-8),parameter=parameter,dt_ms=dt_ms)
