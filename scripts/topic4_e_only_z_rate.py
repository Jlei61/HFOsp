"""Corrected spatial-rate closure with E-only postsynaptic Z, not all-GABA gain."""
from analyze_topic4_corrected_bifurcation import System as AllGabaSystem
import numpy as np


class EOnlySystem(AllGabaSystem):
    def qvec(self,q): return np.broadcast_to(np.asarray(q,float),(self.n,))

    def moments(self,r,q):
        m=self.m;n=self.n;e,i=r[:n],r[n:];te,ti=m.tau_mem_e_ms,m.tau_mem_i_ms;nu=m.nu_ext_per_ms;q=self.qvec(q)
        mu=np.r_[te*(self.ga*(m.w_ee@e+m.j_ext_e_mv*nu)-self.gg*q*(m.w_ei@i)),
                  ti*(self.ga*(m.w_ie@e+m.j_ext_i_mv*nu)-self.gg*(m.w_ii@i))]
        ex=np.r_[te*(m.v_ee@e+m.j_ext_e_mv**2*nu),ti*(m.v_ie@e+m.j_ext_i_mv**2*nu)]
        inh=np.r_[te*q*q*(m.v_ei@i),ti*(m.v_ii@i)]
        return mu,ex,inh

    def blocks(self,r,q):
        m=self.m;n=self.n;u,x,h=self.gains(r,q);te,ti=m.tau_mem_e_ms,m.tau_mem_i_ms;q=self.qvec(q)
        mean=np.block([[te*self.ga*m.w_ee,-te*self.gg*q[:,None]*m.w_ei],
                       [ti*self.ga*m.w_ie,-ti*self.gg*m.w_ii]])
        var=np.block([[x[:n,None]*te*m.v_ee,h[:n,None]*te*(q*q)[:,None]*m.v_ei],
                      [x[n:,None]*ti*m.v_ie,h[n:,None]*ti*m.v_ii]])
        return u,mean,var

    def characteristic(self,z,r,q,pre=None):
        n=self.n;dt=self.dt;m=self.m;u,_,var=self.blocks(r,q) if pre is None else pre;q=self.qvec(q)
        weights=np.exp(-np.arange(1,self.cfg['max_delay_steps']+1)*np.log(complex(z)))
        w={k:(op@weights).reshape(n,n) for k,op in self.block_ops.items()}
        def filt(rise,decay,tm):
            a=np.exp(-dt/rise);b=np.exp(-dt/decay)
            return dt*tm/rise*(1-b)*z*z/((z-a)*(z-b))
        aE=filt(self.ra,self.ta,m.tau_mem_e_ms);gE=filt(self.rg,self.tau,m.tau_mem_e_ms)
        aI=filt(self.ra,self.ta,m.tau_mem_i_ms);gI=filt(self.rg,self.tau,m.tau_mem_i_ms)
        mean=np.block([[aE*w['ee'],-q[:,None]*gE*w['ei']],[aI*w['ie'],-gI*w['ii']]])
        return ((z-1)*np.diag(self.tr)/dt+np.eye(2*n)-var-u[:,None]*mean)

    def char_refined(self,lam,r,q,dt=None):
        n=self.n;m=self.m;u,_,var=self.blocks(r,q);q=self.qvec(q)
        weights=np.exp(-np.arange(1,self.cfg['max_delay_steps']+1)*self.dt*lam)
        w={k:(op@weights).reshape(n,n) for k,op in self.block_ops.items()}
        def filt(rise,decay,tm,area):
            if dt is None:return tm*area/((1+lam*rise)*(1+lam*decay))
            z=np.exp(lam*dt);a=np.exp(-dt/rise);b=np.exp(-dt/decay)
            return tm*area*(1-a)*(1-b)*z*z/((z-a)*(z-b))
        ae=filt(self.ra,self.ta,m.tau_mem_e_ms,self.ga);ge=filt(self.rg,self.tau,m.tau_mem_e_ms,self.gg)
        ai=filt(self.ra,self.ta,m.tau_mem_i_ms,self.ga);gi=filt(self.rg,self.tau,m.tau_mem_i_ms,self.gg)
        mean=np.block([[ae*w['ee'],-q[:,None]*ge*w['ei']],[ai*w['ie'],-gi*w['ii']]])
        tf=lam if dt is None else np.expm1(lam*dt)/dt
        return tf*np.diag(self.tr)+np.eye(2*n)-var-u[:,None]*mean


def native_gaba_variance_factor(system):
    """Current variance = factor * sum_j(w_ij^2 * r_j_per_ms) for independent spikes."""
    dt=system.dt;ar=np.exp(-dt/system.rg);ad=np.exp(-dt/system.tau)
    pref=(system.m.tau_mem_e_ms/system.rg*(1-ad)/(ar-ad))**2
    return dt*pref*(ar*ar/(1-ar*ar)+ad*ad/(1-ad*ad)-2*ar*ad/(1-ar*ad))
