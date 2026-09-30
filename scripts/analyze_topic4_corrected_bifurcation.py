#!/usr/bin/env python3
"""Fixed points and exact native-step delay characteristic of corrected spatial rate."""
from validate_topic4_fixed_rate_base import ROOT, OUT as BASE, read, write
import numpy as np
from scipy import sparse
from scipy.optimize import root
from scipy.linalg import eigvals, eig, svd
from src.topic4_patient_zm_meanfield import load_patient_coarse_model, transfer_rates
from pathlib import Path

OUT=ROOT/'results/topic4_sef_hfo/corrected_rate_bifurcation_v1'


class System:
    def __init__(self,grid=10,tau=20.611550480127335):
        folder=BASE/f'coarse_{grid}';self.m=m=load_patient_coarse_model(folder/'model.npz');self.cfg=cfg=read(folder/'prepared.json')
        self.n=m.n_cells;self.dt=cfg['dt_ms'];self.tau=tau
        self.ra=cfg['tau_r_ampa_ms'];self.rg=cfg['tau_r_gaba_ms'];self.ta=m.tau_ampa_ms
        self.ga=self.dt/(self.ra*(1-np.exp(-self.dt/self.ra)));self.gg=self.dt/(self.rg*(1-np.exp(-self.dt/self.rg)))
        self.tr=np.r_[np.full(self.n,read(BASE/'colored_response_diagnostic.json')['best_tau_rate_ms']),np.full(self.n,read(BASE/'colored_response_diagnostic_I.json')['best_tau_rate_ms'])]
        self.ops={k:sparse.load_npz(folder/f'delay_{k}.npz') for k in ('ee','ei','ie','ii')}
        self.block_ops={k:sparse.csr_matrix((o.data, (o.tocoo().row*self.n+o.tocoo().col%self.n,o.tocoo().col//self.n)),shape=(self.n*self.n,cfg['max_delay_steps'])) for k,o in self.ops.items()}

    def moments(self,r,q):
        m=self.m;n=self.n;e,i=r[:n],r[n:];te,ti=m.tau_mem_e_ms,m.tau_mem_i_ms;nu=m.nu_ext_per_ms
        mu=np.r_[te*(self.ga*(m.w_ee@e+m.j_ext_e_mv*nu)-self.gg*q*(m.w_ei@i)),ti*(self.ga*(m.w_ie@e+m.j_ext_i_mv*nu)-self.gg*q*(m.w_ii@i))]
        ex=np.r_[te*(m.v_ee@e+m.j_ext_e_mv**2*nu),ti*(m.v_ie@e+m.j_ext_i_mv**2*nu)]
        inh=np.r_[te*q*q*(m.v_ei@i),ti*q*q*(m.v_ii@i)]
        return mu,ex,inh

    def phi(self,mu,ex,inh):
        n=self.n;m=self.m;tm=np.r_[np.full(n,m.tau_mem_e_ms),np.full(n,m.tau_mem_i_ms)]
        shift=2.065/2*np.sqrt(np.maximum((ex*(self.ra+self.ta)+inh*(self.rg+self.tau))/tm,1e-16))
        sig=np.sqrt(np.maximum(ex+inh,1e-12));p=transfer_rates(m,mu[:n]-shift[:n],sig[:n],mu[n:]-shift[n:],sig[n:])
        return np.r_[p[0],p[1]]

    def F(self,r,q):return self.phi(*self.moments(r,q))-r

    def gains(self,r,q):
        vals=self.moments(r,q);derivatives=[]
        for k in range(3):
            eps=1e-5*np.maximum(abs(vals[k]),1.);p=[v.copy() for v in vals];a=[v.copy() for v in vals];p[k]+=eps;a[k]-=eps
            derivatives.append((self.phi(*p)-self.phi(*a))/(2*eps))
        return derivatives

    def blocks(self,r,q):
        m=self.m;n=self.n;u,x,h=self.gains(r,q);te,ti=m.tau_mem_e_ms,m.tau_mem_i_ms
        mean=np.block([[te*self.ga*m.w_ee,-te*self.gg*q*m.w_ei],[ti*self.ga*m.w_ie,-ti*self.gg*q*m.w_ii]])
        var=np.block([[x[:n,None]*te*m.v_ee,h[:n,None]*te*q*q*m.v_ei],[x[n:,None]*ti*m.v_ie,h[n:,None]*ti*q*q*m.v_ii]])
        return u,mean,var

    def jac(self,r,q):
        u,mean,var=self.blocks(r,q);return u[:,None]*mean+var-np.eye(2*self.n)

    def solve(self,q,initial):
        sol=root(lambda r:self.F(r,q),initial,jac=lambda r:self.jac(r,q),tol=1e-10)
        err=float(np.max(abs(self.F(sol.x,q))))
        valid=err<1e-8 and sol.x.min()>-1e-8 and sol.x[:self.n].max()<=.5+1e-7 and sol.x[self.n:].max()<=1+1e-7
        return sol.x,err,valid

    def characteristic(self,z,r,q,pre=None):
        # h_d(t)=r(t-d), g(t+1)=a_r*g+b*D*h; c(t+1)=a_d*c+(1-a_d)*g(t+1).
        # Eliminate gating/current/history exactly; poles at a_r/a_d are outside near-unit tests.
        n=self.n;dt=self.dt;m=self.m;u,_,var=self.blocks(r,q) if pre is None else pre
        weights=np.exp(-np.arange(1,self.cfg['max_delay_steps']+1)*np.log(complex(z)))
        w={k:(op@weights).reshape(n,n) for k,op in self.block_ops.items()}
        def filt(rise,decay,tm):
            a=np.exp(-dt/rise);b=np.exp(-dt/decay)
            return dt*tm/rise*(1-b)*z*z/((z-a)*(z-b))
        aE=filt(self.ra,self.ta,m.tau_mem_e_ms);gE=filt(self.rg,self.tau,m.tau_mem_e_ms)
        aI=filt(self.ra,self.ta,m.tau_mem_i_ms);gI=filt(self.rg,self.tau,m.tau_mem_i_ms)
        mean=np.block([[aE*w['ee'],-q*gE*w['ei']],[aI*w['ie'],-q*gI*w['ii']]])
        return ((z-1)*np.diag(self.tr)/dt+np.eye(2*n)-var-u[:,None]*mean)

    def char_refined(self,lam,r,q,dt=None):
        """lambda in inverse ms. Native impulse area held fixed, physical delays unchanged."""
        n=self.n;m=self.m;u,_,var=self.blocks(r,q)
        weights=np.exp(-np.arange(1,self.cfg['max_delay_steps']+1)*self.dt*lam)
        w={k:(op@weights).reshape(n,n) for k,op in self.block_ops.items()}
        def filt(rise,decay,tm,area):
            if dt is None:return tm*area/((1+lam*rise)*(1+lam*decay))
            z=np.exp(lam*dt);a=np.exp(-dt/rise);b=np.exp(-dt/decay)
            return tm*area*(1-a)*(1-b)*z*z/((z-a)*(z-b))
        ae=filt(self.ra,self.ta,m.tau_mem_e_ms,self.ga);ge=filt(self.rg,self.tau,m.tau_mem_e_ms,self.gg)
        ai=filt(self.ra,self.ta,m.tau_mem_i_ms,self.ga);gi=filt(self.rg,self.tau,m.tau_mem_i_ms,self.gg)
        mean=np.block([[ae*w['ee'],-q*ge*w['ei']],[ai*w['ie'],-q*gi*w['ii']]])
        timefactor=lam if dt is None else np.expm1(lam*dt)/dt
        return timefactor*np.diag(self.tr)+np.eye(2*n)-var-u[:,None]*mean


def branches():
    s=System();rows=[];arrays={}
    for name,qs,initial in [('high',np.linspace(.25,1.15,181),np.full(200,.45)),('low',np.linspace(1.25,.25,201),np.full(200,.00005))]:
        r=initial.copy()
        for q in qs:
            r2,err,ok=s.solve(float(q),r)
            if not ok:
                rows.append({'branch':name,'q':float(q),'valid':False,'residual':err});break
            r=r2;j=s.jac(r,q);ev=eigvals(j);key=f'{name}_{q:.6f}';arrays[key]=r
            row={'branch':name,'q':float(q),'valid':True,'residual':err,
                'mean_e_hz':float(np.average(r[:s.n],weights=s.m.count_e)*1000),
                'min_abs_static_eigenvalue':float(np.min(abs(ev))),
                'max_real_static_residual_eigenvalue':float(np.max(ev.real))}
            rows.append(row)
        write(OUT/'fixed_points.json',{'status':'DIRECT_PARAMETER_BRANCHES','rows':rows,
            'static_J_warning':'Jacobian of equilibrium residual only. Its eigenvalues are not full delayed dynamical stability.'})
        np.savez_compressed(OUT/'fixed_points.npz',**arrays)
    print('branches',len(rows),flush=True)


def fold(branch,tau=20.611550480127335,initial=None):
    s=System(tau=tau);n=2*s.n
    if initial is None:
        a=np.load(OUT/'fixed_points.npz');q0=.405 if branch=='high' else .71
        r0=a[f'{branch}_{q0:.6f}'];ev,V=eig(s.jac(r0,q0));v0=V[:,np.argmin(abs(ev))].real;v0/=np.linalg.norm(v0)
    else:r0,q0,v0=initial
    # Extended saddle-node system, no inference from failure of direct q continuation.
    def fun(y):
        r,q,v=y[:n],y[n],y[n+1:]
        return np.r_[s.F(r,q),s.jac(r,q)@v,v@v-1]
    sol=root(fun,np.r_[r0,q0,v0],tol=2e-9);r,q,v=sol.x[:n],sol.x[n],sol.x[n+1:]
    err=float(max(abs(fun(sol.x))));assert err<1e-7,(branch,tau,sol.message,err)
    J=s.jac(r,q);ev,L,R=eig(J,left=True,right=True);j=np.argmin(abs(ev));w=L[:,j].real;v=R[:,j].real;v/=np.linalg.norm(v);w/=w@v
    h=1e-4 if branch=='high' else 1e-7
    fq=(s.F(r,q+1e-5)-s.F(r,q-1e-5))/(2e-5)
    curvature=(s.F(r+h*v,q)-2*s.F(r,q)+s.F(r-h*v,q))/(h*h)
    row={'branch':branch,'tau_gaba_ms':tau,'q':float(q),'residual':err,'mean_e_hz':float(np.average(r[:s.n],weights=s.m.count_e)*1000),
        'static_eigenvalue_nearest_zero':float(ev[j].real),'next_smallest_static_eigenvalue_abs':float(np.sort(abs(ev))[1]),
        'parameter_transversality_wFq':float(w@fq),'quadratic_nondegeneracy_wFrrvv':float(w@curvature),
        'q_definition':'Global I-to-E AND I-to-I jump multiplier; variance scales q squared.'}
    dest=OUT/'folds';dest.mkdir(exist_ok=True);name=f'{branch}_tau{tau:g}';np.savez_compressed(dest/f'{name}.npz',r=r,q=q,v=v,w=w)
    write(dest/f'{name}.json',row);print(row,flush=True);return r,q,v


def arc(branch,steps=60):
    s=System();f=np.load(OUT/'folds'/f'{branch}_tau{s.tau:g}.npz');r,q,v=f['r'],float(f['q']),f['v']
    scale=.3 if branch=='high' else .0001;n=len(r);start=np.r_[r/scale,q];points=[]
    for direction in (-1,1):
        y=start.copy();t=np.r_[v*direction,0.];ds=.12
        for k in range(steps):
            pred=y+ds*t
            def fun(x):return np.r_[s.F(x[:n]*scale,x[n])/scale,(x-pred)@t]
            def jac(x):
                r=x[:n]*scale;q=x[n];fq=(s.F(r,q+1e-5)-s.F(r,q-1e-5))/(2e-5)/scale
                return np.vstack([np.column_stack([s.jac(r,q),fq]),t])
            sol=root(fun,pred,jac=jac,tol=1e-9)
            if max(abs(fun(sol.x)))>1e-7:break
            y=sol.x;r=y[:n]*scale;q=y[n]
            if r.min()<-1e-8 or q<.15 or q>1.5:break
            *_,vh=svd(jac(y)[:-1]);tnew=vh[-1];tnew*=np.sign(tnew@t);t=tnew
            points.append((direction,k,q,r.copy()))
    np.savez_compressed(OUT/f'arc_{branch}.npz',q=np.array([p[2] for p in points]),r=np.array([p[3] for p in points]),direction=np.array([p[0] for p in points]))
    print('arc',branch,len(points),flush=True)


def hopf(tau=20.611550480127335,guess=(.74,4.),dt=.1):
    s=System(tau=tau);archive=np.load(OUT/'fixed_points.npz');r0=archive['low_0.800000'];cache={}
    def equation(x):
        q,f=x;r,err,ok=s.solve(q,r0)
        if not ok:raise RuntimeError(('Equilibrium solve failed in Hopf search',q,err))
        M=s.char_refined(2j*np.pi*f/1000,r,q,dt);ev=eigvals(M);v=ev[np.argmin(abs(ev))]
        cache['r']=r;return [v.real,v.imag]
    sol=root(equation,guess,tol=2e-9);err=max(abs(np.array(equation(sol.x))));assert err<1e-7,(tau,sol.x,err)
    q,f=sol.x;r=cache['r'].copy();lam0=2j*np.pi*f/1000
    ev=eigvals(s.char_refined(lam0,r,q,dt))
    # Independently follow the critical pair at nearby q, retaining its frequency.
    track=[]
    for qq in (q-.002,q-.0002,q,q+.0002,q+.002):
        rr,_,ok=s.solve(qq,r);assert ok
        def fun(x):
            e=eigvals(s.char_refined((x[0]+1j*x[1])/1000,rr,qq,dt));v=e[np.argmin(abs(e))];return [v.real,v.imag]
        ans=root(fun,[0,2*np.pi*f],tol=1e-9);ee=max(abs(np.array(fun(ans.x))));assert ee<1e-7
        track.append({'q':float(qq),'real_per_s':float(ans.x[0]),'frequency_hz':float(ans.x[1]/2/np.pi),'residual':float(ee)})
    row={'tau_gaba_ms':tau,'dt_ms':dt,'q':float(q),'frequency_hz':float(f),'residual':float(err),
         'mean_e_hz':float(np.average(r[:s.n],weights=s.m.count_e)*1000),'next_characteristic_eigenvalue_abs':float(np.sort(abs(ev))[1]),
         'critical_pair_track':track,'d_real_lambda_dq_per_s':(track[3]['real_per_s']-track[1]['real_per_s'])/.0004,
         'interpretation':'Complex unit-circle crossing for native time-step map; imaginary-axis crossing for fixed-area continuous delayed limit. First Lyapunov coefficient not computed; super/subcritical designation withheld.'}
    folder=OUT/'hopf';folder.mkdir(exist_ok=True);name=f'tau{tau:g}_dt{dt}'
    write(folder/f'{name}.json',row);np.savez_compressed(folder/f'{name}.npz',r=r,q=q,frequency_hz=f)
    print('Hopf',row,flush=True);return row


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--fold',choices=['high','low']);p.add_argument('--arc',choices=['high','low']);p.add_argument('--hopf',action='store_true');a=p.parse_args()
    if a.fold:fold(a.fold)
    elif a.arc:arc(a.arc)
    elif a.hopf:hopf()
    else:branches()
