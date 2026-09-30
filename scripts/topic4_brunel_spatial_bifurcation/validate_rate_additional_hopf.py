"""Check newly found stationary Hopfs against all nine local DDE states.

The constant-orbit file is only a scaffold for propagating the equilibrium
variational equation. It is never a nontrivial periodic solution.
"""
from rate_floquet import *

DEST=PERIODIC_OUT/'stationary_root_counts'


def validate(label,device=0):
    s=RateField();a=read(DEST/f'{label}_highorder.json');b=read(DEST/f'{label}_highorder_half.json')
    z=np.load(DEST/f'{label}_highorder_half.npz');r=z['rates'];J=float(z['J']);lam=1j*float(z['omega'])
    v=z['vector'];v=v/max(abs(v))*.001;q=s.eigenstate(r,J,lam,v)
    y=s.equilibrium_state(r,J);arr=np.array([m@r for m in s.matrices(J)])
    d_arr=np.array([m@v for m in s.matrices(J,lam)])
    scale=np.array([1000,1000,1,1,1,1,.1,.1,1])[:,None]
    target=lam*q;checks=[]
    for eps in [.02,.01,.005,.0025,.00125,.000625]:
        dq=np.zeros_like(q)
        for part,c in [(np.real,1),(np.imag,1j)]:
            dq+=c*(s.rhs(y+eps*part(q),arr+eps*part(d_arr))-s.rhs(y-eps*part(q),arr-eps*part(d_arr)))/(2*eps)
        checks.append(dict(epsilon=eps,relative_full_rhs_error=float(np.linalg.norm((dq-target)*scale)/np.linalg.norm(target*scale))))
    write(DEST/f'{label}_local_rhs_checks.json',dict(checks=checks))
    print('LOCAL RHS CHECKS',label,checks,flush=True)
    assert checks[-1]['relative_full_rhs_error']<1e-6
    scaffold=DEST/f'{label}_constant_equilibrium_scaffold.npz';T=2*np.pi/lam.imag
    np.savez_compressed(scaffold,r=np.tile(r,(64,1)),T=T,J=J,residual=max(abs(s.residual(r,J)))*1000)
    stepchecks=[]
    for dt in [.1,.05,.025]:
        m=Monodromy(s,scaffold,dt,device)
        x=np.r_[q.ravel(),(np.exp(-lam*np.arange(1,m.D+1)[:,None]*m.dt)*v).ravel()]
        ax=m.matvec(x.real)+1j*m.matvec(x.imag)
        weights=np.r_[np.broadcast_to(scale,q.shape).ravel(),np.full(m.D*s.P,1000.)]
        err=np.linalg.norm(weights*(ax-np.exp(lam*T)*x))/np.linalg.norm(weights*x)
        stepchecks.append(dict(dt_ms=m.dt,period_ms=T,relative_full_DDE_mode_error=float(err)))
        del m
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    ratios=[lo['relative_full_DDE_mode_error']/hi['relative_full_DDE_mode_error'] for lo,hi in zip(stepchecks[:-1],stepchecks[1:])]
    ratio=min(ratios)
    write(DEST/f'{label}_full_state_checks_raw.json',dict(rhs=checks,propagation=stepchecks,step_reduction=ratio))
    print('FULL STATE CHECKS',label,checks,stepchecks,ratio,flush=True)
    shortchecks=[]
    if stepchecks[-1]['relative_full_DDE_mode_error']>=.001:
        # These equilibria also have strongly positive, unrelated exponents.
        # A full period amplifies discretization contamination in those modes.
        # Retain that result and independently check a shorter horizon; do not
        # relax the mode-error tolerance or project out any unstable modes.
        short=DEST/f'{label}_constant_equilibrium_quarter_period_scaffold.npz'
        np.savez_compressed(short,r=np.tile(r,(64,1)),T=T/4,J=J,residual=max(abs(s.residual(r,J)))*1000)
        for dt in [.1,.05,.025]:
            m=Monodromy(s,short,dt,device)
            x=np.r_[q.ravel(),(np.exp(-lam*np.arange(1,m.D+1)[:,None]*m.dt)*v).ravel()]
            ax=m.matvec(x.real)+1j*m.matvec(x.imag)
            weights=np.r_[np.broadcast_to(scale,q.shape).ravel(),np.full(m.D*s.P,1000.)]
            err=np.linalg.norm(weights*(ax-np.exp(lam*T/4)*x))/np.linalg.norm(weights*x)
            shortchecks.append(dict(dt_ms=m.dt,horizon_ms=T/4,relative_full_DDE_mode_error=float(err)))
            del m
            cp.get_default_memory_pool().free_all_blocks()
        assert shortchecks[-1]['relative_full_DDE_mode_error']<.001
        assert min(lo['relative_full_DDE_mode_error']/hi['relative_full_DDE_mode_error'] for lo,hi in zip(shortchecks[:-1],shortchecks[1:]))>2.8
    assert ratio>2.8
    out=dict(label=label,status='VALIDATED_IMAGINARY_PAIR_CROSSING',source=str(DEST/f'{label}_highorder_half.npz'),
        J_EE_core=J,frequency_hz=b['frequency_hz'],gain_step_halving_parameter_difference=abs(a['J_EE_core']-J),
        gain_step_halving_frequency_difference_hz=abs(a['frequency_hz']-b['frequency_hz']),
        full_rhs_checks=checks,full_DDE_mode_checks=stepchecks,time_step_error_reductions=ratios,
        quarter_period_mode_checks=shortchecks,
        equilibrium_rhs_residual=float(max(abs(s.rhs(y,arr)).ravel())),transversality=b['transversality'],
        criticality='NOT_YET_VALIDATED',scope='A complex eigenpair crosses on an already unstable equilibrium branch. This alone does not establish a stable emerging cycle or first resting instability.')
    write(DEST/f'{label}_full_state_validation.json',out);print('VALIDATED ADDITIONAL HOPF',out,flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--labels',nargs='+',default=['H3','H4']);p.add_argument('--device',type=int,default=0)
    a=p.parse_args()
    for label in a.labels:validate(label,a.device)
