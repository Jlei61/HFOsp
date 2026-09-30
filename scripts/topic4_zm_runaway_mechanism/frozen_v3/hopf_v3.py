"""Hopf points on equilibrium branches (frozen Z, prescribed path) and their numerical criticality.

1. Locate: along a branch, track the leading complex root with refine_root; bisect in D for Re(lambda)=0
   (equilibrium re-solved at each D by Newton from the nearest branch point).
2. Criticality without a normal form: seed small periodic orbits from the critical eigenmode with
   amplitude constraints (first-harmonic projection = eps), solve the phase-fixed BVP with D free, for
   several eps; the sign of dD/d(eps^2) relative to the crossing direction gives super/subcritical; the
   small cycles are then passed to floquet_v3 for their own transverse stability.
"""
from periodic_v3 import *
from root_count_v3 import refine_root
import argparse
def equilibrium_at(s,D,r0):
    s.set_D(D);r,ok,tr=s.solve(r0);assert ok,('newton',D,tr[-1]);return r
def leading_root(s,r,lam0,dynamic_z=False):
    f=refine_root(s,r,lam0,dynamic_z=dynamic_z)
    if f is None:return None
    lam,v,err=f
    if lam.imag<0:lam,v=lam.conjugate(),v.conjugate()
    return lam,v,err
def locate(s,branch_points,lam0,dynamic_z=False,tol=1e-9):
    """branch_points: [(D,r)] two points bracketing the crossing. Returns D*, r*, lam*, v*."""
    (Da,ra),(Db,rb)=branch_points;fa=leading_root(s,ra,lam0,dynamic_z);fb=leading_root(s,rb,fa[0],dynamic_z);assert fa[0].real*fb[0].real<0,(fa[0],fb[0])
    la,lb=fa[0],fb[0];r=ra.copy()
    for it in range(60):
        Dm=Da+(Db-Da)*(-la.real)/(lb.real-la.real) if it%3 else .5*(Da+Db)   # regula falsi with bisection safeguard
        r=equilibrium_at(s,Dm,r);fm=leading_root(s,r,.5*(la+lb),dynamic_z);lm=fm[0]
        if abs(lm.real)<tol or abs(Db-Da)<1e-12:break
        if lm.real*la.real>0:Da,la=Dm,lm
        else:Db,lb=Dm,lm
    return Dm,r,fm
def small_cycles(s,o,D,r,lam,v,amplitudes,device=0):
    """Seed r(t)=r*+eps*Re(v e^{i w t}) (rate units) and solve the BVP with amplitude constraint and D free."""
    N=o.N;w=lam.imag;T=2*np.pi/w;t=np.arange(N)*T/N;q=v/abs(v).max();out=[]
    for eps in amplitudes:
        seed=r[None,:]+eps*1e-3*np.real(q[None,:]*np.exp(1j*w*t)[:,None]);seed=np.maximum(seed,0)
        proj=np.conj(q)   # projection vector for the first harmonic
        sol=o.solve(seed,T,D,amplitude=(proj,eps*1e-3*np.vdot(q,q).real/np.vdot(q,q).real),maxiter=24,tol=1e-9)
        g=sol['r'][:,s.E]@s.mean_weights*1000;out.append(dict(eps=eps,D=sol['D'],T=sol['T'],residual=sol['residual'],global_mean_hz=float(g.mean()),global_min_hz=float(g.min()),global_max_hz=float(g.max()),r=sol['r']))
        log('small cycle eps',eps,'D %.9f T %.4f res %.2e max %.2f Hz'%(sol['D'],sol['T'],sol['residual'],g.max()))
    return out
def main(a):
    s=load_model(a.device);out=PERIODIC_OUT/'hopf';out.mkdir(parents=True,exist_ok=True)
    za=np.load(a.point_a);zb=np.load(a.point_b);D,r,(lam,v,err)=locate(s,[(float(za['D']),za['r']),(float(zb['D']),zb['r'])],complex(a.lam_re,a.lam_im),a.dynamic_z)
    energy=s.sizes*abs(v)**2*s.E;energy/=energy.sum()
    # crossing slope
    h=1e-5;rp=equilibrium_at(s,D+h,r);lp=leading_root(s,rp,lam,a.dynamic_z)[0];rm=equilibrium_at(s,D-h,r);lm=leading_root(s,rm,lam,a.dynamic_z)[0];slope=(lp.real-lm.real)/(2*h)
    row=dict(label=a.label,D=D,lambda_per_ms=[lam.real,lam.imag],frequency_hz=lam.imag*1000/(2*np.pi),residual=err,global_E_hz=s.global_rate(r),regional_hz=s.regional_rates(r),
             dRe_dD=slope,E_energy_A_B_S=[float(energy[s.geo['group_region']==k].sum()) for k in range(3)],dynamic_z=a.dynamic_z)
    np.savez_compressed(out/f'{a.label}.npz',r=r,D=D,lam=lam,v=v,Z=s.Z);log('HOPF',row)
    o=PeriodicV3(s,a.N,a.device);cyc=small_cycles(s,o,D,r,lam,v,a.amplitudes,a.device)
    for c in cyc:np.savez_compressed(out/f'{a.label}_cycle_eps{c["eps"]:g}.npz',r=c.pop('r'),T=c['T'],D=c['D'],residual=c['residual'])
    eps2=np.array([c['eps']**2 for c in cyc]);Ds=np.array([c['D'] for c in cyc]);coef=np.polyfit(eps2,Ds,1)[0] if len(cyc)>1 else None
    # supercritical if the cycle branch emanates toward the side where the equilibrium is unstable (Re lambda>0)
    crit=None
    if coef is not None:
        unstable_side=np.sign(slope);crit='supercritical' if np.sign(coef)==unstable_side else 'subcritical'
    row.update(small_cycles=cyc,dD_deps2=coef,criticality_numerical=crit,note='criticality from the direction of the small-cycle branch relative to the unstable side; transverse stability by floquet_v3 on the saved cycles')
    write(out/f'{a.label}.json',row);log('CRITICALITY',crit,coef)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('point_a');p.add_argument('point_b');p.add_argument('--lam-re',type=float,default=.01);p.add_argument('--lam-im',type=float,default=.15)
    p.add_argument('--label',default='hopf');p.add_argument('--N',type=int,default=128);p.add_argument('--amplitudes',type=float,nargs='+',default=[.5,1.,2.,4.]);p.add_argument('--dynamic-z',action='store_true');p.add_argument('--device',type=int,default=0);main(p.parse_args())
