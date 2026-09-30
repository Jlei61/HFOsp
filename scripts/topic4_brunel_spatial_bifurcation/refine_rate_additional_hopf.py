"""Refine a new Hopf on a folded stationary branch using core-B rate as coordinate.

The coordinate is a continuation constraint, not a model reduction. All 935
equilibrium equations remain in the bordered solve. This avoids switching
equilibria when J reverses near the stationary fold.
"""
from rate_stationary_contour_modes import *
from scipy.optimize import brentq
from scipy.sparse.linalg import spsolve


def equilibrium_at_rate(s,r,J,target,w):
    for it in range(18):
        F=np.r_[s.residual(r,J)*1000,w@r*1000-target];error=max(abs(F))
        if error<2e-11:return r,J,error
        A=sparse.bmat([[s.jacobian(r,J),sparse.csr_matrix(s.parameter_derivative(r,J)[:,None])],
            [sparse.csr_matrix(w[None,:]),sparse.csr_matrix((1,1))]],format='csc')
        d=spsolve(A,-F)
        for back in range(16):
            step=2.**-back;rr=r+d[:-1]*step/1000;JJ=J+d[-1]*step/1000
            # Some strongly inhibited groups underflow to zero. Sparse solves
            # can then add ~1e-27 /ms roundoff; retain the nonnegative domain
            # without rejecting an otherwise converged Newton step.
            if rr.min() < -1e-18 or JJ<=0:continue
            rr=np.maximum(rr,0.)
            FF=np.r_[s.residual(rr,JJ)*1000,w@rr*1000-target]
            if np.linalg.norm(FF)<np.linalg.norm(F):r,J=rr,JJ;break
        else:raise RuntimeError('Bordered stationary branch correction failed')
    raise RuntimeError('Bordered stationary branch correction did not converge')


def main(a):
    s=RateField()
    def gains(mom):
        out=[]
        for k in range(3):
            h=a.gain_step*np.maximum(abs(mom[k]),1.);hi=list(mom);lo=list(mom)
            hi[k]=mom[k]+h;lo[k]=mom[k]-h;d=s.phi(*hi)-s.phi(*lo)
            if a.gain_order==4:
                hi[k]=mom[k]+2*h;lo[k]=mom[k]-2*h;out.append((8*d-s.phi(*hi)+s.phi(*lo))/(12*h))
            else:out.append(d/(2*h))
        return out
    s.gains=gains
    coordinate=getattr(a,'coordinate','core_B');region=0 if coordinate=='core_A' else 1
    mask=s.E&(s.geo['group_region']==region);w=np.zeros(s.P);w[mask]=s.geo['group_size'][mask];w/=w.sum()
    cache=[]
    for i in a.indices:
        z=np.load(DEST/f'{getattr(a,"namespace","")}tracked_mode{a.mode}_branch{i:04d}.npz')
        cache.append(dict(coordinate=float(w@z['rates']*1000),r=z['rates'],J=float(z['J']),lam=complex(z['lam']),v=z['vector']))
    bracket=sorted(q['coordinate'] for q in cache);calls=[]
    def evaluate(c):
        old=min(cache,key=lambda q:abs(q['coordinate']-c));r,J,e=equilibrium_at_rate(s,old['r'].copy(),old['J'],c,w)
        q=refine(s,r,J,old['lam'],old['v'],tol=2e-12);assert q is not None
        lam,v,res=q;row=dict(coordinate=c,r=r,J=J,lam=lam,v=v);cache.append(row)
        calls.append(dict(core_B_rate_Hz=s.regional_rates(r)[1],coordinate_value=c,J_EE_core=J,lambda_per_ms=lam,equilibrium_residual_Hz=e,characteristic_residual=res))
        print('ADDITIONAL HOPF',a.label,c,J,lam,res,flush=True)
        return lam.real,row
    c=brentq(lambda c:evaluate(c)[0],*bracket,xtol=1e-10,rtol=1e-13)
    _,root=evaluate(c);h=1e-4
    left=evaluate(c-h)[1];right=evaluate(c+h)[1]
    slope=(right['lam'].real-left['lam'].real)/(2*h);dJ=(right['J']-left['J'])/(2*h)
    lam=root['lam'];v=root['v'];assert abs(lam.real)<1e-9 and abs(lam.imag)>1e-5 and abs(slope)>1e-6
    mass=s.geo['group_size']*abs(v)**2*s.E;mass/=mass.sum()
    row=dict(label=a.label,J_EE_core=root['J'],frequency_hz=lam.imag*1000/(2*np.pi),
        gain_derivative_order=a.gain_order,gain_derivative_step=a.gain_step,
        core_B_rate_Hz=s.regional_rates(root['r'])[1],coordinate_value=c,rates_hz=s.regional_rates(root['r']),lambda_per_ms=lam,
        equilibrium_residual=float(max(abs(s.residual(root['r'],root['J'])))),
        characteristic_residual=float(np.linalg.norm(s.characteristic(root['r'],root['J'],lam)@v)),
        E_rate_mode_energy_by_region=[mass[s.geo['group_region']==k].sum() for k in range(3)],
        transversality=dict(coordinate=f'Core-{"AB"[region]} neuron-weighted E equilibrium rate, Hz',
            real_exponent_derivative_per_ms_per_Hz=slope,J_derivative_per_Hz=dJ,
            real_exponent_derivative_per_ms_per_J=slope/dJ,coordinate_step_Hz=h,
            real_exponents=[left['lam'].real,right['lam'].real]),
        source_mode=a.mode,source_branch_indices=a.indices,solver_evaluations=calls,
        source_namespace=getattr(a,'namespace',''),
        status='REFINED_HOPF_CANDIDATE',criticality='NOT_COMPUTED',
        scope='A new imaginary-axis crossing on an already unstable equilibrium branch; not the first resting instability. Derivative and independent full-state checks pending.')
    write(DEST/(a.label+'.json'),row)
    np.savez_compressed(DEST/(a.label+'.npz'),rates=root['r'],J=root['J'],omega=lam.imag,vector=v)
    print('HOPF CANDIDATE REFINED',a.label,root['J'],row['frequency_hz'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',type=int,required=True);p.add_argument('--indices',type=int,nargs=2,required=True)
    p.add_argument('--label',required=True);p.add_argument('--gain-order',type=int,choices=[2,4],default=4)
    p.add_argument('--gain-step',type=float,default=1e-4);p.add_argument('--namespace',default='')
    p.add_argument('--coordinate',choices=['core_A','core_B'],default='core_B');main(p.parse_args())
