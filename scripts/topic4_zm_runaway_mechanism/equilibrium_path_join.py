"""Cross a known nonsmooth Z-path interpolation join without inventing a fold.

Solve the same equilibrium at the knot and at diminishing one-sided offsets;
check physical solutions, derivative convergence and return to the knot.
"""
from native_high_equilibrium_branch import parameter_column
from native_path import *
import equilibria_v3 as arc
from scipy.sparse.linalg import eigs,spsolve
import argparse


def main(a):
    s=model();attach_native_path(s);arc.RS=.1;arc.DS=.1;arc.param_derivative=parameter_column
    z=np.load(a.source);D0=float(z['D']);knot=float(s.path_D_knots[np.argmin(abs(s.path_D_knots-D0))])
    assert 0<knot<1 and abs(knot-D0)<1e-8,(D0,knot)
    out=OUT/'equilibria'/a.label;out.mkdir(parents=True,exist_ok=True)
    write(out/'contract.json',dict(source=a.source,knot_D=knot,direction=a.direction,
        offset_steps=[1e-5,5e-6,2.5e-6],equilibrium_tolerance_hz=2e-8,
        question='Is the interrupted branch continuous through the known nonsmooth interpolation join?',
        scope='Same equations and spatial path, held Z and dynamic M; no smoothing or bifurcation inference at a path corner'))
    s.set_D(D0);assert np.max(abs(s.Z-z['Z']))<1e-12
    s.set_D(knot);r0,ok,tr=s.solve(z['r'],tol=1e-12);assert ok,tr[-1]
    J=s.jacobian(r0);ev,V=eigs(J,k=4,sigma=0,tol=1e-11)
    eig_errors=[float(np.linalg.norm(J@V[:,j]-ev[j]*V[:,j])/np.linalg.norm(V[:,j])) for j in range(4)]
    assert max(eig_errors)<1e-8 and min(abs(ev))>1e-5,(ev,eig_errors)
    np.savez_compressed(out/'knot.npz',r=r0,D=knot,Z=s.Z)
    gradients={}
    tangents={}
    for sign in [-1,1]:
        D=knot+sign*1e-9;s.set_D(D);column=parameter_column(s,r0)
        gradients[sign]=-spsolve(J,column)
        tangent=np.r_[gradients[sign]*arc.DS/arc.RS,1.]
        tangents[sign]=tangent/np.linalg.norm(tangent)
    rows=[];accepted={}
    for h in [1e-5,5e-6,2.5e-6]:
        for sign in [-1,1]:
            D=knot+sign*h;s.set_D(D)
            seed=r0+sign*h*gradients[sign];r,ok,tr=s.solve(seed,tol=1e-12);assert ok,tr[-1]
            rD=(r-r0)/(sign*h)
            relative=float(np.linalg.norm(rD-gradients[sign])/np.linalg.norm(gradients[sign]))
            tangent=arc.tangent(s,r,D,direction=a.direction)
            np.savez_compressed(out/f'side{sign:+d}_h{h:g}.npz',r=r,D=D,Z=s.Z,tangent=tangent)
            s.set_D(knot);back,ok,trace=s.solve(r,tol=1e-12);assert ok,trace[-1]
            return_error=float(abs(back-r0).max()*1000);assert return_error<1e-7,return_error
            q=dict(offset=h,side=sign,D=D,global_E_hz=s.global_rate(r),
                residual_hz=tr[-1]*1000,one_sided_derivative_relative_error=relative,
                return_to_knot_max_rate_difference_hz=return_error,
                min_rate_hz=float(r.min()*1000),max_rate_hz=float(r.max()*1000))
            assert r.min()>=0 and np.all(r<1/s.ref)
            rows.append(q);accepted[(h,sign)]=(r,D,tangent);log('PATH JOIN',q)
    for sign in [-1,1]:
        err=[q['one_sided_derivative_relative_error'] for q in rows if q['side']==sign]
        assert err[-1]<.01 and err[-1]<.7*err[0],err
    r,D,t=accepted[(2.5e-6,a.direction)];s.set_D(D)
    np.savez_compressed(out/'continue.npz',r=r,D=D,Z=s.Z,tangent=t)
    q=dict(status='JOIN_CONTINUITY_AND_ONE_SIDED_CHECK_PASS',source=a.source,knot_D=knot,
        global_E_hz_at_knot=s.global_rate(r0),knot_static_eigenvalues=[[v.real,v.imag] for v in ev],
        eigen_residuals=eig_errors,one_sided_tangent_cosine=float(tangents[-1]@tangents[1]),
        rows=rows,continuation_seed=str(out/'continue.npz'),
        interpretation='Both parameter segments meet at the same physical equilibrium. Numerical angle-based continuation failure at this known path corner is not an equilibrium saddle-node; temporal stability remains separately determined.')
    write(out/'result.json',q);log('JOIN RESULT',q)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--label',required=True)
    p.add_argument('--direction',type=int,choices=[-1,1],required=True);main(p.parse_args())
