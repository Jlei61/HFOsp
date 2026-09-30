"""Track one exact-map instability along the newly found local equilibrium arc."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
import argparse
import numpy as np
from analyze_topic4_fig5_D_target_roots import OUT


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--fold',action='store_true')
    parser.add_argument('--fold-review',action='store_true'); args=parser.parse_args()
    if args.fold:
        import topic4_fig5_z_bifurcation_preview as b
        eq=b.Equilibrium(); b.Equilibrium=lambda:eq; b.OUT=OUT
        b.fold_check(branch='target_equilibria_+1',prefix='target_stationary_fold',turn=0)
        return
    from topic4_fig5_z_frozen_v1 import Characteristic
    from topic4_fig5_z_characteristic_root_newton import root
    if args.fold_review:
        c=Characteristic(); a=np.load(OUT/'target_stationary_fold_state.npz')
        r,D=a['r_hz'],float(a['s']); c.at(r,D)
        singular=np.linalg.svd(c.matrix(0),compute_uv=False)
        seed=np.load(OUT/'target_root_unstable_mode.npz')
        lam,mode,err,history=root(c,complex(seed['growth_per_s']),seed['mode'],maxiter=8)
        f=c.eq.evaluate(r,D); checks=[]
        for h in (.4,.2,.1,.05,.025,.0125,.00625):
            v=a['right_mode']
            value=.5*np.dot(a['left_mode'],(c.eq.evaluate(r+h*v,D)-2*f+c.eq.evaluate(r-h*v,D))/(h*h))
            checks.append(dict(h_hz=h,coefficient=float(value)))
        result=dict(D=D,zero_characteristic_smallest_singular_value=float(singular[-1]),
                    second_smallest_singular_value=float(singular[-2]),oscillatory_root_per_s=[lam.real,lam.imag],
                    root_residual=err,oscillatory_instability_persists=bool(err<1e-7 and lam.real>0),
                    quadratic_step_checks=checks,classification='STATIONARY_TURN_OF_UNSTABLE_EQUILIBRIA',
                    smooth_saddle_node_non_degeneracy='NOT_FULLY_CERTIFIED',native_state3_bifurcation='NOT_ESTABLISHED')
        (OUT/'target_stationary_fold_review.json').write_text(json.dumps(result,indent=2)+'\n')
        return
    c=Characteristic(); states=np.load(OUT/'target_equilibria_+1.npz')
    seed=np.load(OUT/'target_root_unstable_mode.npz'); lam=complex(seed['growth_per_s']); mode=seed['mode']
    rows=[]
    for index,(r,D) in enumerate(zip(states['r_hz'],states['s'])):
        c.at(r,D); lam,mode,err,history=root(c,lam,mode,maxiter=8)
        row=dict(branch='target_equilibria_+1',index=index,D=float(D),lambda_per_s=[lam.real,lam.imag],
                 residual=err,status='UNSTABLE_COMPLEX_MULTIPLIER_CERTIFIED' if err<1e-7 and lam.real>1e-5 else 'UNCLASSIFIED')
        rows.append(row);print(row,flush=True)
        (OUT/'target_branch_complex_certificates.json').write_text(json.dumps(rows,indent=2)+'\n')
        if err>1e-7:break
    # Recheck the seed with halved input perturbations for transfer derivatives.
    c.at(seed['rate_hz'],float(seed['D'])); eq=c.eq; m=c.m; r=seed['rate_hz']; n=m.n
    mu,ex,inh=eq.last['mu'],eq.last['ex'],eq.last['inh']; h=1e-4
    c.u=(m.phi_e(mu+h,ex,inh)-m.phi_e(mu-h,ex,inh))/(2*h)
    c.v=(m.phi_e(mu,ex+h,inh)-m.phi_e(mu,ex-h,inh))/(2*h)
    c.w=(m.phi_e(mu,ex,inh+h)-m.phi_e(mu,ex,inh-h))/(2*h)
    re,ri=r[:n]/1000,r[n:]/1000
    mui=m.ti*(m.gaA*(m.w_ie@re+m.ji*m.nu_sig)-m.gaG*(m.w_ii@ri))
    ei=m.ti*(m.v_ie@re+m.ji*m.ji*m.nu_sig); ii=m.ti*(m.v_ii@ri)
    c.ui=(m.phi_i(mui+h,ei,ii)-m.phi_i(mui-h,ei,ii))/(2*h)
    c.vi=(m.phi_i(mui,ei+h,ii)-m.phi_i(mui,ei-h,ii))/(2*h)
    c.wi=(m.phi_i(mui,ei,ii+h)-m.phi_i(mui,ei,ii-h))/(2*h)
    lam,mode,err,history=root(c,complex(seed['growth_per_s']),seed['mode'],maxiter=8)
    (OUT/'target_root_gain_sensitivity.json').write_text(json.dumps(dict(derivative_step=1e-4,
        lambda_per_s=[lam.real,lam.imag],residual=err,unstable=bool(err<1e-7 and lam.real>0)),indent=2)+'\n')


if __name__=='__main__':main()
