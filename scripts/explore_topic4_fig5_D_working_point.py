"""Bounded I->E working-point continuation, including numerical QA."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time
import numpy as np
from scipy.linalg import eig,solve
from topic4_fig5_D_physical_model import Equilibrium,Characteristic,OUT,ROOT,table,transfer
from topic4_fig5_z_characteristic_root_newton import root
from topic4_fig5_z_branch_dynamics import transfer as old_transfer
import siegert_table

PRIOR=ROOT/'results/topic4_sef_hfo/fig5_z_bifurcation_preview_20260915/frozen_filtered_v1'
def write(name,x):(OUT/name).write_text(json.dumps(x,indent=2)+'\n')

def qa():
    c=Characteristic();eq=c.eq;m=c.m;n=m.n
    seed=np.load(PRIOR/'core_b_crossing_state.npz');r=seed['r_hz'];D=float(seed['s'])
    eq.evaluate(r,D);p=eq.last;xs,gs=table();xo,go=siegert_table.table()
    args=(p['mu'],np.repeat(p['ex'],m.K),p['inh'],m.theta_u,m.te,m.tref_e,m.v_reset,m.ra+m.ta,m.w2cv_e,m.gh_x,m.gh_w)
    new=transfer(*args,xs,gs);old=old_transfer(*args,xo,go)
    f,j=eq.evaluate(r,D,True);rng=np.random.default_rng(514);v=rng.normal(size=800);v/=np.linalg.norm(v)
    deriv=[]
    for h in (.01,.001,.0001):deriv.append(dict(h=h,max_jv_error=float(max(abs((eq.evaluate(r+h*v,D)-eq.evaluate(r-h*v,D))/(2*h)-j@v)))))
    fields=[eq.full_z(d) for d in np.linspace(0,1,101)]
    result=dict(smooth_vs_old_transfer_max_hz=float(max(abs(new-old))*1000),jacobian_directional_checks=deriv,
      physical_path_min=float(np.min(fields)),physical_path_max=float(np.max(fields)),D_identity_max_error=float(max(abs(1-z.mean()-d) for z,d in zip(fields,np.linspace(0,1,101)))),
      Z='frozen spatial parameter',M='dynamic; steady M=tau_M*r only for equilibrium equations',q_ie_mean_scaling='q',q_ie_variance_scaling='q^2')
    # Independent 64-node quadrature, including the exact same outer GH mixture.
    ref=np.zeros_like(new);sigma=np.sqrt(args[1]);sigg=np.sqrt(args[2]*m.w2cv_e)
    shift=args[0]-1.0325*np.sqrt(args[1]*(m.ra+m.ta)/m.te)
    for x,w in zip(m.gh_x,m.gh_w):
        ref+=w*siegert_table.lif_rate_legendre(shift+np.sqrt(2)*sigg*x,sigma,m.theta_u,m.te,m.tref_e,m.v_reset,order=64)
    result['smooth_vs_GL64_max_hz']=float(max(abs(new-ref))*1000)
    write('model_qa.json',result);print(result,flush=True)

def hopf(q,core,r,D,lam,vector=None):
    c=Characteristic(q);history=[]
    for it in range(14):
        r,err,ok=c.eq.solve(r,D)
        if not ok:raise RuntimeError(('equilibrium',q,core,D,err))
        c.at(r,D);lam,vector,er,_=root(c,lam,vector)
        print('HOPF',q,core,it,D,lam,err,er,flush=True)
        history.append(dict(D=D,root=[lam.real,lam.imag],equilibrium_error_hz=err,root_error=er))
        if er>1e-6:raise RuntimeError(('root',er))
        if abs(lam.real)<2e-7:break
        h=1e-5;rp,_,ok=c.eq.solve(r,D+h);assert ok
        c.at(rp,D+h);lp,_,_,_=root(c,lam,vector)
        slope=(lp.real-lam.real)/h
        D-=float(np.clip(lam.real/slope,-.025,.025))
    c.at(r,D);h=1e-5
    slopes=[]
    for sign in (-1,1):
        rp,er,ok=c.eq.solve(r,D+sign*h);assert ok
        c.at(rp,D+sign*h);ll,_,er,_=root(c,lam,vector);slopes.append(ll.real)
    c.at(r,D);z=c.eq.full_z(D);v=vector[:c.m.n];m=c.m
    energy=np.abs(v)**2*m.count_e;shares={}
    for key in ('175_0','175_1','175_2'):shares[key]=float(np.sum(abs(v)**2*m.region_w[key])/energy.sum())
    info=dict(q_ie=q,core=core,D=D,mean_e_hz=float(np.average(r[:m.n],weights=m.count_e)),
      frequency_hz=lam.imag/(2*np.pi),lambda_per_s=[lam.real,lam.imag],transversality_per_s_per_D=(slopes[1]-slopes[0])/(2*h),
      equilibrium_residual_hz=float(max(abs(c.eq.evaluate(r,D)))),root_residual=er,spatial_Z_min=float(z.min()),spatial_Z_max=float(z.max()),
      physical=bool(D>=0 and D<=1 and z.min()>=-1e-10 and z.max()<=1+1e-10),mode_E_energy=shares,iterations=history,
      type='OSCILLATORY_UNIT_CIRCLE_CROSSING; nonlinear nondegeneracy pending')
    name=f'q{q:g}_H{core}'
    np.savez_compressed(OUT/f'{name}.npz',r_hz=r,D=D,omega_per_s=lam.imag,mode=vector)
    write(name+'.json',info)
    return r,D,lam,vector,info

def screen():
    allrows=[]
    for core,filename in [('b','core_b_crossing_state.npz'),('a','oscillatory_crossing_state.npz')]:
        seed=np.load(PRIOR/filename);r=seed['r_hz'];D=float(seed['s']);lam=1j*float(seed['omega_per_s']);vector=None
        for q in (1.,1.1,1.25,1.5,1.75,2.):
            r,D,lam,vector,row=hopf(q,core,r,D,lam,vector);allrows.append(row);write('working_point_screen.json',allrows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['qa','screen']);a=p.parse_args()
    qa() if a.mode=='qa' else screen()
