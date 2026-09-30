"""Equilibria and local invariant-cycle branches at the selected physical working point."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json
import numpy as np
from scipy.signal import resample
from topic4_fig5_D_physical_model import Equilibrium,Characteristic,Orbit,OUT,ROOT
from topic4_fig5_z_frozen_v1 import rate_filters
from topic4_fig5_z_cycle_from_crossing import Branch
from topic4_fig5_z_frequency_parallel import install
import topic4_fig5_z_bifurcation_preview as continuation

def equilibrium(q):
    eq=Equilibrium(q);continuation.OUT=OUT
    a=np.load(OUT/f'q{q:g}_Hb.npz');r=a['r_hz'];D=float(a['D'])
    continuation.trace_branch(eq,r,D,-1,f'q{q:g}_low',max_points=70,lower_bound=0,upper_bound=1,step_max=.012,trust_curvature=True)
    continuation.trace_branch(eq,r,D,1,f'q{q:g}_middle',max_points=180,lower_bound=0,upper_bound=1,step_max=.025,trust_curvature=True)
    r,er,ok=eq.solve(np.r_[np.full(eq.n,450.),np.full(eq.n,600.)],1.)
    print('HIGH SEED',er,ok,flush=True)
    if ok:continuation.trace_branch(eq,r,1.,-1,f'q{q:g}_high',max_points=160,lower_bound=0,upper_bound=1.001,step_max=.025,trust_curvature=True)

def cycles(q,core,N,amplitudes):
    install(4);a=np.load(OUT/f'q{q:g}_H{core}.npz');D=float(a['D']);om=float(a['omega_per_s']);r=a['r_hz'];v=a['mode']
    c=Characteristic(q);c.at(r,D);m=c.m;n=m.n;K=m.K;W=c.weights(1j*om/1000);le,li,lm,*_=rate_filters(m,1j*om/1000)
    ve,vi=v[:n],v[n:];mu=m.te*(np.repeat(W['ee']@ve,K)-c.z*np.repeat(W['ei']@vi,K))
    ex=m.te*np.repeat(W['vee']@ve,K);inh=m.te*c.z2*np.repeat(W['vei']@vi,K)
    ru=(c.u*mu+c.v*ex+c.w*inh)/(1/le+c.u*m.eta_M*lm)
    vf=np.r_[ru,vi];idx=int(np.argmax(abs(ru)));vf*=np.exp(-1j*np.angle(vf[idx]));vf/=abs(vf[idx])
    theta=2*np.pi*np.arange(N)/N;wave=(np.exp(1j*theta)[:,None]*vf).real;phase=(1j*np.exp(1j*theta)[:,None]*vf).real
    c.eq.evaluate(r,D);req=np.r_[c.eq.last['r_u'],r[n:]/1000]
    b=Branch.__new__(Branch);b.req=req;b.wave=wave;b.phase=phase;b.T0=2*np.pi/om*1000;b.s0=D;b.N=N;b.core=core
    b.o=Orbit(D,N,q);b.pp=phase/np.sum(phase**2);b.ap=wave/np.sum(wave**2);b.use_preconditioner=True
    b.destination=OUT/f'q{q:g}_cycles_{core}_N{N}';b.destination.mkdir(exist_ok=True)
    last=0.;T=b.T0;rr=np.broadcast_to(req,(N,len(req))).copy();rows=[]
    for amp in amplitudes:
        coarse=OUT/f'q{q:g}_cycles_{core}_N32/amp{amp:g}_N32.npz'
        if N>32 and coarse.exists():
            co=np.load(coarse);rr=resample(co['r'],N,axis=0);T=float(co['T']);D=float(co['s'])
        else:rr=rr+(amp-last)/1000*wave
        rr,T,D,err=b.solve(rr,T,D,amp,maxiter=10)
        dense=resample(rr,4*N,axis=0);macro=(dense[:,:n*K]*m.w_u).reshape(4*N,n,K).sum(2)*1000
        glob=macro@m.count_e/m.count_e.sum()
        row=dict(q_ie=q,core=core,N=N,control_amplitude_hz=amp,D=D,period_ms=T,residual_hz=err,
          global_mean_hz=float(glob.mean()),global_min_hz=float(glob.min()),global_max_hz=float(glob.max()),
          unit_min_hz=float(dense.min()*1000),branch_D_shift_per_amplitude_squared=(D-b.s0)/amp**2,
          filename=str(b.destination/f'amp{amp:g}_N{N}.npz'),stability='PENDING_NORMAL_FORM_OR_FLOQUET')
        rows.append(row);(b.destination/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
        print('ACCEPTED CYCLE',row,flush=True)
        if err>1e-5 or row['unit_min_hz']<-.0001:break
        last=amp

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['equilibria','extend_middle','cycles']);p.add_argument('--q',type=float,default=1.25);p.add_argument('--core',default='a');p.add_argument('--N',type=int,default=32);p.add_argument('--amplitudes',default='.05,.1,.2,.4,.8,1.6,3.2');a=p.parse_args()
    if a.mode=='equilibria':equilibrium(a.q)
    elif a.mode=='cycles':cycles(a.q,a.core,a.N,list(map(float,a.amplitudes.split(','))))
    else:
        eq=Equilibrium(a.q);continuation.OUT=OUT;seed=np.load(OUT/f'q{a.q:g}_middle.npz')
        continuation.trace_branch(eq,seed['r_hz'][-1],float(seed['s'][-1]),1,f'q{a.q:g}_middle_extension',max_points=240,lower_bound=0,upper_bound=1,step_max=.035,initial_tangent=seed['tangent'][-1],trust_curvature=True)
