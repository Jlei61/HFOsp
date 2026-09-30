#!/usr/bin/env python3
"""Exploratory equilibrium continuation and native peak/valley display for Fig.5.

Uses the frozen v1 equations unchanged at equilibrium. This closure failed native
burst correspondence; the resulting branches are NOT native-SNN bifurcations.
No native runs, prior outputs, or accepted protocols are changed.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
from scipy import linalg, optimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/topic4_fig5_z_state'))
from approx_system import ReducedModel

SOURCE = ROOT / 'results/topic4_sef_hfo/fig5_current_network_z_state_v1'
OUT = ROOT / 'results/topic4_sef_hfo/fig5_z_bifurcation_preview_20260915'


def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


class Equilibrium:
    def __init__(self):
        self.m = m = ReducedModel(dict(grid=20, variance='stationary'))
        # Filtered and stationary variance have exactly the same steady limit.
        self.n = m.n
        if (OUT / 'spatial_z_path.npz').exists():
            saved = np.load(OUT / 'spatial_z_path.npz')
            self.fields = saved['z_e']; self.times = list(saved['time_s'])
            self.prepare_path()
            return
        self.fields = [np.ones(32000)]
        self.times = [0.]
        run = SOURCE / 'replay/runs/eta0.0005_s9108401'
        targets = [1000, 2000, 3000, 4000, 5000, 6000, 7000]
        for target in targets:
            candidates = []
            for f in sorted((run / 'fields').glob('*.npz')):
                a = np.load(f)
                steps = a['zm_step']
                k = int(np.argmin(np.abs(steps * .1 - target)))
                if abs(steps[k] * .1 - target) <= 10.:
                    candidates.append((abs(steps[k] * .1-target), steps[k] * .1, a['z'][k]))
            best = min(candidates, key=lambda x: x[0])
            self.times.append(float(best[1]) / 1000.)
            self.fields.append(best[2])
        for t in (8000, 9000, 9300, 9420, 9870, 10370):
            a = np.load(run / 'checkpoints' / f't{t}ms.npz')
            self.times.append(t / 1000.)
            self.fields.append(a['slow__z'][:32000])
        self.fields = np.asarray(self.fields)
        self.prepare_path()
        np.savez_compressed(OUT / 'spatial_z_path.npz', time_s=self.times,
                            depletion=self.ss, z_e=self.fields)

    def prepare_path(self):
        m = self.m
        self.ss = 1 - self.fields.mean(1)
        assert np.all(np.diff(self.ss) > 0)
        self.zu = np.asarray([m.unit_mean(z) for z in self.fields])
        self.z2u = np.asarray([m.unit_mean(z*z) for z in self.fields])
        self.zcross = np.asarray([m.unit_mean(a*b) for a,b in zip(self.fields[:-1],self.fields[1:])])
        self.cache = None

    def z(self, s):
        k = int(np.clip(np.searchsorted(self.ss, s)-1, 0, len(self.ss)-2))
        a = (s-self.ss[k])/(self.ss[k+1]-self.ss[k])
        # Interpolate full per-neuron fields BEFORE computing the second moment.
        return ((1-a)*self.zu[k]+a*self.zu[k+1],
                (1-a)**2*self.z2u[k]+2*a*(1-a)*self.zcross[k]+a*a*self.z2u[k+1])

    def evaluate(self, rhz, s, jac=False):
        m = self.m; n = self.n
        re, ri = np.asarray(rhz[:n])/1000, np.asarray(rhz[n:])/1000
        z,z2 = self.z(s)
        cae = m.te*m.gaA*(m.w_ee@re + m.je*m.nu_sig)
        cge = m.te*m.gaG*(m.w_ei@ri)
        cai = m.ti*m.gaA*(m.w_ie@re + m.ji*m.nu_sig)
        cgi = m.ti*m.gaG*(m.w_ii@ri)
        mu0 = np.repeat(cae,m.K) - z*np.repeat(cge,m.K)
        exe = m.te*(m.v_ee@re + m.je*m.je*m.nu_sig)
        inhe = z2*np.repeat(m.te*(m.v_ei@ri),m.K)
        exi = m.ti*(m.v_ie@re + m.ji*m.ji*m.nu_sig)
        inhi = m.ti*(m.v_ii@ri)
        # Eliminate each unit's steady M exactly: M_u=tau_M*r_u.
        r = m.phi_e(mu0,exe,inhe)
        delta = 2e-4
        for _ in range(12):
            mu = mu0-m.eta_M*m.tau_M*r
            phi = m.phi_e(mu,exe,inhe)
            dmu = (m.phi_e(mu+delta,exe,inhe)-m.phi_e(mu-delta,exe,inhe))/(2*delta)
            step = (r-phi)/(1+m.eta_M*m.tau_M*dmu)
            r -= step
            if np.max(np.abs(step)) < 1e-12:
                break
        mu = mu0-m.eta_M*m.tau_M*r
        raw = m.phi_e(mu,exe,inhe)
        local_error = float(np.max(np.abs(r-raw))*1000)
        pe = (r*m.w_u).reshape(n,m.K).sum(1)
        pi = m.phi_i(cai-cgi,exi,inhi)
        f = np.r_[pe,pi]*1000-rhz
        self.last = dict(r_u=r, mu=mu, ex=exe, inh=inhe, local_error_hz=local_error)
        if not jac:
            return f
        weights = m.w_u.reshape(n,m.K)
        dmu = (m.phi_e(mu+delta,exe,inhe)-m.phi_e(mu-delta,exe,inhe))/(2*delta)
        dex = (m.phi_e(mu,exe+delta,inhe)-m.phi_e(mu,exe-delta,inhe))/(2*delta)
        dinh = (m.phi_e(mu,exe,inhe+delta)-m.phi_e(mu,exe,inhe-delta))/(2*delta)
        den = 1+m.eta_M*m.tau_M*dmu
        def avg(a):
            return (weights*a.reshape(n,m.K)).sum(1)
        am,ae,ag = avg(dmu/den),avg(dex/den),avg(dinh*z2/den)
        az = avg(dmu*z/den)
        jee = am[:,None]*m.te*m.gaA*m.w_ee + ae[:,None]*m.te*m.v_ee
        jei = -az[:,None]*m.te*m.gaG*m.w_ei + ag[:,None]*m.te*m.v_ei
        mui = cai-cgi
        im = (m.phi_i(mui+delta,exi,inhi)-m.phi_i(mui-delta,exi,inhi))/(2*delta)
        ie = (m.phi_i(mui,exi+delta,inhi)-m.phi_i(mui,exi-delta,inhi))/(2*delta)
        ig = (m.phi_i(mui,exi,inhi+delta)-m.phi_i(mui,exi,inhi-delta))/(2*delta)
        jie = im[:,None]*m.ti*m.gaA*m.w_ie+ie[:,None]*m.ti*m.v_ie
        jii = -im[:,None]*m.ti*m.gaG*m.w_ii+ig[:,None]*m.ti*m.v_ii
        return f, np.block([[jee,jei],[jie,jii]])-np.eye(2*n)

    def solve(self, x, s):
        def fj(r):
            return self.evaluate(r,s,True)
        result = optimize.root(fj,x,jac=True,method='hybr',options={'xtol':1e-9,'maxfev':120})
        err = float(np.max(np.abs(self.evaluate(result.x,s))))
        valid = err<2e-6 and result.x.min()>-1e-7
        return result.x, err, valid

    def newton(self, r, s, iterations=20):
        r=np.array(r,copy=True)
        for it in range(iterations):
            f,j=self.evaluate(r,s,True)
            err=float(np.max(np.abs(f)))
            if err < 2e-7:
                return r,err,True
            try: step=linalg.solve(j,-f,check_finite=False)
            except (ValueError,linalg.LinAlgError): break
            old=np.linalg.norm(f)
            for a in (1.,.5,.25,.125,.0625,.03125,.015625):
                rr=r+a*step
                if rr.min() < -1e-6: continue
                ff=self.evaluate(rr,s)
                if np.all(np.isfinite(ff)) and np.linalg.norm(ff)<old:
                    r=rr;break
            else: break
        return r,float(np.max(np.abs(self.evaluate(r,s)))),False


SCALE=1000.


def trace_branch(eq,r,s,direction,name,max_points=150,lower_bound=0.,step_max=.06,initial_tangent=None,upper_bound=None,trust_curvature=False):
    """Pseudo-arclength continuation of the *full* 800-variable residual."""
    start=time.time(); n=len(r); y=np.r_[r/SCALE,s]
    tangent=np.r_[np.zeros(n),float(direction)] if initial_tangent is None else np.array(initial_tangent,copy=True)
    tangent/=np.linalg.norm(tangent);ds=.003
    points=[]; failures=[]
    for idx in range(max_points):
        r=y[:-1]*SCALE;s=float(y[-1]);f,j=eq.evaluate(r,s,True)
        eps=2e-6
        fp=(eq.evaluate(r,s+eps)-eq.evaluate(r,s-eps))/(2*eps)/SCALE
        B=np.empty((n+1,n+1));B[:-1,:-1]=j;B[:-1,-1]=fp;B[-1]=tangent
        try: tn=linalg.solve(B,np.r_[np.zeros(n),1.],check_finite=False)
        except (ValueError,linalg.LinAlgError): break
        tn/=np.linalg.norm(tn)
        if np.dot(tn,tangent)<0:tn=-tn
        tangent=tn
        points.append(dict(index=idx,s=s,mean_z=1-s,
                           mean_e_hz=float(np.average(r[:eq.n],weights=eq.m.count_e)),
                           core_a_hz=float(eq.m.region_rate(r[:eq.n],'175_0')),
                           core_b_hz=float(eq.m.region_rate(r[:eq.n],'175_1')),
                           residual_max_hz=float(np.max(np.abs(f))),tangent_s=float(tangent[-1])))
        # Persist every accepted solution before attempting another correction.
        if idx==0: all_r=[r.copy()]; all_t=[tangent.copy()]
        else: all_r.append(r.copy());all_t.append(tangent.copy())
        tmp=OUT/f'{name}.writing.npz'
        np.savez_compressed(tmp,r_hz=np.array(all_r),s=[p['s'] for p in points],tangent=np.array(all_t))
        os.replace(tmp,OUT/f'{name}.npz')
        write(f'{name}.json',dict(points=points,failures=failures,seconds=time.time()-start,
                                 stability='NOT_COMPUTED',model_correspondence='FAIL_IN_PRIOR_VALIDATION'))
        if idx%5==0:print(name,idx,'Z',1-s,'R',points[-1]['mean_e_hz'],'ds/dl',tangent[-1],flush=True)
        if s<lower_bound or s>(eq.ss[-1]+.008 if upper_bound is None else upper_bound):break
        success=False
        for retry in range(7):
            pred=y+ds*tangent;yn=pred.copy()
            for it in range(14):
                ff,jj=eq.evaluate(yn[:-1]*SCALE,yn[-1],True)
                constraint=np.dot(yn-pred,tangent)
                if np.max(np.abs(ff))<2e-6 and abs(constraint)<1e-9 and yn[:-1].min()>-1e-9:
                    success=True;break
                dpar=(eq.evaluate(yn[:-1]*SCALE,yn[-1]+eps)-eq.evaluate(yn[:-1]*SCALE,yn[-1]-eps))/(2*eps)/SCALE
                A=np.empty_like(B);A[:-1,:-1]=jj;A[:-1,-1]=dpar;A[-1]=tangent
                try: delta=linalg.solve(A,-np.r_[ff/SCALE,constraint],check_finite=False)
                except (ValueError,linalg.LinAlgError):break
                merit=np.linalg.norm(np.r_[ff/SCALE,constraint])
                for alpha in (1.,.5,.25,.125,.0625):
                    test=yn+alpha*delta
                    if test[:-1].min() < -1e-7:continue
                    tf=eq.evaluate(test[:-1]*SCALE,test[-1])
                    tm=np.linalg.norm(np.r_[tf/SCALE,np.dot(test-pred,tangent)])
                    if np.isfinite(tm) and tm<merit:
                        yn=test;break
                else:break
            if success and trust_curvature:
                # Reject a distant intersection with the predictor plane. Such
                # intersections can jump between nearby folded root branches.
                correction=float(np.linalg.norm(yn-pred))
                ff,jj=eq.evaluate(yn[:-1]*SCALE,yn[-1],True)
                dpar=(eq.evaluate(yn[:-1]*SCALE,yn[-1]+eps)-eq.evaluate(yn[:-1]*SCALE,yn[-1]-eps))/(2*eps)/SCALE
                A=np.empty_like(B)
                A[:-1,:-1]=jj;A[:-1,-1]=dpar;A[-1]=tangent
                newtan=linalg.solve(A,np.r_[np.zeros(n),1.],check_finite=False);newtan/=np.linalg.norm(newtan)
                alignment=float(np.dot(newtan,tangent))
                if correction>.5*ds or alignment<.9:success=False
            if success:break
            failures.append(dict(index=idx,step=ds,residual_hz=float(np.max(np.abs(ff)))))
            ds*=.5
            if ds<2e-5:break
        if not success:break
        y=yn
        ds=min(step_max,ds*(1.35 if it<5 else 1.05))
    return points


def continue_all():
    eq=Equilibrium()
    # Warm start from an already converged equilibrium, never from an invented branch.
    a=np.load(OUT/'pilot_s0.3090_init300.npz')
    trace_branch(eq,a['r_hz'],float(a['s']),-1,'recruited_equilibria',max_points=130)
    # An independent low-state search; a failed root is never plotted as a fold.
    trials=[]
    for ratee,ratei in ((.1,.2),(.03,.3),(.01,1.),(1.,2.)):
        rr,err,ok=eq.newton(np.r_[np.full(eq.n,ratee),np.full(eq.n,ratei)],0.)
        trials.append(dict(initial_e_hz=ratee,initial_i_hz=ratei,residual_hz=err,valid=ok))
        print('low search',trials[-1],flush=True)
        if ok:
            trace_branch(eq,rr,0.,1,'low_equilibria',max_points=100)
            break
    write('low_root_search.json',trials)


def core_branch():
    eq=Equilibrium()
    a=np.load(SOURCE/'approx/v1/runs/z8000_h8000_W1/fields.npz')
    initial=a['fields_hz'][-4000:].mean(0).reshape(-1).astype(float)
    s=float(eq.ss[np.argmin(abs(np.array(eq.times)-8.))])
    r,err,ok=eq.newton(initial,s,iterations=35)
    write('core_seed.json',dict(s=s,residual_hz=err,valid=ok))
    print('core seed',s,err,ok,flush=True)
    if ok:trace_branch(eq,r,s,-1,'core_equilibria',max_points=100)


def fold_check(branch='recruited_equilibria',prefix='fold',turn=0):
    eq=Equilibrium();m=eq.m;n=2*eq.n
    a=np.load(OUT/f'{branch}.npz')
    turns=np.flatnonzero(a['tangent'][:-1,-1]*a['tangent'][1:,-1]<0)
    assert len(turns)>0
    k=int(turns[turn]); y0=np.r_[a['r_hz'][k]/SCALE,a['s'][k]]
    y1=np.r_[a['r_hz'][k+1]/SCALE,a['s'][k+1]]
    control=int(np.argmax(abs(y1[:-1]-y0[:-1])))
    saved={}
    def solve_at(theta):
        if theta in saved:return saved[theta]
        target=(1-theta)*y0[control]+theta*y1[control]
        y=(1-theta)*y0+theta*y1
        for it in range(15):
            f,j=eq.evaluate(y[:-1]*SCALE,y[-1],True)
            eps=2e-6
            fs=(eq.evaluate(y[:-1]*SCALE,y[-1]+eps)-eq.evaluate(y[:-1]*SCALE,y[-1]-eps))/(2*eps)/SCALE
            A=np.zeros((n+1,n+1));A[:-1,:-1]=j;A[:-1,-1]=fs;A[-1,control]=1.
            err=max(float(np.max(abs(f))),abs(y[control]-target)*SCALE)
            if err<3e-7:break
            y+=linalg.solve(A,-np.r_[f/SCALE,y[control]-target],check_finite=False)
        assert err<2e-6,(theta,err)
        tangent=linalg.solve(A,np.r_[np.zeros(n),1.],check_finite=False)
        saved[theta]=(float(tangent[-1]),y)
        print('fold refine',theta,tangent[-1],err,flush=True)
        return saved[theta]
    theta=optimize.brentq(lambda th:solve_at(th)[0],0.,1.,xtol=2e-7)
    slope,y=solve_at(theta);r=y[:-1]*SCALE;s=float(y[-1])
    f,j=eq.evaluate(r,s,True)
    eigenvalues,left,right=linalg.eig(j,left=True,right=True,check_finite=False)
    order=np.argsort(abs(eigenvalues));ind=order[0]
    v=right[:,ind].real;v/=np.linalg.norm(v)
    w=left[:,ind].real;w/=np.dot(w,v)
    eps=2e-6
    fs=(eq.evaluate(r,s+eps)-eq.evaluate(r,s-eps))/(2*eps)
    curv=[]
    for h in (.1,.05,.02):
        second=(eq.evaluate(r+h*v,s)-2*f+eq.evaluate(r-h*v,s))/(h*h)
        curv.append(dict(step_hz=h,quadratic_coefficient=float(.5*np.dot(w,second))))
    energy=v[:eq.n]**2
    energy_total=float(np.dot(energy,m.count_e))
    core_energy=[float(np.dot(energy,m.region_w[f'175_{i}'])/energy_total) for i in range(3)]
    trans=float(np.dot(w,fs))
    passed=(max(abs(f))<2e-6 and abs(eigenvalues[ind])<1e-5 and
            abs(eigenvalues[order[1]])>1e-5 and abs(trans)>1e-5 and
            all(abs(c['quadratic_coefficient'])>1e-6 for c in curv) and
            len(set(np.sign(c['quadratic_coefficient']) for c in curv))==1)
    report=dict(status='NUMERICAL_STATIONARY_FOLD_CONFIRMED' if passed else 'FOLD_CANDIDATE',
                s=s,mean_z=1-s,mean_e_hz=float(np.average(r[:eq.n],weights=m.count_e)),
                residual_max_hz=float(max(abs(f))),critical_eigenvalue=[float(eigenvalues[ind].real),float(eigenvalues[ind].imag)],
                second_smallest_eigenvalue_modulus=float(abs(eigenvalues[order[1]])),
                right_null_residual=float(np.linalg.norm(j@v)),left_null_residual=float(np.linalg.norm(w@j)),
                transversality=trans,quadratic_checks=curv,core_a_b_surround_mode_energy=core_energy,
                continuation_bracket=[k,k+1],full_delay_dynamic_stability='NOT_COMPUTED',
                native_irregular_burst_onset='NOT_ESTABLISHED',
                branch=branch,
                physical_z_range=bool(s>=0 and s<=eq.ss[-1]),
                scope='Fold of the frozen v1 stationary equations with equilibrium dynamic M eliminated; v1 failed native bursting correspondence.')
    write(f'{prefix}_check.json',report)
    np.savez_compressed(OUT/f'{prefix}_state.npz',r_hz=r,s=s,right_mode=v,left_mode=w)
    print(json.dumps(report,indent=2),flush=True)


def verify():
    import csv
    eq=Equilibrium();m=eq.m
    qa=[]
    a=np.load(OUT/'recruited_equilibria.npz')
    for index in (0,len(a['s'])//2,len(a['s'])-1):
        r=a['r_hz'][index];s=float(a['s'][index]);f=eq.evaluate(r,s)
        unit_rate=eq.last['r_u'].copy();m.reset()
        re,ri=r[:eq.n]/1000,r[eq.n:]/1000
        m.r_u=unit_rate;m.r_i=ri.copy();m.m_u=m.tau_M*unit_rate
        m.z_u,m.z2_u=eq.z(s)
        m.gAE=m.te*m.gaA*(m.w_ee@re+m.je*m.nu_sig);m.cAE=m.gAE.copy()
        m.gGE=m.te*m.gaG*(m.w_ei@ri);m.cGE=m.gGE.copy()
        m.gAI=m.ti*m.gaA*(m.w_ie@re+m.ji*m.nu_sig);m.cAI=m.gAI.copy()
        m.gGI=m.ti*m.gaG*(m.w_ii@ri);m.cGI=m.gGI.copy()
        m.hE[:]=re;m.hI[:]=ri
        before={k:getattr(m,k).copy() for k in ('r_u','r_i','m_u','gAE','cAE','gGE','cGE','gAI','cAI','gGI','cGI')}
        m.step(np.full(eq.n,m.nu_sig),m.nu_sig)
        diff={k:float(np.max(abs(getattr(m,k)-v))) for k,v in before.items()}
        qa.append(dict(index=index,s=s,stationary_residual_hz=float(max(abs(f))),
                       max_native_step_state_changes=diff))
    with (SOURCE/'native_state_map.csv').open() as f:ref={r['name']:r for r in csv.DictReader(f)}
    observed=json.loads((OUT/'native_extrema.json').read_text())
    errors=[abs(r['mean_rate_hz']-float(ref[r['name']]['tail_all_E_hz'])) for r in observed]
    result=dict(equilibrium_embedding=qa,native_trajectory_count=len(observed),
                native_mean_rate_max_abs_error_hz=max(errors),
                equilibrium_max_residual_hz=float(max(np.max(abs(eq.evaluate(r,s))) for r,s in zip(a['r_hz'],a['s']))),
                note='Equilibria inserted into original v1 synapse/delay/rate/M update; steady filtered-variance limit equals stationary variance. This does not test stability or native correspondence.')
    result['status']='PASS' if result['native_mean_rate_max_abs_error_hz']<1e-10 and result['equilibrium_max_residual_hz']<2e-6 and all(max(q['max_native_step_state_changes'].values())<1e-6 for q in qa) else 'FAIL'
    write('numerical_qa.json',result);print(json.dumps(result,indent=2),flush=True)


def pilot():
    OUT.mkdir(parents=True,exist_ok=True)
    t=time.time(); eq=Equilibrium(); n=2*eq.n
    rng=np.random.default_rng(55)
    x=np.full(n,30.); s=.15
    f,j=eq.evaluate(x,s,True)
    direction=rng.normal(size=n); direction/=np.linalg.norm(direction)
    fd=(eq.evaluate(x+1e-3*direction,s)-eq.evaluate(x-1e-3*direction,s))/(2e-3)
    qa=dict(jacobian_direction_rel_error=float(np.linalg.norm(fd-j@direction)/np.linalg.norm(fd)))
    print('J QA',qa,flush=True)
    rows=[]
    for ss,rate in [(0.,.01),(.15,.01),(.23,.01),(.309,300.)]:
        tt=time.time(); r,err,ok=eq.solve(np.full(n,rate),ss)
        rows.append(dict(s=ss,initial_hz=rate,error_hz=err,valid=ok,
                         mean_e_hz=float(np.average(r[:eq.n],weights=eq.m.count_e)),seconds=time.time()-tt))
        print(rows[-1],flush=True)
        if ok: np.savez_compressed(OUT/f'pilot_s{ss:.4f}_init{rate:g}.npz',r_hz=r,s=ss)
    qa['pilot']=rows;qa['seconds']=time.time()-t
    write('pilot.json',qa)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',nargs='?',default='pilot')
    args=parser.parse_args()
    if args.mode=='pilot':pilot()
    elif args.mode=='continue':continue_all()
    elif args.mode=='core':core_branch()
    elif args.mode=='fold':fold_check()
    elif args.mode=='verify':verify()
