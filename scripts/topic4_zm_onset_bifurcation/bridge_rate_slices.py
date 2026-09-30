"""Locate missing intermediate equilibria using mean rate as a solver coordinate.

D remains the plotted physical model parameter. The added equation selects a
global rate; it is NOT a new forcing or alteration of the model residual.
Independent solutions must not be joined as a verified continuation branch.
"""
from continue_D import *


def solve_rate(s,target,r,D,project_trials=False,unbounded_trials=False):
    x=np.r_[r/RS,D/DS]
    w=np.zeros(s.P);w[s.E]=s.mean_weights
    trace=[]
    for k in range(65):
        r=x[:-1]*RS;D=x[-1]*DS
        f=np.r_[s.residual(r,D),w@r-target/1000]
        error=float(abs(f).max());trace.append(error)
        if error<2e-11:
            rr=np.clip(r,0,np.nextafter(1/s.ref,0))
            if abs(s.residual(rr,D)).max()<2e-11 and abs(w@rr-target/1000)<2e-11:return rr,D,True,trace
        A=sparse.bmat([[s.jacobian(r,D)*RS,sparse.csr_matrix((s.parameter_derivative(r,D)*DS)[:,None])],
            [sparse.csr_matrix((w*RS)[None,:]),sparse.csr_matrix((1,1))]],format='csc')
        dx=spsolve(A,-f);alpha=1.
        for j in range(25):
            trial=x+alpha*dx;rr=trial[:-1]*RS;dd=trial[-1]*DS
            admissible=unbounded_trials or project_trials or (rr.min()>=-1e-12 and np.all(rr<1/s.ref))
            if 0<=dd<=1 and admissible and np.all(np.isfinite(rr)):
                # Projection is only a nonlinear-solver trial step. Acceptance
                # still requires the unmodified original residual and rate
                # constraint, so this does not clip or change the rate model.
                if not unbounded_trials:rr=np.clip(rr,0,np.nextafter(1/s.ref,0));trial[:-1]=rr/RS
                ff=np.r_[s.residual(rr,dd),w@rr-target/1000]
                if np.linalg.norm(ff)<np.linalg.norm(f):x=trial;break
            alpha*=.5
        else:return r,D,False,trace
    return r,D,False,trace


def main(a):
    s=ZMRate();base=DEST/'g20';dest=base/a.label;dest.mkdir(exist_ok=False)
    candidates=[]
    names=['D_gap_lower_v1','D_gap_upper_v1','D_gap_lower_guarded','D_arclength_upper_onset_range_v4']
    if a.label!='D_gap_rate_slices' and (base/'D_gap_rate_slices/result.json').exists():names.append('D_gap_rate_slices')
    for name in names:
        info=read(base/name/'result.json')
        for row in info['rows']:
            if not row.get('converged',True):continue
            candidates.append(dict(rate=row['global_E_hz'],D=row['D'],file=str(base/name/f'point{row["index"]:04d}.npz')))
    candidates.sort(key=lambda q:q['rate']);last=None;rows=[]
    for k,target in enumerate(np.arange(a.start,a.end+.01,a.step)):
        seeds=[]
        if last is not None:seeds.append((*last,'previous_converged_rate'))
        nearest=min(candidates,key=lambda q:abs(q['rate']-target))
        with np.load(nearest['file']) as z:seeds.append((z['r'],float(z['D']),nearest['file']))
        lo=max([q for q in candidates if q['rate']<=target],key=lambda q:q['rate'],default=None)
        hi=min([q for q in candidates if q['rate']>=target],key=lambda q:q['rate'],default=None)
        if lo is not None and hi is not None and hi['rate']>lo['rate']:
            t=(target-lo['rate'])/(hi['rate']-lo['rate'])
            with np.load(lo['file']) as z:rl=z['r'].copy()
            with np.load(hi['file']) as z:rh=z['r'].copy()
            seeds.append(((1-t)*rl+t*rh,(1-t)*lo['D']+t*hi['D'],'interpolated_initial_guess_only'))
        attempts=[]
        for rr,dd,source in seeds:
            r,D,ok,trace=solve_rate(s,target,rr,dd,a.project_trials,a.unbounded_trials)
            attempts.append(dict(source=source,converged=ok,residual=trace[-1],iterations=len(trace)))
            if ok:break
        row=dict(index=k,target_hz=target,converged=ok,attempts=attempts)
        if ok:
            last=(r,D)
            row.update(D=D,global_E_hz=s.global_rate(r),regional_hz=s.regional_rates(r),
                equilibrium_residual_hz=float(abs(s.residual(r,D)).max()*1000))
            np.savez_compressed(dest/f'point{k:04d}.npz',r=r,D=D,Z=s.Z,tangent=tangent(s,r,D))
        rows.append(row)
        write(dest/'result.json',dict(status='RUNNING',rows=rows,
            connection='Independent equilibrium solves; branch connectivity not asserted',M='dynamic'))
        print(k,target,ok,D,trace[-1],flush=True)
    write(dest/'result.json',dict(status='COMPLETE',rows=rows,
        connection='Independent equilibrium solves; branch connectivity not asserted',M='dynamic'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--start',type=float,default=96)
    p.add_argument('--end',type=float,default=216);p.add_argument('--step',type=float,default=2)
    p.add_argument('--label',default='D_gap_rate_slices')
    p.add_argument('--project-trials',action='store_true');p.add_argument('--unbounded-trials',action='store_true');main(p.parse_args())
