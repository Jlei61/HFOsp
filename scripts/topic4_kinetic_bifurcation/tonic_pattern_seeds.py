"""Bounded stationary root predictors from recruited spatial patterns.

Time-average activity in intermittently recruited boundary cells need not be
a stationary rate. Thresholding only supplies initial guesses; all accepted
roots solve the same original equations with every E/I group released.
"""
from equilibrium_predictor import *


def run(args):
    p=EquilibriumProblem(OUT/'stationary_response/degree6_dv0.125','pchip')
    s=np.load(args.seed);original=s['rate_hz'];D=float(s['D']);ids=np.flatnonzero(~p.e)
    root=OUT/'equilibrium_predictors/tonic_pattern_seeds';root.mkdir(parents=True,exist_ok=False)
    results=[]
    for cut in args.thresholds:
        folder=root/f'cut{cut:g}';folder.mkdir();r=original.copy()
        r[p.e & (original<cut)]=.1
        history=[]
        for iteration in range(60):
            f,J,_,_=p.equations(r,D);err=float(np.max(abs(f[ids])));history.append(err)
            if err<1e-8:break
            dx=spsolve(J[ids][:,ids],-f[ids]);accepted=False
            for a in 2.**-np.arange(16):
                trial=r.copy();trial[ids]+=a*dx
                nf=p.equations(trial,D,False)
                if np.max(abs(nf[ids]))<err:r=trial;accepted=True;break
            if not accepted:break
        print('tonic seed',cut,'I residual',history[-1],flush=True)
        solved,qa=p.fixed_D(D,r,maxit=80)
        row=dict(threshold_hz=cut,D=D,initial_I_history=history,full_root_qa=qa,
            mean_E_hz=float(p.eweights@solved),minimum_rate_hz=float(solved.min()),
            scope='Spatial initial-guess diagnostic only; full density correction and stability remain required')
        write(folder/'status.json',row)
        np.savez_compressed(folder/('seed.npz' if qa['converged'] and solved.min()>-1e-7 else 'failed_seed.npz'),D=D,rate_hz=solved)
        results.append(row);write(root/'progress.json',dict(status='RUNNING',pid=os.getpid(),completed=results));print(row,flush=True)
    write(root/'status.json',dict(status='BOUNDED_SEARCH_COMPLETE',results=results))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=Path,required=True)
    ap.add_argument('--thresholds',type=float,nargs='+',default=[100.,200.,300.]);run(ap.parse_args())
