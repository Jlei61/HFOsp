"""Bounded pseudo-time search for a stationary-density root predictor.

The auxiliary rate relaxation is a nonlinear solver, never a physical model
or a source of stability labels. Only r=Phi(B r) at the original D is retained.
Each resulting seed still requires full conditional-density correction.
"""
from equilibrium_predictor import *


def run(args):
    p=EquilibriumProblem(OUT/'stationary_response/degree6_dv0.125','pchip')
    folder=OUT/'equilibrium_predictors'/args.label;folder.mkdir(parents=True,exist_ok=False)
    with np.load(args.seed) as z:r=z['rate_hz'].copy();D=float(z['D'])
    r=np.maximum(r,0.)
    delta=.01;rows=[];started=time.time();status='ITERATION_LIMIT'
    for it in range(args.iterations):
        f,J,_,u=p.equations(r,D)
        error=float(np.linalg.norm(f));maximum=float(np.max(abs(f)))
        row=dict(iteration=it,residual_hz=maximum,residual_L2=error,
            mean_E_hz=float(p.eweights@r),pseudo_step=delta)
        rows.append(row)
        write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),latest=row,wall_s=time.time()-started))
        print(row,flush=True)
        if maximum<1e-8 and r.min()>-1e-7:
            status='ORIGINAL_MODEL_ROOT_FOUND';break
        step=spsolve(J+p.I/delta,-f)
        # Backtracking uses the same original nonlinear residual. A very small
        # pseudo step may initially increase it for a rate-unstable equilibrium;
        # if no descent is available, increase delta toward the Newton limit.
        accepted=False
        for a in 2.**-np.arange(12):
            nr=r+a*step;nf=p.equations(nr,D,False);new_error=float(np.linalg.norm(nf))
            if new_error<error:
                r=nr;delta=min(1e8,max(delta*.7,delta*error/max(new_error,1e-30))*1.3)
                accepted=True;break
        if not accepted:
            delta=min(1e8,delta*10.)
            if delta>=1e8:status='NO_RESIDUAL_DESCENT';break
    name='seed.npz' if status=='ORIGINAL_MODEL_ROOT_FOUND' else 'failed_seed.npz'
    np.savez_compressed(folder/name,D=D,rate_hz=r)
    write(folder/'solver.json',dict(status=status,history=rows,wall_s=time.time()-started,
        role='Root predictor only; auxiliary pseudo time supplies no dynamical inference'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=Path,required=True)
    ap.add_argument('--label',required=True);ap.add_argument('--iterations',type=int,default=250)
    run(ap.parse_args())
