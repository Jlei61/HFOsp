"""Find stationary predictors in net-current rather than rate coordinates.

The equations u=B Phi(u) and r=Phi(B r) have identical fixed points. This
alternative numerical coordinate avoids penalizing legitimate negative net
currents as negative firing rates. It changes no physical model parameter.
Only a root verified in the original rate equations is saved as a candidate.
"""
from equilibrium_predictor import *


def run(args):
    p=EquilibriumProblem(OUT/'stationary_response/degree6_dv0.125','pchip')
    source=dict(np.load(args.seed));D=float(source['D']);r=source['rate_hz']
    z,_=p.resource(D);B=(p.A-sparse.diags(z)@p.G-p.Mcoupling).tocsr()
    u=B@r;folder=OUT/'equilibrium_predictors'/args.label;folder.mkdir(parents=True,exist_ok=False)
    history=[];started=time.time();status='ITERATION_LIMIT'
    for it in range(args.iterations):
        r,derivative=p.response(u);f=u-B@r
        err=float(np.linalg.norm(f));maximum=float(np.max(abs(f)))
        original=float(np.max(abs(p.equations(r,D,False))))
        row=dict(iteration=it,current_residual_max_mv=maximum,current_residual_L2=err,
                 original_rate_residual_max_hz=original,mean_E_hz=float(p.eweights@r))
        history.append(row);write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),latest=row,wall_s=time.time()-started))
        print('input root',row,flush=True)
        if maximum<1e-8 and original<1e-7:status='ORIGINAL_MODEL_ROOT_FOUND';break
        J=p.I-B@sparse.diags(derivative);delta=spsolve(J,-f);accepted=False
        for alpha in 2.**-np.arange(18):
            trial=u+alpha*delta;nr,_=p.response(trial);nf=trial-B@nr
            if np.linalg.norm(nf)<err:
                u=trial;accepted=True;break
        if not accepted:status='NO_CURRENT_RESIDUAL_DESCENT';break
    name='seed.npz' if status=='ORIGINAL_MODEL_ROOT_FOUND' else 'failed_seed.npz'
    np.savez_compressed(folder/name,D=D,rate_hz=r,current_mv=u)
    write(folder/'result.json',dict(status=status,history=history,wall_s=time.time()-started,
        source=str(args.seed),scope='Equivalent-coordinate root search only. Full-density correction and dynamic stability are not supplied by this predictor.'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=Path,required=True)
    ap.add_argument('--label',required=True);ap.add_argument('--iterations',type=int,default=80)
    run(ap.parse_args())
