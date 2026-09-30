"""Low-state stationary predictor parametrized by mean E firing rate.

Mean rate is a numerical continuation coordinate. D remains the physical
parameter, and no current or firing clamp is added to the dynamical model.
This coordinate crosses a D fold without a nearly vertical D arclength step.
"""
from correct_rate_section import bordered_step
from equilibrium_predictor import *


def run(args):
    p=EquilibriumProblem(args.table,'pchip');folder=OUT/'equilibrium_predictors'/args.label
    folder.mkdir(parents=True,exist_ok=False);r,qa=p.fixed_D(0.,np.where(p.e,.1,.3));assert qa['converged'],qa
    D=0.;rows=[];rates=[];Ds=[];start=float(p.eweights@r);started=time.time();status='RATE_LIMIT'
    target=start
    while target<=args.maximum_rate+1e-12:
        if rows:
            # Predictor from the preceding regular bordered tangent.
            dr,dd,_=bordered_step(p,r,D,target)
            r=r+dr;D=D+dd
        if not 0<=D<=1:status='PHYSICAL_BOUNDARY';break
        converged=False
        for iteration in range(15):
            f=p.equations(r,D,False);constraint=float(p.eweights@r-target)
            error=float(np.max(abs(f))+100*abs(constraint))
            if np.max(abs(f))<1e-8 and abs(constraint)<1e-11:converged=True;break
            dr,dd,_=bordered_step(p,r,D,target);accepted=False
            for alpha in 2.**-np.arange(14):
                nr=r+alpha*dr;nd=D+alpha*dd
                if not 0<=nd<=1:continue
                nf=p.equations(nr,nd,False);ne=np.max(abs(nf))+100*abs(p.eweights@nr-target)
                if ne<error:r,D=nr,nd;accepted=True;break
            if not accepted:break
        if not converged:status='SECTION_CORRECTION_FAILED';break
        f,J,fd,u=p.equations(r,D)
        mat=sparse.vstack([sparse.hstack([J,fd[:,None]]),sparse.csr_matrix(np.r_[p.eweights,0.][None,:])]).tocsr()
        tangent=spsolve(mat,np.r_[np.zeros(p.P),1.])
        outside=int(np.sum((u<p.current_grid[0])|(u>p.current_grid[-1])))
        row=dict(index=len(rows),D=float(D),mean_E_hz=float(p.eweights@r),dD_dmeanE=float(tangent[-1]),
                 residual_hz=float(np.max(abs(f))),outside_current_table=outside,iterations=iteration)
        rows.append(row);rates.append(r.copy());Ds.append(D)
        write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),latest=row,wall_s=time.time()-started))
        np.savez_compressed(folder/'branch.npz',D=np.asarray(Ds),rate_hz=np.asarray(rates))
        print('rate predictor',row,flush=True)
        if outside:status='CURRENT_TABLE_BOUNDARY';break
        target+=args.step
    write(folder/'branch.json',dict(status=status,points=rows,table=str(args.table),wall_s=time.time()-started,
        scope='Stationary-density table predictor only. Full-map correction, noise-order convergence and dynamic stability remain required.'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--table',type=Path,required=True);ap.add_argument('--label',required=True)
    ap.add_argument('--step',type=float,default=.001);ap.add_argument('--maximum-rate',type=float,default=.18)
    run(ap.parse_args())
