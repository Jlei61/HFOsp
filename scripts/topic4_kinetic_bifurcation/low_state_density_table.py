"""A denser local stationary predictor for low-state mesh refinement.

The current range is deliberately restricted to the low-state problem; this
table must not be used to infer the middle/high activity branch. Physical
equilibria continue to require the full-map correction.
"""
from stationary_response import *


def run(args):
    folder=OUT/'stationary_response'/f'degree{args.degree}_dv{args.dv:g}_low_dense'
    if args.high_precision:folder=folder.with_name(folder.name+'_high_precision')
    folder.mkdir(parents=True,exist_ok=False)
    currents=np.linspace(-2.,3.,101);thresholds=np.r_[np.linspace(14.,18.,17),18.]
    populations=np.r_[np.zeros(17,np.uint8),np.ones(1,np.uint8)]
    theta=np.repeat(thresholds,len(currents));pop=np.repeat(populations,len(currents));u=np.tile(currents,len(thresholds))
    mode='high_precision' if args.high_precision else 'legacy'
    m=LocalStationaryDensity(theta,pop,u,args.degree,args.dv,args.device,basis_mode=mode)
    write(folder/'config.json',dict(degree=args.degree,voltage_dv=args.dv,dt_ms=DT,
        threshold_spacing_mv=.25,current_spacing_mv=.05,current_bounds_mv=[-2.,3.],
        basis_mode=mode,intended_scope='Low-state root predictor; full density correction required'))
    def progress(it,residual,rate):
        write(folder/'status.json',dict(status='RUNNING',pid=os.getpid(),iteration=it,residual=float(residual.max())))
        print('low table',args.degree,it,residual.max(),flush=True)
    rate,res,diag,hist=m.solve(10000,progress,tolerance=1e-10,acceleration=True)
    shape=(len(thresholds),len(currents))
    np.savez_compressed(folder/'response.npz',rate_hz=rate.reshape(shape),residual=res.reshape(*shape,2),
        theta=thresholds,population=populations,current_mv=currents,iteration_history=hist)
    np.savez_compressed(folder/'density_state.npz',F=cp.asnumpy(m.F),theta=theta,population=pop,current_mv=u,
        edges=m.edges,noise_mass=cp.asnumpy(m.mass))
    write(folder/'status.json',dict(status='STATIONARY_SOLVED' if diag['converged'] else 'INCOMPLETE',diagnostics=diag))
    print(folder,diag,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--degree',type=int,default=8)
    ap.add_argument('--dv',type=float,default=.125);ap.add_argument('--device',type=int,default=0)
    ap.add_argument('--high-precision',action='store_true');run(ap.parse_args())
