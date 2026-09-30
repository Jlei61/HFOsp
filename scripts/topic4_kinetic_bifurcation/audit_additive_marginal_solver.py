"""Verify the stable marginal constraint with the actual stalled local batch."""
from correct_batched_section import *


def run(a):
    cfg=read(a.source/'config.json');p=EquilibriumProblem(Path(cfg['table']),'pchip')
    folder=OUT/'local_stationary_polish'/a.label;folder.mkdir(parents=True,exist_ok=False)
    with np.load(a.source/'latest_rates.npz') as z:
        r=z['rate_hz'];D=float(z['D']);actual=z['actual_rate_hz'];u=z['current_mv']
    dr,dd,_=bordered_step(p,r,D,cfg['global_rate_section_hz'],r-actual,response_slope=p.response(u)[1])
    D+=dd;u=p.equations(r+dr,D)[-1];assert abs(D-read(a.source/'progress.json')['D'])<1e-13
    ids=np.arange(64);disk=np.load(a.source/'density_coefficients.npy',mmap_mode='r')
    m=LocalStationaryDensity(p.theta[ids],p.geo['population'][ids],u[ids],cfg['degree'],cfg['voltage_dv'],
        a.device,basis_mode=cfg.get('basis_mode','legacy'));m.F[:]=cp.asarray(disk[ids,:,:m.width])
    m.map();initial=m.residual();initial_rate=cp.asnumpy(m.flux@m.mass*1000/DT)
    def progress(it,res,rate):
        write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),iteration=it,maximum_residual=float(res.max())))
    rate,res,qa,history=m.solve(4000,progress,tolerance=2e-12,acceleration=True,stable_marginal_projection=True)
    np.savez_compressed(folder/'local_state.npz',F=cp.asnumpy(m.F),current_mv=u[ids],group_indices=ids,D=D,rate_hz=rate)
    write(folder/'result.json',dict(status='ADDITIVE_LOCAL_BATCH_PASS' if qa['converged'] else 'ADDITIVE_LOCAL_BATCH_INCOMPLETE',
        source=str(a.source.resolve()),D=D,group_indices=ids,initial_residuals=initial,final_residuals=res,
        maximum_rate_change_hz=float(np.max(abs(rate-initial_rate))),diagnostics=qa,iterations=history,
        scope='Same original local map at actual stalled outer-1 currents, 64 groups; no network equilibrium claim'))
    print(qa,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--label',required=True)
    ap.add_argument('--device',type=int,default=0);run(ap.parse_args())
