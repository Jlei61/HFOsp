"""Independent constant-current density derivative for equilibrium tangents."""
from stationary_response import *


def run(args):
    source=Path(args.equilibrium);cfg=read(source/'config.json')
    assert read(source/'status.json')['status']=='FULL_MAP_EQUILIBRIUM_CORRECTED'
    s=dict(np.load(source/'stationary_local_state.npz'));n=len(s['theta'])
    folder=source/f'static_response_eps{args.epsilon:g}';folder.mkdir(parents=True,exist_ok=False)
    current=np.tile(s['current_mv'],2)+args.epsilon*np.r_[np.ones(n),-np.ones(n)]
    m=LocalStationaryDensity(np.tile(s['theta'],2),np.tile(s['population'],2),current,cfg['degree'],cfg['voltage_dv'],args.device,basis_mode=cfg.get('basis_mode','legacy'))
    m.F[:]=cp.asarray(np.tile(s['F'],(2,1,1)))
    def progress(it,residual,rate):
        write(folder/'status.json',dict(status='RUNNING',pid=os.getpid(),iteration=it,maximum_residual=float(residual.max())))
        print('stationary slope',cfg['D'],it,residual.max(),flush=True)
    r,_,diag,_=m.solve(8000,progress,tolerance=1e-12,acceleration=True,
        stable_marginal_projection=args.stable_marginal_projection)
    assert diag['converged'],diag
    slope=(r[:n]-r[n:])/(2*args.epsilon)
    reference=s['actual_rate_hz']
    np.savez_compressed(folder/'susceptibility.npz',static_derivative_hz_per_mv=slope,
        plus_rate_hz=r[:n],minus_rate_hz=r[n:],reference_rate_hz=reference,
        forward_derivative_hz_per_mv=(r[:n]-reference)/args.epsilon,
        backward_derivative_hz_per_mv=(reference-r[n:])/args.epsilon,
        second_derivative_hz_per_mv2=(r[:n]+r[n:]-2*reference)/args.epsilon**2)
    write(folder/'status.json',dict(status='LOCAL_STATIONARY_SLOPE_COMPLETE',epsilon_mv=args.epsilon,qa=diag,
        stable_marginal_projection=args.stable_marginal_projection))
    print(folder,diag,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--equilibrium',type=Path,required=True)
    ap.add_argument('--epsilon',type=float,default=.001);ap.add_argument('--device',type=int,default=1)
    ap.add_argument('--stable-marginal-projection',action='store_true');run(ap.parse_args())
