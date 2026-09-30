"""Resolve a suspected noise-quadrature kink near the network turning point.

This is a local numerical-convergence diagnostic, not a replacement neuron
model or a physical bifurcation. Currents and thresholds are held identical
while the conditional-noise degree and voltage mesh are varied.
"""
from stationary_response import *


def run(args):
    source=OUT/'corrected_rate_sections/rE0.15256800_degree6_dv0.125_secant'
    s=dict(np.load(source/'stationary_local_state.npz'));g=args.group
    folder=OUT/'local_threshold_resolution'/f'g{g}_degree{args.degree}_dv{args.dv:g}'
    if args.high_precision:folder=folder.with_name(folder.name+'_high_precision')
    folder.mkdir(parents=True,exist_ok=False)
    u=s['current_mv'][g]+np.linspace(-.04,.04,41)
    theta=np.full(len(u),s['theta'][g]);pop=np.full(len(u),s['population'][g])
    mode='high_precision' if args.high_precision else 'legacy'
    m=LocalStationaryDensity(theta,pop,u,args.degree,args.dv,args.device,basis_mode=mode)
    if not args.high_precision and m.F.shape[1:]==s['F'].shape[1:]:m.F[:]=cp.asarray(np.repeat(s['F'][g:g+1],len(u),axis=0))
    def progress(it,residual,rate):
        write(folder/'status.json',dict(status='RUNNING',pid=os.getpid(),iteration=it,residual=float(residual.max())))
        print('local mesh',args.degree,args.dv,it,residual.max(),flush=True)
    r,_,diag,_=m.solve(10000,progress,tolerance=1e-12,acceleration=True)
    nodes=cp.asnumpy(m.nodes)
    np.savez_compressed(folder/'curve.npz',current_mv=u,rate_hz=r,threshold_mv=theta[0],noise_nodes_mv=nodes)
    write(folder/'status.json',dict(status='COMPLETE' if diag['converged'] else 'INCOMPLETE',qa=diag,basis_mode=mode,
        candidate_node_crossings=theta[0]-nodes[(theta[0]-nodes>=u.min())&(theta[0]-nodes<=u.max())],
        interpretation='Local stationary response mesh sensitivity; no network bifurcation label'))
    print(folder,diag,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--group',type=int,default=395)
    ap.add_argument('--degree',type=int,default=6);ap.add_argument('--dv',type=float,default=.125)
    ap.add_argument('--device',type=int,default=1);ap.add_argument('--high-precision',action='store_true');run(ap.parse_args())
