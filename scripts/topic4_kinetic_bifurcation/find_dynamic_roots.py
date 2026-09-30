"""Locate positive real roots of the full delayed density feedback equation."""
from dynamic_loop_spectrum import *
from scipy.optimize import brentq


def run(args):
    folder=Path(args.response);model=DynamicLoop(folder);history=[]
    rng=np.random.default_rng(6291);v0=rng.normal(size=model.P)
    def evaluate(s,save_mode=False):
        matrix=model.matrix(float(s))
        vals,vecs=eigs(matrix,k=24,which='LM',v0=v0,tol=1e-10,maxiter=4000,ncv=60)
        real=np.flatnonzero(abs(vals.imag)<1e-7)
        if not len(real):raise RuntimeError('No real loop eigenvalue in computed outer spectrum')
        at=real[np.argmax(vals[real].real)]
        row=dict(growth_per_s=float(s),leading_real_loop_eigenvalue=float(vals[at].real),
            residual=float(np.linalg.norm(matrix@vecs[:,at]-vals[at]*vecs[:,at])))
        history.append(row);print(row,flush=True)
        if save_mode:return vals[at],vecs[:,at],row
        return vals[at].real-1.
    a=evaluate(0.)
    if a<=0:
        write(folder/'positive_real_root.json',dict(status='NO_POSITIVE_REAL_ROOT_BRACKET_AT_ZERO',history=history,
            meaning='This does not exclude complex roots or isolated real crossings away from zero'));return
    upper=10.
    while evaluate(upper)>0:
        upper*=2
        if upper>10000:raise RuntimeError('No upper root bracket')
    root=brentq(evaluate,0.,upper,xtol=1e-7)
    value,mode,row=evaluate(root,True)
    geo=model.geo;e=geo['population']==0;groups=geo['group_cell'][e]
    power=abs(mode[groups])**2*geo['group_size'][e]
    fraction=[float(power[geo['group_region'][e]==j].sum()/power.sum()) for j in range(3)]
    np.savez_compressed(folder/'positive_real_mode.npz',growth_per_s=root,mode_EI=mode,
                        map_multiplier=np.exp(root*DT/1000.))
    write(folder/'positive_real_root.json',dict(status='POSITIVE_REAL_DYNAMIC_ROOT_FOUND',D=model.config['D'],
        growth_per_s=root,map_multiplier=float(np.exp(root*DT/1000.)),
        characteristic_eigenvalue=[float(value.real),float(value.imag)],
        E_mode_power_A_B_surround=fraction,history=history,
        meaning='Unstable real exponential mode of the local-density susceptibility plus original delay/synapse/dynamic-M feedback',
        remaining=['direct full-map growth check','kernel-amplitude/duration convergence','density discretization convergence']))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--response',type=Path,required=True);run(ap.parse_args())
