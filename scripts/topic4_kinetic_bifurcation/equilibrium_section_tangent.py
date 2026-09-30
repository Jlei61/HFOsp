"""Actual-density stationary tangent and simple-fold diagnostics.

The local derivative is measured from the conditional-density map, not from
the interpolation table. Removing locally invertible density states yields the
same zero eigenvalue condition. Dynamic stability remains a separate analysis.
"""
from equilibrium_predictor import *
from scipy.sparse.linalg import eigs


def run(args):
    folder=Path(args.response);source=folder.parent;cfg=read(source/'config.json')
    p=EquilibriumProblem(Path(cfg.get('table',OUT/'stationary_response/degree6_dv0.125')),'pchip')
    state=source/'stationary_local_state.npz'
    if not state.exists():state=source/'latest_rates.npz'
    with np.load(state) as z:r=z['rate_hz']
    if (folder/'config.json').exists():
        response_config=read(folder/'config.json')
        assert response_config.get('complete_network',True), 'Selected-group pilot is not a network tangent'
    with np.load(folder/'susceptibility.npz') as z:fp=z['static_derivative_hz_per_mv']
    z,dz=p.resource(cfg['D']);B=p.A-sparse.diags(z)@p.G-p.Mcoupling
    K=sparse.diags(fp)@B;J=p.I-K;fd=fp*dz*(p.G@r)
    border=sparse.vstack([sparse.hstack([J,fd[:,None]]),sparse.csr_matrix(np.r_[p.eweights,0][None,:])]).tocsr()
    tangent=spsolve(border,np.r_[np.zeros(p.P),1.])
    rng=np.random.default_rng(1982);v0=rng.normal(size=p.P)
    values,vecs=eigs(K,k=24,which='LM',v0=v0,tol=1e-10,ncv=60)
    at=np.argmin(abs(values-1.));value=values[at];right=vecs[:,at]
    lv,lefts=eigs(K.T,k=24,which='LM',v0=v0,tol=1e-10,ncv=60)
    left=lefts[:,np.argmin(abs(lv-value))]
    assert abs(value.imag)<1e-7
    right=right.real;left=left.real;right/=np.linalg.norm(right);left/=left@right
    gap=float(np.min(np.delete(abs(values-1.),at)))
    np.savez_compressed(folder/'stationary_tangent.npz',rate_derivative_per_mean_rate=tangent[:-1],
        D_derivative_per_mean_rate=tangent[-1],right_rate_mode=right,left_rate_mode=left,
        loop_eigenvalues=values)
    report=dict(D=cfg['D'],mean_E_hz=float(p.eweights@r),D_derivative_per_mean_rate=float(tangent[-1]),
        nearest_stationary_loop_eigenvalue=float(value.real),other_computed_distance_to_plus_one=gap,
        smallest_computed_loop_modulus=float(np.min(abs(values))),
        left_mode_parameter_projection=float(left@fd),mode_normalization='right Euclidean norm 1, left dot right 1',
        bordered_solve_residual=float(np.max(abs(border@tangent-np.r_[np.zeros(p.P),1.]))),
        source='Actual stationary density susceptibility; table used only for exact network operators',
        bifurcation_type='NOT_YET_CLASSIFIED; locate zero tangent and verify nondegeneracy and discretization')
    write(folder/'stationary_tangent.json',report);print(report,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--response',type=Path,required=True);run(ap.parse_args())
