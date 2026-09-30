"""Check cycle-fold tangents in the full variational delay system.

A fold of autonomous periodic orbits generally has a generalized +1 mode:
    (M - I) v + (dT/ds) f = 0.
The phase direction f already has multiplier +1 everywhere. Checking only
that phase multiplier, or demanding a second independent ordinary eigenvector,
does not validate a cycle fold. The supplied tangent is computed independently
by the phase-fixed periodic BVP, including its period derivative.
"""
from rate_floquet import *
from rate_periodic_state import recover
import gc


def check(critical_json, dt, device, analytic_seed=False, stream_harmonics=None):
    q = read(Path(critical_json)); path = Path(q['orbit']); z = np.load(path)
    s = RateField()
    import cupy as cp
    cp.cuda.Device(device).use()
    if stream_harmonics is None:
        bank=(len(z['r'])//2+1)*sum(len(v[0]) for v in s.raw)*16
        free=int(cp.cuda.runtime.memGetInfo()[0])
        # The fine-step delay history and gains coexist with a second
        # reconstruction object. Stream the already verified exact harmonic
        # actions if their combined reserve would exceed available memory.
        stream_harmonics=free<bank+int(4*1024**3)
    if stream_harmonics:
        assert read(PERIODIC_OUT/'streamed_harmonic_operator_check.json')['status']=='PASS'
    m = Monodromy(s, path, dt, device,stream_harmonics=stream_harmonics)
    tangent_path = PERIODIC_OUT/f'{q["label"]}_tangent_N{q["N"]}.npz'
    tangent = np.load(tangent_path)['tangent']
    r = z['r']; T = float(z['T']); J = float(z['J'])
    dr = tangent[:-2].reshape(r.shape)/1000
    dT = T*tangent[-2]; dJ = tangent[-1]/1000
    scale = max(np.linalg.norm(dr)/np.linalg.norm(r), abs(dT/T), abs(dJ/J), 1e-3)
    eps = min(.01, 1e-5/scale)
    # Reconstruction coexists with the full-history return operator. Reuse
    # its independently verified bounded CSR-index storage as well; leaving
    # this second object unbounded defeats streamed actions on fine meshes.
    o = Periodic(s, len(r), device,
        harmonic_capacity=64 if m.bounded_harmonic_indices else None)
    o.low_memory = True
    o.stream_harmonics=stream_harmonics
    o.harmonic_chunk_size=64;o.derivative_chunk_size=64

    def initial(delta):
        # Keep the physical history times -k*dt fixed while changing T. This
        # includes the phase derivative of delayed history, not only dr/ds.
        period = T + delta*dT
        o.recovery_period = period
        kernels = o.kernels(period, J + delta*dJ)
        cf = cp.fft.rfft(cp.asarray(r + delta*dr), axis=0)/len(r)
        return recover(o, kernels, cf, m.D, m.dt)

    v = (initial(eps)-initial(-eps))/(2*eps)
    vh = (initial(eps/2)-initial(-eps/2))/eps
    reconstruction_change = np.linalg.norm(vh-v)/np.linalg.norm(vh)
    o.cache=None
    del o
    gc.collect(); cp.get_default_memory_pool().free_all_blocks()
    phase = m.phase_vector(); v = vh
    analytic_change = None
    if analytic_seed:
        from rate_fold_tangent_exact import reconstruct
        exact = reconstruct(s, r, T, J, tangent, m.D, m.dt)
        analytic_change = float(np.linalg.norm(exact-vh)/np.linalg.norm(exact))
        v = exact
    mv = m.matvec(v); mp = m.matvec(phase)
    residual = mv-v+dT*phase
    orthogonal = v-phase*(phase@v)/(phase@phase)
    row = dict(label=q['label'], orbit=str(path), tangent_source=str(tangent_path),
        J_EE_core=J, T_ms=T, N=len(r), dt_ms=m.dt,
        dT_dcoordinate_ms=float(dT), dJ_dcoordinate=float(dJ),
        seed_method='analytic harmonic derivative' if analytic_seed else 'centered finite difference',
        analytic_vs_finite_difference_relative_change=analytic_change,
        derivative_step=float(eps), reconstruction_step_halving_relative_change=float(reconstruction_change),
        generalized_plus_one_relative_defect=float(np.linalg.norm(residual)/(np.linalg.norm(v)+abs(dT)*np.linalg.norm(phase))),
        generalized_plus_one_defect_over_tangent=float(np.linalg.norm(residual)/np.linalg.norm(v)),
        phase_relative_defect=float(np.linalg.norm(mp-phase)/np.linalg.norm(phase)),
        tangent_fraction_orthogonal_to_phase=float(np.linalg.norm(orthogonal)/np.linalg.norm(v)),
        streamed_exact_harmonic_actions=bool(stream_harmonics),
        bounded_reconstruction_harmonic_indices=m.bounded_harmonic_indices,
        equation='(M-I)v + T_prime*f = 0; M*f = f',
        scope='Independent full-state/delay-history check of the fold tangent. Not the complete Floquet spectrum or stability of either adjacent branch.')
    suffix='_analytic' if analytic_seed else ''
    prefix=PERIODIC_OUT/f'{q["label"]}_monodromy_check_N{len(r)}_dt{dt:g}{suffix}'
    write(Path(str(prefix)+'.json'), row)
    save_periodic_array(str(prefix)+'.npz',local=v[:9*s.P],history=v[9*s.P:].reshape(m.D,s.P),
        phase_local=phase[:9*s.P],phase_history=phase[9*s.P:].reshape(m.D,s.P),dt=m.dt,dT=dT)
    print('FOLD MONODROMY',row,flush=True)
    return row


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('critical_json',nargs='+')
    p.add_argument('--dt',type=float,nargs='+',default=[.1,.05]);p.add_argument('--device',type=int,default=0)
    p.add_argument('--analytic-seed',action='store_true')
    args=p.parse_args()
    for path in args.critical_json:
        for dt in args.dt:
            check(path,dt,args.device,args.analytic_seed)
            gc.collect()
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
