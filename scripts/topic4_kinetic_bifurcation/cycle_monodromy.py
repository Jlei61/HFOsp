"""Matrix-free leading spectrum of a full-state candidate period return.

The variational equations include all recurrent delays, synaptic currents,
conditional voltage densities and dynamic E-M. The reference is reset before
each Arnoldi product. Ritz values are not promoted to Floquet multipliers until
the candidate orbit, discretization and residual checks have been accepted.
"""
from network_tangent import *
from search_recurrence import capture, separation
import signal


class StateCoordinates:
    """Invertible block scaling on the physical, fixed-noise-marginal subspace."""
    def __init__(self,m):
        self.model=m;self.shapes={k:getattr(m,k).shape for k in STATE_NAMES}
        self.slices={};offset=0;self.scales={}
        w=cp.asarray(m.geo['group_size']/40000.)
        # The metric only conditions Arnoldi; it does not alter the map.
        self.scales['F']=cp.sqrt(w/cp.maximum(cp.sum(m.F*m.F,axis=(1,2)),1e-24))[:,None,None]
        for k in ('qa','ia','qg','ig','M'):
            value=getattr(m,k);rms=cp.sqrt(w@(value*value))
            self.scales[k]=cp.sqrt(w)/cp.maximum(rms,1.)
        rms=cp.sqrt(cp.mean(cp.sum(m.history*m.history*w[None,:],axis=1)))
        self.scales['history']=cp.sqrt(w/m.D)[None,:]/cp.maximum(rms,DT/1000.)
        for k in STATE_NAMES:
            n=int(np.prod(self.shapes[k]));self.slices[k]=slice(offset,offset+n);offset+=n
        self.size=offset

    def pack(self,obj):
        out=cp.empty(self.size,dtype=cp.float64)
        for k in STATE_NAMES:
            a=getattr(obj,k)
            if k=='history':a=a[(self.model.step_index-np.arange(self.model.D))%self.model.D].copy();a[0]=0.
            out[self.slices[k]]=(a*self.scales[k]).ravel()
        return out

    def unpack(self,v,obj,project=False):
        m=self.model
        for k in STATE_NAMES:
            a=v[self.slices[k]].reshape(self.shapes[k])/self.scales[k]
            if k=='history':
                getattr(obj,k)[(m.step_index-np.arange(m.D))%m.D]=a
                getattr(obj,k)[m.step_index%m.D]=0.
            else:getattr(obj,k)[:]=a
        obj.M[m.pop!=0]=0.
        # I refractory padding is not part of the dynamical state.
        obj.F[m.pop!=0,:,m.nv+int(m.refs.min()):]=0.
        if project:obj.project_noise_marginal()


def run(args):
    source=Path(args.source);cfg=read(source/'config.json')
    if args.equilibrium:
        assert read(source/'status.json')['status']=='FULL_MAP_EQUILIBRIUM_CORRECTED'
    folder=OUT/('equilibrium_map_spectra' if args.equilibrium else 'candidate_monodromy')/args.label
    folder.mkdir(parents=True,exist_ok=args.resume)
    storage=Path('/data/hfosp/topic4_sef_hfo/kinetic_population_bifurcation_20260916/krylov')/args.label
    storage.mkdir(parents=True,exist_ok=args.resume)
    m=AutonomousDensity(cfg['D'],cfg['degree'],cfg['voltage_dv'],args.device,basis_mode=cfg.get('basis_mode','legacy'))
    m.restore(source);t=NetworkTangent(m);coordinates=StateCoordinates(m)
    initial=capture(m);initial_index=m.step_index;weights=cp.asarray(m.geo['group_size']/40000.)
    steps=round(args.period/DT);assert abs(steps*DT-args.period)<1e-8
    config=dict(source=str(source.resolve()),D=cfg['D'],degree=cfg['degree'],
        voltage_dv=cfg['voltage_dv'],period_ms=steps*DT,steps=steps,krylov_dimension=args.dimension,seed=args.seed,
        object=('Power of the full original-map Jacobian at a corrected equilibrium' if args.equilibrium else
                'Derivative of the original FP64 map iterated over the candidate return time'),
        scope=('Equilibrium stability candidate; iteration duration is a numerical spectral transformation, not an orbit period' if args.equilibrium else
               'Candidate monodromy; not a certified Floquet spectrum before orbit and numerical-convergence acceptance'),
        excluded_directions=['private noise marginals','I M','I refractory padding','unused delay-ring slot'],
        states=coordinates.size,M='dynamic',Z='fixed spatial path',krylov_storage=str(storage))
    if args.equilibrium:config['equilibrium']=True
    if args.resume:assert read(folder/'config.json')==config
    else:write(folder/'config.json',config)
    rng=np.random.default_rng(args.seed)
    for k in ('qa','ia','qg','ig','M'):
        getattr(t,k)[:]=getattr(m,k)*cp.asarray(rng.normal(size=m.P))
    t.M[m.pop!=0]=0.
    t.history[:]=m.history*cp.asarray(rng.normal(size=m.history.shape))
    q=coordinates.pack(t);q/=cp.linalg.norm(q)
    basis=[q];H=np.zeros((args.dimension+1,args.dimension));started=time.time();last=started
    results=[];resets=[];closure=None
    if args.resume:
        results=read(folder/'spectrum_progress.json')['iterations']
        old=np.load(folder/'arnoldi.npz')['H'];H[:old.shape[0],:old.shape[1]]=old
        basis=[cp.asarray(np.load(storage/f'q{i:03d}.npy')) for i in range(len(results)+1)]
        closure=results[0]['closure'] if results else None
    else:np.save(storage/'q000.npy',cp.asnumpy(basis[0]))
    stop=[False];signal.signal(signal.SIGTERM,lambda signum,frame:stop.__setitem__(0,True))
    def reset():
        for name in initial:
            if name not in ('ordered_history','step_index'):getattr(m,name)[:]=initial[name]
        m.step_index=initial_index
        for name in ('maxneg','minflux','masserror'):getattr(m,name).fill(0.)
    for j in range(len(results),args.dimension):
        reset();coordinates.unpack(basis[j],t,project=True)
        for step in range(steps):
            t.advance()
            if stop[0]:
                write(folder/'interruption.json',dict(status='INTERRUPTED_SIGTERM',completed_columns=len(results),
                    partial_column=j,partial_elapsed_ms=(step+1)*DT,wall_s=time.time()-started,
                    resume='Restart with the same arguments plus --resume; only this partial product is repeated'))
                return
            if time.time()-last>20:
                cp.cuda.get_current_stream().synchronize()
                write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),column=j,
                    completed_period_ms=(step+1)*DT,wall_s=time.time()-started,latest=results[-1] if results else None))
                print('monodromy',args.label,j,(step+1)*DT,flush=True);last=time.time()
        terminal=capture(m);sep=separation(initial,terminal,weights)
        if closure is None:closure=sep
        else:resets.append(abs(sep['score']-closure['score']))
        noise_mass_error=float(cp.max(abs(t.F.sum(2))).get())
        v=coordinates.pack(t);image_norm=float(cp.linalg.norm(v).get())
        # Two modified Gram-Schmidt passes are essential for the strongly
        # non-normal spatial/delay map; all products remain in FP64.
        for _ in range(2):
            for i in range(j+1):
                h=float(cp.dot(basis[i],v).get());H[i,j]+=h;v-=h*basis[i]
        H[j+1,j]=float(cp.linalg.norm(v).get())
        values,vectors=np.linalg.eig(H[:j+1,:j+1]);order=np.argsort(-abs(values));values=values[order];vectors=vectors[:,order]
        residuals=abs(H[j+1,j]*vectors[-1,:])
        row=dict(column=j,dimension=j+1,ritz_real=values.real.tolist(),ritz_imag=values.imag.tolist(),
            ritz_modulus=abs(values).tolist(),ritz_absolute_residual=residuals.tolist(),
            image_norm=image_norm,noise_marginal_tangent_max=noise_mass_error,closure=sep)
        if H[j+1,j]>=1e-13:np.save(storage/f'q{j+1:03d}.npy',cp.asnumpy(v/H[j+1,j]))
        results.append(row);np.savez_compressed(folder/'arnoldi.npz',H=H[:j+2,:j+1])
        write(folder/'spectrum_progress.json',dict(status='RUNNING',iterations=results))
        print('candidate spectrum',args.label,j+1,list(zip(values[:6],residuals[:6])),flush=True)
        if H[j+1,j]<1e-13:break
        basis.append(v/H[j+1,j])
    map_diagnostics=m.diagnostics()
    final=results[-1];gram=np.array([[float(cp.dot(a,b).get()) for b in basis[:len(results)]] for a in basis[:len(results)]])
    # Store a few full spatial modes for independent nonlinear checks. The
    # coordinates and physical components are both available without a model
    # reconstruction from a plotted scalar observable.
    reset()
    for mode in range(min(args.save_modes,len(values))):
        for part,coeff in [('real',vectors[:,mode].real),('imag',vectors[:,mode].imag)]:
            if np.linalg.norm(coeff)<1e-12:continue
            vec=cp.zeros(coordinates.size)
            for i,c in enumerate(coeff):vec+=c*basis[i]
            coordinates.unpack(vec,t,project=True)
            arrays={k:cp.asnumpy(t.canonical(k)) for k in STATE_NAMES}
            np.savez_compressed(folder/f'mode_{mode:02d}_{part}.npz',**arrays)
    interpretation=('Stationary-map Jacobian power. A modulus above one indicates growth only after eigenpair and full-map validation; complex argument is aliased by the iteration duration.' if args.equilibrium else
                    'Candidate periodic return; orbit closure and numerical qualification are required before Floquet interpretation.')
    write(folder/'result.json',dict(status='COMPLETE',iterations=results,final=final,closure=closure,
        repeated_reference_closure_max_difference=max(resets,default=0.),
        krylov_orthogonality_error=float(np.max(abs(gram-np.eye(len(gram))))),
        diagnostics=map_diagnostics,wall_s=time.time()-started,
        acceptance='CANDIDATE_SPECTRUM_ONLY; reference invariance, Ritz convergence and discretization checks required',
        interpretation=interpretation))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--label',required=True)
    ap.add_argument('--period',type=float,required=True);ap.add_argument('--dimension',type=int,default=12)
    ap.add_argument('--seed',type=int,default=2141);ap.add_argument('--save-modes',type=int,default=3)
    ap.add_argument('--equilibrium',action='store_true',help='Source must be a corrected equilibrium; requested duration is a Jacobian-power transform, not a cycle period')
    ap.add_argument('--device',type=int,default=0);ap.add_argument('--resume',action='store_true');run(ap.parse_args())
