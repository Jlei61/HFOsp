"""Stationary density response used only as an equilibrium predictor.

This is not a rate ODE. The original voltage/current/refractory density map is
solved at constant recurrent net current. Under-relaxation changes the fixed-
point iteration, not its fixed points; the pseudo-iteration is not a trajectory.
Full network stability must still come from the dynamic density/delay model.
"""
from autonomous_density import *
from population_field_gpu import CODE


class LocalStationaryDensity:
    def __init__(self, theta, population, current, degree=6, dv=.125, device=0, basis_mode='legacy'):
        cp.cuda.Device(device).use()
        self.theta = np.asarray(theta); self.population = np.asarray(population, np.uint8)
        self.u = np.asarray(current); self.P = len(self.theta); self.degree = degree; self.dv = dv
        self.prep = read(OPERATORS/'prepared.json'); p = self.prep['params']
        self.basis_mode=basis_mode
        if basis_mode=='high_precision':
            from stable_stationary_basis import StationaryBasis
            self.basis=StationaryBasis('E',degree,self.prep['nu_ext_per_ms'])
        else:
            assert basis_mode=='legacy'
            self.basis = MovingBasis('E', degree, self.prep['nu_ext_per_ms'], 'stationary')
        self.A = cp.asarray(self.basis.advance(self.prep['nu_ext_per_ms']))
        self.mass = cp.asarray(self.basis.frame['mass']); self.nodes = cp.asarray(self.basis.frame['nodes'])
        eigenvalues,eigenvectors=np.linalg.eig(cp.asnumpy(self.A))
        at=np.argmin(abs(eigenvalues-1.));marginal=eigenvectors[:,at].real
        marginal/=cp.asnumpy(self.mass)@marginal
        assert abs(eigenvalues[at]-1.)<1e-10 and marginal.min()>0
        self.stationary_noise_marginal=cp.asarray(marginal)
        self.K = len(self.nodes)
        low = np.r_[np.arange(-650.,-30.,40*dv),np.arange(-30.,0.,8*dv)]
        self.edges = np.array([np.r_[low,np.linspace(0,th,round(18/dv)+1)] for th in self.theta])
        self.centers = (self.edges[:,:-1]+self.edges[:,1:])*.5
        self.widths = np.diff(self.edges); self.nv = self.centers.shape[1]
        refs = np.where(self.population==0,round(p['tau_ref_E']/DT),round(p['tau_ref_I']/DT))
        self.width = self.nv+int(refs.max()); self.refs = cp.asarray(refs,dtype=np.int32)
        ratio = np.where(self.population==0,1.,p['tau_m_I']*p['J_ext_I']/(p['tau_m_E']*p['J_ext_E']))
        tm = np.where(self.population==0,p['tau_m_E'],p['tau_m_I'])
        self.ratio = cp.asarray(ratio); self.decay = cp.asarray(np.exp(-DT/tm)); self.drive = cp.asarray(self.u)
        self.de,self.dc,self.dw = [cp.asarray(x) for x in (self.edges,self.centers,self.widths)]
        F = np.zeros((self.P,self.K,self.width)); ids = np.arange(self.P)
        at = (self.centers<=11.).sum(1)-1
        frac = (11.-self.centers[ids,at])/(self.centers[ids,at+1]-self.centers[ids,at])
        mass = cp.asnumpy(self.mass)
        F[ids,:,at] = mass[None,:]*(1-frac[:,None]); F[ids,:,at+1] = mass[None,:]*frac[:,None]
        self.F = cp.asarray(F); self.Q = cp.empty_like(self.F); self.flux = cp.empty((self.P,self.K))
        module = cp.RawModule(code=CODE,options=('--fmad=false',),name_expressions=['voltage'])
        self.voltage = module.get_function('voltage')

    def map(self):
        moved = cp.ascontiguousarray(cp.matmul(self.A,self.F))
        self.voltage((self.P*self.K,),(128,),(moved,self.Q,self.flux,self.de,self.dc,self.dw,
            self.nodes,self.ratio,self.decay,self.drive,self.refs,np.int32(self.K),
            np.int32(self.nv),np.int32(self.width)))
        return self.flux@self.mass*1000/DT

    def residual(self):
        difference = self.Q-self.F
        physical = cp.einsum('k,gkv->gv',self.mass,difference)
        scale = cp.sum(abs(self.F),axis=(1,2))
        return cp.asnumpy(cp.c_[cp.sum(abs(physical),axis=1),
                               cp.sum(abs(difference),axis=(1,2))/cp.maximum(scale,1.)])

    def solve(self, max_iterations=8000, progress=None, tolerance=1e-9, acceleration=False,
              stable_marginal_projection=False):
        started = time.time(); last = started
        history = []; converged = False
        # Anderson acceleration is used only for this stationary root solve.
        # The physical map, its Jacobian and simulated trajectories are unchanged.
        dx_history=[];dr_history=[];previous_start=None;previous_residual=None
        def impose_stationary_marginal():
            if stable_marginal_projection:
                # High-order modal rows can contain large cancelling signed
                # coefficients while their marginal is tiny. Multiplicative
                # normalization then amplifies floating-point marginal errors.
                # An additive constraint projection onto a shared physical PDF
                # imposes the same marginal without dividing by modal mass.
                physical=cp.maximum(cp.einsum('k,gkv->gv',self.mass,self.F),0.)
                distribution=physical/physical.sum(1)[:,None]
                error=self.stationary_noise_marginal[None,:]-self.F.sum(2)
                self.F+=error[:,:,None]*distribution[:,None,:]
            else:
                self.F*=self.stationary_noise_marginal[None,:,None]/cp.sum(self.F,axis=2)[:,:,None]
        if acceleration:
            impose_stationary_marginal()
            macro_start=self.F.copy()
        for it in range(max_iterations):
            rate = self.map()
            if (it+1)%100==0:
                residual = self.residual()
                history.append([it+1,float(residual[:,0].max()),float(residual[:,1].max())])
                if max(residual.max(0)) < tolerance:
                    converged = True; break
                if progress and time.time()-last>20:
                    progress(it+1,residual,cp.asnumpy(rate)); last = time.time()
            self.F *= .5; self.F += .5*self.Q
            if acceleration and (it+1)%20==0:
                residual_macro=self.F-macro_start
                if previous_start is not None:
                    dx_history.append(macro_start-previous_start)
                    dr_history.append(residual_macro-previous_residual)
                    if len(dx_history)>3:dx_history.pop(0);dr_history.pop(0)
                    n=len(dx_history);gram=cp.empty((self.P,n,n));rhs=cp.empty((self.P,n))
                    for j in range(n):
                        rhs[:,j]=cp.sum(dr_history[j]*residual_macro,axis=(1,2))
                        for k in range(j+1):
                            gram[:,j,k]=gram[:,k,j]=cp.sum(dr_history[j]*dr_history[k],axis=(1,2))
                    scale=cp.maximum(cp.trace(gram,axis1=1,axis2=2),1e-300)
                    gram+=cp.eye(n)[None,:,:]*(1e-11*scale[:,None,None])
                    gamma=cp.linalg.solve(gram,rhs[:,:,None])[:,:,0]
                    ok=cp.isfinite(gamma).all(1)&(cp.sum(abs(gamma),axis=1)<100.)
                    gamma=cp.where(ok[:,None],gamma,0.)
                    for j in range(n):self.F-=gamma[:,j,None,None]*(dx_history[j]+dr_history[j])
                # These marginals are fixed analytically by stationary private
                # noise. Anderson combinations must not drift along the neutral
                # probability-mass directions and converge to a wrong mass.
                impose_stationary_marginal()
                previous_start=macro_start;previous_residual=residual_macro
                macro_start=self.F.copy()
        rate = self.map(); residual = self.residual()
        physical = cp.einsum('k,gkv->gv',self.mass,self.F)
        diagnostics = dict(iterations=it+1,converged=converged,seconds=time.time()-started,
            stationary_solver='Anderson3 over 20 relaxed iterations' if acceleration else 'under-relaxed fixed point',
            marginal_projection='Additive shared physical PDF' if stable_marginal_projection else 'Multiplicative modal normalization',
            max_stationary_residual=float(residual.max()),
            maximum_mass_error=float(cp.max(abs(physical.sum(1)-1)).get()),
            max_negative_probability=float(cp.maximum(-physical,0).sum(1).max().get()),
            minimum_rate_hz=float(rate.min().get()),finite=bool(cp.isfinite(self.F).all().get()))
        return cp.asnumpy(rate),residual,diagnostics,np.asarray(history)


def main(args):
    folder = OUT/'stationary_response'/f'degree{args.degree}_dv{args.dv:g}'
    folder.mkdir(parents=True,exist_ok=False)
    currents = np.unique(np.r_[np.arange(-8.,12.001,.25),np.arange(13.,61.,1.),
                               np.arange(70.,201.,10.),np.arange(250.,601.,50.),
                               np.arange(800.,2001.,200.)])
    thresholds = np.r_[np.arange(14.,18.001,.5),18.]
    populations = np.r_[np.zeros(len(thresholds)-1,np.uint8),np.ones(1,np.uint8)]
    theta = np.repeat(thresholds,len(currents)); pop = np.repeat(populations,len(currents))
    bias = np.tile(currents,len(thresholds))
    model = LocalStationaryDensity(theta,pop,bias,args.degree,args.dv,args.device)
    cfg = dict(degree=args.degree,dv=args.dv,dt_ms=DT,nu_per_ms=model.prep['nu_ext_per_ms'],
        thresholds=thresholds,populations=populations,currents_mv=currents,conditions=len(theta),
        meaning='Stationary density solver / equilibrium predictor; no fitted rate dynamics or stability claim',
        iteration='Under-relaxed fixed-point solve of the original local density map')
    write(folder/'config.json',cfg)
    def progress(it,residual,rate):
        write(folder/'status.json',dict(status='RUNNING',pid=os.getpid(),iteration=it,
            maximum_stationary_residual=float(residual.max()),max_rate_hz=float(rate.max())))
        print('stationary iteration',it,'max residual',residual.max(),flush=True)
    rate,residual,diagnostics,history = model.solve(args.iterations,progress)
    shape=(len(thresholds),len(currents))
    np.savez_compressed(folder/'response.npz',rate_hz=rate.reshape(shape),
        residual=residual.reshape(*shape,2),theta=thresholds,population=populations,
        current_mv=currents,iteration_history=history)
    np.savez_compressed(folder/'density_state.npz',F=cp.asnumpy(model.F),theta=theta,population=pop,
                        current_mv=bias,edges=model.edges,noise_mass=cp.asnumpy(model.mass))
    write(folder/'status.json',dict(status='STATIONARY_SOLVED' if diagnostics['converged'] else 'PARTIAL_STATIONARY_CONVERGENCE',
        diagnostics=diagnostics,network_equilibrium='NOT_YET_SOLVED',stability='NOT_COMPUTED'))
    print(folder,diagnostics,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--degree',type=int,default=6)
    ap.add_argument('--dv',type=float,default=.125);ap.add_argument('--device',type=int,default=0)
    ap.add_argument('--iterations',type=int,default=8000);main(ap.parse_args())
