"""Pseudo-arclength predictors for equilibria of the dynamic density model.

Only the stationary local density response is tabulated. No rate time constant
or rate-ODE eigenvalues are used. Each predictor must be corrected/verified in
the full density model before receiving a stability or bifurcation label.
"""
from autonomous_density import *
from scipy import sparse
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, least_squares
from scipy.sparse.linalg import spsolve


class EquilibriumProblem:
    def __init__(self, table, theta_interpolation='linear'):
        self.table_path = Path(table)
        self.theta_interpolation=theta_interpolation
        data = np.load(self.table_path/'response.npz')
        self.geo = dict(np.load(OPERATORS/'geometry.npz'))
        self.prep = read(OPERATORS/'prepared.json'); p = self.prep['params']
        self.P = len(self.geo['population']); self.e = self.geo['population']==0
        self.theta = self.geo['threshold_mv']; self.weights = self.geo['group_size']/40000.
        self.eweights = self.geo['group_size']*self.e/32000.
        self.rscale = 100.
        curves = data['rate_hz']; self.current_grid = data['current_mv']
        self.negative_table_min = float(curves.min())
        assert curves.min() > -1e-7, 'Local stationary predictor has material negative rates.'
        self.tables = [PchipInterpolator(self.current_grid,np.maximum(curve,0.),extrapolate=False) for curve in curves]
        self.slopes = [f.derivative() for f in self.tables]
        theta_grid = data['theta'][:-1];self.theta_grid=theta_grid
        assert self.theta[self.e].min()>=theta_grid.min() and self.theta[self.e].max()<=theta_grid.max()
        self.lo = np.clip(np.searchsorted(theta_grid,self.theta)-1,0,len(theta_grid)-2)
        self.hi = self.lo+1
        self.w = (self.theta-theta_grid[self.lo])/(theta_grid[self.hi]-theta_grid[self.lo])
        self.lo[~self.e]=len(curves)-1; self.hi[~self.e]=len(curves)-1; self.w[~self.e]=0.
        tm = np.where(self.e,p['tau_m_E'],p['tau_m_I'])
        mats=[]
        for name,tau in [('ampa',p['tau_r_AMPA']),('gaba',p['tau_r_GABA'])]:
            W=sparse.load_npz(OPERATORS/f'delay_{name}.npz').tocoo()
            total=sparse.coo_matrix((W.data,(W.row,W.col%self.P)),shape=(self.P,self.P)).tocsr()
            mats.append(sparse.diags(tm/tau/(1-np.exp(-DT/tau))*DT/1000.)@total)
        self.A,self.G=mats
        self.Mcoupling=sparse.diags(.0005*self.e)  # tau_M=1000 ms -> M*=rate_Hz.
        self.I=sparse.eye(self.P,format='csr')
        src=ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/runs/eta0.0005_s9108401/checkpoints/t9420ms.npz'
        with np.load(src) as z:self.h=-np.log(z['slow__z'][:32000])
        self.e_group=self.geo['cell_group'][:32000]

    def resource(self,D):
        assert 0<=D<=1
        if D==1:return np.where(self.e,0.,1.),np.zeros(self.P)
        if D==0:alpha=0.
        else:
            upper=1.
            while np.exp(-upper*self.h).mean()>1-D:upper*=2
            alpha=brentq(lambda a:np.exp(-a*self.h).mean()-(1-D),0.,upper,xtol=1e-13)
        z=np.exp(-alpha*self.h);dz=-self.h*z/np.mean(self.h*z)
        counts=self.geo['group_size']
        zg=np.bincount(self.e_group,weights=z,minlength=self.P)/counts
        derivative=np.bincount(self.e_group,weights=dz,minlength=self.P)/counts
        zg[~self.e]=1.
        assert abs(1-self.eweights@zg-D)<1e-12
        return zg,derivative

    def response(self,u):
        inside=(u>=self.current_grid[0])&(u<=self.current_grid[-1])
        v=np.clip(u,self.current_grid[0],self.current_grid[-1]);ids=np.arange(self.P)
        rates=np.asarray([f(v) for f in self.tables]);slope=np.asarray([f(v) for f in self.slopes])
        f=(1-self.w)*rates[self.lo,ids]+self.w*rates[self.hi,ids]
        df=((1-self.w)*slope[self.lo,ids]+self.w*slope[self.hi,ids])*inside
        if self.theta_interpolation=='pchip':
            active=self.e&(self.w>0)&(self.w<1)
            ai=np.flatnonzero(active);lo=self.lo[active];dx=self.theta[active]-self.theta_grid[lo]
            def interpolate(values):
                coeff=PchipInterpolator(self.theta_grid,values[:-1,active],axis=0).c
                c=coeff[:,lo,np.arange(len(ai))]
                return ((c[0]*dx+c[1])*dx+c[2])*dx+c[3]
            f[active]=interpolate(rates)
            # Differentiate the actual shape-preserving theta interpolation,
            # including its slope selection. This is a predictor Jacobian only.
            eps=1e-4
            plus=np.asarray([fn(np.clip(v+eps,self.current_grid[0],self.current_grid[-1])) for fn in self.tables])
            minus=np.asarray([fn(np.clip(v-eps,self.current_grid[0],self.current_grid[-1])) for fn in self.tables])
            df[active]=(interpolate(plus)-interpolate(minus))/(2*eps)*inside[active]
        return f,df

    def equations(self,r,D,jacobian=True):
        z,dz=self.resource(D)
        B=(self.A-sparse.diags(z)@self.G-self.Mcoupling).tocsr()
        u=B@r;f,fp=self.response(u)
        residual=r-f
        if not jacobian:return residual
        J=self.I-sparse.diags(fp)@B
        FD=fp*dz*(self.G@r)
        return residual,J,FD,u

    def fixed_D(self,D,initial,maxit=50):
        r=np.asarray(initial).copy();history=[]
        for it in range(maxit):
            F,J,_,u=self.equations(r,D);error=np.linalg.norm(F,np.inf);history.append(error)
            if error<1e-8:return r,dict(converged=True,iterations=it,residual=error)
            delta=spsolve(J,-F);accepted=False
            for a in 2.**-np.arange(16):
                trial=r+a*delta
                # Newton iterates are algebraic guesses, not simulated rates.
                # Positivity follows from r=f(u) at the accepted root. Rejecting
                # every negative predictor can pin the solve at silent cells.
                err=np.linalg.norm(self.equations(trial,D,False),np.inf)
                if err<error:
                    r=trial;accepted=True;break
            if not accepted:break
        return r,dict(converged=False,iterations=it,residual=history[-1],history=history)

    def tangent(self,r,D,previous=None):
        _,J,FD,_=self.equations(r,D)
        if previous is None:
            tr=spsolve(J,-FD);td=1.
        else:
            tr0,td0=previous
            row=sparse.csr_matrix(np.r_[self.weights*tr0/self.rscale**2,td0][None,:])
            B=sparse.vstack([sparse.hstack([J,FD[:,None]]),row]).tocsr()
            t=spsolve(B,np.r_[np.zeros(self.P),1.]);tr,td=t[:-1],t[-1]
        norm=np.sqrt(np.dot(self.weights,tr*tr)/self.rscale**2+td*td)
        tr/=norm;td/=norm
        return tr,td

    def arclength(self,r,D,tangent,ds):
        tr,td=tangent;pred=r+ds*tr;pd=D+ds*td
        x=pred.copy();d=pd
        for it in range(12):
            if not 0<=d<=1:return None
            F,J,FD,_=self.equations(x,d)
            arc=np.dot(self.weights*tr,x-pred)/self.rscale**2+td*(d-pd)
            if np.linalg.norm(F,np.inf)<1e-7 and abs(arc)<1e-10:return x,d,it
            row=sparse.csr_matrix(np.r_[self.weights*tr/self.rscale**2,td][None,:])
            B=sparse.vstack([sparse.hstack([J,FD[:,None]]),row]).tocsr()
            delta=spsolve(B,-np.r_[F,arc]);accepted=False
            norm=np.linalg.norm(F,np.inf)+abs(arc)*100.
            for a in 2.**-np.arange(12):
                nr=x+a*delta[:-1];nd=d+a*delta[-1]
                if not 0<=nd<=1:continue
                nf=self.equations(nr,nd,False)
                na=np.dot(self.weights*tr,nr-pred)/self.rscale**2+td*(nd-pd)
                if np.linalg.norm(nf,np.inf)+abs(na)*100<norm:
                    x,d=nr,nd;accepted=True;break
            if not accepted:return None
        return None


def main(args):
    folder=OUT/'equilibrium_predictors'/(args.run_label or args.branch)
    folder.mkdir(parents=True,exist_ok=False)
    problem=EquilibriumProblem(args.table,args.theta_interpolation)
    if args.seed:
        with np.load(args.seed) as z:
            seed=z['rate_hz'];D=float(z['D'])
    elif args.branch=='low':
        D=0.
        old=OUT/'qualification/D0.000000_degree6_dv0.125_1000ms/checkpoint.npz'
        with np.load(old) as z:
            seed=z['history'][(int(z['step_index'])-1)%len(z['history'])]*10000.
    elif args.branch=='high':
        D=1.;seed=np.where(problem.e,500.,700.)
    else:
        D=.25
        source=OUT/'particle_controls/selected_g40/D0.250000_Nscale1_seed1901_4000ms_microscopic/trajectory.npz'
        with np.load(source) as f:mean_E=f['field_1ms'][2000:].mean(0)
        seed=np.full(problem.P,20.)
        seed[problem.e]=mean_E[problem.geo['group_cell'][problem.e]]
        # Solve inhibitory self-consistency with E held at the measured spatial
        # pattern, then release all populations for the actual root solve.
        i=np.flatnonzero(~problem.e)
        def inhibitory(x,jac=False):
            trial=seed.copy();trial[i]=x
            if jac:return problem.equations(trial,D)[1][i][:,i]
            return problem.equations(trial,D,False)[i]
        fit=least_squares(inhibitory,seed[i],jac=lambda x:inhibitory(x,True),
            bounds=(0.,1000.),x_scale='jac',max_nfev=160,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        seed[i]=fit.x
        print('inhibitory seed solve',fit.nfev,'max residual',np.max(abs(fit.fun)),flush=True)
        caps=np.where(problem.e,500.,1000.)
        fit=least_squares(lambda x:problem.equations(x,D,False),np.clip(seed,1e-7,caps-1e-7),
            jac=lambda x:problem.equations(x,D)[1],bounds=(np.zeros(problem.P),caps),
            x_scale='jac',max_nfev=200,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        seed=fit.x
        print('coupled spatial seed solve',fit.nfev,'max residual',np.max(abs(fit.fun)),flush=True)
    r,qa=problem.fixed_D(D,seed)
    write(folder/'seed.json',dict(D=D,solver=qa,table=str(args.table),
        theta_interpolation=args.theta_interpolation,
        meaning='Stationary-density interpolation predictor; stability and full-state correction pending'))
    if not qa['converged']:
        np.savez_compressed(folder/'failed_seed.npz',rate_hz=r,D=D);raise RuntimeError(qa)
    assert r.min()>=-1e-7
    records=[];rates=[];Ds=[];tr,td=problem.tangent(r,D)
    if args.branch in ('high','middle_down'):tr=-tr;td=-td
    ds=.005;status='POINT_LIMIT'
    for step in range(args.points):
        residual,_,_,u=problem.equations(r,D)
        record=dict(index=step,D=D,mean_E_hz=problem.eweights@r,
            max_stationary_interpolation_residual_hz=np.max(abs(residual)),tangent_D=td,
            outside_current_table=int(np.sum((u<problem.current_grid[0])|(u>problem.current_grid[-1]))),
            stability='NOT_COMPUTED',bifurcation_type='NOT_CLASSIFIED')
        records.append(record);rates.append(r.copy());Ds.append(D)
        assert r.min()>=-1e-6
        write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),latest=record))
        print(args.branch,step,'D',round(D,7),'rate',round(record['mean_E_hz'],6),'tD',round(td,6),flush=True)
        if step>2 and ((D<1e-5 and td<0) or (D>1-1e-5 and td>0)):
            status='PHYSICAL_BOUNDARY';break
        effective=min(ds,args.max_rate_step/max(np.max(abs(tr)),1e-30))
        if td>0 and D+effective*td>1:effective=(1-D)/td
        if td<0 and D+effective*td<0:effective=-D/td
        found=None;next_tangent=None
        for attempt in range(15):
            found=problem.arclength(r,D,(tr,td),effective)
            if found is not None:
                nr,nd,iterations=found;nt=problem.tangent(nr,nd,(tr,td))
                cosine=np.dot(problem.weights*tr,nt[0])/problem.rscale**2+td*nt[1]
                correction=np.sqrt(np.dot(problem.weights,(nr-r-effective*tr)**2)/problem.rscale**2+(nd-D-effective*td)**2)/effective
                if cosine>=args.minimum_tangent_cosine and correction<=.5 and np.max(abs(nr-r))<=1.5*args.max_rate_step:
                    next_tangent=nt;break
                found=None
            effective*=.5
        if found is None:status='CORRECTOR_FAILED';break
        nr,nd,iterations=found
        tr,td=next_tangent;r,D=nr,nd
        ds=min(.02,effective*(1.25 if iterations<=3 else 1.))
        if effective<1e-7:status='STEP_TOO_SMALL';break
    np.savez_compressed(folder/'branch.npz',D=np.asarray(Ds),rate_hz=np.asarray(rates))
    write(folder/'branch.json',dict(status=status,points=records,
        scope='Uncertified equilibrium predictors, not a delivered bifurcation branch',
        missing=['direct full-density stationary residual and mesh correction',
            'dynamic stability with delay and M','critical point nondegeneracy and spatial mode']))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--table',type=Path,default=OUT/'stationary_response/degree6_dv0.125')
    ap.add_argument('--branch',choices=['low','high','middle_up','middle_down'],default='low');ap.add_argument('--points',type=int,default=180)
    ap.add_argument('--seed',type=Path);ap.add_argument('--run-label')
    ap.add_argument('--theta-interpolation',choices=['linear','pchip'],default='linear')
    ap.add_argument('--max-rate-step',type=float,default=10.)
    ap.add_argument('--minimum-tangent-cosine',type=float,default=.9)
    main(ap.parse_args())
