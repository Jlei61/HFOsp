"""Find missing equilibrium seeds by a numerical current homotopy.

The auxiliary lambda is a solver coordinate, never a plotted physical
parameter. Only lambda=1 solves the original selected-g40 equilibrium.
"""
from equilibrium_predictor import *


class CurrentHomotopy:
    def __init__(self, problem, D, seed):
        self.p=problem;self.D=D
        z,_=problem.resource(D)
        self.B=(problem.A-sparse.diags(z)@problem.G-problem.Mcoupling).tocsr()
        low=np.full(problem.P,problem.current_grid[0])
        high=np.full(problem.P,problem.current_grid[-1])
        floor=problem.response(low)[0];ceiling=problem.response(high)[0]
        self.seed=np.clip(seed,floor+1e-8,ceiling-1e-8)
        for _ in range(60):
            middle=(low+high)/2;fm=problem.response(middle)[0]
            low=np.where(fm<self.seed,middle,low);high=np.where(fm>=self.seed,middle,high)
        self.anchor=(low+high)/2
        self.seed=problem.response(self.anchor)[0]

    def equations(self,r,lam):
        br=self.B@r
        f,fp=self.p.response(lam*br+(1-lam)*self.anchor)
        return r-f,self.p.I-sparse.diags(lam*fp)@self.B,-fp*(br-self.anchor)

    def tangent(self,r,lam,previous=None):
        _,J,fl=self.equations(r,lam)
        if previous is None:
            tr=spsolve(J,-fl);td=1.
        else:
            row=sparse.csr_matrix(np.r_[self.p.weights*previous[0]/10000,previous[1]][None,:])
            mat=sparse.vstack([sparse.hstack([J,fl[:,None]]),row]).tocsr()
            ans=spsolve(mat,np.r_[np.zeros(self.p.P),1.]);tr,td=ans[:-1],ans[-1]
        norm=np.sqrt(np.dot(self.p.weights,tr*tr)/10000+td*td)
        return tr/norm,td/norm

    def correct(self,r,lam,tangent,ds):
        tr,td=tangent;pred=r+ds*tr;pl=lam+ds*td;x=pred.copy();l=pl
        for it in range(15):
            f,J,fl=self.equations(x,l)
            arc=np.dot(self.p.weights*tr,x-pred)/10000+td*(l-pl)
            err=np.max(abs(f))+100*abs(arc)
            if np.max(abs(f))<1e-7 and abs(arc)<1e-10:return x,l,it
            row=sparse.csr_matrix(np.r_[self.p.weights*tr/10000,td][None,:])
            mat=sparse.vstack([sparse.hstack([J,fl[:,None]]),row]).tocsr()
            dx=spsolve(mat,-np.r_[f,arc]);ok=False
            for a in 2.**-np.arange(14):
                nr=x+a*dx[:-1];nl=l+a*dx[-1]
                nf=self.equations(nr,nl)[0]
                na=np.dot(self.p.weights*tr,nr-pred)/10000+td*(nl-pl)
                if np.max(abs(nf))+100*abs(na)<err:
                    x,l=nr,nl;ok=True;break
            if not ok:return None
        return None


class OutputHomotopy(CurrentHomotopy):
    def __init__(self,problem,D,seed):
        self.p=problem;self.D=D;self.seed=np.asarray(seed).copy()
        z,_=problem.resource(D)
        self.B=(problem.A-sparse.diags(z)@problem.G-problem.Mcoupling).tocsr()

    def equations(self,r,lam):
        f,fp=self.p.response(self.B@r)
        return r-lam*f-(1-lam)*self.seed, self.p.I-sparse.diags(lam*fp)@self.B, self.seed-f


def run(args):
    folder=OUT/'equilibrium_predictors'/args.label;folder.mkdir(parents=True,exist_ok=False)
    p=EquilibriumProblem(OUT/'stationary_response/degree6_dv0.125',args.theta_interpolation)
    if args.resume:
        resumed=dict(np.load(args.resume/'checkpoint.npz'))
        seed=resumed['anchor_rate'];D=float(resumed['D'])
        previous_config=read(args.resume/'config.json')
        assert previous_config['kind']==args.kind and previous_config['theta_interpolation']==args.theta_interpolation
    elif args.seed:
        with np.load(args.seed) as z:seed=z['rate_hz'];D=float(z['D'])
    else:
        D=args.D;seed=p.response(np.zeros(p.P))[0]
    hom=(CurrentHomotopy if args.kind=='current' else OutputHomotopy)(p,D,seed)
    r=hom.seed;lam=0.;tr,td=hom.tangent(r,lam)
    if args.resume:
        r=resumed['rate_hz'];lam=float(resumed['lambda_solver'])
        assert np.max(abs(hom.equations(r,lam)[0]))<1e-7
        tr,td=hom.tangent(r,lam,(resumed['tangent_rate'],float(resumed['tangent_lambda'])))
    ds=.01;rows=[];rates=[];status='POINT_LIMIT';started=time.time()
    write(folder/'config.json',dict(D=D,kind=args.kind,theta_interpolation=args.theta_interpolation,
        seed_source=str(args.seed) if args.seed else 'uncoupled local stationary rates',resumed_from=str(args.resume) if args.resume else None,max_component_step_hz=args.max_rate_step,
        minimum_tangent_cosine=args.minimum_tangent_cosine,maximum_relative_corrector=args.maximum_relative_corrector))
    for it in range(args.points):
        row=dict(index=it,lambda_solver=lam,D=D,mean_E_hz=p.eweights@r,
                 tangent_lambda=td,residual_hz=float(np.max(abs(hom.equations(r,lam)[0]))))
        rows.append(row);rates.append(r.copy());write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),latest=row,wall_s=time.time()-started))
        if it%10==0:np.savez_compressed(folder/'checkpoint.npz',D=D,rate_hz=r,lambda_solver=lam,tangent_rate=tr,tangent_lambda=td,anchor_rate=hom.seed)
        print(row,flush=True)
        if lam>=1:
            root,qa=p.fixed_D(D,r,maxit=100)
            write(folder/'original_model_root_qa.json',qa)
            if qa['converged'] and root.min()>-1e-7:
                np.savez_compressed(folder/'seed.npz',D=D,rate_hz=root)
                status='ORIGINAL_MODEL_ROOT_FOUND';break
        if lam < -.25 or lam>1.25:status='AUXILIARY_RANGE_LIMIT';break
        result=None;effective=min(ds,args.max_rate_step/max(np.max(abs(tr)),1e-30))
        next_tangent=None
        for _ in range(15):
            result=hom.correct(r,lam,(tr,td),effective)
            if result is not None and np.max(abs(result[0]-r))<=1.5*args.max_rate_step:
                nr,nl,its=result
                nt=hom.tangent(nr,nl,(tr,td))
                cosine=np.dot(p.weights*tr,nt[0])/10000.+td*nt[1]
                correction=np.sqrt(np.dot(p.weights,(nr-r-effective*tr)**2)/10000.+(nl-lam-effective*td)**2)/effective
                if cosine>=args.minimum_tangent_cosine and correction<=args.maximum_relative_corrector:
                    next_tangent=nt;break
            result=None
            effective*=.5
        if result is None:status='CORRECTOR_FAILED';break
        nr,nl,its=result
        tr,td=next_tangent;r,lam=nr,nl
        ds=min(.03,effective*(1.3 if its<=3 else 1.))
        if effective<1e-7:status='STEP_TOO_SMALL';break
    np.savez_compressed(folder/'last_auxiliary_state.npz',D=D,rate_hz=r,lambda_solver=lam)
    np.savez_compressed(folder/'trajectory.npz',rate_hz=np.asarray(rates),lambda_solver=np.asarray([q['lambda_solver'] for q in rows]))
    write(folder/'homotopy.json',dict(status=status,points=rows,wall_s=time.time()-started,
        meaning='Auxiliary numerical solve only; no physical branch or stability claim'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=Path);ap.add_argument('--D',type=float,default=.25)
    ap.add_argument('--resume',type=Path)
    ap.add_argument('--kind',choices=['current','output'],default='current')
    ap.add_argument('--theta-interpolation',choices=['linear','pchip'],default='linear')
    ap.add_argument('--max-rate-step',type=float,default=20.)
    ap.add_argument('--minimum-tangent-cosine',type=float,default=.9)
    ap.add_argument('--maximum-relative-corrector',type=float,default=.5)
    ap.add_argument('--label',default='middle_current_homotopy');ap.add_argument('--points',type=int,default=400)
    run(ap.parse_args())
