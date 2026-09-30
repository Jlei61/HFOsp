"""Continue genuine cycle-fold and period-doubling conditions in heterogeneity.

Periodic orbits retain the original Fourier residual and physical delay kernels.
LP uses a zero parameter component of the branch tangent; PD uses an
anti-periodic variational null mode. Neither is inferred from rate extrema.
"""
from common import *
sys.path.append(str(ROOT/'scripts/topic4_core_branch_connections_v5'))
from folds import Chart as BaseChart,metric
from antiperiodic import build
from analytic_orbit import Orbit
import folds,antiperiodic
folds.Orbit=Orbit
antiperiodic.Orbit=Orbit
from scipy.signal import resample
from scipy.sparse.linalg import eigs,LinearOperator,gmres
from scipy.linalg import lu_factor,lu_solve
from scipy.optimize import root_scalar
import argparse,csv,time


class Chart(BaseChart):
    """Fourier coarse-space preconditioning of the unchanged bordered Jacobian."""
    def preconditioner(self,B,coarse=128):
        N=self.N;nc=min(coarse,N);dim=6*nc+2
        def restrict(x):return np.r_[resample(x[:-2].reshape(N,6),nc,axis=0).ravel(),x[-2:]]
        def prolong(x):return np.r_[resample(x[:-2].reshape(nc,6),N,axis=0).ravel(),x[-2:]]
        A=np.empty((dim,dim));e=np.zeros(dim)
        for j in range(dim):
            e[j]=1;A[:,j]=restrict(B@prolong(e));e[j]=0
        lu=lu_factor(A)
        def mv(x):
            coarse_x=restrict(x)
            return x+prolong(lu_solve(lu,coarse_x)-coarse_x)
        return LinearOperator(B.shape,matvec=mv,dtype=float)

    def solve(self,z,coordinate):
        z=z.copy();hist=[];pre=None
        for k in range(18):
            F,B,J,b=self.evaluate(z,coordinate);err=float(abs(F).max());hist.append(err)
            if err<5e-10:break
            if pre is None or k in (5,10):pre=self.preconditioner(B)
            dz,info=gmres(B,-F,M=pre,rtol=min(1e-7,max(1e-10,err*.005)),atol=1e-12,restart=180,maxiter=12)
            linerr=np.linalg.norm(B@dz+F)/max(np.linalg.norm(F),1e-20)
            if linerr>1e-4:
                pre=self.preconditioner(B,256)
                dz,info=gmres(B,-F,M=pre,rtol=1e-8,atol=1e-12,restart=250,maxiter=15)
                linerr=np.linalg.norm(B@dz+F)/max(np.linalg.norm(F),1e-20)
            if linerr>.01:raise RuntimeError(('linear solve inaccurate',linerr,hist))
            alpha=1.
            for _ in range(16):
                trial=z+alpha*dz
                if np.linalg.norm(self.evaluate(trial,coordinate,False))<np.linalg.norm(F):break
                alpha*=.5
            else:raise RuntimeError(('preconditioned line search',hist))
            z=trial
        else:raise RuntimeError(('preconditioned no convergence',hist))
        F,B,J,b=self.evaluate(z,coordinate)
        if pre is None:pre=self.preconditioner(B)
        rhs=np.zeros(len(z));rhs[-1]=1
        tan,info=gmres(B,rhs,M=pre,rtol=2e-9,atol=1e-11,restart=180,maxiter=15)
        if np.linalg.norm(B@tan-rhs)>1e-7:raise RuntimeError('preconditioned tangent failed')
        tan/=np.sqrt(metric(tan,tan,self.N))
        return z,tan,float(abs(F).max()),B,J,b


def seeds():
    out={}
    p=ROOT/'results/topic4_sef_hfo/core_network_bifurcation_v7_20260916/critical_points.csv'
    for row in csv.DictReader(p.open()):
        name=row['label']
        if name=='Cycle fold':name='LPC_onset'
        if name in ['LPC_onset','LP0a','LP0b','LP0c','LP1','PD0','PD1','PD2','PD3']:
            out[name]=resolve(row['source'])
    return out


def load_seed(path,N):
    z=np.load(path);r=resample(z['r'],N,axis=0)
    q=np.r_[(r/.01).ravel(),np.log(float(z['T'])),float(z['g'])/.01]
    if 'tangent' in z:
        v=z['tangent'];t=np.r_[resample(v[:-2].reshape(-1,6),N,axis=0).ravel(),v[-2:]]
    elif 'right_null' in z:
        v=z['right_null'];t=np.r_[resample(v[:-1].reshape(-1,6),N,axis=0).ravel(),v[-1],0.]
    else:t=np.eye(1,len(q),len(q)-1).ravel()
    t/=np.sqrt(metric(t,t,N))
    return q,t


def fold_derivative(h,z,t,N):
    """Differentiate F=0 and the fold solvability condition along h."""
    ch=Chart(System(h),z,t,N,.01);F,B,J,b=ch.evaluate(z);pre=ch.preconditioner(B)
    eps=1e-5;side=-1 if h>=.5 else 1
    other=Chart(System(h+side*eps),z,t,N,.01)
    Fh=(other.evaluate(z,jac=False)[:-1]-F[:-1])/(side*eps)
    rhs=np.r_[-Fh,0.]
    u,info=gmres(B,rhs,M=pre,rtol=1e-9,atol=1e-11,restart=180,maxiter=15)
    if np.linalg.norm(B@u-rhs)>1e-7:raise RuntimeError('fold h particular derivative failed')
    unit=np.zeros(len(z));unit[-1]=1
    w,info=gmres(B.T,unit,rtol=1e-9,atol=1e-11,restart=250,maxiter=20)
    if np.linalg.norm(B.T@w-unit)>1e-7:raise RuntimeError('fold left null solve failed')
    left=w[:-1];v=t[:-1]
    def jv(chart,q):return chart.evaluate(q)[2]@v
    Jhv=(jv(other,z)-J@v)/(side*eps)
    dJu=(jv(ch,z+eps*u)-jv(ch,z-eps*u))/(2*eps)
    dJt=(jv(ch,z+eps*t)-jv(ch,z-eps*t))/(2*eps)
    den=left@dJt
    if abs(den)<1e-9:raise RuntimeError('Degenerate fold curvature')
    shift=-float(left@(Jhv+dJu))/den;dz=u+shift*t
    rhs=np.r_[-(Jhv+dJu+shift*dJt),0.]
    dv,info=gmres(B,rhs,M=pre,rtol=1e-8,atol=1e-10,restart=180,maxiter=15)
    dv[-1]=0
    return dz,dv


def orbit_derivative(h,z,N):
    t=np.zeros_like(z);t[-1]=1
    ch=Chart(System(h),z,t,N,.01);F,B,J,b=ch.evaluate(z);pre=ch.preconditioner(B)
    eps=1e-5;side=-1 if h>=.5 else 1
    other=Chart(System(h+side*eps),z,t,N,.01)
    Fh=(other.evaluate(z,jac=False)-F)/(side*eps)
    u,info=gmres(B,-Fh,M=pre,rtol=1e-9,atol=1e-11,restart=180,maxiter=15)
    if np.linalg.norm(B@u+Fh)>1e-7:raise RuntimeError('orbit h predictor failed')
    rhs=np.zeros(len(z));rhs[-1]=1
    ug,info=gmres(B,rhs,M=pre,rtol=1e-9,atol=1e-11,restart=180,maxiter=15)
    def anti(q,hh):return build(System(hh),q[:-2].reshape(N,6)*.01,np.exp(q[-2]),q[-1]*.01)
    K=anti(z,h)
    vals,vec=eigs(K,k=48,which='LM',ncv=100,tol=1e-9,maxiter=600)
    j=np.argmin(abs(vals-1));v=vec[:,j].real
    val2,vec2=eigs(K.T,k=48,which='LM',ncv=100,tol=1e-9,maxiter=600)
    w=vec2[:,np.argmin(abs(val2-1))].real;w/=w@v
    # Forward/backward derivative uses the available side of the bounded h domain.
    khv=(anti(z+side*eps*u,h+side*eps)@v-K@v)/(side*eps)
    kgv=(anti(z+eps*ug,h)@v-anti(z-eps*ug,h)@v)/(2*eps)
    den=w@kgv
    if abs(den)<1e-9:raise RuntimeError('PD transversality too small')
    delta=-float(w@khv)/den
    return u+delta*ug,np.zeros_like(t)


def correct_fold(h,z,normal,N):
    chart=Chart(System(h),z,normal,N,.01);cache={}
    def at(x):
        key=float(x)
        if key not in cache:
            if cache:
                old=min(cache,key=lambda k:abs(k-key));guess=cache[old][0]+(key-old)*cache[old][1]
            else:guess=z+key*normal
            q,t,err,*_=chart.solve(guess,key)
            if err>1e-8:raise RuntimeError(('orbit residual',err))
            cache[key]=(q,t,err)
        return cache[key]
    def fn(x):return float(at(x)[1][-1]*.01)
    x=0.;f=fn(x)
    for it in range(10):
        if abs(f)<1e-15:break
        eps=.003;der=(fn(x+eps)-fn(x-eps))/(2*eps)
        if abs(der)<1e-12:raise RuntimeError('Fold curvature too small')
        step=np.clip(-f/der,-2.,2.);trial=x+step;ft=fn(trial)
        if abs(ft)>abs(f):
            for _ in range(6):
                step*=.5;trial=x+step;ft=fn(trial)
                if abs(ft)<abs(f):break
        x,f=trial,ft
    q,t,err=at(x)
    if abs(f)>1e-11:raise RuntimeError(('fold critical residual',f))
    F,B,J,b=chart.evaluate(q,x)
    null_error=float(abs(J@t[:-1]).max())
    if null_error>2e-6:raise RuntimeError(('fold null error',null_error))
    bracket=[fn(x-.003),fn(x+.003)]
    if bracket[0]*bracket[1]>=0:raise RuntimeError(('No cycle-fold tangent sign change',bracket))
    row=dict(h=h,g=float(q[-1]*.01),T_ms=float(np.exp(q[-2])),N=N,
             orbit_residual=err,critical_residual=abs(f),null_residual=null_error,
             criterion='Cycle-fold tangent dJ/ds = 0; nontrivial periodic null mode',
             bracket_tangent=bracket)
    return q,t,row


def correct_pd(h,z,N):
    normal=np.zeros_like(z);normal[-1]=1
    chart=Chart(System(h),z,normal,N,.01);cache={};s=chart.s
    def at(g):
        key=float(g);x=(key-z[-1]*.01)/.01
        if key not in cache:
            old=min(cache,key=lambda k:abs(k-key)) if cache else None
            guess=cache[old][0].copy() if old is not None else z.copy();guess[-1]=key/.01
            q,t,err,*_=chart.solve(guess,x)
            if err>1e-8:raise RuntimeError(('orbit residual',err))
            r=q[:-2].reshape(N,6)*.01;T=np.exp(q[-2]);K=build(s,r,T,key)
            vals,vec=eigs(K,k=48,which='LM',ncv=100,tol=2e-9,maxiter=600,
                          v0=np.random.default_rng(51).normal(size=6*N))
            ix=np.argmin(abs(vals-1));val=vals[ix]
            if abs(val.imag)>1e-5 or abs(val-1)>.3:raise RuntimeError(('No nearby real anti mode',val))
            v=vec[:,ix].real;v/=np.linalg.norm(v)
            cache[key]=(q,t,err,float(val.real-1),float(abs(K@v-val.real*v).max()),v)
        return cache[key]
    g=float(z[-1]*.01);f=at(g)[3]
    for it in range(10):
        if abs(f)<1e-8:break
        eps=1e-6;der=(at(g+eps)[3]-at(g-eps)[3])/(2*eps)
        if abs(der)<1e-8:raise RuntimeError('PD crossing derivative too small')
        step=float(np.clip(-f/der,-.008,.008));gg=g+step;ff=at(gg)[3]
        if abs(ff)>abs(f):
            for _ in range(6):
                step*=.5;gg=g+step;ff=at(gg)[3]
                if abs(ff)<abs(f):break
        g,f=gg,ff
    q,t,err,test,res,v=at(g)
    if abs(test)>1e-7:raise RuntimeError(('PD critical residual',test))
    row=dict(h=h,g=g,T_ms=float(np.exp(q[-2])),N=N,orbit_residual=err,
             critical_residual=abs(test),null_residual=res,
             criterion='Antiperiodic variational eigenvalue = 1, equivalent to Floquet multiplier -1',
             critical_mode_percent=(100*(v.reshape(N,6)**2).sum(0)).tolist())
    return q,t,row


def correct_pd_bordered(h,z,N):
    """Smooth minimally augmented PD test, including nearby complex K modes.

    The zero of the bordered scalar still means (I-K)v=0. Unlike following
    a real eigenvalue away from the boundary it survives eigenvalue collisions.
    """
    normal=np.zeros_like(z);normal[-1]=1
    chart=Chart(System(h),z,normal,N,.01);s=chart.s;cache={};border=None
    def at(g):
        nonlocal border
        key=float(g)
        if key in cache:return cache[key]
        old=min(cache,key=lambda k:abs(k-key)) if cache else None
        guess=cache[old][0].copy() if old is not None else z.copy();guess[-1]=key/.01
        q,t,err,*_=chart.solve(guess,(key-z[-1]*.01)/.01)
        K=build(s,q[:-2].reshape(N,6)*.01,np.exp(q[-2]),key);dim=6*N
        if border is None:
            vals,vec=eigs(K,k=48,which='LM',ncv=100,tol=1e-10,maxiter=700)
            vals2,vec2=eigs(K.T,k=48,which='LM',ncv=100,tol=1e-10,maxiter=700)
            ix=np.argmin(abs(vals-1));jx=np.argmin(abs(vals2-vals[ix]))
            right=vec[:,ix].real;right/=np.linalg.norm(right)
            left=vec2[:,jx].real;left/=np.linalg.norm(left)
            border=right,left
        right,left=border
        def mv(x):return np.r_[x[:-1]-K@x[:-1]+left*x[-1],right@x[:-1]]
        B=LinearOperator((dim+1,dim+1),matvec=mv,dtype=float)
        rhs=np.zeros(dim+1);rhs[-1]=1
        sol,info=gmres(B,rhs,rtol=1e-10,atol=1e-12,restart=200,maxiter=20)
        if np.linalg.norm(B@sol-rhs)>1e-7:raise RuntimeError('PD bordered linear residual')
        cache[key]=(q,t,err,float(sol[-1]),sol[:-1],K)
        return cache[key]
    g=float(z[-1]*.01);f=at(g)[3]
    for it in range(16):
        if abs(f)<1e-9:break
        eps=2e-7;der=(at(g+eps)[3]-at(g-eps)[3])/(2*eps)
        step=float(np.clip(-f/der,-.005,.005));gg=g+step;ff=at(gg)[3]
        for _ in range(9):
            if abs(ff)<abs(f):break
            step*=.5;gg=g+step;ff=at(gg)[3]
        else:raise RuntimeError(('PD bordered scalar line search',g,f))
        g,f=gg,ff
    q,t,err,test,v,K=at(g);v/=np.linalg.norm(v)
    res=float(abs(K@v-v).max())
    if abs(test)>1e-7 or res>1e-7:raise RuntimeError(('PD bordered residual',test,res))
    row=dict(h=h,g=g,T_ms=float(np.exp(q[-2])),N=N,orbit_residual=err,
             critical_residual=abs(test),null_residual=res,
             criterion='Bordered antiperiodic null condition; Floquet multiplier -1',
             critical_mode_percent=(100*(v.reshape(N,6)**2).sum(0)).tolist())
    return q,t,row


def run(label,N=512,step=.05,initial=None,start_h=1.,direction=-1.,suffix='',bordered=False,min_step=.0002):
    folder=OUT/'boundaries'/(label+suffix);folder.mkdir(parents=True,exist_ok=True)
    z,t=load_seed(initial or seeds()[label],N);h=start_h;rows=[];failures=[]
    pd_correct=correct_pd_bordered if bordered else correct_pd
    correct=(lambda hh,zz,tt:pd_correct(hh,zz,N)) if label.startswith('PD') else (lambda hh,zz,tt:correct_fold(hh,zz,tt,N))
    q,tt,row=correct(h,z,t);z,t=q,tt
    todo=[h];current_step=step
    while True:
        row.update(label=label,status='REFINED',sigma_A_mV=System(h).original_std_A*h)
        target=folder/f'h{h:.8f}_N{N}.npz'
        np.savez_compressed(target,r=z[:-2].reshape(N,6)*.01,T=np.exp(z[-2]),g=z[-1]*.01,
                            h=h,tangent=t,N=N,residual=row['orbit_residual'])
        row['source']=str(target.relative_to(ROOT));rows.append(row)
        write(folder/f'curve_N{N}.json',dict(label=label,points=rows,failures=failures,status='RUNNING'))
        print(label,'h',h,'g',row['g'],'T',row['T_ms'],'res',row['critical_residual'],flush=True)
        if (direction<0 and h<=1e-10) or (direction>0 and h>=1-1e-10):break
        derivatives=None
        try:derivatives=orbit_derivative(h,z,N) if label.startswith('PD') else fold_derivative(h,z,t,N)
        except Exception as exc:print(label,'PREDICTOR_FAILED',str(exc)[:160],flush=True)
        while True:
            next_h=float(np.clip(h+direction*current_step,0,1))
            try:
                guess=z if derivatives is None else z+(next_h-h)*derivatives[0]
                tangent=t if derivatives is None else t+(next_h-h)*derivatives[1]
                tangent=tangent/np.sqrt(metric(tangent,tangent,N))
                q,tt,newrow=correct(next_h,guess,tangent)
                if not .4<newrow['g']<1.8:raise RuntimeError('Outside requested parameter neighborhood')
                if abs(newrow['g']-row['g'])>.05 or abs(np.log(newrow['T_ms']/row['T_ms']))>.4:
                    raise RuntimeError('Critical branch step too large')
                z,t,row,h=q,tt,newrow,next_h;current_step=min(step,current_step*1.4);break
            except Exception as exc:
                failures.append(dict(h=next_h,step=current_step,error=str(exc)[:400]));current_step/=2
                write(folder/f'progress_N{N}.json',dict(last_h=h,next_h=next_h,error=str(exc),step=current_step))
                print(label,'RETRY',next_h,current_step,str(exc)[:180],flush=True)
                if current_step<min_step:
                    write(folder/f'curve_N{N}.json',dict(label=label,points=rows,failures=failures,status='PARTIAL_UNRESOLVED_ENDPOINT'))
                    return rows
    write(folder/f'curve_N{N}.json',dict(label=label,points=rows,failures=failures,status='DOMAIN_COVERED'))
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label',choices=list(seeds()));p.add_argument('--N',type=int,default=512);p.add_argument('--step',type=float,default=.05)
    p.add_argument('--initial');p.add_argument('--start-h',type=float,default=1.);p.add_argument('--direction',type=float,default=-1.)
    p.add_argument('--suffix',default='');p.add_argument('--bordered',action='store_true');p.add_argument('--min-step',type=float,default=.0002)
    a=p.parse_args();run(a.label,a.N,a.step,a.initial,a.start_h,a.direction,a.suffix,a.bordered,a.min_step)
