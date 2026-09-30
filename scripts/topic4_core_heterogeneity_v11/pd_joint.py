"""Joint orbital/PD Newton correction with a minimally augmented scalar."""
from periodic_boundaries import *


def correct(h,z,N):
    q=z.copy();normal=np.zeros_like(q);normal[-1]=1
    chart=Chart(System(h),z,normal,N,.01);s=chart.s
    def anti(q):return build(s,q[:-2].reshape(N,6)*.01,np.exp(q[-2]),q[-1]*.01)
    K=anti(q);dim=6*N
    vals,vec=eigs(K,k=48,which='LM',ncv=100,tol=1e-10,maxiter=700)
    vals2,vec2=eigs(K.T,k=48,which='LM',ncv=100,tol=1e-10,maxiter=700)
    ix=np.argmin(abs(vals-1));jx=np.argmin(abs(vals2-vals[ix]))
    right=vec[:,ix].real;right/=np.linalg.norm(right)
    left=vec2[:,jx].real;left/=np.linalg.norm(left)
    def test(q,der=False):
        K=anti(q)
        def mv(x):return np.r_[x[:-1]-K@x[:-1]+left*x[-1],right@x[:-1]]
        def rmv(x):return np.r_[x[:-1]-K.T@x[:-1]+right*x[-1],left@x[:-1]]
        A=LinearOperator((dim+1,dim+1),matvec=mv,rmatvec=rmv,dtype=float)
        rhs=np.zeros(dim+1);rhs[-1]=1
        v,info=gmres(A,rhs,rtol=2e-10,atol=1e-12,restart=250,maxiter=20)
        if np.linalg.norm(A@v-rhs)>1e-7:raise RuntimeError('joint PD null solve')
        if not der:return v[-1]
        w,info=gmres(A.T,rhs,rtol=2e-10,atol=1e-12,restart=250,maxiter=20)
        if np.linalg.norm(A.T@w-rhs)>1e-7:raise RuntimeError('joint PD adjoint solve')
        return v[-1],v[:-1],w[:-1]
    history=[]
    for iteration in range(20):
        F,B,J,b=chart.evaluate(q);F[-1]=0
        scalar,v,w=test(q,True);err=float(abs(F).max());history.append([err,float(scalar)])
        print('PD JOINT',h,iteration,q[-1]*.01,err,scalar,flush=True)
        if err<5e-10 and abs(scalar)<1e-8:break
        pre=chart.preconditioner(B);rhs=np.zeros(len(q));rhs[-1]=1
        u,info=gmres(B,-F,M=pre,rtol=1e-9,atol=1e-12,restart=200,maxiter=15)
        ug,info=gmres(B,rhs,M=pre,rtol=1e-9,atol=1e-12,restart=200,maxiter=15)
        if np.linalg.norm(B@u+F)>1e-7 or np.linalg.norm(B@ug-rhs)>1e-7:raise RuntimeError('joint orbital linear residual')
        def deriv(d):
            eps=min(1e-4,1e-4/max(abs(d).max(),1e-20))
            return float(w@((anti(q+eps*d)@v-anti(q-eps*d)@v)/(2*eps)))
        dg=(-scalar-deriv(u))/deriv(ug);dq=u+dg*ug
        alpha=1.;merit=np.linalg.norm(F)+abs(scalar)
        for _ in range(16):
            trial=q+alpha*dq
            if abs(trial[-2]-q[-2])<.4 and abs(trial[-1]-q[-1])<4:
                Ft=chart.evaluate(trial,jac=False)[:-1];st=test(trial)
                if np.linalg.norm(Ft)+abs(st)<merit:break
            alpha*=.5
        else:raise RuntimeError(('joint PD line search',history))
        q=trial
    else:raise RuntimeError(('joint PD nonconvergence',history))
    v/=np.linalg.norm(v);res=float(abs(anti(q)@v-v).max())
    if res>1e-7:raise RuntimeError(('joint PD null residual',res))
    row=dict(h=h,g=float(q[-1]*.01),T_ms=float(np.exp(q[-2])),N=N,orbit_residual=err,
             critical_residual=float(abs(scalar)),null_residual=res,
             criterion='Joint orbital and bordered antiperiodic null equations; Floquet -1',
             critical_mode_percent=(100*(v.reshape(N,6)**2).sum(0)).tolist())
    return q,normal,row


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--h',type=float,default=.999);p.add_argument('--initial');p.add_argument('--N',type=int,default=1024);p.add_argument('--tag',default='PD2_joint')
    a=p.parse_args();z,t=load_seed(a.initial or seeds()['PD2'],a.N)
    q,t,row=correct(a.h,z,a.N);dest=OUT/'refined_scan'/f'{a.tag}_h{a.h:.8f}_N{a.N}.npz';dest.parent.mkdir(exist_ok=True)
    np.savez_compressed(dest,r=q[:-2].reshape(a.N,6)*.01,T=np.exp(q[-2]),g=q[-1]*.01,h=a.h,tangent=t,N=a.N)
    write(dest.with_suffix('.json'),row);print('PD JOINT REFINED',row,flush=True)
