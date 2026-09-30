"""Joint periodic-orbit/cycle-fold Newton correction with a bordered null test."""
from periodic_boundaries import *


def correct(h,z,normal,N):
    q=z.copy();chart=Chart(System(h),z,normal,N,.01);unit=np.zeros(len(q));unit[-1]=1
    def evaluate(q,adj=False):
        F,B,J,b=chart.evaluate(q);F[-1]=0
        pre=chart.preconditioner(B)
        v,info=gmres(B,unit,M=pre,rtol=1e-10,atol=1e-12,restart=200,maxiter=15)
        if np.linalg.norm(B@v-unit)>1e-7:raise RuntimeError('fold joint right solve')
        if not adj:return F,v[-1]
        w,info=gmres(B.T,unit,rtol=1e-10,atol=1e-12,restart=250,maxiter=25)
        if np.linalg.norm(B.T@w-unit)>1e-7:raise RuntimeError('fold joint left solve')
        return F,v[-1],v,w,B,pre
    history=[]
    for iteration in range(20):
        F,scalar,v,w,B,pre=evaluate(q,True);err=float(abs(F).max());history.append([err,float(scalar)])
        print('FOLD JOINT',h,iteration,q[-1]*.01,err,scalar,flush=True)
        if err<5e-10 and abs(scalar)<1e-8:break
        if iteration>=4 and err>1e-7 and err>.995*history[-4][0]:
            raise RuntimeError(('fold joint stalled correction',history))
        u,info=gmres(B,-F,M=pre,rtol=1e-10,atol=1e-12,restart=200,maxiter=15)
        if np.linalg.norm(B@u+F)>1e-7:raise RuntimeError('fold joint particular residual')
        def derivative(d):
            eps=min(1e-4,1e-4/max(abs(d).max(),1e-20))
            return -float(w@((chart.evaluate(q+eps*d)[1]@v-chart.evaluate(q-eps*d)[1]@v)/(2*eps)))
        shift=(-scalar-derivative(u))/derivative(v);dq=u+shift*v
        alpha=1.;merit=np.linalg.norm(F)+abs(scalar)
        for _ in range(14):
            trial=q+alpha*dq
            if abs(trial[-2]-q[-2])<.5 and abs(trial[-1]-q[-1])<5:
                Ft,st=evaluate(trial)
                if np.linalg.norm(Ft)+abs(st)<merit:break
            alpha*=.5
        else:raise RuntimeError(('fold joint line search',history))
        q=trial
    else:raise RuntimeError(('fold joint nonconvergence',history))
    t=v/np.sqrt(metric(v,v,N));J=chart.evaluate(q)[2];res=float(abs(J@t[:-1]).max())
    if res>1e-7:raise RuntimeError(('fold joint null residual',res))
    # A tiny parameter tangent on a flat long-period tail alone is insufficient.
    # Recheck an actual local tangent sign change with the strict corrector.
    q,t,geometry=correct_fold(h,q,t,N)
    row=dict(h=h,g=float(q[-1]*.01),T_ms=float(np.exp(q[-2])),N=N,orbit_residual=err,
             critical_residual=float(abs(scalar)),null_residual=res,
             criterion='Joint orbital and bordered fixed-parameter null equations; cycle fold',
             bracket_tangent=geometry['bracket_tangent'])
    return q,t,row


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--h',type=float,default=.95);p.add_argument('--initial');p.add_argument('--N',type=int,default=1024);p.add_argument('--tag',default='LP1_joint')
    a=p.parse_args();z,t=load_seed(a.initial or seeds()['LP1'],a.N);q,t,row=correct(a.h,z,t,a.N)
    dest=OUT/'refined_scan'/f'{a.tag}_h{a.h:.8f}_N{a.N}.npz';dest.parent.mkdir(exist_ok=True)
    np.savez_compressed(dest,r=q[:-2].reshape(a.N,6)*.01,T=np.exp(q[-2]),g=q[-1]*.01,h=a.h,tangent=t,N=a.N)
    write(dest.with_suffix('.json'),row);print('FOLD JOINT REFINED',row,flush=True)
