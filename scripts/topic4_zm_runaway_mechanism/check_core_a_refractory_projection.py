"""Independent convex-QP check and one actual rejected proposal geometry."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.optimize import minimize,LinearConstraint,Bounds
from common import OUT,write,model
from onset_variational_return import Coordinates
from onset_poincare_corrector import SectionReturn
from core_a_refractory_projection import project


def attach_admissible(A):
    A.admissible=lambda x:SectionReturn.admissible(A,x)
    return A


def main():
    folder=OUT/'core_a_bifurcation_type_20260924/numerical_checks/refractory_projection'
    folder.mkdir(exist_ok=True);assert not(folder/'result.json').exists()
    rng=np.random.default_rng(92491);rows=[];P=3;depth=12;N=(47+depth)*P
    E=np.array([True,True,False]);scale=np.ones((47+depth,1));weight=np.ones((1,P))
    c=SimpleNamespace(P=P,scale=scale,weight=weight)
    for trial in range(6):
        x=rng.uniform(.1,.2,N);x.reshape(-1,P)[11:47]=rng.normal(0,.1,(36,P));x.reshape(-1,P)[4,~E]=0
        normal=rng.normal(size=N);normal/=np.linalg.norm(normal)
        A=attach_admissible(SimpleNamespace(c=c,e=SimpleNamespace(s=SimpleNamespace(E=E),dt=.5),normal=normal,xref=x))
        assert A.admissible(x)
        raw=x+rng.normal(0,.6,N);diagnostic={};actual=project(A,raw,x,diagnostics=diagnostic)
        assert actual is not None,diagnostic
        pos=np.zeros((47+depth,P),bool);pos[:11]=True;pos[47:]=True;pos=pos.ravel()
        lo=np.where(pos,0.,-np.inf);hi=np.full(N,np.inf);lo[4*P+2]=hi[4*P+2]=0.
        C=[]
        for g,n in enumerate([4,4,2]):
            for start in range(depth-n+1):
                a=np.zeros(N);a[(47+np.arange(start,start+n))*P+g]=.5;C.append(a)
        C=np.array(C)
        opt=minimize(lambda y:.5*np.sum((y-raw)**2),x,jac=lambda y:y-raw,
            method='SLSQP',bounds=Bounds(lo,hi),constraints=[LinearConstraint(normal[None],normal@x,normal@x),LinearConstraint(C,-np.inf,1.)],
            options={'ftol':1e-12,'maxiter':1000})
        assert opt.success,opt.message
        err=float(np.linalg.norm(actual-opt.x)/max(1.,np.linalg.norm(opt.x)))
        assert err<1e-7,err
        rows.append(dict(trial=trial,relative_to_independent_SLSQP=err,**diagnostic))
        write(folder/'toy_progress.json',rows)
    s=model(40);parent=OUT/'core_a_bifurcation_type_20260924/near_returns/above70s_A_sustained_B_cycle'
    base=dict(np.load(parent/'exact_single_B_cycle/node00.npz'));c=Coordinates(base,s)
    f=parent/'multiple_single_B_cyclicM/iteration01';x=c.pack(dict(np.load(f/'node00.npz')))
    A=attach_admissible(SimpleNamespace(c=c,e=SimpleNamespace(s=s,dt=.05),normal=np.load(f/'phase_normal.npy'),xref=c.pack(base)))
    raw=x+np.load(f/'first_proposal_delta.npy')[:c.size];diagnostic={}
    y=project(A,raw,x,diagnostics=diagnostic)
    assert y is not None,diagnostic
    assert A.admissible(y) and abs(A.normal@(y-A.xref))<1e-9
    np.save(folder/'actual_projected_proposal.npy',y)
    write(folder/'result.json',dict(status='PASS',toy_rows=rows,actual_proposal=diagnostic,
        actual_source=str(f),scope='Independent convex-QP numerical geometry only. No network flow was changed or clipped; a projected initial-state proposal still requires actual nonlinear residual and variational prediction acceptance.',model_promoted=False))
    print('REFRACTORY PROJECTION QA PASS',diagnostic,flush=True)


if __name__=='__main__':main()
