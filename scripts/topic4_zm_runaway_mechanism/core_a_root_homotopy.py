"""Numerical pseudo-arclength homotopy to a root of the unchanged network.

The auxiliary homotopy parameter is exclusively a root-solver device. Only
its value one solves the original physical model; no other point is a result
about the network or a bifurcation branch.
"""
from common import OUT,np,read,write,log
from physical_delay_conditional_drift import PhysicalDelayConditionalDrift
from current_rate_logit_root import value_and_jacobian
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.special import logit
import os,time,argparse

DEST=OUT/'core_a_bifurcation_type_20260924/numerical_root_homotopy'
SOURCE=OUT/'core_a_resource_bifurcation_20260923'


def main(seed='mean'):
    global DEST
    if seed!='mean':DEST=DEST.with_name(DEST.name+'_'+seed)
    DEST.mkdir(exist_ok=True);assert not (DEST/'jobs.json').exists()
    s=PhysicalDelayConditionalDrift();Z=np.load(SOURCE/'fields.npz')['coreA10370_background9000'];s.set_Z(Z)
    mean=np.load(SOURCE/'coreA_depleted/block01.npz')['group_rate_hz'].mean(0).astype(float)/1000
    if seed=='instant':mean=np.load(SOURCE/'coreA_depleted/final_state.npz')['rate'].copy()
    if seed=='lowB':mean[s.E&(s.geo['group_region']==1)]=.001
    q0=logit(np.clip(mean*s.ref,1e-9,1-1e-9));P=s.P;eye=sparse.eye(P,format='csc');ws=4.
    write(DEST/'contract.json',dict(question='Construct a stationary root at the SAME local Z endpoint after direct Newton stalls.',
        method='Auxiliary H(q,a)=(1-a)*(q0-q)+a*F(q), F the exact physical logit equilibrium residual. Start at a=0,q=q0; pseudo-arclength continuation follows H=0 through numerical turning points toward a=1. No auxiliary point is a physical equilibrium.',
        unchanged='Original physical graph, response, Z field and M equilibrium law. At a=1 independently require original rate residual<1e-11/ms, then verify actual flow before any dynamics claim.',numerical_initial_guess=seed,
        limits='At most160 accepted points/250trials, 14Newton corrections per point. This is solver construction, not a physical parameter campaign; auxiliary folds never receive SN labels.',model_promoted=False))
    def evaluate(q,a,jac=True):
        value=value_and_jacobian(s,q,jac);r,F,op=value[:3]
        H=(1-a)*(q0-q)+a*F
        if not jac:return H
        return H,(a*value[3]-(1-a)*eye).tocsc(),F-(q0-q)
    def tangent(q,a,previous=None):
        H,J,col=evaluate(q,a)
        if previous is None:
            v=np.r_[spsolve(J,-col),1.]
        else:
            row=np.r_[previous[:-1]/P,ws*ws*previous[-1]]
            B=sparse.vstack([sparse.hstack([J,sparse.csc_matrix(col[:,None])]),sparse.csc_matrix(row[None,:])],format='csc')
            v=spsolve(B,np.r_[np.zeros(P),1.])
        v/=np.sqrt(v[:-1]@v[:-1]/P+(ws*v[-1])**2)
        return v
    y=np.r_[q0,0.];v=tangent(q0,0.);ds=.08;rows=[];trials=[];start=time.time();status='TARGET_NOT_REACHED'
    for trial in range(250):
        predicted=y+ds*v;x=predicted.copy();success=False;trace=[]
        arcrow=np.r_[v[:-1]/P,ws*ws*v[-1]]
        for it in range(14):
            H,J,col=evaluate(x[:-1],x[-1]);arc=float(arcrow@(x-predicted))
            err=float(np.sqrt(H@H/P+arc*arc));trace.append(err)
            if max(float(abs(H).max()),abs(arc))<2e-9:success=True;break
            B=sparse.vstack([sparse.hstack([J,sparse.csc_matrix(col[:,None])]),sparse.csc_matrix(arcrow[None,:])],format='csc')
            delta=spsolve(B,-np.r_[H,arc]);alpha=min(1.,4/max(abs(delta[:-1]).max(),1e-12))
            for back in range(18):
                nxt=x+alpha*delta;hh=evaluate(nxt[:-1],nxt[-1],False);aa=float(arcrow@(nxt-predicted))
                if np.sqrt(hh@hh/P+aa*aa)<err:x=nxt;break
                alpha*=.5
            else:break
        record=dict(trial=trial,accepted=success,auxiliary_a=float(x[-1]),step=ds,newton_errors=trace)
        trials.append(record);write(DEST/'trials.json',trials)
        if not success:
            ds*=.5
            if ds<1e-4:status='NUMERICAL_MIN_STEP';break
            continue
        previous=y.copy();y=x;v=tangent(y[:-1],y[-1],v)
        row=dict(index=len(rows),auxiliary_a=float(y[-1]),iterations=len(trace),H_infinity=float(abs(evaluate(y[:-1],y[-1],False)).max()),seconds=time.time()-start)
        rows.append(row);write(DEST/'accepted.json',rows);write(DEST/'jobs.json',dict(status='RUNNING',pid=os.getpid(),latest=row))
        np.savez_compressed(DEST/'latest_auxiliary.npz',q=y[:-1],a=y[-1],tangent=v,Z=Z,q0=q0)
        log('CORE A NUMERICAL HOMOTOPY',row)
        if (previous[-1]-1)*(y[-1]-1)<=0:
            a=(1-previous[-1])/(y[-1]-previous[-1]);q=(1-a)*previous[:-1]+a*y[:-1]
            for _ in range(15):
                r,F,op,J=value_and_jacobian(s,q)
                if abs(op['rate']-r).max()<1e-11:break
                step=spsolve(J,-F);norm=np.linalg.norm(F)
                for alpha in 2.**-np.arange(20):
                    qt=q+alpha*step
                    if np.linalg.norm(value_and_jacobian(s,qt,False)[1])<norm:q=qt;break
                else:break
            r,F,op=value_and_jacobian(s,q,False);error=float(abs(s.residual(r)).max())
            np.savez_compressed(DEST/'physical_candidate.npz',q=q,r=r,Z=Z)
            if error<1e-11:status='ROOT_PASS';break
        if len(rows)>=160:break
        ds=min(.4,ds*(1.4 if len(trace)<=4 else (1.1 if len(trace)<=7 else .7)))
    result=dict(status=status,auxiliary_endpoint=float(y[-1]),accepted_points=len(rows),trials=len(trials),model_promoted=False,stability='NOT_COMPUTED')
    if status=='ROOT_PASS':result.update(residual_per_ms=error,global_rate_hz=s.global_rate(r),regional_rates_hz=s.regional_rates(r))
    write(DEST/'result.json',result);write(DEST/'jobs.json',dict(status='COMPLETE',pid=os.getpid(),scientific_result=status));log('CORE A HOMOTOPY COMPLETE',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',choices=['mean','instant','lowB'],default='mean');a=p.parse_args();main(a.seed)
